"""Deterministic guard that keeps model-written prose tied to verified figures.

Every number in a sentence must match (within its own rounding precision) a number present in the
verified context the model was given; sentences citing anything else are removed. Scale words are
reconciled with the $-millions convention used across the pipeline ("$716.9 billion" == 716,924).
Small integers (counts, "5-year", item numbers) and calendar years are not treated as figures.

The same matcher attributes verified figures to their sources: `build_figure_index` records which source
each known figure came from, and `annotate_sentences` appends a citation marker to each sentence.
"""

import bisect
import re
from dataclasses import dataclass
from typing import Callable, Generic, Hashable, Iterable, List, Optional, Tuple, TypeVar

_FIGURE = re.compile(
    r"(?<![\w.])\$?(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?\s*"
    r"(%|x\b|bps\b|pp\b|trillion\b|billion\b|million\b|thousand\b|[TBMK]\b)?",
    re.IGNORECASE,
)
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z*\"(\[])")
_LINE_PREFIX = re.compile(r"^(\s*(?:[-*+]|\d+\.)\s+|\s*)")
_SCALE_TO_MILLIONS = {"trillion": 1e6, "t": 1e6, "billion": 1e3, "b": 1e3, "million": 1.0, "m": 1.0, "thousand": 1e-3, "k": 1e-3}
RELATIVE_TOLERANCE = 0.002
SMALL_INTEGER_MAX = 31


def _parse(match: re.Match) -> Tuple[float, int, str, bool]:
    whole, frac, unit = match.group(1), match.group(2) or "", (match.group(3) or "").lower()
    value = float(whole.replace(",", "") + frac)
    decimals = len(frac) - 1 if frac else 0
    return value, decimals, unit, match.group(0).startswith("$")


def collect_known_figures(context: str) -> List[float]:
    """Sorted absolute values of every number in the verified context, in $ millions when a scale word is given."""
    known = set()
    for match in _FIGURE.finditer(context or ""):
        value, _, unit, _ = _parse(match)
        known.add(value * _SCALE_TO_MILLIONS.get(unit, 1.0))
    return sorted(known)


def _is_figure(value: float, decimals: int, unit: str, dollar: bool) -> bool:
    if unit or dollar or decimals:
        return True
    if 1900 <= value <= 2100:
        return False
    return value > SMALL_INTEGER_MAX


def _window(value: float, decimals: int, unit: str) -> Tuple[float, float]:
    """Inclusive $-millions interval a figure written with `decimals` precision can stand for."""
    scale = _SCALE_TO_MILLIONS.get(unit, 1.0)
    candidate = value * scale
    tolerance = 0.5 * 10 ** (-decimals) * scale + RELATIVE_TOLERANCE * candidate
    return candidate - tolerance, candidate + tolerance


def _matches(known: List[float], value: float, decimals: int, unit: str) -> bool:
    lo, hi = _window(value, decimals, unit)
    i = bisect.bisect_left(known, lo)
    return i < len(known) and known[i] <= hi


def sentence_is_verified(sentence: str, known: List[float]) -> bool:
    for match in _FIGURE.finditer(sentence):
        value, decimals, unit, dollar = _parse(match)
        if _is_figure(value, decimals, unit, dollar) and not _matches(known, value, decimals, unit):
            return False
    return True


def strip_unverified_sentences(text: str, known: List[float]) -> Tuple[str, int]:
    """Removes sentences containing figures absent from `known`; returns the cleaned text and removal count."""
    if not text:
        return text, 0
    removed = 0
    out_lines: List[str] = []
    for line in text.split("\n"):
        prefix = _LINE_PREFIX.match(line).group(1)
        body = line[len(prefix):]
        kept = []
        for sentence in _SENTENCE_BOUNDARY.split(body):
            if sentence_is_verified(sentence, known):
                kept.append(sentence)
            else:
                removed += 1
        if kept:
            out_lines.append(prefix + " ".join(kept))
        elif not body.strip():
            out_lines.append(line)
    return "\n".join(out_lines).strip(), removed


K = TypeVar("K", bound=Optional[Hashable])
_TRAILING_PUNCT = re.compile(r"[.!?]+[\"')\]*_]*\s*$")


@dataclass
class FigureIndex(Generic[K]):
    """Known figures (in $ millions when scaled) with their source and whether they are percentages.

    Lower priority wins. A `None` source marks a known figure that has no filing passage (e.g. WACC,
    terminal growth): it still wins its window, so it blocks accidental matches with unrelated rows.
    """

    values: List[float]
    entries: List[Tuple[float, int, K, bool]]


def build_figure_index(entries: Iterable[Tuple[str, K]]) -> FigureIndex[K]:
    """Indexes every figure in each (text, source) entry; the entry's position is its priority."""
    rows: List[Tuple[float, int, K, bool]] = []
    for priority, (text, key) in enumerate(entries):
        for match in _FIGURE.finditer(text or ""):
            value, _, unit, _ = _parse(match)
            rows.append((value * _SCALE_TO_MILLIONS.get(unit, 1.0), priority, key, unit == "%"))
    rows.sort(key=lambda row: (row[0], row[1]))
    return FigureIndex(values=[row[0] for row in rows], entries=rows)


def attribute_sentence(sentence: str, index: FigureIndex[K], limit: int = 3) -> List[K]:
    """Best (highest-priority) source for each figure in `sentence`, in sentence order, at most `limit`.

    Percentages only match percentage entries and amounts only match amounts; a figure whose best match
    is an unciteable (`None`) entry gets no source.
    """
    found: List[K] = []
    for match in _FIGURE.finditer(sentence):
        value, decimals, unit, dollar = _parse(match)
        if not _is_figure(value, decimals, unit, dollar):
            continue
        lo, hi = _window(value, decimals, unit)
        is_pct = unit == "%"
        window = [
            row for row in index.entries[bisect.bisect_left(index.values, lo):bisect.bisect_right(index.values, hi)]
            if row[3] == is_pct
        ]
        if not window:
            continue
        best = min(window, key=lambda row: row[1])[2]
        if best is not None and best not in found:
            found.append(best)
        if len(found) >= limit:
            break
    return found


def _attach(sentence: str, mark: str) -> str:
    """Places `mark` before the sentence's closing punctuation (and any closing emphasis/quotes)."""
    m = _TRAILING_PUNCT.search(sentence)
    if m:
        return sentence[:m.start()] + mark + sentence[m.start():]
    return sentence.rstrip() + mark


def annotate_sentences(text: str, index: FigureIndex[K], render: Callable[[List[K]], str]) -> str:
    """Appends `render(sources)` to every sentence whose figures are attributable; other text is unchanged."""
    if not text:
        return text
    out_lines: List[str] = []
    for line in text.split("\n"):
        prefix = _LINE_PREFIX.match(line).group(1)
        body = line[len(prefix):]
        sentences = []
        for sentence in _SENTENCE_BOUNDARY.split(body):
            keys = attribute_sentence(sentence, index)
            mark = render(keys) if keys else ""
            sentences.append(_attach(sentence, mark) if mark else sentence)
        out_lines.append(prefix + " ".join(sentences))
    return "\n".join(out_lines)
