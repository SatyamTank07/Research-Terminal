"""Deterministic guard that keeps model-written prose tied to verified figures.

Every number in a sentence must match (within its own rounding precision) a number present in the
verified context the model was given; sentences citing anything else are removed. Scale words are
reconciled with the $-millions convention used across the pipeline ("$716.9 billion" == 716,924).
Small integers (counts, "5-year", item numbers) and calendar years are not treated as figures.
"""

import bisect
import re
from typing import List, Tuple

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


def _matches(known: List[float], value: float, decimals: int, unit: str) -> bool:
    scale = _SCALE_TO_MILLIONS.get(unit, 1.0)
    candidate = value * scale
    tolerance = 0.5 * 10 ** (-decimals) * scale + RELATIVE_TOLERANCE * candidate
    i = bisect.bisect_left(known, candidate - tolerance)
    return i < len(known) and known[i] <= candidate + tolerance


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
