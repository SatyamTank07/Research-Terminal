"""Checks that an LLM-proposed quote and its numbers really appear in the filing chunks it cites."""

import re
from collections import defaultdict
from typing import Any, List, Mapping, Optional, Sequence

from app.agents.shared.filing.sentences import normalize_text, split_words

QUOTE_FUZZY_COVERAGE = 0.85
MIN_QUOTE_WORDS = 4

_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")


def verify_quote(quote: str, chunk_text: str, allow_fuzzy: bool = True) -> bool:
    """True if `quote` appears in `chunk_text` verbatim (after normalization) or near-verbatim.

    Ellipses split the quote into segments that must each match. With `allow_fuzzy`, a segment
    that is not an exact substring passes if >= 85% of its words appear in a same-length window
    of the chunk.
    """
    if not isinstance(quote, str) or not isinstance(chunk_text, str):
        return False
    haystack = normalize_text(chunk_text)
    segments = [s.strip(" .,;:\"'") for s in re.split(r"\.\.\.|…|\[\.\.\.\]", normalize_text(quote))]
    segments = [s for s in segments if s]
    if not segments or sum(len(s.split()) for s in segments) < MIN_QUOTE_WORDS:
        return False
    hay_words = split_words(haystack)
    for segment in segments:
        if segment in haystack:
            continue
        if not allow_fuzzy:
            return False
        seg_words = split_words(segment)
        if not seg_words or not _fuzzy_window_match(seg_words, hay_words):
            return False
    return True


def _fuzzy_window_match(needle: List[str], hay: List[str]) -> bool:
    """Near-verbatim match: >= 85% of words in a same-length window, and every number exact."""
    size = len(needle)
    if size > len(hay):
        return False
    target = defaultdict(int)
    for w in needle:
        target[w] += 1
    numbers = [w for w in target if any(ch.isdigit() for ch in w)]
    for start in range(0, len(hay) - size + 1):
        window = defaultdict(int)
        for w in hay[start:start + size + 2]:
            window[w] += 1
        if any(window.get(n, 0) < target[n] for n in numbers):
            continue
        hit = sum(min(cnt, window.get(w, 0)) for w, cnt in target.items())
        if hit / size >= QUOTE_FUZZY_COVERAGE:
            return True
    return False


def _canonical_number(token: str) -> str:
    return token.replace(",", "").rstrip("0").rstrip(".") if "." in token else token.replace(",", "")


def numbers_supported(text: Optional[str], evidence_texts: Sequence[str]) -> bool:
    """True if every number in `text` also appears in the evidence (commas / trailing zeros ignored)."""
    if not text:
        return True
    claimed = {_canonical_number(n) for n in _NUMBER.findall(text)}
    if not claimed:
        return True
    available = {_canonical_number(n) for t in evidence_texts for n in _NUMBER.findall(t or "")}
    return claimed <= available


def find_quote_chunk(
    quote: str, claimed_chunk_id: str, pool: Mapping[str, Mapping[str, Any]]
) -> Optional[Mapping[str, Any]]:
    """The pooled chunk that holds `quote`: the claimed chunk (near-verbatim allowed), else the first
    other chunk that holds it exactly."""
    claimed = pool.get(claimed_chunk_id)
    if claimed is not None and verify_quote(quote, claimed["content"]):
        return claimed
    return next(
        (c for cid, c in pool.items() if cid != claimed_chunk_id and verify_quote(quote, c["content"], allow_fuzzy=False)),
        None,
    )


__all__ = ["MIN_QUOTE_WORDS", "QUOTE_FUZZY_COVERAGE", "find_quote_chunk", "numbers_supported", "verify_quote"]
