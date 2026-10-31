"""Text normalization and sentence / word splitting for filing text."""

import re
from typing import List

MIN_SENTENCE_CHARS = 60

_SENTENCE_SPLIT = re.compile(r"(?<=[.;:!?])\s+(?=[A-Z•\"“(])|\s*•\s*")
_WORD = re.compile(r"[a-z0-9]+")
_CHAR_MAP = str.maketrans({
    "‘": "'", "’": "'", "‛": "'", "′": "'",
    "“": '"', "”": '"', "‟": '"', "″": '"',
    "–": "-", "—": "-", "−": "-", " ": " ",
})


def normalize_text(text: str) -> str:
    """Lowercases, maps curly quotes/dashes to ASCII and collapses whitespace."""
    return re.sub(r"\s+", " ", (text or "").translate(_CHAR_MAP)).strip().lower()


def split_words(text: str) -> List[str]:
    """Lowercase alphanumeric words of the normalized text (normalizing twice is a no-op)."""
    return _WORD.findall(normalize_text(text))


def split_sentences(text: str) -> List[str]:
    """Splits disclosure text into sentences / bullets, keeping substantive ones only."""
    flat = re.sub(r"\s+", " ", text or "").strip()
    return [s.strip() for s in _SENTENCE_SPLIT.split(flat) if len(s.strip()) >= MIN_SENTENCE_CHARS]


__all__ = ["MIN_SENTENCE_CHARS", "normalize_text", "split_sentences", "split_words"]
