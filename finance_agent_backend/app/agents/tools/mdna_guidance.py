"""Deterministic Item 7 (MD&A) guidance screen for the Forecasting Analyst.

Counts forward-looking sentences (excluding safe-harbor and disclosure boilerplate) and extracts the
quantified ones. The agent only asks the LLM for a guided forecast when the filing has at least one
quantified outlook statement among enough forward-looking sentences.
"""

import re
from typing import List, Sequence, Set

from pydantic import BaseModel, Field

MIN_FORWARD_LOOKING_SENTENCES = 2
MAX_GUIDANCE_SENTENCES = 8
MAX_SENTENCE_CHARS = 400

FORWARD_LOOKING = re.compile(
    r"\b(?:expects?|expected to|anticipates?|anticipated|outlook|(?<!accounting )guidance|intends? to|"
    r"plans? to|will (?:continue|increase|decrease|grow|decline|remain)|forecasts?|going forward|"
    r"over the next)\b",
    re.IGNORECASE,
)
BOILERPLATE = re.compile(
    r"forward-looking statements|safe harbor|Private Securities Litigation Reform Act|"
    r"actual results (?:could|may) differ|undue reliance",
    re.IGNORECASE,
)
# Forward-looking phrasing that is disclosure boilerplate, not an operating outlook.
NOT_OUTLOOK = re.compile(
    r"expected to be recognized|unrecognized compensation|remaining performance obligations|"
    r"(?:will be|are) sufficient|sufficient to (?:meet|fund)|hypothetical|pension|postretirement|"
    r"(?:issued|accounting|new|recent) guidance|(?:stable|negative|positive|developing) outlook|"
    r"credit rating|expected to adopt",
    re.IGNORECASE,
)
QUANTIFIED = re.compile(r"\$\s?\d|\d+(?:\.\d+)?\s?(?:%|percent|basis points|billion|million)", re.IGNORECASE)
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[])")


class GuidanceSentence(BaseModel):
    chunk_id: str
    sentence: str


class MdnaAssessment(BaseModel):
    chunk_count: int = 0
    forward_looking_count: int = 0
    quantified_guidance: List[GuidanceSentence] = Field(default_factory=list)
    is_informative: bool = False


def assess_mdna_guidance(chunks: Sequence) -> MdnaAssessment:
    forward = 0
    quantified: List[GuidanceSentence] = []
    seen: Set[str] = set()
    for chunk in chunks:
        for raw in SENTENCE_SPLIT.split(chunk.content or ""):
            sentence = " ".join(raw.split())
            if (
                not sentence
                or sentence in seen
                or BOILERPLATE.search(sentence)
                or NOT_OUTLOOK.search(sentence)
                or not FORWARD_LOOKING.search(sentence)
            ):
                continue
            seen.add(sentence)
            forward += 1
            if QUANTIFIED.search(sentence) and len(quantified) < MAX_GUIDANCE_SENTENCES:
                quantified.append(GuidanceSentence(chunk_id=chunk.chunk_id, sentence=sentence[:MAX_SENTENCE_CHARS]))
    return MdnaAssessment(
        chunk_count=len(chunks),
        forward_looking_count=forward,
        quantified_guidance=quantified,
        is_informative=forward >= MIN_FORWARD_LOOKING_SENTENCES and bool(quantified),
    )
