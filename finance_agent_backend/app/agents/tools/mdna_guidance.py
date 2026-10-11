"""Deterministic Item 7 (MD&A) guidance screen for the Forecasting Analyst.

Counts forward-looking sentences (excluding safe-harbor and disclosure boilerplate) and extracts the
quantified ones. The agent only asks the LLM for a guided forecast when the filing has at least one
quantified outlook statement among enough forward-looking sentences.

Quantified guidance must be an annual or long-range outlook for an operating driver (revenue, growth,
margins, operating costs, CapEx, volumes). These never count, whatever numbers they carry:
- hedging boilerplate ("results are inherently unpredictable and may be materially affected by many
  factors ..."), which is excluded from the forward-looking count as well;
- foreign-exchange impact notes ("guidance anticipates a favorable impact of approximately 180 basis
  points from foreign exchange rates");
- single-quarter guidance ("first quarter 2026 ... net sales are expected to be between ...").

Sentences are split on terminal punctuation (also when followed by a closing quote or parenthesis), on
blank lines and on bullets, so a heading or bullet list cannot glue boilerplate onto the next statement.
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
    r"actual results (?:could|may) differ|undue reliance|inherently (?:unpredictable|uncertain)|"
    r"may be materially (?:affected|impacted)|(?:many|numerous|a number of|various) factors|"
    r"subject to (?:substantial|significant) (?:risks|uncertaint)|risks and uncertainties",
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
# An operating driver the forecast actually uses (revenue, growth, margins, costs, CapEx, volumes).
OPERATING_DRIVER = re.compile(
    r"\b(?:revenues?|net sales|sales|grow(?:th)?|margins?|operating (?:income|expenses?|costs?|profit|loss)|"
    r"gross profit|cost of (?:sales|revenues?|goods sold)|capital (?:expenditures?|spending|investments?)|capex|"
    r"research and development|deliveries|shipments|production|capacity|units?)\b",
    re.IGNORECASE,
)
FX_IMPACT = re.compile(r"foreign (?:exchange|currenc)|exchange rates?|currency (?:translation|fluctuation|movements?)",
                       re.IGNORECASE)
SINGLE_QUARTER = re.compile(
    r"\b(?:first|second|third|fourth|current|next|coming|this|the) (?:fiscal )?quarter\b|\bQ[1-4]\b|"
    r"\bquarter ending\b|\bthree months ending\b",
    re.IGNORECASE,
)
MARGIN_LANGUAGE = re.compile(
    r"\bmargins?\b|operating (?:income|expenses?|costs?|leverage|profit)|gross profit|"
    r"cost of (?:sales|revenues?|goods sold)|\bpricing\b|price (?:increases?|reductions?)|"
    r"cost (?:reductions?|savings|efficienc\w*|pressures?)|efficienc(?:y|ies)|restructuring|\bmix\b|"
    r"depreciation|input costs",
    re.IGNORECASE,
)
SENTENCE_SPLIT = re.compile(
    r"(?:(?<=[.!?])|(?<=[.!?][\"'”’)\]]))\s+(?=[\"'“‘(\[]?[A-Z])|\n\s*\n|\s*[•●▪]\s*"
)


class GuidanceSentence(BaseModel):
    chunk_id: str
    sentence: str


class MdnaAssessment(BaseModel):
    chunk_count: int = 0
    forward_looking_count: int = 0
    quantified_guidance: List[GuidanceSentence] = Field(default_factory=list)
    is_informative: bool = False


def split_sentences(text: str) -> List[str]:
    """Whitespace-normalized sentences; blank lines and bullets are hard boundaries."""
    return [s for s in (" ".join(raw.split()) for raw in SENTENCE_SPLIT.split(text or "")) if s]


def is_outlook_sentence(sentence: str) -> bool:
    """Forward-looking and not safe-harbor or disclosure boilerplate."""
    return (
        bool(FORWARD_LOOKING.search(sentence))
        and not BOILERPLATE.search(sentence)
        and not NOT_OUTLOOK.search(sentence)
    )


def is_quantified_guidance(sentence: str) -> bool:
    """An outlook sentence with a figure about an operating driver; FX-impact and one-quarter notes excluded."""
    return (
        is_outlook_sentence(sentence)
        and bool(QUANTIFIED.search(sentence))
        and bool(OPERATING_DRIVER.search(sentence))
        and not FX_IMPACT.search(sentence)
        and not SINGLE_QUARTER.search(sentence)
    )


def assess_mdna_guidance(chunks: Sequence) -> MdnaAssessment:
    forward = 0
    quantified: List[GuidanceSentence] = []
    seen: Set[str] = set()
    for chunk in chunks:
        for sentence in split_sentences(chunk.content):
            if sentence in seen or not is_outlook_sentence(sentence):
                continue
            seen.add(sentence)
            forward += 1
            if is_quantified_guidance(sentence) and len(quantified) < MAX_GUIDANCE_SENTENCES:
                quantified.append(GuidanceSentence(chunk_id=chunk.chunk_id, sentence=sentence[:MAX_SENTENCE_CHARS]))
    return MdnaAssessment(
        chunk_count=len(chunks),
        forward_looking_count=forward,
        quantified_guidance=quantified,
        is_informative=forward >= MIN_FORWARD_LOOKING_SENTENCES and bool(quantified),
    )


def margin_evidence_chunk_ids(chunks: Sequence) -> Set[str]:
    """Chunks with at least one forward-looking (non-boilerplate) sentence about margins or costs.

    Only these can lift the margin-trend corridor in `screen_forecast_drivers`.
    """
    return {
        chunk.chunk_id
        for chunk in chunks
        if any(is_outlook_sentence(s) and MARGIN_LANGUAGE.search(s) for s in split_sentences(chunk.content))
    }
