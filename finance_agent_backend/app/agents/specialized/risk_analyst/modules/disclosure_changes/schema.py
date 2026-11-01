"""The year-over-year Item 1A comparison and its new or removed passages."""

from typing import List

from pydantic import BaseModel, Field


class DisclosurePassage(BaseModel):
    """A contiguous run of new (current year) or removed (prior year) Item 1A sentences."""

    chunk_id: str = Field(..., description="Chunk the passage came from (current filing for new, prior filing for removed)")
    text: str = Field(..., description="Passage text, trimmed")
    similarity: float = Field(..., description="Best shingle-Jaccard similarity to the other year's sentences")


class RiskDisclosureDiff(BaseModel):
    """Sentence-level year-over-year comparison of Item 1A disclosures."""

    prior_fiscal_year: int
    current_sentences: int
    prior_sentences: int
    unchanged_sentences: int
    modified_sentences: int
    new_sentences: int
    removed_sentences: int
    change_ratio_pct: float = Field(..., description="(new + modified) / current sentences, in percent")
    new_passages: List[DisclosurePassage] = Field(default_factory=list)
    removed_passages: List[DisclosurePassage] = Field(default_factory=list)
