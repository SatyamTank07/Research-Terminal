"""Result types of the management-guidance module."""

from typing import List

from pydantic import BaseModel, Field


class GuidanceSentence(BaseModel):
    chunk_id: str
    sentence: str


class MdnaAssessment(BaseModel):
    chunk_count: int = 0
    forward_looking_count: int = 0
    quantified_guidance: List[GuidanceSentence] = Field(default_factory=list)
    is_informative: bool = False
