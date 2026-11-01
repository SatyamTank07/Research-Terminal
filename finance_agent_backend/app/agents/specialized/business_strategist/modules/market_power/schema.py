"""Filing sentences on customer concentration and pricing, found by code outside Item 1."""

from typing import Optional

from pydantic import BaseModel, Field

NOT_DISCLOSED = "Not explicitly disclosed in 10-K"

class DisclosureSentence(BaseModel):
    """A filing sentence surfaced by a deterministic screen, with its chunk and (inferred) item."""

    chunk_id: str
    item: Optional[str] = None
    text: str
    kind: str = Field(..., description="'revenue_concentration' | 'receivables_concentration' | 'pricing'")
