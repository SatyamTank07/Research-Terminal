"""An Item 1A sentence that puts a number on an exposure."""

from pydantic import BaseModel


class QuantifiedDisclosure(BaseModel):
    """An Item 1A sentence stating a magnitude (revenue share, charge, regional share...)."""

    chunk_id: str
    text: str
