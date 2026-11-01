"""The judge's evidence rule: a claim keeps only quotes that exist in the filing and evidence its mechanism.

Every judged module (business model, competition and moat, market power) checks its claims' quotes here and
records what it proposed, kept and dropped in a `VerifyTally` that assembly adds to the data-quality record.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, List, Literal, Optional

from pydantic import BaseModel, Field

from app.agents.shared.filing.sentences import normalize_text
from app.agents.shared.quotes import find_quote_chunk
from app.agents.specialized.business_strategist.context import Pool

MAX_QUOTE_CHARS = 300
MAX_EVIDENCE_PER_CLAIM = 2


class StrategyEvidence(BaseModel):
    """A verbatim filing quote that survived verification, attached to the claim it supports."""

    claim: Literal[
        "primary_moat", "secondary_moat", "segment_moat", "competitive_threat", "pricing_power", "customer_concentration"
    ]
    subject: Optional[str] = Field(None, description="Moat type, segment name or threat the quote supports")
    chunk_id: str
    quote: str
    item: Optional[str] = Field(None, description="True 10-K item of the quoted chunk")
    breadcrumb: Optional[str] = None


@dataclass
class VerifyTally:
    """Claims and quotes one verify step proposed, kept and dropped, and the notes explaining each drop."""

    claims_proposed: int = 0
    claims_dropped: int = 0
    quotes_verified: int = 0
    quotes_rejected: int = 0
    quotes_off_topic: int = 0
    notes: List[str] = field(default_factory=list)


def verify_quotes(
    raw_evidence: Any,
    pool: Pool,
    tally: VerifyTally,
    claim: str,
    subject: Optional[str],
    supports: Optional[Callable[[str, str], bool]] = None,
) -> List[StrategyEvidence]:
    """Verifies up to MAX_EVIDENCE_PER_CLAIM quotes: the claimed chunk (near-verbatim), else any chunk (exact).

    `supports` additionally requires the quote to evidence the claim's mechanism (claim_relevance); a real
    but off-topic quote is counted in `quotes_off_topic` and not kept.
    """
    if isinstance(raw_evidence, dict):
        raw_evidence = [raw_evidence]
    if not isinstance(raw_evidence, list):
        return []
    verified: List[StrategyEvidence] = []
    seen = set()
    for ev in raw_evidence[:MAX_EVIDENCE_PER_CLAIM]:
        if not isinstance(ev, dict):
            continue
        quote = ev.get("quote")
        chunk_id = str(ev.get("chunk_id") or "").strip()
        if not isinstance(quote, str) or not quote.strip():
            continue
        quote = quote.strip()[:MAX_QUOTE_CHARS]
        hit = find_quote_chunk(quote, chunk_id, pool)
        if hit is None:
            tally.quotes_rejected += 1
            continue
        if supports is not None and not supports(quote, hit.get("content") or ""):
            tally.quotes_off_topic += 1
            continue
        key = (hit["chunk_id"], normalize_text(quote))
        if key in seen:
            continue
        seen.add(key)
        tally.quotes_verified += 1
        verified.append(
            StrategyEvidence(
                claim=claim,  # type: ignore[arg-type]
                subject=subject,
                chunk_id=hit["chunk_id"],
                quote=quote,
                item=hit.get("item"),
                breadcrumb=hit.get("breadcrumb"),
            )
        )
    return verified


def text_of(value: Any) -> str:
    """The judge's text field, stripped; "" when missing or not a string."""
    return value.strip() if isinstance(value, str) and value.strip() else ""


def not_assessed(reason: str) -> str:
    return f"Not assessed: {reason}."
