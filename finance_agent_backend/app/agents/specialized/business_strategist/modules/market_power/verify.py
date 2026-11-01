"""Checks the judge's pricing-power and customer-concentration claims against filing quotes.

A pricing claim needs a verified quote; a claim OF pricing power needs a quote about raising prices or premiums,
and its figures must appear in the quoted chunks, else the quote itself is reported. Customer concentration
without a verified, figure-consistent quote falls back to the code-screened filing sentence, verbatim.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

from app.agents.shared.quotes import numbers_supported
from app.agents.specialized.business_strategist.claim_relevance import pricing_claim_supported
from app.agents.specialized.business_strategist.context import Pool
from app.agents.specialized.business_strategist.evidence import (
    MAX_QUOTE_CHARS,
    StrategyEvidence,
    VerifyTally,
    not_assessed,
    text_of,
    verify_quotes,
)
from app.agents.specialized.business_strategist.modules.market_power.schema import NOT_DISCLOSED, DisclosureSentence

_NOT_DISCLOSED_HINTS = ("not explicitly disclosed", "not disclosed", "no disclosure", "not discussed", "not stated")


@dataclass(frozen=True)
class MarketPower:
    pricing_power_assessment: str
    customer_concentration: str
    evidence: List[StrategyEvidence]
    tally: VerifyTally


def verify_market_power(
    data: Dict[str, Any], pool: Pool, disclosures: Sequence[DisclosureSentence], assessed: bool, reason: str
) -> MarketPower:
    """Pricing power and customer concentration, kept only where verified quotes support them."""
    tally = VerifyTally()
    evidence: List[StrategyEvidence] = []
    pricing = verify_pricing(data, pool, evidence, tally) if assessed else not_assessed(reason)
    concentration = verify_concentration(data, pool, disclosures, evidence, tally, assessed, reason)
    return MarketPower(pricing_power_assessment=pricing, customer_concentration=concentration, evidence=evidence, tally=tally)


def is_not_disclosed(text: str) -> bool:
    low = text.lower()
    return any(h in low for h in _NOT_DISCLOSED_HINTS)


def verify_pricing(data: Dict[str, Any], pool: Pool, evidence: List[StrategyEvidence], tally: VerifyTally) -> str:
    raw = data.get("pricing_power") if isinstance(data.get("pricing_power"), dict) else {}
    assessment = text_of(raw.get("assessment"))
    if not assessment or is_not_disclosed(assessment):
        return NOT_DISCLOSED
    tally.claims_proposed += 1
    verified = verify_quotes(raw.get("evidence"), pool, tally, "pricing_power", None)
    if not verified:
        tally.claims_dropped += 1
        tally.notes.append("Pricing-power assessment dropped: no verified filing quote.")
        return NOT_DISCLOSED
    evidence.extend(verified)
    if not any(pricing_claim_supported(assessment, e.quote) for e in verified):
        tally.claims_dropped += 1
        tally.notes.append("Pricing-power claim not evidenced by its quotes (they describe pricing, not pricing power); "
                           "replaced by the verified quote.")
        return f'Filing states: "{verified[0].quote}"'
    if not numbers_supported(assessment, [pool[e.chunk_id]["content"] for e in verified]):
        tally.notes.append("Pricing-power text cited figures absent from its quotes; replaced by the verified quote.")
        return f'Filing states: "{verified[0].quote}"'
    return assessment


def verify_concentration(
    data: Dict[str, Any],
    pool: Pool,
    disclosures: Sequence[DisclosureSentence],
    evidence: List[StrategyEvidence],
    tally: VerifyTally,
    assessed: bool,
    reason: str,
) -> str:
    raw = data.get("customer_concentration") if isinstance(data.get("customer_concentration"), dict) else {}
    disclosure = text_of(raw.get("disclosure"))
    if assessed and disclosure and not is_not_disclosed(disclosure):
        tally.claims_proposed += 1
        verified = verify_quotes(raw.get("evidence"), pool, tally, "customer_concentration", None)
        if verified and numbers_supported(disclosure, [pool[e.chunk_id]["content"] for e in verified]):
            evidence.extend(verified)
            return disclosure
        tally.claims_dropped += 1
        tally.notes.append("Customer-concentration text lacked a verified quote or supported figures; replaced deterministically.")

    # Deterministic fallback: the screened filing sentence itself, verbatim.
    revenue = [d for d in disclosures if d.kind == "revenue_concentration"]
    receivables = [d for d in disclosures if d.kind == "receivables_concentration"]
    chosen = revenue[0] if revenue else receivables[0] if receivables else None
    if chosen is not None and chosen.chunk_id in pool:
        chunk = pool[chosen.chunk_id]
        evidence.append(
            StrategyEvidence(
                claim="customer_concentration",
                subject="code_screened",
                chunk_id=chosen.chunk_id,
                quote=chosen.text[:MAX_QUOTE_CHARS],
                item=chunk.get("item"),
                breadcrumb=chunk.get("breadcrumb"),
            )
        )
        if revenue:
            return f'Filing states: "{chosen.text}"'
        return f'{NOT_DISCLOSED} for revenue. Receivables: "{chosen.text}"'
    return NOT_DISCLOSED if assessed else not_assessed(reason)
