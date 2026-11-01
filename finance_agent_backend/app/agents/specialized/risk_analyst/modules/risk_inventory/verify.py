"""What are the material risks, and does the filing back each claim? Verifies the judge's risk inventory: raw
proposals sanitized into valid risks, every quote matched to the filing, exposures and mitigations kept only when the
quotes support them, impact capped for immaterial, legal-outside-Item-1A, unsubstantiated and boilerplate risks, and
duplicates folded into the higher-ranked risk."""

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from app.agents.shared.filing.sentences import normalize_text
from app.agents.shared.quotes import find_quote_chunk, numbers_supported
from app.agents.specialized.risk_analyst.context import RISK_SECTION_ITEM
from app.agents.specialized.risk_analyst.modules.quantified_exposures import QUANTIFIED_FIGURE
from app.agents.specialized.risk_analyst.modules.risk_inventory.schema import (
    FINANCIAL_TRANSMISSIONS,
    RISK_CATEGORIES,
    RiskEvidence,
    RiskItem,
)
from app.agents.specialized.risk_analyst.modules.risk_inventory.thresholds import (
    DUPLICATE_TITLE_JACCARD,
    DUPLICATE_TITLE_JACCARD_SHARED_CHUNK,
    MAX_QUOTE_CHARS,
    MAX_SIGNPOSTS,
)
from app.agents.specialized.risk_analyst.ratings import derive_severity, higher_level, normalize_level, rank_key

Pool = Dict[str, Dict[str, Any]]

IMMATERIALITY_PATTERNS = re.compile(
    r"not\s+(?:currently\s+)?(?:expected|believed|anticipated|likely)\s+to\s+(?:have|result\s+in|be)\s+(?:a\s+)?material"
    r"|(?:will|would|does|did|could)\s+not\s+(?:have|result\s+in)\s+(?:a\s+)?material\s+adverse"
    r"|(?:do|does|did)\s+not\s+believe\s+[^.]{0,160}?material"
    r"|\bimmaterial\b",
    re.IGNORECASE,
)

BOILERPLATE_PATTERNS = re.compile(
    r"(?:general|adverse|unfavorable|weak|deteriorating)\s+(?:macro)?economic\s+conditions"
    r"|economic\s+(?:downturns?|slowdowns?|uncertainty|instability)"
    r"|economic\s+conditions\s+and\s+market\s+volatility"
    r"|stock\s+price\s+volatility"
    r"|volatil\w*\s+(?:of|in)\s+(?:our\s+)?(?:stock|share)\s+price"
    r"|(?:stock|share)\s+price\s+(?:may|could)\s+(?:be\s+)?(?:volatile|fluctuate)"
    r"|\bmarket\s+volatility\b"
    r"|forward-looking\s+statements",
    re.IGNORECASE,
)

REALIZED_EFFECT_PATTERNS = re.compile(
    r"\b(?:we|the\s+company)\s+(?:have\s+|has\s+|had\s+)?(?:incurred|recorded|recognized|experienced|lost)\b"
    r"|\b(?:have|has|had)\s+(?:adversely\s+|negatively\s+|materially\s+)?(?:affected|impacted|harmed)\b"
    r"|\bresulted\s+in\b|\bwrite-?(?:down|off)s?\b",
    re.IGNORECASE,
)

_TITLE_STOPWORDS = {
    "risk", "risks", "the", "and", "of", "to", "in", "on", "for", "our", "from", "with",
    "related", "due", "its", "their", "a", "an", "by", "or", "potential", "exposure",
}

_CATEGORY_KEYWORDS: List[Tuple[str, str]] = [
    (r"regulat|legal|litigation|lawsuit|antitrust|export\s+control|compliance|government|sanction", "Regulatory & Legal"),
    (r"cyber|technolog|privacy|intellectual\s+property|\bip\b|obsolesc|software|security\s+breach", "Technological & Cybersecurity"),
    (r"supply|supplier|manufactur|foundry|concentrat|customer|single.source|sole.source", "Supply Chain & Concentration"),
    (r"compet|demand|pricing|market\s+share|substitut|cyclical", "Competitive & Demand"),
    (r"macro|geopolit|currency|foreign\s+exchange|tariff|trade|interest\s+rate|inflation|recession", "Macroeconomic & Geopolitical"),
    (r"debt|liquidity|capital\s+structure|financing|credit|leverage|tax|impairment", "Financial & Capital Structure"),
]

_NOTE_IMMATERIAL = "Impact capped at Medium: the cited filing text states the matter is not expected to be material."
_NOTE_LEGAL_OUTSIDE = "Impact capped at Medium: legal matter sourced outside Item 1A without a quantified exposure."
_NOTE_BOILERPLATE = "Impact capped at Low: generic boilerplate risk without a company-specific exposure."
_NOTE_UNSUBSTANTIATED = "Impact capped at Medium: no filing-stated magnitude or realized effect supports a High impact."
_NOTE_EXPOSURE_UNIT = "Quantified exposure removed: not stated as a percentage or currency amount."
_NOTE_EXPOSURE = "Quantified exposure removed: its figures do not appear in the cited filing text."
_NOTE_MITIGATION = "Mitigation removed: no verified filing quote supports it."


@dataclass(frozen=True)
class InventoryResult:
    """Verified risks after the duplicate merge, which risk absorbed which (by id of the absorbed risk), and counts."""

    survivors: List[RiskItem]
    absorbed: Dict[int, RiskItem]
    dropped_unverified: int
    quotes_rejected: int


# ------------------------------------------------------------------
# Screens
# ------------------------------------------------------------------
def states_immaterial(text: str) -> bool:
    return bool(text) and bool(IMMATERIALITY_PATTERNS.search(text))


def is_boilerplate(text: str) -> bool:
    return bool(text) and bool(BOILERPLATE_PATTERNS.search(text))


def substantiates_high_impact(quotes: Sequence[str], quantified_exposure: Optional[str]) -> bool:
    """High impact needs a filing-stated magnitude or a realized (not hypothetical) effect in the quotes."""
    if quantified_exposure:
        return True
    return any(QUANTIFIED_FIGURE.search(q) or REALIZED_EFFECT_PATTERNS.search(q) for q in quotes if q)


# ------------------------------------------------------------------
# Sanitize: raw judge output into valid risk fields
# ------------------------------------------------------------------
def sanitize_risk(item: Any, idx: int) -> Optional[Dict[str, Any]]:
    """One proposed risk as RiskItem fields, or None when it has no title or summary."""
    if not isinstance(item, dict):
        return None
    title = str(item.get("risk_title") or "").strip()
    summary = str(item.get("risk_summary") or "").strip()
    if not title or not summary:
        return None

    adjustments: List[str] = []
    likelihood = normalize_level(item.get("likelihood"))
    impact = normalize_level(item.get("impact"))
    if likelihood is None:
        likelihood = "Medium"
        adjustments.append("Likelihood not rated by the model; defaulted to Medium.")
    if impact is None:
        impact = "Medium"
        adjustments.append("Impact not rated by the model; defaulted to Medium.")

    transmissions = item.get("financial_transmission") or []
    if isinstance(transmissions, str):
        transmissions = [transmissions]
    lookup = {t.lower(): t for t in FINANCIAL_TRANSMISSIONS}
    transmission = list(dict.fromkeys(
        lookup[t.strip().lower()] for t in transmissions if isinstance(t, str) and t.strip().lower() in lookup
    ))

    signposts = item.get("monitoring_signposts") or []
    if isinstance(signposts, str):
        signposts = [signposts]

    return {
        "risk_id": str(item.get("risk_id") or f"R{idx + 1}"),
        "risk_category": coerce_category(item.get("risk_category"), f"{title} {summary}"),
        "risk_title": title,
        "risk_summary": summary,
        "likelihood": likelihood,
        "impact": impact,
        "severity": derive_severity(likelihood, impact),
        "financial_transmission": transmission,
        "quantified_exposure": _clean_str(item.get("quantified_exposure")),
        "mitigating_factors": _clean_str(item.get("mitigating_factors")),
        "mitigation_evidence": _sanitize_evidence(item.get("mitigation_evidence")),
        "evidence": [e for e in (_sanitize_evidence(x) for x in (item.get("evidence") or [])) if e],
        "monitoring_signposts": [s.strip() for s in signposts if isinstance(s, str) and s.strip()][:MAX_SIGNPOSTS],
        "adjustments": adjustments,
    }


def _clean_str(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    v = value.strip()
    return None if not v or v.lower() in ("null", "none", "n/a", "not disclosed") else v


def _sanitize_evidence(value: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(value, dict):
        return None
    chunk_id, quote = value.get("chunk_id"), value.get("quote")
    if not isinstance(chunk_id, str) or not isinstance(quote, str) or not quote.strip():
        return None
    return {"chunk_id": chunk_id.strip(), "quote": quote.strip()[:MAX_QUOTE_CHARS]}


def coerce_category(category: Any, fallback_text: str) -> str:
    """The listed category the model named, else the first keyword match in its label or text, else Operational."""
    if isinstance(category, str):
        for c in RISK_CATEGORIES:
            if category.strip().lower() == c.lower():
                return c
    for text in (category if isinstance(category, str) else "", fallback_text):
        for pattern, mapped in _CATEGORY_KEYWORDS:
            if text and re.search(pattern, text, re.IGNORECASE):
                return mapped
    return "Operational"


# ------------------------------------------------------------------
# Verify: quotes, exposure, mitigation, impact caps, duplicates
# ------------------------------------------------------------------
def verify_inventory(risks: Sequence[RiskItem], pool: Pool) -> InventoryResult:
    """Verifies each risk in place; a risk with no verified quote is dropped. Survivors then fold their duplicates."""
    verified: List[RiskItem] = []
    dropped = rejected = 0
    for risk in risks:
        kept, misses = _verified_quotes(risk, pool)
        rejected += misses
        if not kept:
            dropped += 1
            continue
        _apply_screens(risk, kept, pool)
        verified.append(risk)
    survivors, absorbed = merge_duplicates(verified)
    return InventoryResult(survivors=survivors, absorbed=absorbed, dropped_unverified=dropped, quotes_rejected=rejected)


def verify_evidence(ev: Optional[RiskEvidence], pool: Pool) -> Optional[RiskEvidence]:
    """Matches a quote to its claimed chunk (exact or near-verbatim), else to any pooled chunk (exact)."""
    if ev is None or not pool:
        return None
    hit = find_quote_chunk(ev.quote, ev.chunk_id, pool)
    if hit is None:
        return None
    return RiskEvidence(
        chunk_id=hit["chunk_id"],
        quote=ev.quote,
        item=hit.get("item"),
        breadcrumb=hit.get("breadcrumb"),
        verified=True,
    )


def _verified_quotes(risk: RiskItem, pool: Pool) -> Tuple[List[RiskEvidence], int]:
    """The risk's quotes that match the filing (duplicates once), and how many did not match."""
    kept: List[RiskEvidence] = []
    rejected = 0
    seen = set()
    for ev in risk.evidence:
        hit = verify_evidence(ev, pool)
        if hit is None:
            rejected += 1
            continue
        key = (hit.chunk_id, normalize_text(hit.quote))
        if key not in seen:
            seen.add(key)
            kept.append(hit)
    return kept, rejected


def _apply_screens(risk: RiskItem, kept: List[RiskEvidence], pool: Pool) -> None:
    """Keeps only supported exposure and mitigation, caps the impact the quotes do not carry, re-derives severity."""
    risk.evidence = kept
    chunk_texts = [pool[e.chunk_id]["content"] for e in kept]

    if risk.quantified_exposure and not QUANTIFIED_FIGURE.search(risk.quantified_exposure):
        risk.quantified_exposure = None
        risk.adjustments.append(_NOTE_EXPOSURE_UNIT)
    elif risk.quantified_exposure and not numbers_supported(risk.quantified_exposure, chunk_texts):
        risk.quantified_exposure = None
        risk.adjustments.append(_NOTE_EXPOSURE)

    mitigation = verify_evidence(risk.mitigation_evidence, pool) if risk.mitigation_evidence else None
    if risk.mitigating_factors and mitigation is None:
        risk.mitigating_factors = None
        risk.adjustments.append(_NOTE_MITIGATION)
    risk.mitigation_evidence = mitigation if risk.mitigating_factors else None

    outside = [e for e in kept if e.item != RISK_SECTION_ITEM]
    immaterial = any(states_immaterial(e.quote) for e in kept) or any(
        states_immaterial(pool[e.chunk_id]["content"]) for e in outside
    )
    if risk.impact == "High" and immaterial:
        risk.impact = "Medium"
        risk.adjustments.append(_NOTE_IMMATERIAL)
    elif (
        risk.impact == "High"
        and risk.risk_category == "Regulatory & Legal"
        and len(outside) == len(kept)
        and not risk.quantified_exposure
    ):
        risk.impact = "Medium"
        risk.adjustments.append(_NOTE_LEGAL_OUTSIDE)

    if risk.impact == "High" and not substantiates_high_impact([e.quote for e in kept], risk.quantified_exposure):
        risk.impact = "Medium"
        risk.adjustments.append(_NOTE_UNSUBSTANTIATED)

    if risk.impact != "Low" and not risk.quantified_exposure and is_boilerplate(f"{risk.risk_title} {risk.risk_summary}"):
        risk.impact = "Low"
        risk.adjustments.append(_NOTE_BOILERPLATE)

    risk.severity = derive_severity(risk.likelihood, risk.impact)


def merge_duplicates(risks: List[RiskItem]) -> Tuple[List[RiskItem], Dict[int, RiskItem]]:
    """Folds duplicate risks into the higher-ranked one. Returns (survivors, {id(absorbed): survivor})."""
    survivors: List[RiskItem] = []
    absorbed: Dict[int, RiskItem] = {}
    for risk in sorted(risks, key=rank_key):
        base = next((s for s in survivors if _is_duplicate(s, risk)), None)
        if base is None:
            survivors.append(risk)
            continue
        _absorb(base, risk)
        absorbed[id(risk)] = base
    return survivors, absorbed


def _title_tokens(title: str) -> set:
    return {w for w in re.findall(r"[a-z0-9]+", normalize_text(title)) if len(w) > 2 and w not in _TITLE_STOPWORDS}


def _jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def _is_duplicate(a: RiskItem, b: RiskItem) -> bool:
    similarity = _jaccard(_title_tokens(a.risk_title), _title_tokens(b.risk_title))
    if similarity >= DUPLICATE_TITLE_JACCARD:
        return True
    shared = {e.chunk_id for e in a.evidence} & {e.chunk_id for e in b.evidence}
    return bool(shared) and similarity >= DUPLICATE_TITLE_JACCARD_SHARED_CHUNK


def _absorb(base: RiskItem, other: RiskItem) -> None:
    seen = {(e.chunk_id, normalize_text(e.quote)) for e in base.evidence}
    for e in other.evidence:
        key = (e.chunk_id, normalize_text(e.quote))
        if key not in seen:
            seen.add(key)
            base.evidence.append(e)
    base.likelihood = higher_level(base.likelihood, other.likelihood)
    base.impact = higher_level(base.impact, other.impact)
    base.severity = derive_severity(base.likelihood, base.impact)
    base.financial_transmission = list(dict.fromkeys(base.financial_transmission + other.financial_transmission))
    base.monitoring_signposts = list(dict.fromkeys(base.monitoring_signposts + other.monitoring_signposts))[:MAX_SIGNPOSTS]
    if not base.quantified_exposure and other.quantified_exposure:
        base.quantified_exposure = other.quantified_exposure
    if not base.mitigating_factors and other.mitigating_factors:
        base.mitigating_factors = other.mitigating_factors
        base.mitigation_evidence = other.mitigation_evidence
    base.adjustments.append(f"Merged duplicate risk: '{other.risk_title}'.")
