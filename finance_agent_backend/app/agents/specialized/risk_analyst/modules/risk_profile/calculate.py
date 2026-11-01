"""Which risk matters most, and how risky is the company overall? Ranks the verified risks (severity, then impact,
then likelihood), keeps at most 8, makes sure the primary threat is among the most severe, and rates the overall
profile from the severity mix."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from app.agents.specialized.risk_analyst.modules.risk_inventory import RiskItem
from app.agents.specialized.risk_analyst.modules.risk_profile.thresholds import MAX_RISKS, MIN_RISKS
from app.agents.specialized.risk_analyst.ratings import rank_key


@dataclass(frozen=True)
class RankedRisks:
    """The kept risks, numbered R1..Rn in rank order, and the data-quality notes about the cut."""

    risks: List[RiskItem]
    notes: Tuple[str, ...]


@dataclass(frozen=True)
class PrimaryThreat:
    """The primary threat's risk id (None without risks), why it threatens the business, and a re-point note."""

    risk_id: Optional[str]
    statement: str
    note: Optional[str]


def rank_risks(survivors: Sequence[RiskItem]) -> RankedRisks:
    """Most severe first, capped at MAX_RISKS, renumbered R1..Rn; notes a cut and a short list."""
    ranked = sorted(survivors, key=rank_key)
    notes: List[str] = []
    if len(ranked) > MAX_RISKS:
        notes.append(f"{len(ranked) - MAX_RISKS} lower-ranked verified risks omitted (cap of {MAX_RISKS}).")
        ranked = ranked[:MAX_RISKS]
    if len(ranked) < MIN_RISKS:
        notes.append(f"Only {len(ranked)} risks survived evidence verification (target {MIN_RISKS}-{MAX_RISKS}).")
    for i, risk in enumerate(ranked, start=1):
        risk.risk_id = f"R{i}"
    return RankedRisks(risks=ranked, notes=tuple(notes))


def resolve_primary_threat(
    ranked: Sequence[RiskItem],
    chosen: Optional[RiskItem],
    absorbed: Dict[int, RiskItem],
    proposed_id: Optional[str],
    proposed_statement: str,
) -> PrimaryThreat:
    """Keeps the model's primary threat when it survived (directly or merged) at the top severity, else the top risk."""
    if not ranked:
        return PrimaryThreat(risk_id=None, statement="", note=None)
    if chosen is not None:
        chosen = absorbed.get(id(chosen), chosen)
    top_severity = ranked[0].severity
    if chosen is not None and any(r is chosen for r in ranked) and chosen.severity == top_severity:
        statement = proposed_statement or f"{chosen.risk_title}: {chosen.risk_summary}"
        return PrimaryThreat(risk_id=chosen.risk_id, statement=statement, note=None)

    top = ranked[0]
    note = None
    if chosen is not None:
        note = (
            f"Primary threat re-pointed to {top.risk_id} ('{top.risk_title}'): the model's choice "
            f"('{chosen.risk_title}') was not among the most severe verified risks."
        )
    elif proposed_id:
        note = f"Primary threat re-pointed to {top.risk_id}: the model's choice did not survive verification."
    return PrimaryThreat(risk_id=top.risk_id, statement=f"{top.risk_title}: {top.risk_summary}", note=note)


def derive_overall_profile(severities: Sequence[str]) -> str:
    """Aggregate rating; "Not Assessed" when no verified risk exists (never a reassuring "Low")."""
    if not severities:
        return "Not Assessed"
    severe = sum(1 for s in severities if s == "Severe")
    moderate = sum(1 for s in severities if s == "Moderate")
    if severe >= 2:
        return "High"
    if severe == 1 or moderate >= 3:
        return "Moderate"
    return "Low"
