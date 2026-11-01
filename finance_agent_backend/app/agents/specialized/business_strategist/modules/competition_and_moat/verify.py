"""Checks the judge's moat and competition claims: each keeps only quotes that exist and name its mechanism.

An unsupported primary moat is replaced by a verified secondary, else downgraded to None / Indeterminate;
"Wide" needs two distinct verified moat quotes; segment moats must name a code-extracted segment; competitive
threats need a quote about competition.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from app.agents.shared.filing.sentences import normalize_text
from app.agents.specialized.business_strategist.claim_relevance import (
    moat_quote_supports,
    segment_quote_mentions,
    threat_quote_supports,
)
from app.agents.specialized.business_strategist.context import Pool
from app.agents.specialized.business_strategist.evidence import (
    StrategyEvidence,
    VerifyTally,
    not_assessed,
    text_of,
    verify_quotes,
)
from app.agents.specialized.business_strategist.modules.competition_and_moat.schema import MOAT_TYPES, SegmentMoat
from app.agents.specialized.business_strategist.modules.competition_and_moat.thresholds import (
    MAX_SECONDARY_MOATS,
    MAX_THREATS,
    WIDE_MIN_QUOTES,
)
from app.agents.specialized.business_strategist.modules.revenue_mix import (
    SegmentExtraction,
    clean_name,
    match_name,
    name_key,
    reportable_segment_names,
    top_level_line_names,
)

_DURABILITY = {"wide": "Wide", "narrow": "Narrow", "none": "None", "indeterminate": "Indeterminate"}
_TRAJECTORY = {"expanding": "Expanding", "stable": "Stable", "deteriorating": "Deteriorating", "indeterminate": "Indeterminate"}
_MOAT_ALIASES: Tuple[Tuple[str, str], ...] = (
    ("network", "Network Effects"),
    ("switching", "High Switching Costs"),
    ("cost advantage", "Cost Advantage"),
    ("low cost", "Cost Advantage"),
    ("intangible", "Intangible Assets / Brand"),
    ("brand", "Intangible Assets / Brand"),
    ("patent", "Intangible Assets / Brand"),
    ("efficient scale", "Efficient Scale"),
    ("no moat", "None"),
)


@dataclass(frozen=True)
class CompetitionAndMoat:
    """The verified moat fields and competitive threats, their evidence and what the checks dropped."""

    economic_moat_type: str
    secondary_moat_types: List[str]
    segment_specific_moats: List[SegmentMoat]
    moat_durability: str
    moat_trajectory: str
    moat_rationale: str
    moat_assessment: str
    competitive_threats: List[str] = field(default_factory=list)
    evidence: List[StrategyEvidence] = field(default_factory=list)
    tally: VerifyTally = field(default_factory=VerifyTally)

    def moat_fields(self) -> Dict[str, Any]:
        """The moat fields of the Strategist's output."""
        return {
            "economic_moat_type": self.economic_moat_type,
            "secondary_moat_types": self.secondary_moat_types,
            "segment_specific_moats": self.segment_specific_moats,
            "moat_durability": self.moat_durability,
            "moat_trajectory": self.moat_trajectory,
            "moat_rationale": self.moat_rationale,
            "moat_assessment": self.moat_assessment,
        }


def not_assessed_competition(reason: str) -> CompetitionAndMoat:
    """No judgement ran: no moat, no threats."""
    return CompetitionAndMoat(
        economic_moat_type="None",
        secondary_moat_types=[],
        segment_specific_moats=[],
        moat_durability="Indeterminate",
        moat_trajectory="Indeterminate",
        moat_rationale=not_assessed(reason),
        moat_assessment="not_assessed",
    )


def verify_competition_and_moat(data: Dict[str, Any], pool: Pool, extraction: SegmentExtraction) -> CompetitionAndMoat:
    """The judge's moats and competitive threats, kept only where verified filing quotes evidence them."""
    tally = VerifyTally()
    evidence: List[StrategyEvidence] = []
    segment_names = reportable_segment_names(extraction)
    line_names = top_level_line_names(extraction)
    moat = verify_moats(data, pool, segment_names, line_names, evidence, tally)
    threats = verify_threats(data, pool, evidence, tally)
    return CompetitionAndMoat(**moat, competitive_threats=threats, evidence=evidence, tally=tally)


def coerce_moat(value: Any) -> Optional[str]:
    """The moat type the judge named, mapped onto MOAT_TYPES (aliases allowed); None when unrecognised."""
    if not isinstance(value, str) or not value.strip():
        return None
    v = value.strip()
    for t in MOAT_TYPES:
        if v.lower() == t.lower():
            return t
    low = v.lower()
    if low in ("none", "no moat", "n/a"):
        return "None"
    for needle, mapped in _MOAT_ALIASES:
        if needle in low:
            return mapped
    return None


def verify_moats(
    data: Dict[str, Any],
    pool: Pool,
    segment_names: List[str],
    line_names: List[str],
    evidence: List[StrategyEvidence],
    tally: VerifyTally,
) -> Dict[str, Any]:
    primary = data.get("primary_moat") if isinstance(data.get("primary_moat"), dict) else {}
    p_type = coerce_moat(primary.get("type"))
    p_rationale = text_of(primary.get("rationale"))
    if p_type and p_type != "None":
        tally.claims_proposed += 1
    p_evidence = (
        verify_quotes(primary.get("evidence"), pool, tally, "primary_moat", p_type, lambda q, _c: moat_quote_supports(p_type, q))
        if p_type and p_type != "None" else []
    )

    secondaries: List[Tuple[str, str, List[StrategyEvidence]]] = []
    raw_secondary = data.get("secondary_moats") if isinstance(data.get("secondary_moats"), list) else []
    for item in raw_secondary:
        if not isinstance(item, dict):
            continue
        s_type = coerce_moat(item.get("type"))
        if not s_type or s_type == "None":
            continue
        tally.claims_proposed += 1
        s_evidence = verify_quotes(
            item.get("evidence"), pool, tally, "secondary_moat", s_type, lambda q, _c, t=s_type: moat_quote_supports(t, q)
        )
        if not s_evidence:
            tally.claims_dropped += 1
            tally.notes.append(f"Secondary moat '{s_type}' dropped: no verified filing quote evidencing its mechanism.")
            continue
        secondaries.append((s_type, text_of(item.get("rationale")), s_evidence))

    assessment = "verified"
    if p_type and p_type != "None" and p_evidence:
        evidence.extend(p_evidence)
    elif secondaries:
        promoted = secondaries.pop(0)
        if p_type and p_type != "None":
            tally.claims_dropped += 1
            tally.notes.append(
                f"Primary moat '{p_type}' had no verified quote evidencing its mechanism; verified secondary moat "
                f"'{promoted[0]}' promoted to primary."
            )
        p_type, p_rationale, p_evidence = promoted
        p_evidence = [e.model_copy(update={"claim": "primary_moat"}) for e in p_evidence]
        evidence.extend(p_evidence)
    elif p_type == "None":
        assessment = "no_moat"
    else:
        assessment = "unsupported"
        if p_type:
            tally.claims_dropped += 1
        tally.notes.append(
            f"Primary moat {repr(p_type) if p_type else '(missing)'} had no verified filing quote evidencing its "
            "mechanism; moat downgraded to None / Indeterminate."
        )

    kept_secondary: List[str] = []
    for s_type, _, s_evidence in secondaries:
        if s_type != p_type and s_type not in kept_secondary and len(kept_secondary) < MAX_SECONDARY_MOATS:
            kept_secondary.append(s_type)
            evidence.extend(s_evidence)

    segment_moats = verify_segment_moats(data, pool, segment_names, line_names, evidence, tally)

    durability = _DURABILITY.get(str(data.get("moat_durability") or "").strip().lower(), "Indeterminate")
    trajectory = _TRAJECTORY.get(str(data.get("moat_trajectory") or "").strip().lower(), "Indeterminate")
    if assessment == "verified":
        moat_quotes = {(e.chunk_id, normalize_text(e.quote)) for e in evidence if e.claim in ("primary_moat", "secondary_moat")}
        if durability == "Wide" and len(moat_quotes) < WIDE_MIN_QUOTES:
            durability = "Narrow"
            tally.notes.append(f"Durability capped at Narrow: 'Wide' needs at least {WIDE_MIN_QUOTES} distinct verified moat quotes.")
        elif durability == "None":
            durability = "Indeterminate"
            tally.notes.append("Durability 'None' contradicts a verified moat; set to Indeterminate.")
        rationale = p_rationale or f"{p_type} evidenced by the quoted filing disclosures."
    elif assessment == "no_moat":
        durability = durability if durability in ("None", "Indeterminate") else "None"
        trajectory = "Indeterminate" if trajectory == "Expanding" else trajectory
        rationale = p_rationale or "The filing evidences no structural competitive advantage."
    else:
        p_type, durability, trajectory = "None", "Indeterminate", "Indeterminate"
        kept_secondary = []
        rationale = not_assessed("no proposed moat was supported by a verified filing quote that evidences its mechanism")

    return {
        "economic_moat_type": p_type if assessment == "verified" else "None",
        "secondary_moat_types": kept_secondary,
        "segment_specific_moats": segment_moats,
        "moat_durability": durability,
        "moat_trajectory": trajectory,
        "moat_rationale": rationale,
        "moat_assessment": assessment,
    }


def verify_segment_moats(
    data: Dict[str, Any],
    pool: Pool,
    segment_names: List[str],
    line_names: List[str],
    evidence: List[StrategyEvidence],
    tally: VerifyTally,
) -> List[SegmentMoat]:
    targets = segment_names or line_names
    lookup = {name_key(n): n for n in targets}
    raw = data.get("segment_moats") if isinstance(data.get("segment_moats"), list) else []
    kept: List[SegmentMoat] = []
    seen = set()
    for item in raw:
        if not isinstance(item, dict):
            continue
        m_type = coerce_moat(item.get("type"))
        name = clean_name(str(item.get("segment_name") or ""))
        if not m_type or not name:
            continue
        tally.claims_proposed += 1
        canonical = match_name(name, lookup) if lookup else name
        if canonical is None:
            tally.claims_dropped += 1
            tally.notes.append(f"Segment moat for '{name}' dropped: not a code-extracted segment or revenue line.")
            continue
        if name_key(canonical) in seen:
            continue
        seg_evidence = verify_quotes(
            item.get("evidence"), pool, tally, "segment_moat", canonical,
            # The segment must be named in the quote or in the chunk it came from.
            lambda q, c, t=m_type, n=canonical: moat_quote_supports(t, q) and segment_quote_mentions(n, f"{q} {c}"),
        )
        if not seg_evidence:
            tally.claims_dropped += 1
            tally.notes.append(
                f"Segment moat for '{canonical}' dropped: no verified quote naming the segment and evidencing {m_type}."
            )
            continue
        seen.add(name_key(canonical))
        evidence.extend(seg_evidence)
        kept.append(
            SegmentMoat(
                segment_name=canonical,
                moat_type=m_type,  # type: ignore[arg-type]
                moat_rationale=text_of(item.get("rationale")) or f"{m_type} evidenced by the quoted disclosure.",
            )
        )
    return kept


def verify_threats(data: Dict[str, Any], pool: Pool, evidence: List[StrategyEvidence], tally: VerifyTally) -> List[str]:
    raw = data.get("competitive_threats") if isinstance(data.get("competitive_threats"), list) else []
    kept: List[str] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        threat = text_of(item.get("threat"))
        if not threat:
            continue
        tally.claims_proposed += 1
        t_evidence = verify_quotes(item.get("evidence"), pool, tally, "competitive_threat", threat,
                                   lambda q, _c: threat_quote_supports(q))
        if not t_evidence or len(kept) >= MAX_THREATS:
            tally.claims_dropped += 1
            continue
        kept.append(threat)
        evidence.extend(t_evidence)
    if raw and not kept:
        tally.notes.append("No proposed competitive threat was supported by a verified filing quote.")
    return kept
