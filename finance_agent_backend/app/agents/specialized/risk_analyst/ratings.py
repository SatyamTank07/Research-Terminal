"""The Risk Analyst's rating scale: High / Medium / Low likelihood and impact, and the likelihood x impact matrix
that turns them into a Severe / Moderate / Low severity."""

from typing import Any, Optional, Tuple

SEVERITY_ORDER = {"Severe": 0, "Moderate": 1, "Low": 2}
LEVEL_ORDER = {"High": 0, "Medium": 1, "Low": 2}

_SEVERITY_MATRIX = {
    ("High", "High"): "Severe",
    ("Medium", "High"): "Severe",
    ("Low", "High"): "Moderate",
    ("High", "Medium"): "Moderate",
    ("Medium", "Medium"): "Moderate",
    ("Low", "Medium"): "Low",
    ("High", "Low"): "Low",
    ("Medium", "Low"): "Low",
    ("Low", "Low"): "Low",
}


def normalize_level(value: Any) -> Optional[str]:
    """Maps free-form LLM ratings onto High / Medium / Low (None if unrecognizable)."""
    if not isinstance(value, str):
        return None
    v = value.strip().lower()
    if v in ("high", "severe", "critical", "very high", "elevated"):
        return "High"
    if v in ("medium", "moderate", "mid", "med"):
        return "Medium"
    if v in ("low", "minor", "limited", "very low"):
        return "Low"
    return None


def higher_level(a: str, b: str) -> str:
    """Returns the higher of two levels on the High > Medium > Low scale."""
    return a if LEVEL_ORDER[a] <= LEVEL_ORDER[b] else b


def derive_severity(likelihood: str, impact: str) -> str:
    return _SEVERITY_MATRIX.get((likelihood, impact), "Moderate")


def rank_key(risk: Any) -> Tuple[int, int, int]:
    """Most severe first, then the higher impact, then the higher likelihood (a rated risk item)."""
    return SEVERITY_ORDER[risk.severity], LEVEL_ORDER[risk.impact], LEVEL_ORDER[risk.likelihood]
