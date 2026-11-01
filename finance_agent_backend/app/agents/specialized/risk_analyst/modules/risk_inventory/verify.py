"""What are the material risks, and does the filing back each claim? Screens on the judge's risk inventory: language
saying a matter is immaterial, generic boilerplate risks, and whether a High impact rests on a stated magnitude or a
realized effect."""

import re
from typing import Optional, Sequence

from app.agents.specialized.risk_analyst.modules.quantified_exposures import QUANTIFIED_FIGURE

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


def states_immaterial(text: str) -> bool:
    return bool(text) and bool(IMMATERIALITY_PATTERNS.search(text))


def is_boilerplate(text: str) -> bool:
    return bool(text) and bool(BOILERPLATE_PATTERNS.search(text))


def substantiates_high_impact(quotes: Sequence[str], quantified_exposure: Optional[str]) -> bool:
    """High impact needs a filing-stated magnitude or a realized (not hypothetical) effect in the quotes."""
    if quantified_exposure:
        return True
    return any(QUANTIFIED_FIGURE.search(q) or REALIZED_EFFECT_PATTERNS.search(q) for q in quotes if q)
