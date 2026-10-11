"""Deterministic check that a verified filing quote actually evidences the claim it is attached to.

Quote verification proves a sentence exists in the filing; it does not prove the sentence supports the
claim. A live run attached "we focus on selection, price, and convenience" to Network Effects and
"one way we offer lower prices" to pricing power. Each claim type therefore needs a quote that names its
economic mechanism (generic, industry-neutral vocabulary), and standard intellectual-property
protection boilerplate, which appears in nearly every 10-K, never counts as moat evidence.
"""

import re
from typing import Optional

MOAT_MECHANISMS = {
    "Network Effects": re.compile(
        r"network effect|two[- ]sided|multi[- ]sided|flywheel|community|user base|installed base|"
        r"\bmore (?:customers|users|sellers|buyers|merchants|developers|participants|members|subscribers|advertisers)\b|"
        r"\battract(?:s|ing)? (?:more |additional )?(?:customers|users|sellers|buyers|merchants|developers|participants|advertisers)\b|"
        r"\bnetwork of (?:users|customers|merchants|members|partners|developers)\b|platform (?:participants|ecosystem)|ecosystem",
        re.IGNORECASE,
    ),
    "High Switching Costs": re.compile(
        r"switch|migrat|integrat|embedded|lock[- ]?in|retention|renewal|long[- ]term (?:contracts?|agreements?|relationships?)|"
        r"multi[- ]year (?:contracts?|agreements?)|recurring|subscription|installed base|ecosystem|interoperab|compatib|"
        r"mission[- ]critical|training|certif",
        re.IGNORECASE,
    ),
    "Cost Advantage": re.compile(
        r"\bscale\b|economies of scale|cost structure|low[- ]cost|cost (?:advantage|leadership)|cost to serve|unit costs?|"
        r"\blower(?:s|ed|ing)? (?:our |its |the )?(?:costs?|prices?|cost to serve|expenses)\b|reduc\w* (?:our |its )?costs?|"
        r"efficien|purchasing power|vertical(?:ly)? integrat|automation|fixed costs?|operating leverage|"
        r"\bproprietary (?:process|manufacturing|technology)\b|infrastructure",
        re.IGNORECASE,
    ),
    "Intangible Assets / Brand": re.compile(
        r"\bbrand|reputation|trust|patent|licen[cs]e|regulatory approval|exclusive|franchise|trade secret|"
        r"\bproprietary\b|copyright|trademark|clinical|approval",
        re.IGNORECASE,
    ),
    "Efficient Scale": re.compile(
        r"limited number of (?:competitors|providers|suppliers)|natural monopol|barriers? to entry|high capital (?:requirements|costs)|"
        r"concession|regulated (?:utility|monopoly|returns?)|only (?:provider|supplier)|sole (?:provider|supplier)|"
        r"capacity|right[- ]of[- ]way|market share",
        re.IGNORECASE,
    ),
}

IP_BOILERPLATE = re.compile(
    r"\b(?:regard|consider)\b[^.]{0,200}\b(?:intellectual property|proprietary rights|trademarks?|patents?)\b[^.]{0,120}"
    r"\b(?:critical|important|valuable|essential)\b|"
    r"\brely (?:up)?on a combination of\b|"
    r"\b(?:may|might) not be (?:able|adequate|sufficient)\b|"
    r"\bprotect(?:ion of)? our (?:intellectual property|proprietary rights)\b",
    re.IGNORECASE,
)

COMPETITION_MECHANISM = re.compile(
    r"compet|rival|new entrants?|entry|substitut|alternative|disrupt|new (?:technolog|business models?|products)|"
    r"pricing pressure|price competition|commoditi|consolidat|market share|lose (?:customers|share)|"
    r"greater resources|lower prices",
    re.IGNORECASE,
)

PRICING_POWER_CLAIM = re.compile(
    r"pricing power|premium|raise (?:its |our )?prices|price increases?|higher prices|maintains? (?:its )?pricing|"
    r"pass(?:es|ed)? (?:on|through)",
    re.IGNORECASE,
)
PRICING_POWER_SUPPORT = re.compile(
    r"price increases?|increase(?:d)? (?:our |its )?prices|rais(?:e|ed|ing) (?:our |its )?prices|premium|higher prices|"
    r"pricing power|price realization|favorable pricing|pricing actions|pass(?:ed)? (?:on|through)",
    re.IGNORECASE,
)


def is_ip_boilerplate(quote: str) -> bool:
    return bool(IP_BOILERPLATE.search(quote or ""))


def moat_quote_supports(moat_type: Optional[str], quote: str) -> bool:
    """True when the quote names the economic mechanism of `moat_type` and is not IP-protection boilerplate."""
    pattern = MOAT_MECHANISMS.get(moat_type or "")
    if pattern is None or not quote:
        return False
    return not is_ip_boilerplate(quote) and bool(pattern.search(quote))


def threat_quote_supports(quote: str) -> bool:
    return bool(quote) and bool(COMPETITION_MECHANISM.search(quote))


def pricing_claim_supported(assessment: str, quote: str) -> bool:
    """A claim OF pricing power needs a quote evidencing it; descriptions of price pressure or low-price strategy pass."""
    if not PRICING_POWER_CLAIM.search(assessment or ""):
        return True
    return bool(PRICING_POWER_SUPPORT.search(quote or ""))
