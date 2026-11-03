"""Picks the report layout from the route and the outputs that exist, and collects the run notes it discloses."""

import logging
from typing import Any, Dict, Iterable

from app.agents.specialized.lead_synthesizer.modules.report_coverage.schema import ReportCoverage

logger = logging.getLogger("finance_agent.agents.lead_synthesizer")

FULL_REPORT = "full_10k_report"
REPORT_LAYOUTS = (FULL_REPORT, "dcf_valuation_only", "financial_audit_only", "business_moat_only", "risk_factors_only")
# A single-topic report without its one topic has nothing to show: it falls back to the full layout with notes.
TOPIC_OUTPUT_OF_ROUTE = {
    "financial_audit_only": "financial_audit",
    "business_moat_only": "business_moat",
    "risk_factors_only": "risk_audit",
}


def assess_report_coverage(route: str, outputs: Dict[str, Any], notes: Iterable[str]) -> ReportCoverage:
    """The route the prose is written for, the layout that shows it, and the run notes, each once."""
    required = TOPIC_OUTPUT_OF_ROUTE.get(route)
    if required and outputs.get(required) is None:
        logger.warning(f"Route {route} has no {required}; using the full report layout with notes.")
        route = FULL_REPORT
    layout = route if route in REPORT_LAYOUTS else FULL_REPORT
    return ReportCoverage(route=route, layout=layout, notes=tuple(dict.fromkeys(notes)))
