"""Golden scenarios: one chat turn each, with the canned structured LLM answers that turn needs.

The Supervisor's routing and the Synthesizer's prose are the only LLM answers with content; every other
LLM call gets "{}", so agents take their deterministic paths (the code being moved).
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

# One sentence carries a figure found in no filing, so the figure screen must drop it.
SYNTHESIS_PROSE: Dict[str, str] = {
    "executive_summary": "The filing supports a balanced view of the business. Revenue reached $999,999 million last year.",
    "bottom_line": "The stance depends on whether margins hold.",
    "pillar_1_business_moat": "The business relies on its installed base and its brand.",
    "pillar_2_financial_durability": "Cash generation covers reinvestment needs.",
    "pillar_3_valuation_asymmetry": "The valuation case rests on the base-case cash flows.",
    "business_commentary": "Segment disclosure frames the revenue mix.",
    "earnings_quality_commentary": "Reported and normalized earnings are compared year by year.",
    "cash_flow_balance_sheet_commentary": "Free cash flow funds buybacks and dividends.",
    "forecast_commentary": "Growth fades toward the terminal rate.",
    "valuation_commentary": "The discount rate drives most of the value range.",
    "risk_commentary": "Risk disclosures focus on competition and regulation.",
    "bear_case_commentary": "A demand slowdown is the main bear case.",
    "disclosure_change_commentary": "Year-over-year risk changes are limited.",
}


@dataclass(frozen=True)
class Scenario:
    name: str
    query: str
    extraction: Dict[str, Any]
    session_state: Optional[Dict[str, Any]] = None

    def structured_responses(self) -> Dict[str, Dict[str, Any]]:
        return {"SupervisorExtraction": self.extraction, "ResearchSynthesisPayload": SYNTHESIS_PROSE}


def _research(query_type: str, ticker: Optional[str], year: Optional[int]) -> Dict[str, Any]:
    return {
        "query_type": query_type,
        "extracted_ticker": ticker,
        "extracted_year": year,
        "routing_reasoning": "golden scenario",
    }


# Pins today's behaviour: the session year (FY2024) is lost because the orchestrator passes fiscal_year=0,
# so the run uses FY2025. PR 12 fixes this and regenerates the snapshot.
_AAPL_SESSION = {
    "active_ticker": "AAPL",
    "active_company": "Apple Inc.",
    "active_fiscal_year": 2024,
    "available_fiscal_years": [2025, 2024],
    "last_query_type": "full_10k_report",
    "valuation_overrides": {"ticker": "AAPL", "terminal_growth": 0.03},
}

SCENARIOS: Dict[str, Scenario] = {
    scenario.name: scenario
    for scenario in [
        Scenario("full_aapl_2025", "Write a full 10-K research report on Apple for FY2025",
                 _research("full_10k_report", "AAPL", 2025)),
        Scenario("full_amzn_2025", "Write a full 10-K research report on Amazon for FY2025",
                 _research("full_10k_report", "AMZN", 2025)),
        Scenario("full_nvda_2026", "Write a full 10-K research report on NVIDIA for FY2026",
                 _research("full_10k_report", "NVDA", 2026)),
        Scenario("full_tsla_2025", "Write a full 10-K research report on Tesla for FY2025",
                 _research("full_10k_report", "TSLA", 2025)),
        Scenario("dcf_aapl_2025_overrides", "Run a DCF on Apple with a WACC of 9% and terminal growth of 3%",
                 _research("dcf_valuation_only", "AAPL", 2025)),
        Scenario("audit_aapl_2025", "Audit Apple's FY2025 financial statements",
                 _research("financial_audit_only", "AAPL", 2025)),
        Scenario("moat_nvda_2026", "What is NVIDIA's business model and economic moat?",
                 _research("business_moat_only", "NVDA", 2026)),
        Scenario("risk_tsla_2025", "What are Tesla's main risk factors?",
                 _research("risk_factors_only", "TSLA", 2025)),
        Scenario("followup_dcf_aapl_session", "Now run the DCF again with a beta of 1.4",
                 _research("dcf_valuation_only", None, None), session_state=_AAPL_SESSION),
        Scenario("chat_what_is_dcf", "What is a discounted cash flow valuation?", {
            "query_type": "conversational",
            "extracted_ticker": None,
            "extracted_year": None,
            "conversational_response": "A DCF values a business as the present value of its future free cash flows.",
            "routing_reasoning": "golden scenario",
        }),
    ]
}
