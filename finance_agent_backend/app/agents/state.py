"""Shared State & Structured Output Schemas for Multi-Agent Equity Research.

Consolidates sub-agent contracts (Auditor, Forecaster, Valuation Specialist,
Moat Strategist, Risk Analyst, Synthesizer) and defines the centralized
LangGraph TypedDict state graph.
"""

from typing import Any, Dict, List, Optional, TypedDict

# ==============================================================================
# 1. Financial Auditor & Statement Analyst Output Schema
# ==============================================================================
from app.agents.specialized.financial_auditor.state_financial_auditor import (
    BalanceSheetSnapshot,
    FinancialAuditOutput,
    YearFinancials,
)

# ==============================================================================
# 2. DCF Valuation Specialist Output Schema
# ==============================================================================
from app.agents.specialized.valuation_specialist.state_valuation_specialist import (
    DCFValuationOutput,
    WACCAudit,
)

# ==============================================================================
# 3. Financial Forecasting Analyst Output Schema
# ==============================================================================
from app.agents.specialized.forecasting_analyst.state_forecasting_analyst import (
    ForecastOutput,
    ForecastYear,
    GuidanceSource,
)

# ==============================================================================
# 4. Business & Moat Strategist Output Schema
# ==============================================================================
from app.agents.specialized.business_strategist.state_business_strategist import (
    BusinessMoatOutput,
    SegmentDetail,
)

# ==============================================================================
# 5. Risk & Red Flag Analyst Output Schema
# ==============================================================================
from app.agents.specialized.risk_analyst.state_risk_analyst import (
    RiskAuditOutput,
    RiskItem,
)

# ==============================================================================
# 6. Routing & Lead Supervisor Output Schema
# ==============================================================================
from app.agents.specialized.supervisor.state_supervisor import (
    QueryType,
    RoutingPlan,
)

# ==============================================================================
# 7. Lead Synthesizer & Final Report Output Schema
# ==============================================================================
from app.agents.specialized.lead_synthesizer.state_lead_synthesizer import (
    Final10KResearchReport,
    ThreePillarThesis,
)


# ==============================================================================
# 8. Central LangGraph State Graph Definition (Multi-Agent Coordinator)
# ==============================================================================
class EquityResearchState(TypedDict, total=False):
    """Centralized TypedDict tracking the state across all specialized agents."""

    # Routing & Session Context
    user_query: str
    ticker: str
    company_name: str
    fiscal_year: int
    document_id: str
    query_type: str
    routing_plan: Optional[RoutingPlan]
    messages: Optional[List[Dict[str, str]]]
    session_state: Optional[Dict[str, Any]]
    updated_session_state: Optional[Dict[str, Any]]

    # Sub-agent structured payloads
    business_moat: Optional[BusinessMoatOutput]
    financial_audit: Optional[FinancialAuditOutput]
    forecast: Optional[ForecastOutput]
    dcf_valuation: Optional[DCFValuationOutput]
    risk_audit: Optional[RiskAuditOutput]

    # Final compiled output
    final_report: Optional[Final10KResearchReport]
    sources: List[Dict[str, Any]]
    error_message: Optional[str]


__all__ = [
    # Auditor
    "FinancialAuditOutput",
    "YearFinancials",
    "BalanceSheetSnapshot",
    # Valuation
    "WACCAudit",
    "DCFValuationOutput",
    # Forecaster
    "ForecastYear",
    "GuidanceSource",
    "ForecastOutput",
    # Business Moat
    "SegmentDetail",
    "BusinessMoatOutput",
    # Risk
    "RiskItem",
    "RiskAuditOutput",
    # Supervisor
    "QueryType",
    "RoutingPlan",
    # Synthesizer
    "ThreePillarThesis",
    "Final10KResearchReport",
    # Central Graph State
    "EquityResearchState",
]
