"""Tools package for specialized agents."""

from app.agents.tools.financial_math_tools import (
    AnnualFinancialInput,
    BalanceSheetInput,
    PriorBalanceSheetInput,
    NormalizationAdjustment,
    ForensicFinding,
    FinancialAuditResult,
    calculate_financial_ratios,
    calculate_quality_of_earnings,
    detect_forensic_findings,
    audit_financial_metrics,
    audit_financial_metrics_tool,
)
from app.agents.tools.statement_extraction import (
    ExtractionResult,
    build_statement_grids,
    extract_audit_inputs,
)
from app.agents.tools.risk_disclosure_tools import RiskDisclosureDiff, diff_risk_disclosures
from app.agents.tools.tavily_search import get_tavily_tool
from app.agents.tools.catalog_tools import search_sec_catalog

__all__ = [
    "AnnualFinancialInput",
    "BalanceSheetInput",
    "PriorBalanceSheetInput",
    "NormalizationAdjustment",
    "ForensicFinding",
    "FinancialAuditResult",
    "calculate_financial_ratios",
    "calculate_quality_of_earnings",
    "detect_forensic_findings",
    "ExtractionResult",
    "build_statement_grids",
    "extract_audit_inputs",
    "audit_financial_metrics",
    "audit_financial_metrics_tool",
    "RiskDisclosureDiff",
    "diff_risk_disclosures",
    "get_tavily_tool",
    "search_sec_catalog",
]

