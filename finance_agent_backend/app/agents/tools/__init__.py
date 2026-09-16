"""Tools package for specialized agents."""

from app.agents.tools.rag_table_tools import (
    TableChunkResult,
    retrieve_10k_tables,
    retrieve_multiyear_financial_series,
    retrieve_10k_tables_tool,
)
from app.agents.tools.dcf_tools import (
    DCFCalculationResult,
    calculate_dcf_with_sensitivity,
    calculate_dcf_tool,
)
from app.agents.tools.financial_math_tools import (
    AnnualFinancialInput,
    BalanceSheetInput,
    FinancialAuditResult,
    calculate_financial_ratios,
    detect_forensic_red_flags,
    audit_financial_metrics,
    audit_financial_metrics_tool,
)
from app.agents.tools.tavily_search import get_tavily_tool

__all__ = [
    "TableChunkResult",
    "retrieve_10k_tables",
    "retrieve_multiyear_financial_series",
    "retrieve_10k_tables_tool",
    "DCFCalculationResult",
    "calculate_dcf_with_sensitivity",
    "calculate_dcf_tool",
    "AnnualFinancialInput",
    "BalanceSheetInput",
    "FinancialAuditResult",
    "calculate_financial_ratios",
    "detect_forensic_red_flags",
    "audit_financial_metrics",
    "audit_financial_metrics_tool",
    "get_tavily_tool",
]
