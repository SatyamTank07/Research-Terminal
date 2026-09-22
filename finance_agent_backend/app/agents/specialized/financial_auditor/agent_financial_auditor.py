"""Financial Auditor & Statement Analyst Agent.

Specialized autonomous agent that audits 10-K financial statements (Item 8),
inspects cross-filing time-series for restatements, normalizes accounting scales,
executes deterministic math tools (Zero Math Hallucination), and emits a typed
FinancialAuditOutput schema.
"""

import json
import logging
from typing import Any, Dict, List, Optional

from app.agents.base import StructuredAgent
from app.agents.registry import AgentRegistry
from app.agents.state import FinancialAuditOutput
from app.agents.tools.financial_math_tools import (
    BalanceSheetResult,
    ProfitabilityRatiosResult,
    SolvencyRatiosResult,
    YearFinancialsResult,
    audit_financial_metrics_tool,
)
from app.agents.tools.rag_table_tools import (
    retrieve_10k_tables_tool,
    retrieve_multiyear_financial_series_tool,
)

logger = logging.getLogger("finance_agent.agents.financial_auditor")


@AgentRegistry.register("financial_auditor")
class FinancialAuditorAgent(StructuredAgent[FinancialAuditOutput]):
    """Autonomous tool-calling agent auditing multi-year 10-K financial statements."""

    prompt_name = "financial_auditor"
    tools = [
        retrieve_10k_tables_tool,
        retrieve_multiyear_financial_series_tool,
        audit_financial_metrics_tool,
    ]
    output_schema = FinancialAuditOutput
    default_recursion_limit = 25

    def audit(self, ticker: str, fiscal_year: int) -> FinancialAuditOutput:
        """
        Direct programmatic interface for LangGraph orchestrator and standalone tests.
        Audits 10-K financial statements and returns a validated FinancialAuditOutput instance.
        """
        query = (
            f"Audit the SEC 10-K financial statements for {ticker.upper()} for fiscal year {fiscal_year}.\n"
            f"1. Retrieve Item 8 tables (Income Statement, Balance Sheet, Cash Flows) for FY{fiscal_year}.\n"
            f"2. Inspect multi-year series across filings to audit consistency and note any restatements in restatement_notes.\n"
            f"3. Extract the 3 contiguous years ({fiscal_year-2}, {fiscal_year-1}, {fiscal_year}) and latest balance sheet "
            f"standardized to $ Millions and shares in Millions.\n"
            f"4. Call audit_financial_metrics_tool EXACTLY ONCE with these extracted inputs.\n"
            f"5. Emit the complete FinancialAuditOutput artifact."
        )

        fallback_defaults = {
            "auditor_summary": "Audit completed.",
            "forensic_red_flags": [],
            "restatement_notes": [],
            "income_statement_markdown_table": "",
            "balance_sheet_markdown_table": "",
            "cash_flow_markdown_table": "",
        }

        return self.execute_structured(
            query,
            ticker=ticker,
            fiscal_year=fiscal_year,
            fallback_defaults=fallback_defaults,
        )

    def _pre_validate_data(
        self,
        data: Dict[str, Any],
        result: Dict[str, Any],
        **kwargs,
    ) -> Dict[str, Any]:
        """Overlays deterministic tool figures onto the dictionary before Pydantic validation."""
        messages = result.get("messages", [])
        tool_audit_data = self._extract_tool_audit_data(messages)
        if tool_audit_data:
            self._overlay_tool_data(data, tool_audit_data)
        return data

    def _post_process_output(
        self,
        output: FinancialAuditOutput,
        result: Dict[str, Any],
        **kwargs,
    ) -> FinancialAuditOutput:
        """Enforces deterministic tool math and backfills clean database markdown tables."""
        messages = result.get("messages", [])

        # 1. Backfill raw markdown tables from tool messages if empty
        if (
            not output.income_statement_markdown_table
            or not output.balance_sheet_markdown_table
            or not output.cash_flow_markdown_table
        ):
            extracted_tables = self._extract_tables_from_messages(messages)
            if not output.income_statement_markdown_table and "income_statement" in extracted_tables:
                output.income_statement_markdown_table = extracted_tables["income_statement"]
            if not output.balance_sheet_markdown_table and "balance_sheet" in extracted_tables:
                output.balance_sheet_markdown_table = extracted_tables["balance_sheet"]
            if not output.cash_flow_markdown_table and "cash_flow" in extracted_tables:
                output.cash_flow_markdown_table = extracted_tables["cash_flow"]

        # 2. Ensure mathematical fidelity: Always enforce deterministic math from the tool
        tool_audit_data = self._extract_tool_audit_data(messages)
        if tool_audit_data:
            if "multi_year_history" in tool_audit_data:
                output.multi_year_history = [
                    YearFinancialsResult.model_validate(y) for y in tool_audit_data["multi_year_history"]
                ]
            if "balance_sheet" in tool_audit_data:
                output.balance_sheet = BalanceSheetResult.model_validate(tool_audit_data["balance_sheet"])
            if "profitability_and_return_ratios" in tool_audit_data:
                output.profitability_and_return_ratios = ProfitabilityRatiosResult.model_validate(
                    tool_audit_data["profitability_and_return_ratios"]
                )
            if "solvency_and_liquidity_ratios" in tool_audit_data:
                output.solvency_and_liquidity_ratios = SolvencyRatiosResult.model_validate(
                    tool_audit_data["solvency_and_liquidity_ratios"]
                )
            if "forensic_red_flags" in tool_audit_data and not output.forensic_red_flags:
                output.forensic_red_flags = tool_audit_data["forensic_red_flags"]

        return output

    def _extract_tool_audit_data(self, messages: List[Any]) -> Optional[Dict[str, Any]]:
        """Finds and parses audit_financial_metrics tool output from messages."""
        for msg in messages:
            msg_name = getattr(msg, "name", "") or ""
            if "audit_financial_metrics" in msg_name:
                raw = getattr(msg, "content", "")
                if isinstance(raw, str):
                    try:
                        return json.loads(raw)
                    except Exception:
                        pass
                elif isinstance(raw, dict):
                    return raw
        return None

    def _extract_tables_from_messages(self, messages: List[Any]) -> Dict[str, str]:
        """Extracts pristine markdown tables directly from tool message payloads."""
        tables: Dict[str, str] = {}
        for msg in messages:
            msg_name = getattr(msg, "name", "") or ""
            if "retrieve_10k_tables" in msg_name or "retrieve_multiyear" in msg_name:
                raw_content = getattr(msg, "content", "")
                if isinstance(raw_content, str):
                    try:
                        parsed = json.loads(raw_content)
                        if isinstance(parsed, list):
                            for item in parsed:
                                if isinstance(item, dict) and "table_markdown" in item:
                                    tbl = item["table_markdown"]
                                    stype = item.get("statement_type")
                                    if stype and stype not in tables:
                                        tables[stype] = tbl
                                    tbl_lower = tbl.lower()
                                    if "operations" in tbl_lower or "net sales" in tbl_lower:
                                        tables.setdefault("income_statement", tbl)
                                    elif "balance sheets" in tbl_lower or "total assets" in tbl_lower:
                                        tables.setdefault("balance_sheet", tbl)
                                    elif "cash flows" in tbl_lower or "operating activities" in tbl_lower:
                                        tables.setdefault("cash_flow", tbl)
                    except Exception:
                        pass
        return tables

    def _overlay_tool_data(self, target: Dict[str, Any], tool_data: Dict[str, Any]):
        """Overlays verified mathematical figures from audit_financial_metrics_tool onto parsed dict."""
        for key in [
            "multi_year_history",
            "balance_sheet",
            "profitability_and_return_ratios",
            "solvency_and_liquidity_ratios",
        ]:
            if key in tool_data:
                target[key] = tool_data[key]
        if "forensic_red_flags" in tool_data and not target.get("forensic_red_flags"):
            target["forensic_red_flags"] = tool_data["forensic_red_flags"]
