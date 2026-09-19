"""Financial Auditor & Statement Analyst Agent (Milestone 2).

Specialized autonomous agent that audits 10-K financial statements (Item 8),
inspects cross-filing time-series for restatements, normalizes accounting scales,
executes deterministic math tools (Zero Math Hallucination), and emits a typed
FinancialAuditOutput schema.
"""

import json
import logging
from typing import Any, Dict, List, Optional, Union
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from app.agents.base import AgentOutput, BaseAgent
from app.agents.prompts import render_prompt
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
class FinancialAuditorAgent(BaseAgent):
    """Autonomous tool-calling agent auditing multi-year 10-K financial statements."""

    def __init__(
        self,
        model_name: str = "openai:gpt-4o-mini",
        recursion_limit: int = 25,
    ):
        self.model_name = model_name
        self.recursion_limit = recursion_limit
        self._cached_agent = None

    def _get_or_create_agent(self):
        load_dotenv(override=True)
        if self._cached_agent is not None:
            return self._cached_agent

        tools = [
            retrieve_10k_tables_tool,
            retrieve_multiyear_financial_series_tool,
            audit_financial_metrics_tool,
        ]

        system_prompt = render_prompt("financial_auditor")

        model_clean = self.model_name.replace("openai:", "")
        llm = ChatOpenAI(
            model=model_clean,
            temperature=0,
            max_retries=5,
        )

        agent = create_agent(
            model=llm,
            tools=tools,
            system_prompt=system_prompt,
        )

        self._cached_agent = agent
        return agent


    def audit(self, ticker: str, fiscal_year: int) -> FinancialAuditOutput:
        """
        Direct programmatic interface for LangGraph orchestrator and standalone tests.
        Audits 10-K financial statements and returns a validated FinancialAuditOutput instance.
        """
        active_agent = self._get_or_create_agent()
        query = (
            f"Audit the SEC 10-K financial statements for {ticker.upper()} for fiscal year {fiscal_year}.\n"
            f"1. Retrieve Item 8 tables (Income Statement, Balance Sheet, Cash Flows) for FY{fiscal_year}.\n"
            f"2. Inspect multi-year series across filings to audit consistency and note any restatements in restatement_notes.\n"
            f"3. Extract the 3 contiguous years ({fiscal_year-2}, {fiscal_year-1}, {fiscal_year}) and latest balance sheet "
            f"standardized to $ Millions and shares in Millions.\n"
            f"4. Call audit_financial_metrics_tool EXACTLY ONCE with these extracted inputs.\n"
            f"5. Emit the complete FinancialAuditOutput artifact."
        )

        result = active_agent.invoke(
            {"messages": [{"role": "user", "content": query}]},
            config={"recursion_limit": self.recursion_limit},
        )


        return self._extract_audit_output(result, ticker=ticker, fiscal_year=fiscal_year)

    def run(self, messages: List[Dict[str, str]]) -> AgentOutput:
        """Executes the agent with conversational history conforming to BaseAgent."""
        active_agent = self._get_or_create_agent()
        result = active_agent.invoke(
            {"messages": messages},
            config={"recursion_limit": self.recursion_limit},
        )

        structured = result.get("structured_response")
        if isinstance(structured, FinancialAuditOutput):
            content = json.dumps(structured.model_dump(), indent=2)
            sources = structured.citations
        elif isinstance(structured, dict):
            content = json.dumps(structured, indent=2)
            sources = structured.get("citations", [])
        else:
            last_msg = result["messages"][-1]
            content = getattr(last_msg, "content", str(last_msg))
            sources = self._extract_citations_from_messages(result.get("messages", []))

        return AgentOutput(content=content, sources=sources)

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

    def _extract_audit_output(
        self,
        result: Dict[str, Any],
        ticker: str,
        fiscal_year: int,
    ) -> FinancialAuditOutput:
        """Extracts and validates FinancialAuditOutput from agent execution results with fallback safeguards."""
        messages = result.get("messages", [])

        # 1. Extract verified mathematical outputs from audit_financial_metrics_tool
        tool_audit_data = None
        for msg in messages:
            msg_name = getattr(msg, "name", "") or ""
            if "audit_financial_metrics" in msg_name:
                raw = getattr(msg, "content", "")
                if isinstance(raw, str):
                    try:
                        tool_audit_data = json.loads(raw)
                    except Exception:
                        pass

        structured = result.get("structured_response")

        if isinstance(structured, FinancialAuditOutput):
            audit_output = structured
        elif isinstance(structured, dict):
            if tool_audit_data:
                self._overlay_tool_data(structured, tool_audit_data)
            audit_output = FinancialAuditOutput.model_validate(structured)
        else:
            # Fallback: Parse from last message content if model emitted JSON string
            last_msg = messages[-1] if messages else None
            raw_content = getattr(last_msg, "content", "") if last_msg else ""
            if isinstance(raw_content, list):
                raw_content = "".join(
                    item.get("text", "") if isinstance(item, dict) else str(item)
                    for item in raw_content
                )
            
            clean_json = raw_content.strip()
            if clean_json.startswith("```"):
                lines = clean_json.split("\n")
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].startswith("```"):
                    lines = lines[:-1]
                clean_json = "\n".join(lines).strip()

            parsed = {}
            if clean_json:
                try:
                    parsed = json.loads(clean_json)
                except Exception:
                    pass

            if tool_audit_data:
                self._overlay_tool_data(parsed, tool_audit_data)

            parsed.setdefault("ticker", ticker.upper())
            parsed.setdefault("fiscal_year", fiscal_year)
            parsed.setdefault("auditor_summary", raw_content[:1000] if raw_content else "Audit completed.")

            try:
                audit_output = FinancialAuditOutput.model_validate(parsed)
            except Exception as e:
                logger.error(f"Failed to parse structured audit output: {e}. Raw content: {raw_content[:500]}")
                raise ValueError(f"Agent did not return a valid FinancialAuditOutput: {e}")

        # Post-processing safeguards: backfill tables or citations from tool calls if empty
        collected_citations = self._extract_citations_from_messages(messages)
        if not audit_output.citations and collected_citations:
            audit_output.citations = collected_citations

        # Backfill raw markdown tables from tool messages if empty
        if (
            not audit_output.income_statement_markdown_table
            or not audit_output.balance_sheet_markdown_table
            or not audit_output.cash_flow_markdown_table
        ):
            extracted_tables = self._extract_tables_from_messages(messages)
            if not audit_output.income_statement_markdown_table and "income_statement" in extracted_tables:
                audit_output.income_statement_markdown_table = extracted_tables["income_statement"]
            if not audit_output.balance_sheet_markdown_table and "balance_sheet" in extracted_tables:
                audit_output.balance_sheet_markdown_table = extracted_tables["balance_sheet"]
            if not audit_output.cash_flow_markdown_table and "cash_flow" in extracted_tables:
                audit_output.cash_flow_markdown_table = extracted_tables["cash_flow"]

        # Ensure mathematical fidelity: Always enforce deterministic math from the tool
        if tool_audit_data:
            if "multi_year_history" in tool_audit_data:
                audit_output.multi_year_history = [
                    YearFinancialsResult.model_validate(y) for y in tool_audit_data["multi_year_history"]
                ]
            if "balance_sheet" in tool_audit_data:
                audit_output.balance_sheet = BalanceSheetResult.model_validate(tool_audit_data["balance_sheet"])
            if "profitability_and_return_ratios" in tool_audit_data:
                audit_output.profitability_and_return_ratios = ProfitabilityRatiosResult.model_validate(
                    tool_audit_data["profitability_and_return_ratios"]
                )
            if "solvency_and_liquidity_ratios" in tool_audit_data:
                audit_output.solvency_and_liquidity_ratios = SolvencyRatiosResult.model_validate(
                    tool_audit_data["solvency_and_liquidity_ratios"]
                )
            if "forensic_red_flags" in tool_audit_data and not audit_output.forensic_red_flags:
                audit_output.forensic_red_flags = tool_audit_data["forensic_red_flags"]


        # Ensure ticker and fiscal year match the requested target
        if not audit_output.ticker:
            audit_output.ticker = ticker.upper()
        if not audit_output.fiscal_year:
            audit_output.fiscal_year = fiscal_year

        return audit_output

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


    def _extract_citations_from_messages(self, messages: List[Any]) -> List[Dict[str, Any]]:
        """Extracts table chunk citation references from tool messages."""
        citations: List[Dict[str, Any]] = []
        seen_chunk_ids = set()

        for msg in messages:
            msg_name = getattr(msg, "name", "") or ""
            if "retrieve_10k_tables" in msg_name or "retrieve_multiyear" in msg_name:
                raw_content = getattr(msg, "content", "")
                if isinstance(raw_content, str):
                    try:
                        parsed = json.loads(raw_content)
                        if isinstance(parsed, list):
                            for item in parsed:
                                if isinstance(item, dict) and "chunk_id" in item:
                                    cid = item.get("chunk_id")
                                    if cid and cid not in seen_chunk_ids:
                                        seen_chunk_ids.add(cid)
                                        citations.append({
                                            "chunk_id": cid,
                                            "ticker": item.get("ticker"),
                                            "fiscal_year": item.get("fiscal_year"),
                                            "item": item.get("item"),
                                            "breadcrumb": item.get("breadcrumb"),
                                            "sub_section": item.get("sub_section"),
                                        })
                    except Exception:
                        pass

        return citations
