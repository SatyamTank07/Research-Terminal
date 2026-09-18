"""Risk & Red Flag Analyst Agent (Milestone 5).

Specialized qualitative agent that analyzes material operational, regulatory,
supply chain, macroeconomic, and technological risks exclusively from audited
SEC 10-K Item 1A disclosures (with gated Item 3 fallback for explicit litigation cross-references),
emitting a typed RiskAuditOutput schema.
"""

import json
import logging
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from app.agents.base import AgentOutput, BaseAgent
from app.agents.prompts import render_prompt
from app.agents.registry import AgentRegistry
from app.agents.state import RiskAuditOutput, RiskItem
from app.agents.tools.rag_narrative_tools import retrieve_10k_narrative_tool

logger = logging.getLogger("finance_agent.agents.risk_analyst")


@AgentRegistry.register("risk_analyst")
class RiskAnalystAgent(BaseAgent):
    """Autonomous qualitative agent analyzing Item 1A risk disclosures and existential threats."""

    def __init__(
        self,
        model_name: str = "openai:gpt-4o-mini",
        recursion_limit: int = 50,
    ):
        self.model_name = model_name
        self.recursion_limit = recursion_limit
        self._cached_agent = None

    def _get_or_create_agent(self):
        load_dotenv(override=True)
        if self._cached_agent is not None:
            return self._cached_agent

        tools = [retrieve_10k_narrative_tool]
        system_prompt = render_prompt("risk_analyst")

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

    def analyze(self, ticker: str, fiscal_year: int) -> RiskAuditOutput:
        """
        Direct programmatic interface for LangGraph orchestrator and standalone tests.
        Analyzes 10-K Item 1A narrative and returns a validated RiskAuditOutput instance.
        """
        active_agent = self._get_or_create_agent()
        query = (
            f"Analyze the risk profile for {ticker.upper()} from its fiscal year {fiscal_year} SEC 10-K filing.\n"
            f"1. Query Item 1A narrative across supply chain, regulatory, technological, and macroeconomic vectors.\n"
            f"2. If an active major lawsuit or regulatory action explicitly cross-references Item 3, apply the gated check on Item 3.\n"
            f"3. Curate 5 to 8 non-boilerplate risks sorted by severity (Severe -> Moderate -> Low) with mitigating factors.\n"
            f"4. Identify the single primary existential threat and assign the overall risk profile.\n"
            f"5. Emit the complete RiskAuditOutput artifact with all citations."
        )

        result = active_agent.invoke(
            {"messages": [{"role": "user", "content": query}]},
            config={"recursion_limit": self.recursion_limit},
        )

        return self._extract_output(result, ticker=ticker, fiscal_year=fiscal_year)

    def run(self, messages: List[Dict[str, str]]) -> AgentOutput:
        """Executes the agent with conversational history conforming to BaseAgent."""
        active_agent = self._get_or_create_agent()
        result = active_agent.invoke(
            {"messages": messages},
            config={"recursion_limit": self.recursion_limit},
        )

        structured = result.get("structured_response")
        if isinstance(structured, RiskAuditOutput):
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

    def _extract_output(
        self,
        result: Dict[str, Any],
        ticker: str,
        fiscal_year: int,
    ) -> RiskAuditOutput:
        """Extracts and validates RiskAuditOutput from agent execution results with fallback safeguards."""
        messages = result.get("messages", [])
        structured = result.get("structured_response")

        if isinstance(structured, RiskAuditOutput):
            output = structured
        elif isinstance(structured, dict):
            output = RiskAuditOutput.model_validate(structured)
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

            parsed.setdefault("ticker", ticker.upper())
            parsed.setdefault("fiscal_year", fiscal_year)
            parsed.setdefault("identified_risks", [])
            parsed.setdefault("primary_existential_threat", "Risk audit based on 10-K Item 1A disclosures.")
            parsed.setdefault("overall_risk_profile", "Moderate")

            try:
                output = RiskAuditOutput.model_validate(parsed)
            except Exception as e:
                logger.error(f"Failed to parse structured RiskAuditOutput: {e}. Raw content: {raw_content[:500]}")
                raise ValueError(f"Agent did not return a valid RiskAuditOutput: {e}")

        # Post-processing safeguards: backfill citations if empty
        collected_citations = self._extract_citations_from_messages(messages)
        if not output.citations and collected_citations:
            output.citations = collected_citations

        # Ensure ticker and fiscal year match the requested target
        if not output.ticker:
            output.ticker = ticker.upper()
        if not output.fiscal_year:
            output.fiscal_year = fiscal_year

        # Sort identified risks by severity descending: Severe -> Moderate -> Low
        severity_rank = {"Severe": 0, "Moderate": 1, "Low": 2}
        output.identified_risks.sort(key=lambda r: severity_rank.get(r.severity, 3))

        return output

    def _extract_citations_from_messages(self, messages: List[Any]) -> List[Dict[str, Any]]:
        """Extracts narrative chunk citation references from tool messages."""
        citations: List[Dict[str, Any]] = []
        seen_chunk_ids = set()

        for msg in messages:
            msg_name = getattr(msg, "name", "") or ""
            if "retrieve_10k_narrative" in msg_name:
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
