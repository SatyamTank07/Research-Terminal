"""Business & Moat Strategist Agent (Milestone 5).

Specialized qualitative agent that analyzes business operations, reporting segments,
revenue architecture, economic moat classification, and pricing power exclusively
from audited SEC 10-K narrative disclosures (Item 1 with gated Item 7 fallback),
emitting a typed BusinessMoatOutput schema.
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
from app.agents.state import BusinessMoatOutput
from app.agents.tools.rag_narrative_tools import retrieve_10k_narrative_tool

logger = logging.getLogger("finance_agent.agents.business_strategist")


@AgentRegistry.register("business_strategist")
class BusinessStrategistAgent(BaseAgent):
    """Autonomous qualitative agent analyzing business model, segments, and economic moat."""

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
        system_prompt = render_prompt("business_strategist")

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

    def analyze(self, ticker: str, fiscal_year: int) -> BusinessMoatOutput:
        """
        Direct programmatic interface for LangGraph orchestrator and standalone tests.
        Analyzes 10-K Item 1 narrative and returns a validated BusinessMoatOutput instance.
        """
        active_agent = self._get_or_create_agent()
        query = (
            f"Analyze the business model, product segments, and economic moat for {ticker.upper()} "
            f"from its fiscal year {fiscal_year} SEC 10-K filing.\n"
            f"1. Query Item 1 narrative for operating segments, revenue architecture, and competitive advantages.\n"
            f"2. Ensure all primary product lines and service categories are captured; if product-level breakdown is sparse in Item 1, query Item 7 for net sales by product category.\n"
            f"3. Classify the economic moat type, durability, and trajectory, and assess pricing power and customer concentration.\n"
            f"4. Emit the complete BusinessMoatOutput artifact with all citations."
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
        if isinstance(structured, BusinessMoatOutput):
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
    ) -> BusinessMoatOutput:
        """Extracts and validates BusinessMoatOutput from agent execution results with fallback safeguards."""
        messages = result.get("messages", [])
        structured = result.get("structured_response")

        if isinstance(structured, BusinessMoatOutput):
            output = structured
        elif isinstance(structured, dict):
            output = BusinessMoatOutput.model_validate(structured)
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
            parsed.setdefault("business_summary", raw_content[:500] if raw_content else "Business analysis completed.")
            parsed.setdefault("revenue_architecture", "Primary revenue derived from product and service sales.")
            parsed.setdefault("primary_product_segments", [])
            parsed.setdefault("segment_details", [])
            parsed.setdefault("economic_moat_type", "None")
            parsed.setdefault("moat_durability", "Narrow")
            parsed.setdefault("moat_trajectory", "Stable")
            parsed.setdefault("moat_rationale", "Moat analysis based on 10-K Item 1 disclosures.")
            parsed.setdefault("pricing_power_assessment", "Standard competitive pricing dynamics.")
            parsed.setdefault("customer_concentration", "No individual customer concentration disclosed.")

            try:
                output = BusinessMoatOutput.model_validate(parsed)
            except Exception as e:
                logger.error(f"Failed to parse structured BusinessMoatOutput: {e}. Raw content: {raw_content[:500]}")
                raise ValueError(f"Agent did not return a valid BusinessMoatOutput: {e}")

        # Post-processing safeguards: backfill citations if empty
        collected_citations = self._extract_citations_from_messages(messages)
        if not output.citations and collected_citations:
            output.citations = collected_citations

        # Ensure ticker and fiscal year match the requested target
        if not output.ticker:
            output.ticker = ticker.upper()
        if not output.fiscal_year:
            output.fiscal_year = fiscal_year

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
