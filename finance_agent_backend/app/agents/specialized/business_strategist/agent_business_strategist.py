"""Business & Moat Strategist Agent.

Specialized qualitative agent that analyzes business operations, reporting segments (ASC 280),
disaggregated revenue streams (ASC 606), multi-faceted economic moats, competitive threats,
and pricing power strictly from audited SEC 10-K filings, emitting a typed BusinessMoatOutput artifact.
"""

from typing import Any, List, Optional
from app.agents.base import StructuredAgent
from app.agents.registry import AgentRegistry
from app.agents.specialized.prompts import render_prompt
from app.agents.specialized.business_strategist.state_business_strategist import BusinessMoatOutput
from app.agents.tools.rag_narrative_tools import retrieve_10k_narrative_tool
from app.agents.tools.rag_table_tools import retrieve_10k_tables_tool


@AgentRegistry.register("business_strategist")
class BusinessStrategistAgent(StructuredAgent[BusinessMoatOutput]):
    """Autonomous qualitative agent analyzing business model, segments, and economic moat."""

    prompt_name = "business_strategist"
    tools = [retrieve_10k_narrative_tool, retrieve_10k_tables_tool]
    output_schema = BusinessMoatOutput

    def analyze(self, ticker: str, fiscal_year: int, callbacks: Optional[list] = None) -> BusinessMoatOutput:
        """
        Direct programmatic interface for LangGraph orchestrator and standalone tests.
        Analyzes 10-K Item 1 narrative and segment disclosures and returns a validated BusinessMoatOutput instance.
        """
        query = render_prompt(
            "prompt_business_strategist_query.j2",
            ticker=ticker.upper(),
            fiscal_year=fiscal_year,
        )

        fallback_defaults = {
            "business_summary": "Business operations analysis completed.",
            "revenue_architecture": "Primary revenue derived from commercial product and service sales.",
            "reportable_segments": [],
            "primary_product_segments": [],
            "segment_details": [],
            "economic_moat_type": "None",
            "secondary_moat_types": [],
            "segment_specific_moats": [],
            "moat_durability": "Indeterminate",
            "moat_trajectory": "Indeterminate",
            "moat_rationale": "Moat analysis based on 10-K Item 1 disclosures.",
            "competitive_threats": [],
            "pricing_power_assessment": "Not explicitly disclosed in 10-K.",
            "customer_concentration": "Not explicitly disclosed in 10-K.",
            "citations": [],
        }

        return self.execute_structured(
            query,
            ticker=ticker,
            fiscal_year=fiscal_year,
            fallback_defaults=fallback_defaults,
            callbacks=callbacks,
        )
