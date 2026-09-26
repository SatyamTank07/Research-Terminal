"""Business & Moat Strategist Agent.

Specialized qualitative agent that analyzes business operations, reporting segments,
revenue architecture, economic moat classification, and pricing power exclusively
from audited SEC 10-K narrative disclosures (Item 1 with gated Item 7 fallback),
emitting a typed BusinessMoatOutput schema.
"""

from typing import Any, List, Optional
from app.agents.base import StructuredAgent
from app.agents.registry import AgentRegistry
from app.agents.specialized.prompts import render_prompt
from app.agents.specialized.business_strategist.state_business_strategist import BusinessMoatOutput
from app.agents.tools.rag_narrative_tools import retrieve_10k_narrative_tool



@AgentRegistry.register("business_strategist")
class BusinessStrategistAgent(StructuredAgent[BusinessMoatOutput]):
    """Autonomous qualitative agent analyzing business model, segments, and economic moat."""

    prompt_name = "business_strategist"
    tools = [retrieve_10k_narrative_tool]
    output_schema = BusinessMoatOutput

    def analyze(self, ticker: str, fiscal_year: int, callbacks: Optional[list] = None) -> BusinessMoatOutput:
        """
        Direct programmatic interface for LangGraph orchestrator and standalone tests.
        Analyzes 10-K Item 1 narrative and returns a validated BusinessMoatOutput instance.
        """
        query = render_prompt(
            "prompt_business_strategist_query.j2",
            ticker=ticker.upper(),
            fiscal_year=fiscal_year,
        )

        fallback_defaults = {
            "business_summary": "Business analysis completed.",
            "revenue_architecture": "Primary revenue derived from product and service sales.",
            "primary_product_segments": [],
            "segment_details": [],
            "economic_moat_type": "None",
            "moat_durability": "Narrow",
            "moat_trajectory": "Stable",
            "moat_rationale": "Moat analysis based on 10-K Item 1 disclosures.",
            "pricing_power_assessment": "Standard competitive pricing dynamics.",
            "customer_concentration": "No individual customer concentration disclosed.",
        }

        return self.execute_structured(
            query,
            ticker=ticker,
            fiscal_year=fiscal_year,
            fallback_defaults=fallback_defaults,
            callbacks=callbacks,
        )
