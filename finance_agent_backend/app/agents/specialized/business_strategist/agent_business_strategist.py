"""Business & Moat Strategist Agent.

Specialized qualitative agent that analyzes business operations, reporting segments,
revenue architecture, economic moat classification, and pricing power exclusively
from audited SEC 10-K narrative disclosures (Item 1 with gated Item 7 fallback),
emitting a typed BusinessMoatOutput schema.
"""

from app.agents.base import StructuredAgent
from app.agents.registry import AgentRegistry
from app.agents.state import BusinessMoatOutput
from app.agents.tools.rag_narrative_tools import retrieve_10k_narrative_tool


@AgentRegistry.register("business_strategist")
class BusinessStrategistAgent(StructuredAgent[BusinessMoatOutput]):
    """Autonomous qualitative agent analyzing business model, segments, and economic moat."""

    prompt_name = "business_strategist"
    tools = [retrieve_10k_narrative_tool]
    output_schema = BusinessMoatOutput

    def analyze(self, ticker: str, fiscal_year: int) -> BusinessMoatOutput:
        """
        Direct programmatic interface for LangGraph orchestrator and standalone tests.
        Analyzes 10-K Item 1 narrative and returns a validated BusinessMoatOutput instance.
        """
        query = (
            f"Analyze the business model, product segments, and economic moat for {ticker.upper()} "
            f"from its fiscal year {fiscal_year} SEC 10-K filing.\n"
            f"1. Query Item 1 narrative for operating segments, revenue architecture, and competitive advantages.\n"
            f"2. Ensure all primary product lines and service categories are captured; if product-level breakdown is sparse in Item 1, query Item 7 for net sales by product category.\n"
            f"3. Classify the economic moat type, durability, and trajectory, and assess pricing power and customer concentration.\n"
            f"4. Emit the complete BusinessMoatOutput artifact with all citations."
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
        )
