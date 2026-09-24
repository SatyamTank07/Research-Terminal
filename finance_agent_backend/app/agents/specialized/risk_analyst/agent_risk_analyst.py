"""Risk & Red Flag Analyst Agent.

Specialized qualitative agent that analyzes material operational, regulatory,
supply chain, macroeconomic, and technological risks exclusively from audited
SEC 10-K Item 1A disclosures (with gated Item 3 fallback for explicit litigation cross-references),
emitting a typed RiskAuditOutput schema.
"""

from typing import Any, Dict, List, Optional
from app.agents.base import StructuredAgent
from app.agents.registry import AgentRegistry
from app.agents.specialized.risk_analyst.state_risk_analyst import RiskAuditOutput
from app.agents.tools.rag_narrative_tools import retrieve_10k_narrative_tool



@AgentRegistry.register("risk_analyst")
class RiskAnalystAgent(StructuredAgent[RiskAuditOutput]):
    """Autonomous qualitative agent analyzing Item 1A risk disclosures and existential threats."""

    prompt_name = "risk_analyst"
    tools = [retrieve_10k_narrative_tool]
    output_schema = RiskAuditOutput

    def analyze(self, ticker: str, fiscal_year: int, callbacks: Optional[List[Any]] = None) -> RiskAuditOutput:
        """
        Direct programmatic interface for LangGraph orchestrator and standalone tests.
        Analyzes 10-K Item 1A narrative and returns a validated RiskAuditOutput instance.
        """
        query = (
            f"Analyze the risk profile for {ticker.upper()} from its fiscal year {fiscal_year} SEC 10-K filing.\n"
            f"1. Query Item 1A narrative across supply chain, regulatory, technological, and macroeconomic vectors.\n"
            f"2. If an active major lawsuit or regulatory action explicitly cross-references Item 3, apply the gated check on Item 3.\n"
            f"3. Curate 5 to 8 non-boilerplate risks sorted by severity (Severe -> Moderate -> Low) with mitigating factors.\n"
            f"4. Identify the single primary existential threat and assign the overall risk profile.\n"
            f"5. Emit the complete RiskAuditOutput artifact with all citations."
        )

        fallback_defaults = {
            "identified_risks": [],
            "primary_existential_threat": "Risk audit based on 10-K Item 1A disclosures.",
            "overall_risk_profile": "Moderate",
        }

        return self.execute_structured(
            query,
            ticker=ticker,
            fiscal_year=fiscal_year,
            fallback_defaults=fallback_defaults,
            callbacks=callbacks,
        )

    def _post_process_output(
        self,
        output: RiskAuditOutput,
        result: Dict[str, Any],
        **kwargs,
    ) -> RiskAuditOutput:
        """Sorts identified risks by severity descending: Severe -> Moderate -> Low."""
        severity_rank = {"Severe": 0, "Moderate": 1, "Low": 2}
        output.identified_risks.sort(key=lambda r: severity_rank.get(r.severity, 3))
        return output
