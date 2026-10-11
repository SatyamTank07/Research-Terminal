"""State and Output Schemas for DCF Valuation Specialist Agent."""

from typing import Literal

from pydantic import Field

from app.agents.tools.valuation_model import (
    MarketInputs,
    ScenarioValuation,
    ValuationBuild,
    ValuationDataQuality,
)
from app.agents.tools.wacc_tools import WACCCalculationResult

WACCAudit = WACCCalculationResult

__all__ = [
    "DCFValuationOutput",
    "MarketInputs",
    "ScenarioValuation",
    "ValuationDataQuality",
    "WACCAudit",
]


class DCFValuationOutput(ValuationBuild):
    """Structured artifact emitted by the DCF Valuation Specialist. Every number is code-built."""

    valuation_summary: str = Field(
        ...,
        description="Commentary: valuation bridge and TV concentration, hurdle rate, scenario/sensitivity range, margin of safety",
    )
    commentary_source: Literal["llm", "template"] = Field(
        ..., description="'llm' = tool-free commentary call; 'template' = deterministic fallback"
    )
