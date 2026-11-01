"""Output schema of the Valuation Specialist: every figure is code-built; only valuation_summary is written by the LLM."""

from typing import List, Literal, Optional

from pydantic import BaseModel, Field

from app.agents.specialized.valuation_specialist.modules.cost_of_capital.schema import WACCCalculationResult
from app.agents.specialized.valuation_specialist.modules.market_implied_expectations.schema import ReverseDCFStatus
from app.agents.specialized.valuation_specialist.modules.scenario_values.schema import ScenarioValuation
from app.agents.specialized.valuation_specialist.modules.valuation_stance.schema import ValuationStance

WACCAudit = WACCCalculationResult

__all__ = [
    "DCFValuationOutput",
    "ReverseDCFStatus",
    "ScenarioValuation",
    "ValuationBuild",
    "ValuationDataQuality",
    "ValuationStance",
    "WACCAudit",
]


class ValuationDataQuality(BaseModel):
    forecast_source: Literal["forecasting_analyst", "naive_5pct_fallback"]
    market_data_source: str
    share_price_source: str
    warnings: List[str] = Field(default_factory=list)


class ValuationBuild(BaseModel):
    """Every deterministic figure of the DCF valuation ($ Millions unless stated)."""

    ticker: str
    fiscal_year: int
    valuation_date: str = Field(..., description="ISO date the value is stated at")
    balance_sheet_date: Optional[str] = Field(None, description="ISO fiscal year-end of the anchor 10-K")
    roll_forward_years: float = 0.0

    wacc_audit: WACCCalculationResult
    terminal_growth_rate: float
    terminal_tax_rate_pct: float = Field(..., description="Marginal tax rate applied to terminal NOPAT and the debt shield")
    discounting_convention: str
    terminal_method: str
    terminal_fcf: float
    ronic_used_pct: Optional[float] = None
    implied_terminal_ev_ebitda: Optional[float] = Field(
        None, description="Terminal value / final-year EBITDA; warned outside 5x-30x"
    )
    implied_terminal_ev_nopat: Optional[float] = Field(
        None, description="Terminal value / final-year NOPAT; warned outside 6x-35x"
    )

    projected_fcfs: List[float]
    pv_explicit_fcfs: float
    pv_terminal_value: float
    terminal_value_pct_of_ev: float
    enterprise_value: float
    net_debt: float = Field(..., description="Negative indicates a net cash surplus")
    noncontrolling_interests: float = 0.0
    equity_value: float = Field(..., description="Equity value at the valuation date")
    diluted_shares: float

    implied_fair_value_per_share: float = Field(..., description="Base-case fair value per share ($)")
    current_share_price: Optional[float] = None
    upside_downside_pct: Optional[float] = None
    valuation_stance: Optional[ValuationStance] = None
    implied_ev_ebitda: Optional[float] = Field(None, description="EV / audited base-year EBITDA")

    scenario_valuations: List[ScenarioValuation] = Field(default_factory=list)
    probability_weighted_fair_value: Optional[float] = None
    bear_fair_value: Optional[float] = None
    bull_fair_value: Optional[float] = None

    market_implied_growth_pct: Optional[float] = Field(
        None,
        description="Reverse DCF: constant explicit-period revenue growth (whole %) at which the base DCF equals "
        "the share price, all other inputs held; None unless reverse_dcf_status == 'solved'",
    )
    modeled_revenue_cagr_pct: Optional[float] = Field(
        None, description="Base-case revenue CAGR over the same explicit horizon (whole %), for comparison"
    )
    reverse_dcf_horizon_years: Optional[int] = Field(None, description="Explicit years the implied growth applies to")
    market_implied_margin_shift_pp: Optional[float] = Field(
        None,
        description="Reverse DCF companion: uniform operating-margin shift (percentage points, every explicit year) "
        "at modeled growth that equates the base DCF to the price; None without a price or a root in -30..+50 pp",
    )
    modeled_final_margin_pct: Optional[float] = Field(
        None, description="Base-case final explicit-year operating margin (whole %), for the margin comparison"
    )
    reverse_dcf_status: ReverseDCFStatus = Field(
        "price_unavailable",
        description="'solved'; 'price_above_range' / 'price_below_range' (no growth in -20%..60% reaches the price); "
        "'no_solution' (no finite DCF values); 'price_unavailable'; 'forecast_unavailable'",
    )
    reverse_dcf_summary: str = Field("", description="One-line reverse-DCF result or the reason there is none")
    market_implied_wacc_pct: Optional[float] = Field(
        None, description="Discount rate at which the base-case cash flows and terminal value equal the share price (%)"
    )

    sensitivity_matrix_markdown: str
    ronic_sensitivity_matrix_markdown: str = Field(
        "", description="WACC x terminal RONIC grid of fair value per share ('' unless the TV is value-driver)"
    )
    equity_bridge_markdown: str
    scenario_table_markdown: str = ""
    data_quality: ValuationDataQuality


class DCFValuationOutput(ValuationBuild):
    """Structured artifact emitted by the DCF Valuation Specialist. Every number is code-built."""

    valuation_summary: str = Field(
        ...,
        description="Commentary: valuation bridge and TV concentration, hurdle rate, scenario/sensitivity range, margin of safety",
    )
    commentary_source: Literal["llm", "template"] = Field(
        ..., description="'llm' = tool-free commentary call; 'template' = deterministic fallback"
    )
