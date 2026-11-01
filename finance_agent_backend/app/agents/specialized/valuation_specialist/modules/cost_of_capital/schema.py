"""Result types of the cost-of-capital module."""

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

from pydantic import BaseModel, Field

CostOfDebtSource = Literal[
    "explicit_provided",
    "derived_from_10k_interest_expense",
    "interest_expense_out_of_band_fallback",
    "institutional_credit_spread_fallback",
    "zero_debt_exemption",
]
CapitalStructureSource = Literal["market_value", "book_equity_fallback", "all_equity_no_equity_value"]
WACCSource = Literal["capm_blended", "user_override"]


class WACCCalculationResult(BaseModel):
    """Structured Pydantic model for WACC calculation output and institutional audit trail."""
    wacc: float = Field(..., description="Discount rate applied (user override or CAPM blend) as decimal")
    wacc_pct: float = Field(..., description="Discount rate applied as percentage")
    wacc_source: WACCSource
    capm_wacc_pct: float = Field(..., description="CAPM-blended WACC % (shown even when overridden)")
    risk_free_rate_pct: float
    risk_free_rate_source: str
    equity_risk_premium_pct: float
    equity_risk_premium_source: str
    beta_raw: float = Field(..., description="Beta as supplied")
    beta: float = Field(..., description="Beta used in CAPM (Blume-adjusted when requested)")
    beta_source: str
    cost_of_equity: float = Field(..., description="Cost of Equity (Ke) as decimal via CAPM")
    cost_of_equity_pct: float
    cost_of_debt_pre_tax: float
    cost_of_debt_pre_tax_pct: float
    cost_of_debt_after_tax: float
    cost_of_debt_after_tax_pct: float
    cost_of_debt_source: CostOfDebtSource
    cost_of_debt_note: Optional[str] = Field(None, description="Why a derived Kd was rejected, if it was")
    effective_tax_rate: float
    effective_tax_rate_pct: float
    equity_weight_pct: float
    debt_weight_pct: float
    capital_structure_source: CapitalStructureSource
    market_cap: Optional[float] = Field(None, description="Equity value (E) used for weighting in $ Millions")
    total_debt: float
    total_capital: float
    formula_breakdown_markdown: str


@dataclass(frozen=True)
class WACCComponents:
    """Unrounded CAPM and debt components. render.py formats these, never the rounded result."""

    risk_free_rate: float
    risk_free_rate_source: str
    equity_risk_premium: float
    equity_risk_premium_source: str
    beta_raw: float
    beta: float
    beta_source: str
    blume_adjusted: bool
    cost_of_equity: float
    cost_of_debt_pre_tax: float
    cost_of_debt_source: CostOfDebtSource
    cost_of_debt_note: Optional[str]
    interest_expense: Optional[float]
    credit_spread: float
    tax_rate: float
    cost_of_debt_after_tax: float
    equity_value: Optional[float]
    total_debt: float
    total_capital: float
    equity_weight: float
    debt_weight: float
    capital_structure_source: CapitalStructureSource
    capm_wacc: float
    wacc: float
    wacc_source: WACCSource


@dataclass(frozen=True)
class CostOfCapitalInputs:
    """Everything CAPM and the cost of debt need, with the source of each input."""

    beta: float
    beta_source: str
    apply_blume_adjustment: bool
    total_debt: float
    prior_total_debt: Optional[float]
    interest_expense: Optional[float]
    equity_value: Optional[float]
    equity_value_source: str
    risk_free_rate: float
    risk_free_rate_source: str
    equity_risk_premium: float
    equity_risk_premium_source: str
    marginal_tax_rate: float
    wacc_override: Optional[float]
    warnings: Tuple[str, ...]


@dataclass(frozen=True)
class CostOfCapital:
    """The discount rate with its audit trail, the marginal tax rate later modules reuse, and data-quality warnings."""

    wacc_audit: WACCCalculationResult
    marginal_tax_rate: float
    warnings: Tuple[str, ...]

    @property
    def wacc(self) -> float:
        return self.wacc_audit.wacc
