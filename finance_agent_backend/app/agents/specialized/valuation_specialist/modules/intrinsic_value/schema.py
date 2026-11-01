"""Result types of the intrinsic-value module."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple

if TYPE_CHECKING:
    from app.agents.specialized.forecasting_analyst.state_forecasting_analyst import ForecastDiagnostics, ForecastYear

TerminalMethod = Literal["value_driver", "last_fcf_growth"]
ForecastSource = Literal["forecasting_analyst", "naive_5pct_fallback"]


@dataclass(frozen=True)
class DCFCase:
    """Validated inputs that fix one DCF; value_point evaluates it at any discount rate and growth."""

    fcfs: Tuple[float, ...]
    wacc: float
    terminal_growth_rate: float
    terminal_method: TerminalMethod
    terminal_nopat: Optional[float]
    ronic: Optional[float]
    offset: float                     # 0.5 mid-year, 0.0 year-end
    net_debt: float                   # negative = net cash
    noncontrolling_interests: float
    roll_forward_years: float
    claims_roll_factor: float         # (1 + Rf) ^ years, applied to net debt and NCI
    diluted_shares: float


@dataclass(frozen=True)
class DCFPoint:
    """One unrounded DCF evaluation ($ millions; per_share in $)."""

    tv_fcf: float
    pv_fcfs: float
    tv: float
    pv_tv: float
    ev: float
    equity_bs: float
    equity: float
    per_share: float


@dataclass(frozen=True)
class DCFTerms:
    """The discount-rate, perpetuity and bridge terms every DCF of one valuation shares."""

    wacc: float
    terminal_growth_rate: float
    net_debt: float
    diluted_shares: float
    ronic: Optional[float]
    noncontrolling_interests: float
    roll_forward_years: float
    risk_free_rate: float
    mid_year_convention: bool = True

    def as_kwargs(self, *, without: Tuple[str, ...] = ()) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if k not in without}


@dataclass(frozen=True)
class DCFValue:
    """A DCF's rounded figures ($ millions; per-share in $) plus the unrounded case and base point."""

    enterprise_value: float
    pv_explicit_fcfs: float
    terminal_value: float                       # undiscounted perpetuity value at year N
    pv_terminal_value: float
    terminal_value_pct_of_ev: float
    terminal_method: TerminalMethod
    terminal_fcf: float                         # year N+1 cash flow capitalized in the terminal value
    ronic: Optional[float]                      # set only for the value-driver terminal value
    implied_terminal_ev_ebitda: Optional[float]
    implied_terminal_ev_nopat: Optional[float]
    net_debt: float                             # negative = net cash surplus
    noncontrolling_interests: float
    equity_value_at_balance_sheet_date: float
    roll_forward_years: float
    roll_forward_factor: float                  # (1 + WACC) ^ years, applied to enterprise value
    claims_roll_forward_factor: float           # (1 + Rf) ^ years, applied to net debt and NCI
    equity_value: float                         # at the valuation date
    diluted_shares: float
    fair_value_per_share: float
    wacc: float
    terminal_growth_rate: float
    forecast_years_count: int
    discounting_convention: str                 # 'mid_year' or 'year_end'
    equity_bridge_markdown: str
    warnings: Tuple[str, ...]
    case: DCFCase
    point: DCFPoint


@dataclass(frozen=True)
class IntrinsicValueInputs:
    """What the base-case DCF values, read from the forecast and the audited balance sheet."""

    projected_fcfs: List[float]
    forecast_source: ForecastSource
    cash_flow_warnings: Tuple[str, ...]         # placeholder-forecast warning
    final_year: Optional["ForecastYear"]
    forecast_diagnostics: Optional["ForecastDiagnostics"]
    net_debt: float
    noncontrolling_interests: float
    diluted_shares: float
    base_ebitda: Optional[float]
    balance_sheet_date: Optional[date]
    valuation_date: date
    risk_free_rate: float


@dataclass(frozen=True)
class IntrinsicValue:
    """The base-case DCF, the terms every later DCF of this valuation reuses, and its data-quality warnings."""

    dcf: DCFValue
    terms: DCFTerms
    projected_fcfs: List[float]
    forecast_source: ForecastSource
    terminal_nopat: Optional[float]             # final-year NOPAT re-taxed at the marginal rate
    balance_sheet_date: Optional[date]
    implied_ev_ebitda: Optional[float]          # EV / audited base-year EBITDA
    cash_flow_warnings: Tuple[str, ...]
    warnings: Tuple[str, ...]                   # roll-forward, terminal RONIC, DCF sanity, in that order
