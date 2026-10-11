"""Deterministic Weighted Average Cost of Capital (WACC) & CAPM Engine.

Enforces Architectural Principle #2: Zero Mathematical Hallucination.
Offloads all discount rate derivations to pure, deterministic Python functions.
Computes Cost of Equity (Ke) via the Capital Asset Pricing Model (CAPM) with an optional
Blume beta adjustment, evaluates pre-tax and after-tax Cost of Debt (Kd) from interest
expense over average debt (with a sanity band and a credit-spread fallback), weights the
capital structure, and records the provenance of every input.
"""

import logging
import math
from typing import Any, Dict, Literal, Optional

from langchain_core.tools import tool
from pydantic import AliasChoices, BaseModel, Field, model_validator

from app.config import settings

logger = logging.getLogger("finance_agent.tools.wacc")

KD_CEILING = 0.15
KD_FLOOR_BELOW_RF = 0.01
BLUME_RAW_WEIGHT = 0.67


# ==============================================================================
# 1. Validation Helpers
# ==============================================================================
def _validate_finite_float(val: Any, name: str) -> float:
    """Strictly validates that a numeric input is a real, finite number (not NaN or Inf)."""
    if val is None:
        raise ValueError(f"'{name}' cannot be None.")
    try:
        f_val = float(val)
    except (ValueError, TypeError):
        raise ValueError(f"'{name}' must be a valid numeric float, got: {val!r}")

    if math.isnan(f_val):
        raise ValueError(f"'{name}' cannot be NaN.")
    if math.isinf(f_val):
        raise ValueError(f"'{name}' cannot be infinite.")
    return f_val


def _normalize_rate(val: Optional[float], name: str, default: Optional[float] = None) -> Optional[float]:
    """
    Normalizes rate inputs between decimal (e.g. 0.085) and percentage (e.g. 8.5).
    If val > 1.0, treats it as a percentage and divides by 100.0.
    """
    if val is None:
        return default
    f_val = _validate_finite_float(val, name)
    if f_val < 0.0:
        raise ValueError(f"'{name}' cannot be negative, got: {f_val}")
    if f_val > 1.0:
        # User passed 4.2 instead of 0.042 (4.2%)
        return f_val / 100.0
    return f_val


def blume_adjust(raw_beta: float) -> float:
    """Blume (1971) mean-reversion adjustment used by Bloomberg: 0.67 x raw + 0.33 x 1.0."""
    return BLUME_RAW_WEIGHT * raw_beta + (1.0 - BLUME_RAW_WEIGHT)


# ==============================================================================
# 2. Input and Output Schemas
# ==============================================================================
CostOfDebtSource = Literal[
    "explicit_provided",
    "derived_from_10k_interest_expense",
    "interest_expense_out_of_band_fallback",
    "institutional_credit_spread_fallback",
    "zero_debt_exemption",
]
CapitalStructureSource = Literal["market_value", "book_equity_fallback", "all_equity_no_equity_value"]
WACCSource = Literal["capm_blended", "user_override"]


class WACCInput(BaseModel):
    """
    Structured input container for WACC calculation.
    Supports passing either market_cap directly OR (share_price + diluted_shares).
    When neither is available the capital structure is weighted 100% equity.
    """
    beta: float = Field(..., validation_alias=AliasChoices("beta", "equity_beta"))
    total_debt: float = Field(..., ge=0.0, validation_alias=AliasChoices("total_debt", "gross_debt", "debt"))
    prior_total_debt: Optional[float] = Field(None, ge=0.0)
    market_cap: Optional[float] = Field(None, validation_alias=AliasChoices("market_cap", "mkt_cap"))
    share_price: Optional[float] = Field(None, validation_alias=AliasChoices("share_price", "stock_price"))
    diluted_shares: Optional[float] = Field(None, validation_alias=AliasChoices("diluted_shares", "shares_outstanding"))
    risk_free_rate: float = Field(settings.DEFAULT_RISK_FREE_RATE, validation_alias=AliasChoices("risk_free_rate", "rf"))
    equity_risk_premium: float = Field(
        settings.DEFAULT_EQUITY_RISK_PREMIUM, validation_alias=AliasChoices("equity_risk_premium", "erp")
    )
    tax_rate: float = Field(0.21, validation_alias=AliasChoices("tax_rate", "effective_tax_rate"))
    cost_of_debt: Optional[float] = Field(None, validation_alias=AliasChoices("cost_of_debt", "kd"))
    interest_expense: Optional[float] = Field(None, validation_alias=AliasChoices("interest_expense", "interest"))
    credit_spread_fallback: float = Field(settings.DEFAULT_CREDIT_SPREAD)

    @model_validator(mode="before")
    @classmethod
    def validate_and_normalize(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        # 1. Finite float validation
        for field_name in ["beta", "total_debt"]:
            if field_name in data and data[field_name] is not None:
                data[field_name] = _validate_finite_float(data[field_name], field_name)
        if data.get("prior_total_debt") is not None:
            data["prior_total_debt"] = _validate_finite_float(data["prior_total_debt"], "prior_total_debt")

        beta_val = data.get("beta", 0.0)
        if beta_val <= 0.0 or beta_val > 5.0:
            raise ValueError(
                f"beta ({beta_val:.2f}) is outside realistic equity market bounds (0.0 < beta <= 5.0). "
                f"Real-world public equity betas virtually never exceed 3.5-4.0. "
                f"Please verify that another metric (such as P/E, revenue, or share price) was not misread as beta."
            )
        if data.get("total_debt", 0.0) < 0.0:
            raise ValueError(f"total_debt must be non-negative (>= 0), got: {data.get('total_debt')}")

        # 2. Rate auto-normalization
        for name, default in (
            ("risk_free_rate", settings.DEFAULT_RISK_FREE_RATE),
            ("equity_risk_premium", settings.DEFAULT_EQUITY_RISK_PREMIUM),
            ("tax_rate", 0.21),
            ("credit_spread_fallback", settings.DEFAULT_CREDIT_SPREAD),
        ):
            data[name] = _normalize_rate(data.get(name), name, default=default)

        if data.get("cost_of_debt") is not None:
            data["cost_of_debt"] = _normalize_rate(data["cost_of_debt"], "cost_of_debt")
            if data["cost_of_debt"] > 0.30:
                raise ValueError(
                    f"cost_of_debt ({data['cost_of_debt']*100:.2f}%) exceeds realistic borrowing limits (max 30.0%). "
                    f"Please verify the pre-tax borrowing rate."
                )

        if data.get("interest_expense") is not None:
            data["interest_expense"] = abs(_validate_finite_float(data["interest_expense"], "interest_expense"))

        # 3. Market Cap resolution (None => all-equity weighting downstream)
        mkt_cap = data.get("market_cap")
        price = data.get("share_price")
        shares = data.get("diluted_shares")

        if mkt_cap is not None:
            mkt_cap_val = _validate_finite_float(mkt_cap, "market_cap")
            if mkt_cap_val <= 0.0:
                raise ValueError(f"market_cap must be strictly positive (> 0), got: {mkt_cap_val}")
            data["market_cap"] = mkt_cap_val
        elif price is not None and shares is not None:
            price_val = _validate_finite_float(price, "share_price")
            shares_val = _validate_finite_float(shares, "diluted_shares")
            if price_val <= 0.0:
                raise ValueError(f"share_price must be strictly positive (> 0), got: {price_val}")
            if shares_val <= 0.0:
                raise ValueError(f"diluted_shares must be strictly positive (> 0), got: {shares_val}")
            if shares_val > 100_000.0:
                raise ValueError(
                    f"diluted_shares ({shares_val:,.2f}) appears to be in raw units or thousands. "
                    f"Must be passed in MILLIONS (e.g. 15004.7 for Apple)."
                )
            data["market_cap"] = round(price_val * shares_val, 2)

        return data


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


# ==============================================================================
# 3. Core Deterministic WACC Calculation Function
# ==============================================================================
def calculate_wacc(
    beta: float,
    total_debt: float,
    market_cap: Optional[float] = None,
    share_price: Optional[float] = None,
    diluted_shares: Optional[float] = None,
    risk_free_rate: Optional[float] = None,
    equity_risk_premium: Optional[float] = None,
    tax_rate: float = 0.21,
    cost_of_debt: Optional[float] = None,
    interest_expense: Optional[float] = None,
    credit_spread_fallback: Optional[float] = None,
    *,
    prior_total_debt: Optional[float] = None,
    apply_blume_adjustment: bool = False,
    beta_source: str = "provided",
    risk_free_rate_source: str = "house_default_config",
    equity_risk_premium_source: str = "house_assumption_config",
    equity_value_source: str = "market_value",
    wacc_override: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Calculates the Weighted Average Cost of Capital (WACC) using CAPM and institutional weighting.

    Kd precedence: zero debt -> explicit rate -> interest expense / average debt (accepted only inside
    [Rf - 1pp, 15%], since a coupon far below Rf is a stale historical rate, not the marginal cost) ->
    Rf + credit spread. When wacc_override is given it replaces the blended rate, but every CAPM
    component is still computed and reported.

    Returns:
        Structured dictionary matching WACCCalculationResult.
    """
    validated = WACCInput(
        beta=beta,
        total_debt=total_debt,
        prior_total_debt=prior_total_debt,
        market_cap=market_cap,
        share_price=share_price,
        diluted_shares=diluted_shares,
        risk_free_rate=risk_free_rate,
        equity_risk_premium=equity_risk_premium,
        tax_rate=tax_rate,
        cost_of_debt=cost_of_debt,
        interest_expense=interest_expense,
        credit_spread_fallback=credit_spread_fallback,
    )

    b_raw = validated.beta
    b = blume_adjust(b_raw) if apply_blume_adjustment else b_raw
    d = validated.total_debt
    e = validated.market_cap
    rf = validated.risk_free_rate
    erp = validated.equity_risk_premium
    t = validated.tax_rate
    spread = validated.credit_spread_fallback

    # 1. Cost of Equity (Ke) via CAPM
    ke = rf + (b * erp)

    # 2. Pre-tax Cost of Debt (Kd) & Provenance
    kd_note: Optional[str] = None
    if d == 0.0:
        kd, kd_source = 0.0, "zero_debt_exemption"
    elif validated.cost_of_debt is not None:
        kd, kd_source = validated.cost_of_debt, "explicit_provided"
    elif validated.interest_expense is not None:
        prior = validated.prior_total_debt
        avg_debt = (d + prior) / 2.0 if prior is not None and prior > 0 else d
        derived = validated.interest_expense / avg_debt
        floor = max(0.0, rf - KD_FLOOR_BELOW_RF)
        if floor <= derived <= KD_CEILING:
            kd, kd_source = derived, "derived_from_10k_interest_expense"
        else:
            kd, kd_source = rf + spread, "interest_expense_out_of_band_fallback"
            kd_note = (
                f"Interest expense / average debt = {derived*100:.2f}% is outside the "
                f"[{floor*100:.2f}%, {KD_CEILING*100:.0f}%] band (stale coupons or a scale mismatch); "
                f"used Rf + {spread*100:.2f}% spread instead."
            )
    else:
        kd, kd_source = rf + spread, "institutional_credit_spread_fallback"

    kd_after_tax = kd * max(0.0, 1.0 - t)

    # 3. Capital Structure Weights
    if e is None:
        we, wd, v = 1.0, 0.0, d
        structure_source = "all_equity_no_equity_value"
    else:
        v = e + d
        we, wd = e / v, d / v
        structure_source = "book_equity_fallback" if equity_value_source == "book_equity_fallback" else "market_value"

    capm_wacc = (we * ke) + (wd * kd_after_tax)
    if wacc_override is not None:
        wacc = _normalize_rate(wacc_override, "wacc_override")
        wacc_source = "user_override"
    else:
        wacc = capm_wacc
        wacc_source = "capm_blended"

    if wacc <= 0.0 or wacc > 0.50:
        logger.warning(f"Calculated WACC ({wacc*100:.2f}%) is outside typical institutional boundaries (0% - 50%).")

    # 4. Formatted Institutional Markdown Table
    kd_desc = {
        "explicit_provided": f"Explicitly provided input ({kd*100:.2f}%)",
        "derived_from_10k_interest_expense": f"10-K interest expense ${validated.interest_expense or 0:,.1f}M / average debt",
        "interest_expense_out_of_band_fallback": f"Rf + {spread*100:.2f}% spread (derived rate out of band)",
        "institutional_credit_spread_fallback": f"Rf + {spread*100:.2f}% spread (no interest expense disclosed)",
        "zero_debt_exemption": "Zero debt exemption (Firm has $0 gross debt)",
    }[kd_source]
    beta_desc = f"{beta_source}" + (f" (Blume-adjusted from raw {b_raw:.2f})" if apply_blume_adjustment else "")
    equity_desc = {
        "market_value": "Market capitalization",
        "book_equity_fallback": "Book equity (no market price available)",
        "all_equity_no_equity_value": "No equity value available: weighted 100% equity",
    }[structure_source]
    e_cell = f"${e:,.1f}M" if e is not None else "n/a"

    rows = [
        "| WACC Parameter | Value | Source / Methodology |",
        "| :--- | :---: | :--- |",
        f"| **Risk-Free Rate ($R_f$)** | {rf*100:.2f}% | {risk_free_rate_source} |",
        f"| **Equity Risk Premium (ERP)** | {erp*100:.2f}% | {equity_risk_premium_source} |",
        f"| **Beta ($\\beta$)** | {b:.2f} | {beta_desc} |",
        f"| **Cost of Equity ($K_e$)** | **{ke*100:.2f}%** | CAPM: $R_f + (\\beta \\times \\text{{ERP}})$ |",
        f"| **Pre-Tax Cost of Debt ($K_d$)** | {kd*100:.2f}% | {kd_desc} |",
        f"| **Tax Rate ($t$)** | {t*100:.2f}% | Normalized forecast tax rate |",
        f"| **After-Tax Cost of Debt** | **{kd_after_tax*100:.2f}%** | $K_d \\times (1 - t)$ |",
        f"| **Equity Value ($E$)** | {e_cell} | {equity_desc} |",
        f"| **Gross Total Debt ($D$)** | ${d:,.1f}M | 10-K balance sheet |",
        f"| **Equity Weight ($W_e$)** | {we*100:.2f}% | $E / (E + D)$ |",
        f"| **Debt Weight ($W_d$)** | {wd*100:.2f}% | $D / (E + D)$ |",
        f"| **CAPM Blended WACC** | {capm_wacc*100:.2f}% | $(W_e \\times K_e) + (W_d \\times K_d \\times (1 - t))$ |",
    ]
    if wacc_source == "user_override":
        rows.append(f"| **Applied Discount Rate** | **{wacc*100:.2f}%** | User-specified override |")
    else:
        rows.append(f"| **Applied Discount Rate** | **{wacc*100:.2f}%** | CAPM blended WACC |")

    result = WACCCalculationResult(
        wacc=round(wacc, 6),
        wacc_pct=round(wacc * 100.0, 2),
        wacc_source=wacc_source,
        capm_wacc_pct=round(capm_wacc * 100.0, 2),
        risk_free_rate_pct=round(rf * 100.0, 3),
        risk_free_rate_source=risk_free_rate_source,
        equity_risk_premium_pct=round(erp * 100.0, 3),
        equity_risk_premium_source=equity_risk_premium_source,
        beta_raw=round(b_raw, 4),
        beta=round(b, 4),
        beta_source=beta_source,
        cost_of_equity=round(ke, 6),
        cost_of_equity_pct=round(ke * 100.0, 2),
        cost_of_debt_pre_tax=round(kd, 6),
        cost_of_debt_pre_tax_pct=round(kd * 100.0, 2),
        cost_of_debt_after_tax=round(kd_after_tax, 6),
        cost_of_debt_after_tax_pct=round(kd_after_tax * 100.0, 2),
        cost_of_debt_source=kd_source,
        cost_of_debt_note=kd_note,
        effective_tax_rate=round(t, 6),
        effective_tax_rate_pct=round(t * 100.0, 2),
        equity_weight_pct=round(we * 100.0, 2),
        debt_weight_pct=round(wd * 100.0, 2),
        capital_structure_source=structure_source,
        market_cap=round(e, 2) if e is not None else None,
        total_debt=round(d, 2),
        total_capital=round(v, 2),
        formula_breakdown_markdown="\n".join(rows),
    )

    return result.model_dump()


# ==============================================================================
# 4. LangChain Agent Tool Decorator
# ==============================================================================
@tool
def calculate_wacc_tool(
    beta: float,
    total_debt: float,
    market_cap: Optional[float] = None,
    share_price: Optional[float] = None,
    diluted_shares: Optional[float] = None,
    risk_free_rate: Optional[float] = None,
    equity_risk_premium: Optional[float] = None,
    tax_rate: float = 0.21,
    cost_of_debt: Optional[float] = None,
    interest_expense: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Calculate the Weighted Average Cost of Capital (WACC) discount hurdle rate using CAPM.

    Args:
        beta: Systematic market risk factor (e.g. 1.10).
        total_debt: Gross total debt in $ Millions (must be positive or zero).
        market_cap: Total market cap in $ Millions (optional if share_price and diluted_shares passed).
        share_price: Stock price in $ (used with diluted_shares to compute market cap).
        diluted_shares: Diluted shares in Millions (e.g. 15004.7 for Apple).
        risk_free_rate: 10-Yr US Treasury rate as decimal (defaults to the configured house rate).
        equity_risk_premium: Expected market risk premium as decimal (defaults to the configured house ERP).
        tax_rate: Tax rate as decimal (default 0.21).
        cost_of_debt: Pre-tax borrowing rate as decimal (optional).
        interest_expense: Annual 10-K interest expense in $ Millions (optional, used to derive Kd).

    Returns:
        Structured dictionary containing wacc, components, weights and provenance tags.
    """
    try:
        return calculate_wacc(
            beta=beta,
            total_debt=total_debt,
            market_cap=market_cap,
            share_price=share_price,
            diluted_shares=diluted_shares,
            risk_free_rate=risk_free_rate,
            equity_risk_premium=equity_risk_premium,
            tax_rate=tax_rate,
            cost_of_debt=cost_of_debt,
            interest_expense=interest_expense,
        )
    except Exception as e:
        return {"error": f"calculate_wacc_tool failed: {e}"}
