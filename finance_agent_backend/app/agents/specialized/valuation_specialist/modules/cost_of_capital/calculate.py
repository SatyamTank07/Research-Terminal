"""What return do investors require? CAPM cost of equity (optional Blume beta), cost of debt from interest
over average debt inside a sanity band (else Rf + spread), capital-structure weights, and the provenance of
every input. A user override replaces the blended rate but every CAPM component is still computed.
"""

import logging
import math
from typing import Any, Optional

from pydantic import AliasChoices, BaseModel, Field, model_validator

from app.agents.specialized.valuation_specialist.modules.cost_of_capital.render import render_wacc_build_up
from app.agents.specialized.valuation_specialist.modules.cost_of_capital.schema import (
    CostOfCapital,
    CostOfCapitalInputs,
    WACCCalculationResult,
    WACCComponents,
)
from app.agents.specialized.valuation_specialist.modules.cost_of_capital.thresholds import (
    BLUME_RAW_WEIGHT,
    DEFAULT_TAX_RATE,
    KD_CEILING,
    KD_FLOOR_BELOW_RF,
    MAX_BETA,
    MAX_COST_OF_DEBT_RATE,
    MAX_DILUTED_SHARES_MILLIONS,
    WACC_WARNING_CEILING_RATE,
)
from app.config import settings

logger = logging.getLogger("finance_agent.agents.valuation_specialist")


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
    tax_rate: float = Field(DEFAULT_TAX_RATE, validation_alias=AliasChoices("tax_rate", "effective_tax_rate"))
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
        if beta_val <= 0.0 or beta_val > MAX_BETA:
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
            ("tax_rate", DEFAULT_TAX_RATE),
            ("credit_spread_fallback", settings.DEFAULT_CREDIT_SPREAD),
        ):
            data[name] = _normalize_rate(data.get(name), name, default=default)

        if data.get("cost_of_debt") is not None:
            data["cost_of_debt"] = _normalize_rate(data["cost_of_debt"], "cost_of_debt")
            if data["cost_of_debt"] > MAX_COST_OF_DEBT_RATE:
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
            if shares_val > MAX_DILUTED_SHARES_MILLIONS:
                raise ValueError(
                    f"diluted_shares ({shares_val:,.2f}) appears to be in raw units or thousands. "
                    f"Must be passed in MILLIONS (e.g. 15004.7 for Apple)."
                )
            data["market_cap"] = round(price_val * shares_val, 2)

        return data


def calculate_wacc(
    beta: float,
    total_debt: float,
    market_cap: Optional[float] = None,
    share_price: Optional[float] = None,
    diluted_shares: Optional[float] = None,
    risk_free_rate: Optional[float] = None,
    equity_risk_premium: Optional[float] = None,
    tax_rate: float = DEFAULT_TAX_RATE,
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
) -> WACCCalculationResult:
    """
    Calculates the Weighted Average Cost of Capital (WACC) using CAPM and institutional weighting.

    Kd precedence: zero debt -> explicit rate -> interest expense / average debt (accepted only inside
    [Rf - 1pp, 15%], since a coupon far below Rf is a stale historical rate, not the marginal cost) ->
    Rf + credit spread. When wacc_override is given it replaces the blended rate, but every CAPM
    component is still computed and reported.

    Returns:
        WACCCalculationResult: rounded figures, provenance tags and the WACC build-up table.
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

    if wacc <= 0.0 or wacc > WACC_WARNING_CEILING_RATE:
        logger.warning(f"Calculated WACC ({wacc*100:.2f}%) is outside typical institutional boundaries (0% - 50%).")

    components = WACCComponents(
        risk_free_rate=rf,
        risk_free_rate_source=risk_free_rate_source,
        equity_risk_premium=erp,
        equity_risk_premium_source=equity_risk_premium_source,
        beta_raw=b_raw,
        beta=b,
        beta_source=beta_source,
        blume_adjusted=apply_blume_adjustment,
        cost_of_equity=ke,
        cost_of_debt_pre_tax=kd,
        cost_of_debt_source=kd_source,
        cost_of_debt_note=kd_note,
        interest_expense=validated.interest_expense,
        credit_spread=spread,
        tax_rate=t,
        cost_of_debt_after_tax=kd_after_tax,
        equity_value=e,
        total_debt=d,
        total_capital=v,
        equity_weight=we,
        debt_weight=wd,
        capital_structure_source=structure_source,
        capm_wacc=capm_wacc,
        wacc=wacc,
        wacc_source=wacc_source,
    )

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
        formula_breakdown_markdown=render_wacc_build_up(components),
    )

    return result


def estimate_cost_of_capital(inputs: CostOfCapitalInputs) -> CostOfCapital:
    """CAPM WACC from the gathered inputs; a derived Kd rejected by its band adds its note to the warnings."""
    wacc_audit = calculate_wacc(
        beta=inputs.beta,
        total_debt=inputs.total_debt,
        prior_total_debt=inputs.prior_total_debt,
        market_cap=inputs.equity_value,
        risk_free_rate=inputs.risk_free_rate,
        equity_risk_premium=inputs.equity_risk_premium,
        tax_rate=inputs.marginal_tax_rate,
        interest_expense=inputs.interest_expense,
        apply_blume_adjustment=inputs.apply_blume_adjustment,
        beta_source=inputs.beta_source,
        risk_free_rate_source=inputs.risk_free_rate_source,
        equity_risk_premium_source=inputs.equity_risk_premium_source,
        equity_value_source=inputs.equity_value_source,
        wacc_override=inputs.wacc_override,
    )
    note = (wacc_audit.cost_of_debt_note,) if wacc_audit.cost_of_debt_note else ()
    return CostOfCapital(wacc_audit=wacc_audit, marginal_tax_rate=inputs.marginal_tax_rate, warnings=inputs.warnings + note)
