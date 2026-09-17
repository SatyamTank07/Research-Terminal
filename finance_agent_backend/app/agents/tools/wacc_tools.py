"""Deterministic Weighted Average Cost of Capital (WACC) & CAPM Engine.

Enforces Architectural Principle #2: Zero Mathematical Hallucination.
Offloads all discount rate derivations to pure, deterministic Python functions.
Computes Cost of Equity (Ke) via the Capital Asset Pricing Model (CAPM),
evaluates pre-tax and after-tax Cost of Debt (Kd) with transparent provenance tagging,
and weights capital structure to determine the final hurdle discount rate.
"""

import logging
import math
from typing import Optional, Dict, Any, Literal
from pydantic import BaseModel, Field, AliasChoices, model_validator
from langchain_core.tools import tool

logger = logging.getLogger("finance_agent.tools.wacc")


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


# ==============================================================================
# 2. Input and Output Schemas
# ==============================================================================
CostOfDebtSource = Literal[
    "explicit_provided",
    "derived_from_10k_interest_expense",
    "institutional_credit_spread_fallback",
    "zero_debt_exemption"
]


class WACCInput(BaseModel):
    """
    Structured input container for WACC calculation.
    Supports passing either market_cap directly OR (share_price + diluted_shares).
    Equipped with Pydantic AliasChoices for flexible agent calling.
    """
    beta: float = Field(
        ...,
        description="Equity beta (systematic market risk factor, e.g. 1.10 for AAPL, 2.00 for TSLA)",
        validation_alias=AliasChoices("beta", "equity_beta")
    )
    total_debt: float = Field(
        ...,
        ge=0.0,
        description="Gross total debt (short-term + long-term debt + finance leases) in $ Millions",
        validation_alias=AliasChoices("total_debt", "gross_debt", "debt")
    )
    market_cap: Optional[float] = Field(
        None,
        description="Total market capitalization in $ Millions",
        validation_alias=AliasChoices("market_cap", "equity_value_market", "mkt_cap")
    )
    share_price: Optional[float] = Field(
        None,
        description="Current common stock share price in $ (used with diluted_shares to compute market_cap)",
        validation_alias=AliasChoices("share_price", "stock_price", "current_share_price")
    )
    diluted_shares: Optional[float] = Field(
        None,
        description="Diluted shares outstanding in Millions (used with share_price to compute market_cap)",
        validation_alias=AliasChoices("diluted_shares", "shares_outstanding", "diluted_shares_outstanding")
    )
    risk_free_rate: float = Field(
        0.042,
        description="Risk-free rate as decimal (default 0.042 for 4.2% 10-Yr US Treasury)",
        validation_alias=AliasChoices("risk_free_rate", "rf", "risk_free")
    )
    equity_risk_premium: float = Field(
        0.050,
        description="Equity Risk Premium as decimal (default 0.050 for 5.0% Damodaran US ERP)",
        validation_alias=AliasChoices("equity_risk_premium", "erp")
    )
    tax_rate: float = Field(
        0.21,
        description="Corporate tax rate as decimal (default 0.21 for 21.0% statutory rate or 10-K effective rate)",
        validation_alias=AliasChoices("tax_rate", "effective_tax_rate", "marginal_tax_rate")
    )
    cost_of_debt: Optional[float] = Field(
        None,
        description="Pre-tax cost of debt as decimal (e.g. 0.045 for 4.5%)",
        validation_alias=AliasChoices("cost_of_debt", "kd", "pre_tax_cost_of_debt")
    )
    interest_expense: Optional[float] = Field(
        None,
        description="Annual interest expense in $ Millions (used to derive Kd if cost_of_debt is omitted)",
        validation_alias=AliasChoices("interest_expense", "interest")
    )
    credit_spread_fallback: float = Field(
        0.0125,
        description="Credit spread benchmark over risk_free_rate (default 0.0125 for 1.25% spread, i.e. 5.45% Kd)",
        validation_alias=AliasChoices("credit_spread_fallback", "credit_spread")
    )

    @model_validator(mode="before")
    @classmethod
    def validate_and_normalize(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        # 1. Finite float validation
        for field_name in ["beta", "total_debt"]:
            if field_name in data and data[field_name] is not None:
                data[field_name] = _validate_finite_float(data[field_name], field_name)

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
        data["risk_free_rate"] = _normalize_rate(data.get("risk_free_rate", 0.042), "risk_free_rate", default=0.042)
        data["equity_risk_premium"] = _normalize_rate(data.get("equity_risk_premium", 0.050), "equity_risk_premium", default=0.050)
        data["tax_rate"] = _normalize_rate(data.get("tax_rate", 0.21), "tax_rate", default=0.21)
        data["credit_spread_fallback"] = _normalize_rate(data.get("credit_spread_fallback", 0.0125), "credit_spread_fallback", default=0.0125)

        if "cost_of_debt" in data and data["cost_of_debt"] is not None:
            data["cost_of_debt"] = _normalize_rate(data["cost_of_debt"], "cost_of_debt")
            if data["cost_of_debt"] > 0.30:
                raise ValueError(
                    f"cost_of_debt ({data['cost_of_debt']*100:.2f}%) exceeds realistic borrowing limits (max 30.0%). "
                    f"Please verify the pre-tax borrowing rate."
                )

        if "interest_expense" in data and data["interest_expense"] is not None:
            data["interest_expense"] = _validate_finite_float(data["interest_expense"], "interest_expense")
            if data["interest_expense"] < 0.0:
                # Absolute magnitude if passed negative
                data["interest_expense"] = abs(data["interest_expense"])

        # 3. Market Cap resolution
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
        else:
            raise ValueError(
                "Either 'market_cap' must be provided, or both 'share_price' and 'diluted_shares' must be provided."
            )

        return data


class WACCCalculationResult(BaseModel):
    """Structured Pydantic model for WACC calculation output and institutional audit trail."""
    wacc: float = Field(..., description="Weighted Average Cost of Capital as decimal (for DCF input, e.g. 0.0845)")
    wacc_pct: float = Field(..., description="Weighted Average Cost of Capital as percentage (e.g. 8.45)")
    cost_of_equity: float = Field(..., description="Cost of Equity (Ke) as decimal via CAPM (e.g. 0.0970)")
    cost_of_equity_pct: float = Field(..., description="Cost of Equity (Ke) as percentage (e.g. 9.70)")
    cost_of_debt_pre_tax: float = Field(..., description="Pre-tax Cost of Debt (Kd) as decimal (e.g. 0.0450)")
    cost_of_debt_pre_tax_pct: float = Field(..., description="Pre-tax Cost of Debt (Kd) as percentage (e.g. 4.50)")
    cost_of_debt_after_tax: float = Field(..., description="After-tax Cost of Debt Kd*(1-t) as decimal (e.g. 0.0356)")
    cost_of_debt_after_tax_pct: float = Field(..., description="After-tax Cost of Debt as percentage (e.g. 3.56)")
    effective_tax_rate: float = Field(..., description="Tax rate (t) applied to debt tax shield as decimal (e.g. 0.2100)")
    effective_tax_rate_pct: float = Field(..., description="Tax rate as percentage (e.g. 21.00)")
    equity_weight_pct: float = Field(..., description="Proportion of equity We = E/V as percentage (e.g. 97.30)")
    debt_weight_pct: float = Field(..., description="Proportion of debt Wd = D/V as percentage (e.g. 2.70)")
    market_cap: float = Field(..., description="Market value of equity (E) in $ Millions")
    total_debt: float = Field(..., description="Gross total debt (D) in $ Millions")
    total_capital: float = Field(..., description="Total enterprise capital (V = E + D) in $ Millions")
    cost_of_debt_source: CostOfDebtSource = Field(
        ...,
        description="Explicit provenance tag describing how the pre-tax cost of debt was established"
    )
    formula_breakdown_markdown: str = Field(
        ...,
        description="Institutional markdown table detailing formula components and provenance citations"
    )


# ==============================================================================
# 3. Core Deterministic WACC Calculation Function
# ==============================================================================
def calculate_wacc(
    beta: float,
    total_debt: float,
    market_cap: Optional[float] = None,
    share_price: Optional[float] = None,
    diluted_shares: Optional[float] = None,
    risk_free_rate: float = 0.042,
    equity_risk_premium: float = 0.050,
    tax_rate: float = 0.21,
    cost_of_debt: Optional[float] = None,
    interest_expense: Optional[float] = None,
    credit_spread_fallback: float = 0.0125,
) -> Dict[str, Any]:
    """
    Calculates the Weighted Average Cost of Capital (WACC) using CAPM and institutional weighting.

    Args:
        beta: Systematic market risk factor (> 0).
        total_debt: Gross book debt in $ Millions (>= 0).
        market_cap: Market capitalization in $ Millions (optional if share_price + diluted_shares provided).
        share_price: Current share price in $ (used with diluted_shares).
        diluted_shares: Diluted shares outstanding in Millions (used with share_price).
        risk_free_rate: 10-Year US Treasury yield as decimal (default 0.042).
        equity_risk_premium: Market risk premium as decimal (default 0.050).
        tax_rate: Corporate effective/statutory tax rate as decimal (default 0.21).
        cost_of_debt: Explicit pre-tax cost of debt as decimal (optional).
        interest_expense: Annual 10-K interest expense in $ Millions (used to derive Kd if cost_of_debt omitted).
        credit_spread_fallback: Default credit spread over Rf if no debt rate is available (default 0.0125).

    Returns:
        Structured dictionary matching WACCCalculationResult.
    """
    # 1. Parse and validate inputs via Pydantic model
    validated = WACCInput(
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
        credit_spread_fallback=credit_spread_fallback,
    )

    b = validated.beta
    d = validated.total_debt
    e = validated.market_cap
    rf = validated.risk_free_rate
    erp = validated.equity_risk_premium
    t = validated.tax_rate

    # 2. Cost of Equity (Ke) via CAPM
    # Ke = Rf + Beta * ERP
    ke = rf + (b * erp)

    # 3. Pre-tax Cost of Debt (Kd) & Provenance
    if d == 0.0:
        kd = 0.0
        kd_source: CostOfDebtSource = "zero_debt_exemption"
    elif validated.cost_of_debt is not None:
        kd = validated.cost_of_debt
        kd_source = "explicit_provided"
    elif validated.interest_expense is not None:
        if d > 0.0:
            kd = validated.interest_expense / d
            if kd > 0.20:
                raise ValueError(
                    f"Derived pre-tax cost of debt ({kd*100:.2f}%) exceeds realistic corporate borrowing ceiling (max 20.0%). "
                    f"Interest expense (${validated.interest_expense:,.1f}M) relative to total debt (${d:,.1f}M) indicates "
                    f"a scale mismatch (e.g. thousands vs millions) or trivial debt distortion. "
                    f"Please verify the inputs or omit interest_expense to use the institutional credit spread fallback."
                )
            kd_source = "derived_from_10k_interest_expense"
        else:
            kd = 0.0
            kd_source = "zero_debt_exemption"
    else:
        # Institutional benchmark credit spread fallback
        kd = rf + validated.credit_spread_fallback
        kd_source = "institutional_credit_spread_fallback"

    # 4. After-tax Cost of Debt
    # Debt tax shield: Kd * (1 - t)
    kd_after_tax = kd * max(0.0, 1.0 - t)

    # 5. Capital Structure Weights
    v = e + d
    if v <= 0.0:
        raise ValueError(f"Total enterprise capital (E + D) must be strictly positive, got: {v}")

    we = e / v
    wd = d / v

    # 6. Blended WACC
    wacc = (we * ke) + (wd * kd_after_tax)

    # Sanity check bounds
    if wacc <= 0.0 or wacc > 0.50:
        logger.warning(f"Calculated WACC ({wacc*100:.2f}%) is outside typical institutional boundaries (0% - 50%).")

    # 7. Formatted Institutional Markdown Table
    if kd_source == "explicit_provided":
        kd_desc = f"Explicitly provided input ({kd*100:.2f}%)"
    elif kd_source == "derived_from_10k_interest_expense":
        exp = validated.interest_expense if validated.interest_expense is not None else 0.0
        kd_desc = f"Derived from 10-K (${exp:,.1f}M interest / ${d:,.1f}M debt)"
    elif kd_source == "institutional_credit_spread_fallback":
        kd_desc = f"Credit spread benchmark ({rf*100:.2f}% Rf + {validated.credit_spread_fallback*100:.2f}% spread)"
    else:
        kd_desc = "Zero debt exemption (Firm has $0 gross debt)"

    markdown_audit = (
        "| WACC Parameter | Value | Institutional Source / Methodology |\n"
        "| :--- | :---: | :--- |\n"
        f"| **Risk-Free Rate ($R_f$)** | {rf*100:.2f}% | 10-Year US Treasury Benchmark |\n"
        f"| **Equity Risk Premium (ERP)** | {erp*100:.2f}% | Damodaran US Historical Market ERP |\n"
        f"| **Systematic Risk Beta ($\\beta$)** | {b:.2f} | Market Volatility Covariance Factor |\n"
        f"| **Cost of Equity ($K_e$)** | **{ke*100:.2f}%** | CAPM: $R_f + (\\beta \\times \\text{{ERP}})$ |\n"
        f"| **Pre-Tax Cost of Debt ($K_d$)** | {kd*100:.2f}% | {kd_desc} |\n"
        f"| **Effective Tax Rate ($t$)** | {t*100:.2f}% | 10-K Audited Tax Provision / Statutory |\n"
        f"| **After-Tax Cost of Debt** | **{kd_after_tax*100:.2f}%** | $K_d \\times (1 - t)$ (Interest Tax Shield) |\n"
        f"| **Market Capitalization ($E$)** | ${e:,.1f}M | Total Market Value of Equity |\n"
        f"| **Gross Total Debt ($D$)** | ${d:,.1f}M | 10-K Audited Balance Sheet Total Debt |\n"
        f"| **Total Capital ($V = E + D$)** | ${v:,.1f}M | Combined Enterprise Capital Base |\n"
        f"| **Equity Weight ($W_e$)** | {we*100:.2f}% | $E / (E + D)$ |\n"
        f"| **Debt Weight ($W_d$)** | {wd*100:.2f}% | $D / (E + D)$ |\n"
        f"| **Blended WACC Hurdle Rate** | **{wacc*100:.2f}%** | **$(W_e \\times K_e) + (W_d \\times K_d \\times (1 - t))$** |"
    )

    result = WACCCalculationResult(
        wacc=round(wacc, 6),
        wacc_pct=round(wacc * 100.0, 2),
        cost_of_equity=round(ke, 6),
        cost_of_equity_pct=round(ke * 100.0, 2),
        cost_of_debt_pre_tax=round(kd, 6),
        cost_of_debt_pre_tax_pct=round(kd * 100.0, 2),
        cost_of_debt_after_tax=round(kd_after_tax, 6),
        cost_of_debt_after_tax_pct=round(kd_after_tax * 100.0, 2),
        effective_tax_rate=round(t, 6),
        effective_tax_rate_pct=round(t * 100.0, 2),
        equity_weight_pct=round(we * 100.0, 2),
        debt_weight_pct=round(wd * 100.0, 2),
        market_cap=round(e, 2),
        total_debt=round(d, 2),
        total_capital=round(v, 2),
        cost_of_debt_source=kd_source,
        formula_breakdown_markdown=markdown_audit,
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
    risk_free_rate: float = 0.042,
    equity_risk_premium: float = 0.050,
    tax_rate: float = 0.21,
    cost_of_debt: Optional[float] = None,
    interest_expense: Optional[float] = None,
    credit_spread_fallback: float = 0.0125,
) -> Dict[str, Any]:
    """
    Calculate the Weighted Average Cost of Capital (WACC) discount hurdle rate using CAPM.

    Accepts parameters from the Financial Auditor (total_debt, tax_rate, diluted_shares),
    market data (beta, share_price or market_cap), and macro benchmarks.
    Returns the exact decimal 'wacc' for use in calculate_dcf_tool along with an institutional audit breakdown.

    Args:
        beta: Systematic market risk factor (e.g. 1.10 for AAPL, 2.00 for TSLA).
        total_debt: Gross total debt in $ Millions (must be positive or zero).
        market_cap: Total market cap in $ Millions (optional if share_price and diluted_shares passed).
        share_price: Stock price in $ (used with diluted_shares to compute market cap).
        diluted_shares: Diluted shares in Millions (e.g. 15004.7 for Apple).
        risk_free_rate: 10-Yr US Treasury rate as decimal (default 0.042 for 4.2%).
        equity_risk_premium: Expected market risk premium as decimal (default 0.050 for 5.0%).
        tax_rate: Effective tax rate from 10-K or statutory rate as decimal (default 0.21 for 21.0%).
        cost_of_debt: Pre-tax borrowing rate as decimal (optional).
        interest_expense: Annual 10-K interest expense in $ Millions (optional, used to derive Kd).
        credit_spread_fallback: Credit spread over Rf if no debt rate is available (default 0.0125).

    Returns:
        Structured dictionary containing wacc, wacc_pct, cost_of_equity, cost_of_debt,
        weights, cost_of_debt_source provenance tag, and formula_breakdown_markdown.
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
            credit_spread_fallback=credit_spread_fallback,
        )
    except Exception as e:
        return {"error": f"calculate_wacc_tool failed: {e}"}
