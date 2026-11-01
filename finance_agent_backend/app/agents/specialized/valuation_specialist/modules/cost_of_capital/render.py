"""The WACC build-up table: formats the unrounded components so no figure is rounded twice."""

from app.agents.specialized.valuation_specialist.modules.cost_of_capital.schema import WACCComponents


def render_wacc_build_up(c: WACCComponents) -> str:
    kd_desc = {
        "explicit_provided": f"Explicitly provided input ({c.cost_of_debt_pre_tax*100:.2f}%)",
        "derived_from_10k_interest_expense": f"10-K interest expense ${c.interest_expense or 0:,.1f}M / average debt",
        "interest_expense_out_of_band_fallback": f"Rf + {c.credit_spread*100:.2f}% spread (derived rate out of band)",
        "institutional_credit_spread_fallback": f"Rf + {c.credit_spread*100:.2f}% spread (no interest expense disclosed)",
        "zero_debt_exemption": "Zero debt exemption (Firm has $0 gross debt)",
    }[c.cost_of_debt_source]
    beta_desc = f"{c.beta_source}" + (f" (Blume-adjusted from raw {c.beta_raw:.2f})" if c.blume_adjusted else "")
    equity_desc = {
        "market_value": "Market capitalization",
        "book_equity_fallback": "Book equity (no market price available)",
        "all_equity_no_equity_value": "No equity value available: weighted 100% equity",
    }[c.capital_structure_source]
    e_cell = f"${c.equity_value:,.1f}M" if c.equity_value is not None else "n/a"

    rows = [
        "| WACC Parameter | Value | Source / Methodology |",
        "| :--- | :---: | :--- |",
        f"| **Risk-Free Rate ($R_f$)** | {c.risk_free_rate*100:.2f}% | {c.risk_free_rate_source} |",
        f"| **Equity Risk Premium (ERP)** | {c.equity_risk_premium*100:.2f}% | {c.equity_risk_premium_source} |",
        f"| **Beta ($\\beta$)** | {c.beta:.2f} | {beta_desc} |",
        f"| **Cost of Equity ($K_e$)** | **{c.cost_of_equity*100:.2f}%** | CAPM: $R_f + (\\beta \\times \\text{{ERP}})$ |",
        f"| **Pre-Tax Cost of Debt ($K_d$)** | {c.cost_of_debt_pre_tax*100:.2f}% | {kd_desc} |",
        f"| **Tax Rate ($t$)** | {c.tax_rate*100:.2f}% | Normalized forecast tax rate |",
        f"| **After-Tax Cost of Debt** | **{c.cost_of_debt_after_tax*100:.2f}%** | $K_d \\times (1 - t)$ |",
        f"| **Equity Value ($E$)** | {e_cell} | {equity_desc} |",
        f"| **Gross Total Debt ($D$)** | ${c.total_debt:,.1f}M | 10-K balance sheet |",
        f"| **Equity Weight ($W_e$)** | {c.equity_weight*100:.2f}% | $E / (E + D)$ |",
        f"| **Debt Weight ($W_d$)** | {c.debt_weight*100:.2f}% | $D / (E + D)$ |",
        f"| **CAPM Blended WACC** | {c.capm_wacc*100:.2f}% | $(W_e \\times K_e) + (W_d \\times K_d \\times (1 - t))$ |",
    ]
    if c.wacc_source == "user_override":
        rows.append(f"| **Applied Discount Rate** | **{c.wacc*100:.2f}%** | User-specified override |")
    else:
        rows.append(f"| **Applied Discount Rate** | **{c.wacc*100:.2f}%** | CAPM blended WACC |")
    return "\n".join(rows)
