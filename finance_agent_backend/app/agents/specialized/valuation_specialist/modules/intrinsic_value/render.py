"""The EV-to-equity bridge table: formats the unrounded base point so no figure is rounded twice."""

from app.agents.specialized.valuation_specialist.modules.intrinsic_value.schema import DCFCase, DCFPoint


def render_equity_bridge(
    case: DCFCase, point: DCFPoint, *, discounting_convention: str, roll_forward_factor: float, claims_rate: float
) -> str:
    net_debt, nci = case.net_debt, case.noncontrolling_interests
    lines = [
        "| Valuation Bridge | Value |",
        "|:---|---:|",
        f"| PV of explicit UFCF ({len(case.fcfs)} yrs, {discounting_convention}) | ${point.pv_fcfs:,.1f}M |",
        f"| PV of terminal value ({case.terminal_method}) | ${point.pv_tv:,.1f}M |",
        f"| **Enterprise value** | **${point.ev:,.1f}M** |",
        f"| {'Plus: net cash' if net_debt < 0 else 'Less: net debt'} | ${abs(net_debt):,.1f}M |",
        f"| Less: noncontrolling interests | ${nci:,.1f}M |",
        f"| Equity value at balance-sheet date | ${point.equity_bs:,.1f}M |",
    ]
    if case.roll_forward_years > 0:
        lines += [
            f"| Enterprise value rolled forward {case.roll_forward_years:.2f} yrs @ WACC {case.wacc * 100:.2f}% | "
            f"${point.ev * roll_forward_factor:,.1f}M |",
            f"| {'Net cash' if net_debt < 0 else 'Net debt'} and NCI rolled forward @ Rf {claims_rate * 100:.2f}% | "
            f"${-(net_debt + nci) * case.claims_roll_factor:,.1f}M |",
        ]
    lines += [
        f"| **Equity value at valuation date** | **${point.equity:,.1f}M** |",
        f"| Diluted shares | {case.diluted_shares:,.2f}M |",
        f"| **Fair value per share** | **${point.per_share:.2f}** |",
    ]
    return "\n".join(lines)
