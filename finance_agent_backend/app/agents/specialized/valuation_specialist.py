"""DCF Valuation Specialist Agent.

Specialized autonomous agent that ingests audited balance sheet metrics and cash flows,
derives the Weighted Average Cost of Capital (WACC) via CAPM, computes intrinsic fair value
using a 2-stage Gordon Growth DCF model with mid-year discounting, generates a 5x5 sensitivity
matrix, and emits a structured DCFValuationOutput artifact.
"""

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from app.agents.base import StructuredAgent
from app.agents.registry import AgentRegistry
from app.agents.state import DCFValuationOutput, FinancialAuditOutput, WACCAudit
from app.agents.tools.dcf_tools import calculate_dcf_tool
from app.agents.tools.wacc_tools import calculate_wacc_tool

logger = logging.getLogger("finance_agent.agents.valuation_specialist")


@AgentRegistry.register("valuation_specialist")
class ValuationSpecialistAgent(StructuredAgent[DCFValuationOutput]):
    """Autonomous tool-calling agent executing deterministic DCF valuation and sensitivity modeling."""

    prompt_name = "valuation_specialist"
    tools = [calculate_wacc_tool, calculate_dcf_tool]
    output_schema = DCFValuationOutput
    default_recursion_limit = 50

    def value(
        self,
        ticker: str,
        fiscal_year: int,
        financial_audit: FinancialAuditOutput,
        projected_fcfs: List[float],
        beta: float,
        share_price: Optional[float] = None,
        market_cap: Optional[float] = None,
        terminal_growth_rate: float = 0.025,
        risk_free_rate: float = 0.042,
        equity_risk_premium: float = 0.050,
        cost_of_debt: Optional[float] = None,
        base_year_ebitda: Optional[float] = None,
    ) -> DCFValuationOutput:
        """Direct programmatic interface for LangGraph orchestrator and verification tests.

        Ingests FinancialAuditOutput and market inputs, executes deterministic tools,
        and returns a validated DCFValuationOutput instance.
        """
        # Extract audited figures from FinancialAuditOutput
        total_debt = financial_audit.balance_sheet.total_debt
        net_debt = financial_audit.balance_sheet.net_debt
        diluted_shares = financial_audit.balance_sheet.diluted_shares_outstanding

        # Resolve effective tax rate as decimal (e.g. 0.1561)
        ratios = financial_audit.profitability_and_return_ratios
        tax_rate_val = getattr(ratios, "effective_tax_rate_pct", None)
        if tax_rate_val is not None:
            tax_rate = (tax_rate_val / 100.0) if tax_rate_val > 1.0 else tax_rate_val
        else:
            tax_rate = getattr(ratios, "effective_tax_rate", 0.21)

        # Resolve latest operating income (EBIT)
        latest_operating_income = None
        if financial_audit.multi_year_history:
            latest_operating_income = financial_audit.multi_year_history[-1].operating_income

        # Resolve audited EBITDA from Solvency ratios (only if real EBITDA is available)
        if (
            base_year_ebitda is None
            and hasattr(financial_audit, "solvency_and_liquidity_ratios")
            and financial_audit.solvency_and_liquidity_ratios
        ):
            base_year_ebitda = getattr(
                financial_audit.solvency_and_liquidity_ratios, "ebitda", None
            )

        # Fallback to book stockholders_equity or target capital structure if neither price nor market_cap passed
        if share_price is None and market_cap is None:
            book_equity = getattr(financial_audit.balance_sheet, "stockholders_equity", None)
            if book_equity is not None and book_equity > 0:
                market_cap = float(book_equity)
            else:
                market_cap = max(float(total_debt) * 4.0, 10000.0)
            logger.info(
                f"Neither share_price nor market_cap provided for {ticker}. "
                f"Defaulted capital structure equity to: ${market_cap:,.2f}M"
            )

        market_context_str = ""
        if share_price is not None:
            market_context_str += f"   - Current Share Price: ${share_price:.2f}\n"
        if market_cap is not None:
            market_context_str += f"   - Market Capitalization: ${market_cap:,.2f}M\n"

        ebit_str = (
            f"   - Base Year Operating Income (EBIT): ${latest_operating_income:,.2f}M\n"
            if latest_operating_income is not None
            else ""
        )
        ebitda_str = (
            f"   - Base Year EBITDA (Operating Income + D&A): ${base_year_ebitda:,.2f}M\n"
            if base_year_ebitda is not None
            else "   - Base Year EBITDA: Not available (D&A omitted in 10-K extraction)\n"
        )
        kd_str = (
            f"   - Explicit Pre-Tax Cost of Debt: {cost_of_debt*100:.2f}%\n"
            if cost_of_debt is not None
            else ""
        )

        query = (
            f"Perform an institutional 2-stage DCF valuation for {ticker.upper()} for fiscal year {fiscal_year}.\n"
            f"1. Audited Balance Sheet Inputs (from Financial Auditor):\n"
            f"   - Total Debt: ${total_debt:,.2f}M\n"
            f"   - Net Debt: ${net_debt:,.2f}M ({'Net Cash Surplus' if net_debt < 0 else 'Net Indebtedness'})\n"
            f"   - Diluted Shares: {diluted_shares:,.2f} Million shares (pass diluted_shares={diluted_shares:.2f} EXACTLY, do NOT divide by 1000)\n"
            f"   - Effective Tax Rate: {tax_rate*100:.2f}%\n"
            f"{ebit_str}"
            f"{ebitda_str}"
            f"2. Market & Macro Inputs:\n"
            f"   - Equity Beta: {beta:.2f}\n"
            f"{market_context_str}"
            f"   - Risk-Free Rate (Rf): {risk_free_rate*100:.2f}%\n"
            f"   - Equity Risk Premium (ERP): {equity_risk_premium*100:.2f}%\n"
            f"{kd_str}"
            f"3. 5-Year Explicit Forecast UFCFs ($ Millions): {projected_fcfs}\n"
            f"4. Perpetual Terminal Growth Rate: {terminal_growth_rate*100:.2f}%\n"
            f"5. Execution Sequence:\n"
            f"   Step 1: Call calculate_wacc_tool EXACTLY ONCE to derive WACC (pass beta={beta:.2f}, total_debt={total_debt:.2f}"
            + (f", share_price={share_price:.2f}" if share_price is not None else "")
            + (f", market_cap={market_cap:.2f}" if market_cap is not None else "")
            + f", diluted_shares={diluted_shares:.2f}, tax_rate={tax_rate:.4f}"
            + (f", cost_of_debt={cost_of_debt:.4f}" if cost_of_debt is not None else "")
            + f", risk_free_rate={risk_free_rate:.4f}, equity_risk_premium={equity_risk_premium:.4f}).\n"
            f"   Step 2: Call calculate_dcf_tool EXACTLY ONCE with projected_fcfs={projected_fcfs}, wacc from Step 1, terminal_growth_rate={terminal_growth_rate:.4f}, net_debt={net_debt:.2f}, diluted_shares={diluted_shares:.2f}.\n"
            f"   Step 3: Stop calling tools and emit the complete DCFValuationOutput artifact as valid JSON."
        )

        fallback_defaults = {
            "valuation_summary": "Valuation completed.",
            "discounting_convention": "mid_year",
        }

        return self.execute_structured(
            query,
            ticker=ticker,
            fiscal_year=fiscal_year,
            fallback_defaults=fallback_defaults,
            projected_fcfs=projected_fcfs,
            share_price=share_price,
            terminal_growth_rate=terminal_growth_rate,
            base_year_ebitda=base_year_ebitda,
        )

    def _extract_tool_valuation_data(
        self, messages: List[Any]
    ) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """Extracts calculate_wacc and calculate_dcf tool outputs from message history."""
        tool_wacc_data = None
        tool_dcf_data = None
        for msg in messages:
            msg_name = getattr(msg, "name", "") or ""
            if "calculate_wacc" in msg_name:
                raw = getattr(msg, "content", "")
                if isinstance(raw, str):
                    try:
                        tool_wacc_data = json.loads(raw)
                    except Exception:
                        pass
                elif isinstance(raw, dict):
                    tool_wacc_data = raw
            elif "calculate_dcf" in msg_name:
                raw = getattr(msg, "content", "")
                if isinstance(raw, str):
                    try:
                        tool_dcf_data = json.loads(raw)
                    except Exception:
                        pass
                elif isinstance(raw, dict):
                    tool_dcf_data = raw
        return tool_wacc_data, tool_dcf_data

    def _pre_validate_data(
        self,
        data: Dict[str, Any],
        result: Dict[str, Any],
        ticker: Optional[str] = None,
        fiscal_year: Optional[int] = None,
        projected_fcfs: Optional[List[float]] = None,
        share_price: Optional[float] = None,
        terminal_growth_rate: float = 0.025,
        base_year_ebitda: Optional[float] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Overlays deterministic tool figures onto parsed dictionary prior to Pydantic validation."""
        messages = result.get("messages", [])
        tool_wacc_data, tool_dcf_data = self._extract_tool_valuation_data(messages)
        self._overlay_tool_data(
            data,
            tool_wacc_data=tool_wacc_data,
            tool_dcf_data=tool_dcf_data,
            ticker=ticker or data.get("ticker", ""),
            fiscal_year=fiscal_year or data.get("fiscal_year", 0),
            projected_fcfs=projected_fcfs or [],
            share_price=share_price,
            terminal_growth_rate=terminal_growth_rate,
            base_year_ebitda=base_year_ebitda,
        )
        return data

    def _post_process_output(
        self,
        output: DCFValuationOutput,
        result: Dict[str, Any],
        projected_fcfs: Optional[List[float]] = None,
        share_price: Optional[float] = None,
        terminal_growth_rate: float = 0.025,
        base_year_ebitda: Optional[float] = None,
        **kwargs,
    ) -> DCFValuationOutput:
        """Enforces exact deterministic values from calculate_wacc_tool and calculate_dcf_tool."""
        messages = result.get("messages", [])
        tool_wacc_data, tool_dcf_data = self._extract_tool_valuation_data(messages)

        # Enforce exact mathematical values from tools if available
        if tool_wacc_data:
            output.wacc_audit = WACCAudit.model_validate(tool_wacc_data)
        if tool_dcf_data:
            output.enterprise_value = tool_dcf_data["enterprise_value"]
            output.pv_explicit_fcfs = tool_dcf_data["pv_explicit_fcfs"]
            output.pv_terminal_value = tool_dcf_data["pv_terminal_value"]
            output.terminal_value_pct_of_ev = tool_dcf_data["terminal_value_pct_of_ev"]
            output.net_debt = tool_dcf_data["net_debt"]
            output.equity_value = tool_dcf_data["equity_value"]
            output.diluted_shares = tool_dcf_data["diluted_shares"]
            output.implied_fair_value_per_share = tool_dcf_data["fair_value_per_share"]
            output.sensitivity_matrix_markdown = tool_dcf_data["sensitivity_matrix_markdown"]
            output.discounting_convention = tool_dcf_data.get("discounting_convention", "mid_year")

        if projected_fcfs is not None:
            output.projected_fcfs = projected_fcfs

        # Re-derive upside/downside and valuation stance if share price is supplied
        if share_price is not None and share_price > 0:
            output.current_share_price = share_price
            fair = output.implied_fair_value_per_share
            upside = round(((fair - share_price) / share_price) * 100.0, 2)
            output.upside_downside_pct = upside
            if upside > 10.0:
                output.valuation_stance = "Undervalued"
            elif upside < -10.0:
                output.valuation_stance = "Overvalued"
            else:
                output.valuation_stance = "Fairly Valued"
        else:
            output.current_share_price = None
            output.upside_downside_pct = None
            output.valuation_stance = None

        # Re-derive implied EV/EBITDA multiple cross-check strictly if base year EBITDA provided
        if base_year_ebitda is not None and base_year_ebitda > 0:
            output.implied_ev_ebitda = round(output.enterprise_value / base_year_ebitda, 2)
            output.ev_ebitda_source = "derived_from_10k_ebit_plus_depreciation"
        else:
            output.implied_ev_ebitda = None
            output.ev_ebitda_source = None

        return output

    def _overlay_tool_data(
        self,
        target: Dict[str, Any],
        tool_wacc_data: Optional[Dict[str, Any]],
        tool_dcf_data: Optional[Dict[str, Any]],
        ticker: str,
        fiscal_year: int,
        projected_fcfs: List[float],
        share_price: Optional[float],
        terminal_growth_rate: float,
        base_year_ebitda: Optional[float],
    ):
        """Overlays verified mathematical figures from tools onto parsed dict."""
        target["ticker"] = ticker.upper()
        target["fiscal_year"] = fiscal_year
        target["projected_fcfs"] = projected_fcfs
        target["terminal_growth_rate"] = terminal_growth_rate
        target.setdefault("discounting_convention", "mid_year")

        if tool_wacc_data:
            target["wacc_audit"] = tool_wacc_data

        if tool_dcf_data:
            target["enterprise_value"] = tool_dcf_data["enterprise_value"]
            target["pv_explicit_fcfs"] = tool_dcf_data["pv_explicit_fcfs"]
            target["pv_terminal_value"] = tool_dcf_data["pv_terminal_value"]
            target["terminal_value_pct_of_ev"] = tool_dcf_data["terminal_value_pct_of_ev"]
            target["net_debt"] = tool_dcf_data["net_debt"]
            target["equity_value"] = tool_dcf_data["equity_value"]
            target["diluted_shares"] = tool_dcf_data["diluted_shares"]
            target["implied_fair_value_per_share"] = tool_dcf_data["fair_value_per_share"]
            target["sensitivity_matrix_markdown"] = tool_dcf_data["sensitivity_matrix_markdown"]
            target["discounting_convention"] = tool_dcf_data.get("discounting_convention", "mid_year")

        if share_price is not None and share_price > 0:
            target["current_share_price"] = share_price
            fair = target.get("implied_fair_value_per_share", 0.0)
            if fair > 0:
                upside = round(((fair - share_price) / share_price) * 100.0, 2)
                target["upside_downside_pct"] = upside
                if upside > 10.0:
                    target["valuation_stance"] = "Undervalued"
                elif upside < -10.0:
                    target["valuation_stance"] = "Overvalued"
                else:
                    target["valuation_stance"] = "Fairly Valued"
        else:
            target["current_share_price"] = None
            target["upside_downside_pct"] = None
            target["valuation_stance"] = None

        if base_year_ebitda is not None and base_year_ebitda > 0:
            ev = target.get("enterprise_value", 0.0)
            if ev > 0:
                target["implied_ev_ebitda"] = round(ev / base_year_ebitda, 2)
                target["ev_ebitda_source"] = "derived_from_10k_ebit_plus_depreciation"
            else:
                target["implied_ev_ebitda"] = None
                target["ev_ebitda_source"] = None
        else:
            target["implied_ev_ebitda"] = None
            target["ev_ebitda_source"] = None
