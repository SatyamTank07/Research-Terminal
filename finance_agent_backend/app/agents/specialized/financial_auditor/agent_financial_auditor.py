"""Financial Auditor & Statement Analyst Agent.

Deterministic-first audit of 10-K financial statements (Item 8):

1. Code pre-fetches the income statement, balance sheet and cash flow tables for the target
   filing (plus up to two prior filings), parses them, maps line items with generic US-GAAP
   synonyms, checks accounting identities and computes a baseline audit — no LLM involved.
2. The LLM reviews the baseline against the clean tables, may submit cited corrections and
   normalization adjustments through `refine_financial_audit_tool`, reviews restatements and
   writes the quantified auditor narrative.
3. When statements cannot be extracted deterministically, the LLM transcribes inputs into
   `audit_financial_metrics_tool` (llm_fallback); any line item the parser did find fills the
   gaps the LLM left, and every override is logged.

The final artifact's numbers always come from the last deterministic tool result.
"""

import json
import logging
from typing import Any, Dict, List, Optional

from langchain_core.tools import StructuredTool, tool

from app.agents.base import StructuredAgent
from app.agents.pipeline.contract import RunContext, StatusLine, Upstream
from app.agents.registry import AgentRegistry
from app.agents.shared.filing.table_search import retrieve_10k_tables_tool, retrieve_multiyear_financial_series_tool
from app.agents.shared.prompts import render_prompt
from app.agents.specialized.financial_auditor.assemble import (
    assemble_audit,
    assemble_earnings_quality,
    assemble_profitability,
)
from app.agents.specialized.financial_auditor.context import (
    AnnualFinancialInput,
    BalanceSheetInput,
    ExtractionResult,
    PriorBalanceSheetInput,
    build_statement_grids,
    check_audit_window,
    extract_audit_inputs,
    fetch_statement_tables,
    gap_fill_transcribed_inputs,
    resolve_audit_window,
    validate_audit_inputs,
)
from app.agents.specialized.financial_auditor.findings import rank_findings
from app.agents.specialized.financial_auditor.modules.balance_sheet_strength import (
    classify_book_equity,
    measure_balance_sheet,
    measure_solvency,
    screen_balance_sheet_strength,
)
from app.agents.specialized.financial_auditor.modules.capital_allocation import (
    measure_capital_intensity,
    measure_capital_returns,
    measure_share_count,
    screen_capital_allocation,
)
from app.agents.specialized.financial_auditor.modules.cash_conversion import measure_cash_conversion, screen_cash_conversion
from app.agents.specialized.financial_auditor.modules.earnings_normalization import (
    NormalizationAdjustment,
    normalize_earnings,
    screen_earnings_normalization,
    screen_normalization_adjustments,
)
from app.agents.specialized.financial_auditor.modules.financial_history import build_financial_history
from app.agents.specialized.financial_auditor.modules.margins_and_costs import measure_cost_structure, screen_margins_and_costs
from app.agents.specialized.financial_auditor.modules.revenue_quality import screen_revenue_quality
from app.agents.specialized.financial_auditor.modules.statement_integrity import (
    InputCorrection,
    StatementIntegrity,
    apply_input_corrections,
    assess_statement_integrity,
    summarize_for_judge,
)
from app.agents.specialized.financial_auditor.modules.tax_quality import (
    latest_tax_rate,
    measure_tax_quality,
    screen_tax_quality,
)
from app.agents.specialized.financial_auditor.modules.working_capital import measure_working_capital, screen_working_capital
from app.agents.specialized.financial_auditor.state_financial_auditor import FinancialAuditOutput, FinancialAuditResult

logger = logging.getLogger("finance_agent.agents.financial_auditor")

AUDIT_TOOL_MARKERS = ("audit_financial_metrics", "refine_financial_audit")

REFINE_TOOL_DESCRIPTION = """Re-run the deterministic audit on the extracted 10-K inputs with cited corrections and/or normalization adjustments.

Use ONLY when (a) a baseline value contradicts the statement tables, or (b) reported earnings contain
non-recurring / non-core items that must be normalized. Never re-transcribe inputs; the extracted
inputs are already bound to this tool.

Args:
    normalization_adjustments: Items to strip from reported earnings. amount = POSITIVE magnitude in
        $ millions as disclosed. direction = "inflated_reported_earnings" (gains, credit sales, tax
        benefits, valuation-allowance releases -> normalized earnings fall) or
        "depressed_reported_earnings" (impairments, restructuring, litigation charges -> normalized
        earnings rise). affects = "operating_income" (pre-tax, inside EBIT) or "net_income"
        (below-the-line / tax items).
    corrections: Cited overrides of extracted values: target ("annual" | "balance_sheet" |
        "prior_balance_sheet"), fiscal_year, field (input field name), value, reason, source_chunk_id.
"""


def build_audit(
    annual_financials: List[AnnualFinancialInput],
    balance_sheet: BalanceSheetInput,
    prior_balance_sheet: Optional[PriorBalanceSheetInput] = None,
    normalization_adjustments: Optional[List[NormalizationAdjustment]] = None,
) -> Dict[str, Any]:
    """The audit checklist: every module after statement_integrity, in MODULES order; returns a FinancialAuditResult dict."""
    check_audit_window(annual_financials, balance_sheet, prior_balance_sheet)

    # financial_history
    history = build_financial_history(annual_financials)
    latest_input = sorted(annual_financials, key=lambda x: x.fiscal_year)[-1]
    latest = history[-1]

    # margins_and_costs, working_capital, revenue_quality
    cost_structure = measure_cost_structure(annual_financials, history)
    working_capital = measure_working_capital(annual_financials)

    # cash_conversion
    prior_total_assets = prior_balance_sheet.total_assets if prior_balance_sheet else None
    cash_conversion = measure_cash_conversion(annual_financials, history, prior_total_assets)

    # tax_quality
    tax_years = measure_tax_quality(annual_financials)
    latest_tax = latest_tax_rate(latest_input)

    # earnings_normalization
    screened = screen_normalization_adjustments(annual_financials, normalization_adjustments)
    normalized = normalize_earnings(annual_financials, history, screened["accepted"])

    # balance_sheet_strength
    balance = measure_balance_sheet(balance_sheet, prior_balance_sheet)
    equity_status = classify_book_equity(balance_sheet.stockholders_equity)
    solvency = measure_solvency(balance, latest, latest_input, equity_status)

    # capital_allocation
    returns = measure_capital_returns(latest, balance, prior_balance_sheet, latest_tax.rate, equity_status)
    capital_intensity = measure_capital_intensity(annual_financials, history)
    share_count = measure_share_count(annual_financials)

    findings = rank_findings([
        *screen_margins_and_costs(history, cost_structure),
        *screen_working_capital(annual_financials, history, working_capital),
        *screen_revenue_quality(annual_financials, history),
        *screen_cash_conversion(history, cash_conversion),
        *screen_tax_quality(annual_financials, tax_years),
        *screen_earnings_normalization(normalized),
        *screen_balance_sheet_strength(balance_sheet),
        *screen_capital_allocation(history, capital_intensity, share_count),
    ])

    result = FinancialAuditResult(
        multi_year_history=history,
        balance_sheet=balance,
        profitability_and_return_ratios=assemble_profitability(latest_tax, equity_status, returns, latest),
        solvency_and_liquidity_ratios=solvency,
        earnings_quality=assemble_earnings_quality(cash_conversion, tax_years, normalized),
        working_capital=working_capital,
        capital_intensity=capital_intensity,
        cost_structure=cost_structure,
        share_count_history=share_count,
        forensic_findings=findings,
        rejected_normalization_adjustments=screened["rejected"],
    )
    logger.info(
        f"build_audit: FY{balance_sheet.fiscal_year} net debt ${result.balance_sheet.net_debt:,.2f}M | "
        f"ROIC {result.profitability_and_return_ratios.roic_pct}% | findings {len(findings)}"
    )
    return result.model_dump()


def build_audit_from_inputs(
    inputs: Dict[str, Any],
    normalization_adjustments: Optional[List[Any]] = None,
) -> Dict[str, Any]:
    """Validates plain-dict audit inputs (extraction, corrections, gap-fill) and runs `build_audit`."""
    annual_financials, balance_sheet, prior_balance_sheet = validate_audit_inputs(inputs)
    return build_audit(
        annual_financials=annual_financials,
        balance_sheet=balance_sheet,
        prior_balance_sheet=prior_balance_sheet,
        normalization_adjustments=[
            a if isinstance(a, NormalizationAdjustment) else NormalizationAdjustment.model_validate(a)
            for a in (normalization_adjustments or [])
        ],
    )


@tool
def audit_financial_metrics_tool(
    annual_financials: List[AnnualFinancialInput],
    balance_sheet: BalanceSheetInput,
    prior_balance_sheet: Optional[PriorBalanceSheetInput] = None,
    normalization_adjustments: Optional[List[NormalizationAdjustment]] = None,
) -> Dict[str, Any]:
    """
    Audit 10-K financial statements: deterministic ratios, quality-of-earnings analytics and forensic findings.

    All dollar amounts in $ MILLIONS, shares in MILLIONS, CapEx as a POSITIVE magnitude.

    Args:
        annual_financials: Contiguous fiscal years. Required per year: fiscal_year, revenue, operating_income,
            net_income (attributable to common), operating_cash_flow, capital_expenditures. Supply every other
            line item that exists in the statements: gross_profit or cost_of_revenue, operating_expenses,
            pretax_income, income_tax_expense, net_income_total, depreciation_amortization,
            stock_based_compensation, deferred_income_taxes, accounts_receivable, inventories, accounts_payable,
            total_assets, diluted_weighted_shares, period_end_shares_outstanding.
        balance_sheet: Latest-year snapshot: cash_and_equivalents, marketable_securities (ALL short-term
            investments + non-current marketable securities), short_term_debt, long_term_debt,
            stockholders_equity, weighted_diluted_shares and/or period_end_shares_outstanding, current_assets,
            current_liabilities, total_assets.
        prior_balance_sheet: Prior-year snapshot (enables average-capital ROIC/ROE and accruals).
        normalization_adjustments: Non-recurring / non-core items: positive `amount` plus `direction`
            ("inflated_reported_earnings" for gains/credits/tax benefits, "depressed_reported_earnings"
            for impairments/restructuring/litigation charges).
    """
    return build_audit(
        annual_financials=annual_financials,
        balance_sheet=balance_sheet,
        prior_balance_sheet=prior_balance_sheet,
        normalization_adjustments=normalization_adjustments,
    )


@AgentRegistry.register("financial_auditor")
class FinancialAuditorAgent(StructuredAgent[FinancialAuditOutput]):
    """Deterministic-first auditor of multi-year 10-K financial statements."""

    prompt_name = "financial_auditor"
    tools = [retrieve_10k_tables_tool, retrieve_multiyear_financial_series_tool]
    output_schema = FinancialAuditOutput
    default_recursion_limit = 25

    def __init__(self, model_name: str = "openai:gpt-4o-mini", recursion_limit: Optional[int] = None):
        super().__init__(model_name=model_name, recursion_limit=recursion_limit)
        self._run_tools: Optional[List[Any]] = None

    def get_tools(self) -> List[Any]:
        return self._run_tools if self._run_tools is not None else self.tools + [audit_financial_metrics_tool]

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def audit(
        self,
        ticker: str,
        fiscal_year: int,
        available_fiscal_years: Optional[List[int]] = None,
        callbacks: Optional[List[Any]] = None,
    ) -> FinancialAuditOutput:
        """Audits the 10-K statements for `ticker` / `fiscal_year` and returns a validated FinancialAuditOutput."""
        ticker = ticker.upper()
        target_years = resolve_audit_window(fiscal_year, available_fiscal_years)

        tables = fetch_statement_tables(ticker, fiscal_year, available_fiscal_years)
        extraction: Optional[ExtractionResult] = None
        if tables.get("income_statement"):
            try:
                extraction = extract_audit_inputs(build_statement_grids(tables), fiscal_year, target_years)
            except Exception as e:
                logger.warning(f"Statement extraction failed for {ticker} FY{fiscal_year}: {e}")
        integrity = assess_statement_integrity(extraction) if extraction is not None else None

        baseline_inputs: Optional[Dict[str, Any]] = None
        baseline: Optional[Dict[str, Any]] = None
        if extraction is not None and integrity.is_usable:
            baseline_inputs = extraction.to_tool_kwargs()
            try:
                baseline = build_audit_from_inputs(baseline_inputs)
            except Exception as e:
                extraction.warnings.append(f"Baseline audit rejected extracted inputs: {e}")
                logger.warning(f"Baseline audit failed for {ticker} FY{fiscal_year}: {e}")
                baseline_inputs = None

        mode = "deterministic" if baseline is not None else "llm_fallback"
        self._configure_tools(mode, baseline_inputs)
        logger.info(
            f"FinancialAuditor {ticker} FY{fiscal_year}: mode={mode}, "
            f"completeness={integrity.completeness_score if integrity else None}"
        )

        query = render_prompt(
            "financial_auditor", "prompt_financial_auditor_query.j2",
            ticker=ticker,
            fiscal_year=fiscal_year,
            extraction_mode=mode,
            audit_years=extraction.years if mode == "deterministic" else target_years,
            available_years_str=", ".join(map(str, sorted(available_fiscal_years))) if available_fiscal_years else None,
            baseline_json=json.dumps(baseline, default=str) if baseline else None,
            extraction_json=json.dumps(summarize_for_judge(extraction, integrity), default=str) if extraction else None,
            clean_tables=extraction.clean_tables if extraction else {},
        )

        resolution = dict(
            extraction=extraction,
            integrity=integrity,
            extraction_mode=mode,
            baseline=baseline,
            baseline_inputs=baseline_inputs,
            prefetched_tables=tables,
        )
        fallback_defaults = {
            "auditor_summary": "Auditor narrative unavailable; refer to the deterministic analytics and findings.",
            "restatement_notes": [],
        }
        try:
            return self.execute_structured(
                query,
                ticker=ticker,
                fiscal_year=fiscal_year,
                fallback_defaults=fallback_defaults,
                callbacks=callbacks,
                **resolution,
            )
        except Exception as e:
            if baseline is None:
                raise
            # The LLM step only refines and narrates; the code-built baseline is a complete audit on its own.
            logger.warning(f"FinancialAuditor {ticker} FY{fiscal_year}: LLM step failed ({e!r}); using deterministic baseline.")
            fallback_defaults["auditor_summary"] = (
                "The auditor narrative step failed, so this audit is the deterministic baseline computed from the "
                "parsed statements without LLM refinement; refer to the tables and forensic findings."
            )
            empty: Dict[str, Any] = {"messages": []}
            output = self._extract_structured_output(
                empty, ticker=ticker, fiscal_year=fiscal_year, fallback_defaults=fallback_defaults, **resolution,
            )
            return self._post_process_output(output, empty, ticker=ticker, fiscal_year=fiscal_year, **resolution)

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------
    def _configure_tools(self, mode: str, baseline_inputs: Optional[Dict[str, Any]]) -> None:
        if mode == "deterministic" and baseline_inputs is not None:
            self._run_tools = self.tools + [self._build_refine_tool(baseline_inputs)]
        else:
            self._run_tools = self.tools + [audit_financial_metrics_tool]
        self._cached_agent = None

    @staticmethod
    def _build_refine_tool(baseline_inputs: Dict[str, Any]) -> StructuredTool:
        def refine_financial_audit(
            normalization_adjustments: Optional[List[NormalizationAdjustment]] = None,
            corrections: Optional[List[InputCorrection]] = None,
        ) -> Dict[str, Any]:
            corrected = apply_input_corrections(baseline_inputs, corrections)
            audit = build_audit_from_inputs(corrected["inputs"], normalization_adjustments)
            audit["applied_corrections"] = corrected["applied"]
            audit["rejected_corrections"] = corrected["rejected"]
            return audit

        return StructuredTool.from_function(
            func=refine_financial_audit,
            name="refine_financial_audit_tool",
            description=REFINE_TOOL_DESCRIPTION,
        )

    # ------------------------------------------------------------------
    # Output lifecycle hooks
    # ------------------------------------------------------------------
    def _pre_validate_data(self, data: Dict[str, Any], result: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        return self._apply_resolution(data, result, **kwargs)

    def _post_process_output(
        self, output: FinancialAuditOutput, result: Dict[str, Any], **kwargs
    ) -> FinancialAuditOutput:
        merged = self._apply_resolution(output.model_dump(), result, **kwargs)
        return FinancialAuditOutput.model_validate(merged)

    def _apply_resolution(
        self,
        data: Dict[str, Any],
        result: Dict[str, Any],
        extraction: Optional[ExtractionResult] = None,
        integrity: Optional[StatementIntegrity] = None,
        extraction_mode: str = "llm_fallback",
        baseline: Optional[Dict[str, Any]] = None,
        baseline_inputs: Optional[Dict[str, Any]] = None,
        prefetched_tables: Optional[Dict[str, List[Dict[str, Any]]]] = None,
        **_: Any,
    ) -> Dict[str, Any]:
        """Picks the authoritative audit (last tool result, baseline or gap-filled re-run) and assembles the output."""
        messages = result.get("messages", [])
        log: List[str] = []
        stale = False

        tool_payload = self._last_audit_payload(messages)
        if extraction_mode == "deterministic":
            audit = tool_payload or baseline
            if tool_payload is None:
                log.append("No refinement requested by the auditor; deterministic baseline retained.")
            else:
                log.extend(self._describe_refinement(tool_payload))
        else:
            audit = tool_payload
            if tool_payload is not None and extraction is not None:
                args = self._last_audit_call_args(messages)
                if args:
                    gap_fill = gap_fill_transcribed_inputs(args, extraction)
                    log.extend(gap_fill["log"])
                    if gap_fill["changed"]:
                        audit = build_audit_from_inputs(gap_fill["inputs"], gap_fill["adjustments"])
                        stale = True

        return assemble_audit(
            data,
            audit=audit,
            log=log,
            stale=stale,
            extraction=extraction,
            integrity=integrity,
            extraction_mode=extraction_mode,
            prefetched_tables=prefetched_tables,
            messages=messages,
        )

    # ------------------------------------------------------------------
    # Resolution helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _last_audit_payload(messages: List[Any]) -> Optional[Dict[str, Any]]:
        """Most recent successful audit / refine tool result."""
        for msg in reversed(messages):
            name = getattr(msg, "name", "") or ""
            if not any(marker in name for marker in AUDIT_TOOL_MARKERS):
                continue
            raw = getattr(msg, "content", "")
            payload = raw if isinstance(raw, dict) else None
            if isinstance(raw, str):
                try:
                    payload = json.loads(raw)
                except Exception:
                    payload = None
            if isinstance(payload, dict) and "multi_year_history" in payload:
                return payload
        return None

    @staticmethod
    def _last_audit_call_args(messages: List[Any]) -> Optional[Dict[str, Any]]:
        for msg in reversed(messages):
            for call in reversed(getattr(msg, "tool_calls", None) or []):
                if "audit_financial_metrics" in (call.get("name") or ""):
                    return call.get("args") or None
        return None

    @staticmethod
    def _describe_refinement(payload: Dict[str, Any]) -> List[str]:
        log = []
        for c in payload.get("applied_corrections", []):
            log.append(
                f"Corrected {c['target']} FY{c['fiscal_year']}.{c['field']}: {c.get('previous_value')} -> "
                f"{c['value']} ({c['reason']}; source {c.get('source_chunk_id') or 'n/a'})."
            )
        for c in payload.get("rejected_corrections", []):
            log.append(f"Rejected correction {c.get('target')} FY{c.get('fiscal_year')}.{c.get('field')}: {c.get('rejection')}.")
        for a in payload.get("rejected_normalization_adjustments", []):
            log.append(f"Rejected normalization FY{a.get('fiscal_year')} '{a.get('label')}': {a.get('rejection')}.")
        for year in payload.get("earnings_quality", []):
            for adj in year.get("adjustments", []):
                effect = "added back" if adj["direction"] == "depressed_reported_earnings" else "removed"
                log.append(
                    f"Normalized FY{adj['fiscal_year']} {adj['affects']}: {effect} '{adj['label']}' "
                    f"({adj['amount']:,.1f}M, {adj['direction']}) — {adj.get('rationale') or 'no rationale given'}."
                )
        return log


def audit_financial_statements(run: RunContext, upstream: Upstream) -> FinancialAuditOutput:
    """Pipeline entry: multi-year Item 8 audit anchored on the run's filing."""
    return FinancialAuditorAgent().audit(
        ticker=run.ticker,
        fiscal_year=run.fiscal_year,
        available_fiscal_years=list(run.available_fiscal_years),
        callbacks=run.callbacks,
    )


def summarize_audit(audit: Optional[FinancialAuditOutput]) -> StatusLine:
    if audit is None:
        return "Financial statement audit unavailable", None
    severe_cnt = sum(1 for f in audit.forensic_findings if f.severity in ("high", "medium"))
    return (
        f"Audited financial statements & ratios ({severe_cnt} high/medium forensic findings, "
        f"{audit.data_quality.extraction_mode} extraction)",
        None,
    )
