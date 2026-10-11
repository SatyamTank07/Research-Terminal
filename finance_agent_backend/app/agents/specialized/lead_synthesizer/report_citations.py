"""Filing sources for the code-assembled research report.

Maps specialist artifacts to `SourceSpec`s. Verified strategist and risk quotes are verbatim sources. Audit,
forecast and valuation figures are derived from the statement-table chunks the auditor recorded in
`field_provenance`. Market inputs carry their provider tag. The report templates call the marker methods
(each returns an inline marker or "", and never raises), and `figure_entries` seeds the index that
attributes LLM prose figures.
"""

import logging
from functools import wraps
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from app.agents.tools.citation_registry import CitationRegistry, SourceSpec

logger = logging.getLogger("finance_agent.agents.lead_synthesizer.citations")

IS, BS, CF = "income_statement", "balance_sheet", "cash_flow"
STATEMENT_NAMES = {IS: "income statement", BS: "balance sheet", CF: "cash flow statement"}

# Audit metric -> statements whose rows it is computed from (every attribute rendered by audit_tables.j2).
ROW_STATEMENTS: Dict[str, Tuple[str, ...]] = {
    # Table A
    "revenue": (IS,), "revenue_growth_pct": (IS,), "gross_margin_pct": (IS,), "operating_income": (IS,),
    "operating_margin_pct": (IS,), "net_income": (IS,), "net_margin_pct": (IS,),
    "operating_cash_flow": (CF,), "capital_expenditures": (CF,), "free_cash_flow": (CF,),
    "fcf_conversion_pct": (CF, IS), "stock_based_compensation": (CF,), "sbc_adjusted_fcf": (CF,),
    # Table B
    "normalized_operating_income": (IS,), "normalized_operating_margin_pct": (IS,), "normalized_net_income": (IS,),
    "adjustments_pct_of_operating_income": (IS,), "non_operating_income": (IS,),
    "non_operating_to_operating_income_pct": (IS,), "effective_tax_rate_pct": (IS,),
    "deferred_tax_to_net_income_pct": (CF, IS), "accruals": (IS, CF), "sloan_accruals_ratio_pct": (IS, CF, BS),
    # Table C
    "gross_margin_change_bps": (IS,), "operating_margin_change_bps": (IS,), "operating_expenses": (IS,),
    "operating_expense_growth_pct": (IS,), "opex_growth_minus_revenue_growth_pp": (IS,),
    "dso_days": (BS, IS), "dio_days": (BS, IS), "dpo_days": (BS, IS), "cash_conversion_cycle_days": (BS, IS),
    # Table D
    "capex_to_revenue_pct": (CF, IS), "capex_to_depreciation": (CF,), "ocf_change": (CF,), "capex_change": (CF,),
    "fcf_change": (CF,), "fcf_change_from_capex_pct": (CF,), "diluted_weighted_shares": (IS,),
    "period_end_shares_outstanding": (BS,), "period_end_share_growth_pct": (BS,),
    # Table E (balance-sheet snapshot) and valuation bridge inputs
    "cash_and_equivalents": (BS,), "marketable_securities": (BS,), "total_liquid_cash": (BS,), "total_debt": (BS,),
    "net_debt": (BS,), "stockholders_equity": (BS,), "valuation_shares_outstanding": (BS,), "current_ratio": (BS,),
    "ebitda": (IS, CF), "net_debt_to_ebitda": (BS, IS, CF), "total_debt_to_equity": (BS,),
    "roic_pct": (IS, BS), "roic_on_gross_capital_pct": (IS, BS), "roe_pct": (IS, BS),
}

VALUATION_FIGURES = (
    "implied_fair_value_per_share", "bear_fair_value", "bull_fair_value", "probability_weighted_fair_value",
    "enterprise_value", "equity_value", "terminal_value_pct_of_ev",
)
SEGMENT_FIGURES = ("revenue", "segment_profit", "revenue_share_pct", "profit_margin_pct", "profit_share_pct", "revenue_growth_pct")
REVENUE_LINE_FIGURES = ("revenue", "revenue_share_pct", "revenue_growth_pct")
MARKET_INPUTS: Dict[str, Tuple[str, Callable[[Any], Optional[str]]]] = {
    "price": ("Share price", lambda v: v.data_quality.share_price_source),
    "rf": ("Risk-free rate", lambda v: v.wacc_audit.risk_free_rate_source),
    "erp": ("Equity risk premium", lambda v: v.wacc_audit.equity_risk_premium_source),
    "beta": ("Beta", lambda v: v.wacc_audit.beta_source),
}
MARKET_FIGURES: Tuple[Tuple[str, Callable[[Any], Any]], ...] = (
    ("price", lambda v: v.current_share_price),
    ("rf", lambda v: v.wacc_audit.risk_free_rate_pct),
    ("erp", lambda v: v.wacc_audit.equity_risk_premium_pct),
    ("beta", lambda v: v.wacc_audit.beta),
)
_AUDIT_SERIES = ("multi_year_history", "earnings_quality", "cost_structure", "working_capital", "capital_intensity", "share_count_history")
_AUDIT_SNAPSHOTS = ("balance_sheet", "profitability_and_return_ratios", "solvency_and_liquidity_ratios")


def _safe(method: Callable[..., str]) -> Callable[..., str]:
    """A citation failure must never break the report: log it and emit no marker."""

    @wraps(method)
    def wrapper(self: "ReportCiter", *args: Any, **kwargs: Any) -> str:
        try:
            return method(self, *args, **kwargs)
        except Exception as e:
            logger.warning(f"Citation marker {method.__name__} failed ({e!r}); emitting none.")
            return ""

    return wrapper


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _statement_chunk_map(financial_audit: Any) -> Dict[Tuple[str, int], List[str]]:
    """(statement, fiscal year) -> statement-table chunk ids, from the auditor's field provenance."""
    out: Dict[Tuple[str, int], List[str]] = {}
    provenance = getattr(getattr(financial_audit, "data_quality", None), "field_provenance", None) or {}
    for key, entry in provenance.items():
        if not isinstance(key, str) or not key.startswith("FY") or "." not in key or not isinstance(entry, dict):
            continue
        try:
            year = int(key[2:key.index(".")])
        except ValueError:
            continue
        chunk_id, statement = entry.get("chunk_id"), entry.get("statement")
        if chunk_id and statement:
            ids = out.setdefault((statement, year), [])
            if chunk_id not in ids:
                ids.append(chunk_id)
    return out


class NullCiter:
    """Stand-in when citations cannot be built: every marker is empty."""

    def __getattr__(self, name: str) -> Callable[..., str]:
        return lambda *args, **kwargs: ""


class ReportCiter:
    """Jinja-facing helper returning inline markers for report elements."""

    def __init__(
        self,
        registry: CitationRegistry,
        fiscal_year: int,
        financial_audit: Any = None,
        business_moat: Any = None,
        risk_audit: Any = None,
        forecast: Any = None,
        dcf_valuation: Any = None,
    ):
        self.registry = registry
        self._anchor = fiscal_year
        self._fa = financial_audit
        self._moat = business_moat
        self._risk = risk_audit
        self._forecast = forecast
        self._val = dcf_valuation
        self._statement_chunks = _statement_chunk_map(financial_audit)

    # --- specs ------------------------------------------------------------
    def _statement_spec(self, statements: Sequence[str], years: Iterable[Optional[int]]) -> Optional[SourceSpec]:
        span_years = sorted({y for y in years if y is not None})
        ids: List[str] = []
        for statement in statements:
            for year in span_years:
                ids += [c for c in self._statement_chunks.get((statement, year), []) if c not in ids]
        if not ids:
            return None
        names = [STATEMENT_NAMES[s] for s in statements if s in STATEMENT_NAMES]
        listed = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]
        # No year span: derived sources dedupe on their chunk set, so one label serves every row built from it.
        return SourceSpec(kind="derived", chunk_ids=tuple(ids), derived_from=f"Computed from the audited {listed}")

    def _field_spec(self, attr: str, year: Optional[int] = None) -> Optional[SourceSpec]:
        return self._statement_spec(ROW_STATEMENTS.get(attr, ()), [year or self._anchor])

    def _all_statements_spec(self, year: Optional[int] = None) -> Optional[SourceSpec]:
        return self._statement_spec((IS, BS, CF), [year or self._anchor])

    @staticmethod
    def _quote_spec(chunk_id: Optional[str], quote: Optional[str]) -> Optional[SourceSpec]:
        if not chunk_id or not quote:
            return None
        return SourceSpec(kind="verbatim", chunk_ids=(chunk_id,), quote=quote)

    def _evidence_specs(self, claim: str, subject: Optional[str] = None) -> List[Optional[SourceSpec]]:
        evidence = getattr(self._moat, "evidence", None) or []
        return [
            self._quote_spec(e.chunk_id, e.quote)
            for e in evidence
            if e.claim == claim and (subject is None or (e.subject or "") == subject)
        ]

    @staticmethod
    def _chunks_spec(chunk_ids: Iterable[Optional[str]], derived_from: str) -> Optional[SourceSpec]:
        ids = tuple(dict.fromkeys(str(c) for c in chunk_ids if c))
        return SourceSpec(kind="derived", chunk_ids=ids, derived_from=derived_from) if ids else None

    def _segments_spec(self) -> Optional[SourceSpec]:
        figures = getattr(self._moat, "segment_financials", None) or []
        return self._chunks_spec((f.source_chunk_id for f in figures), "Reported in the segment note (ASC 280)")

    def _revenue_mix_spec(self) -> Optional[SourceSpec]:
        figures = getattr(self._moat, "revenue_lines", None) or []
        return self._chunks_spec((f.source_chunk_id for f in figures), "Reported in the revenue disaggregation table (ASC 606)")

    def _forecast_spec(self) -> Optional[SourceSpec]:
        citations = getattr(self._forecast, "citations", None) or []
        ids = (c.get("chunk_id") for c in citations if isinstance(c, dict))
        return self._chunks_spec(ids, "MD&A passages behind the forecast drivers")

    def _market_spec(self, name: str) -> Optional[SourceSpec]:
        if self._val is None or name not in MARKET_INPUTS:
            return None
        label, provider_of = MARKET_INPUTS[name]
        provider = provider_of(self._val)
        if not provider or provider == "unavailable":
            return None
        return SourceSpec(kind="market", label=label, provider=provider)

    # --- template markers -------------------------------------------------
    @_safe
    def row(self, attr: str, items: Any) -> str:
        years = [getattr(item, "fiscal_year", None) for item in (items or [])]
        return self.registry.cite(self._statement_spec(ROW_STATEMENTS.get(attr, ()), years))

    @_safe
    def field(self, attr: str, year: Optional[int] = None) -> str:
        return self.registry.cite(self._field_spec(attr, year))

    @_safe
    def statements(self, year: Optional[int] = None) -> str:
        return self.registry.cite(self._all_statements_spec(year))

    @_safe
    def statement(self, name: str, year: Optional[int] = None) -> str:
        return self.registry.cite(self._statement_spec((name,), [year or self._anchor]))

    @_safe
    def evidence(self, claim: str, subject: Optional[str] = None) -> str:
        return self.registry.cite_all(self._evidence_specs(claim, subject))

    @_safe
    def risk(self, risk_item: Any) -> str:
        return self.registry.cite_all(self._quote_spec(e.chunk_id, e.quote) for e in risk_item.evidence if e.verified)

    @_safe
    def mitigation(self, risk_item: Any) -> str:
        ev = risk_item.mitigation_evidence
        return self.registry.cite(self._quote_spec(ev.chunk_id, ev.quote) if ev is not None and ev.verified else None)

    @_safe
    def segments(self) -> str:
        return self.registry.cite(self._segments_spec())

    @_safe
    def revenue_mix(self) -> str:
        return self.registry.cite(self._revenue_mix_spec())

    @_safe
    def forecast_drivers(self) -> str:
        return self.registry.cite(self._forecast_spec())

    @_safe
    def market(self, name: str) -> str:
        return self.registry.cite(self._market_spec(name))

    # --- prose attribution ----------------------------------------------
    def _audit_items(self) -> List[Tuple[Optional[int], Any]]:
        fa = self._fa
        if fa is None:
            return []
        items: List[Tuple[Optional[int], Any]] = []
        for name in _AUDIT_SERIES:
            for item in getattr(fa, name, None) or []:
                items.append((getattr(item, "fiscal_year", None), item))
        snapshot_year = getattr(getattr(fa, "balance_sheet", None), "fiscal_year", None) or self._anchor
        for name in _AUDIT_SNAPSHOTS:
            item = getattr(fa, name, None)
            if item is not None:
                items.append((snapshot_year, item))
        return sorted(items, key=lambda pair: -(pair[0] or 0))

    def figure_entries(self) -> List[Tuple[str, SourceSpec]]:
        """(text, source) pairs in attribution priority: audit data newest year first, valuation, segments, quotes."""
        entries: List[Tuple[str, SourceSpec]] = []

        def add(value: Any, spec: Optional[SourceSpec]) -> None:
            if spec is not None and _is_number(value):
                entries.append((f"{float(value):.6f}", spec))

        for year, item in self._audit_items():
            for attr, value in item.model_dump().items():
                if attr in ROW_STATEMENTS:
                    add(value, self._field_spec(attr, year))
        val = self._val
        if val is not None:
            for attr in VALUATION_FIGURES:
                add(getattr(val, attr, None), self._all_statements_spec())
            add(getattr(val, "net_debt", None), self._field_spec("net_debt"))
            add(getattr(val, "diluted_shares", None), self._field_spec("valuation_shares_outstanding"))
            for name, value_of in MARKET_FIGURES:
                add(value_of(val), self._market_spec(name))
        moat = self._moat
        if moat is not None:
            for figure in moat.segment_financials:
                for attr in SEGMENT_FIGURES:
                    add(getattr(figure, attr, None), self._segments_spec())
            for figure in moat.revenue_lines:
                for attr in REVENUE_LINE_FIGURES:
                    add(getattr(figure, attr, None), self._revenue_mix_spec())
            for e in moat.evidence:
                spec = self._quote_spec(e.chunk_id, e.quote)
                if spec is not None:
                    entries.append((e.quote, spec))
        if self._risk is not None:
            for risk_item in self._risk.identified_risks:
                for e in risk_item.evidence:
                    spec = self._quote_spec(e.chunk_id, e.quote) if e.verified else None
                    if spec is not None:
                        entries.append((e.quote, spec))
        return entries
