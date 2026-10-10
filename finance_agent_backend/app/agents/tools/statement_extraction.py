"""Deterministic 10-K Statement Line-Item Extraction.

Maps parsed Item 8 statement tables (income statement, balance sheet, cash flows)
onto the canonical inputs of `audit_financial_metrics`, so the Financial Auditor's
numbers come from Python rather than LLM transcription.

Design rules:
- Generic US-GAAP label synonyms only; no company-specific line items.
- Latest Filing Precedence: grids are supplied newest filing first, and the first
  filing that reports a value for a (field, year) wins. Divergent values reported
  by older filings are surfaced as `cross_filing_differences` (restatement candidates).
- Accounting identities (balance sheet, gross profit, net income tie-out) are checked
  and reported, never silently "fixed".
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from app.services.financial_tables import (
    StatementGrid,
    StatementRow,
    parse_statement_table,
    render_clean_statement,
)

logger = logging.getLogger("finance_agent.tools.statement_extraction")

# Share counts above this are assumed to be reported in thousands (normalized to millions).
SHARE_THOUSANDS_THRESHOLD = 100_000.0
IDENTITY_TOLERANCE_PCT = 0.5
CROSS_FILING_TOLERANCE_PCT = 0.5

REQUIRED_ANNUAL_FIELDS = [
    "revenue",
    "operating_income",
    "net_income",
    "operating_cash_flow",
    "capital_expenditures",
]
REQUIRED_BALANCE_FIELDS = ["cash_and_equivalents", "stockholders_equity"]
SHARE_COUNT_FIELDS = ["weighted_diluted_shares", "period_end_shares_outstanding"]

# Fields that an institutional audit expects to find; drives the completeness score.
EXPECTED_LATEST_FIELDS = REQUIRED_ANNUAL_FIELDS + [
    "gross_profit_or_cost_of_revenue",
    "pretax_income",
    "income_tax_expense",
    "depreciation_amortization",
    "stock_based_compensation",
    "diluted_weighted_shares",
    "current_assets",
    "current_liabilities",
    "total_assets",
]


# ==============================================================================
# 1. Generic US-GAAP Label Synonyms (ordered: earlier patterns win)
# ==============================================================================
INCOME_STATEMENT_SYNONYMS: Dict[str, List[str]] = {
    "revenue": [
        r"^total (net )?revenues?$",
        r"^total net sales$",
        r"^net sales$",
        r"^total net revenues?$",
        r"^net revenues?$",
        r"^revenues?$",
        r"^total revenues? and other income$",
        r"^sales$",
    ],
    "cost_of_revenue": [
        r"^total cost of (revenues?|sales|goods sold)$",
        r"^total costs? of (revenues?|sales)$",
        r"^cost of (revenues?|sales|goods sold)( \(.*\))?$",
    ],
    "gross_profit": [r"^(total )?gross (profit|margin)$"],
    "operating_expenses": [r"^total operating expenses$"],
    "operating_income": [
        r"^(total )?operating income( \(loss\))?$",
        r"^(income|loss)( \(loss\))? from operations$",
        r"^operating (profit|loss|income \(loss\))$",
    ],
    "interest_income": [r"^interest income(, net)?$", r"^interest and dividend income$"],
    "pretax_income": [r"^(income|earnings|loss)( \(loss\))?.* before .*tax"],
    "income_tax_expense": [
        r"^(?!.*\bbefore\b)(?!.*net of tax)(?=.*\b(provision|benefit|expense)\b).*\bincome tax",
        r"^(provision for )?income taxes$",
    ],
    "net_income_total": [r"^net (income|earnings|loss)( \(loss\))?$", r"^net income \(loss\)$"],
    "net_income_attributable": [
        r"^net (income|earnings)( \(loss\))? attributable to (?!non|redeemable)",
    ],
}

CASH_FLOW_SYNONYMS: Dict[str, List[str]] = {
    "net_income_cf": [r"^net (income|earnings|loss)( \(loss\))?( including .*)?$"],
    "depreciation_amortization": [r"depreciation"],
    "stock_based_compensation": [r"(stock|share)[- ]based compensation"],
    "deferred_income_taxes": [r"^deferred (income )?tax"],
    "operating_cash_flow": [r"cash.*(provided|generated|from|used).*operating activities"],
    "capital_expenditures": [
        r"^capital expenditures",
        r"^(purchases?|payments? for( the)?( acquisition of)?|additions? to|acquisitions? of|expenditures? for)\b"
        r".*(property|plant|equipment|fixed assets)",
    ],
}

BALANCE_SHEET_SYNONYMS: Dict[str, List[str]] = {
    "cash_and_equivalents": [r"^cash and cash equivalents$", r"^cash and equivalents$", r"^cash$"],
    "accounts_receivable": [r"^accounts receivable", r"^(trade )?receivables?(, net)?$", r"^trade accounts receivable"],
    "inventories": [r"^inventor(y|ies)"],
    "current_assets": [r"^total current assets$"],
    "total_assets": [r"^total assets$"],
    "accounts_payable": [r"^accounts payable"],
    "current_liabilities": [r"^total current liabilities$"],
    "total_liabilities": [r"^total liabilities$"],
    "stockholders_equity": [
        r"^total (stockholders|shareholders)'?s?'? (equity|deficit|investment)",
        r"^total .*(stockholders|shareholders)'?s?'? equity$",
        r"^total equity$",
    ],
    "noncontrolling_interests": [r"^non-?controlling interests?( in (consolidated )?subsidiaries)?$"],
    "redeemable_noncontrolling_interests": [r"^redeemable non-?controlling"],
    "total_liabilities_and_equity": [r"^total liabilities(,)? .*and .*equity"],
}

SECURITIES_PATTERNS = [
    r"^short-term (investments|marketable securities)",
    r"^(current |non-?current )?marketable securities$",
    r"^investments in marketable securities",
    r"^available-for-sale (debt )?securities",
]
DEBT_ROW_PATTERN = re.compile(
    r"\b(debt|borrowings|commercial paper|notes payable|term loans?|senior notes|convertible .*notes)\b"
)
DEBT_ROW_EXCLUDE = re.compile(r"^total|deferred|issuance|interest payable|accrued")
SHARES_OUTSTANDING_CAPTION = re.compile(
    r"([\d,]+(?:\.\d+)?) and ([\d,]+(?:\.\d+)?) shares (?:were )?(?:issued and )?outstanding"
)
DILUTED_SHARE_SECTION = re.compile(r"weighted|shares used|number of shares|shares outstanding|denominator")
DILUTED_SHARE_LABEL = re.compile(r"^diluted( shares)?$|weighted.average.*diluted|diluted.*weighted|^diluted .*shares")


# ==============================================================================
# 2. Extraction Result Container
# ==============================================================================
@dataclass
class ExtractionResult:
    """Deterministically extracted audit inputs with provenance and quality diagnostics."""

    fiscal_year: int
    years: List[int] = field(default_factory=list)
    annual: Dict[int, Dict[str, float]] = field(default_factory=dict)
    balance: Dict[int, Dict[str, float]] = field(default_factory=dict)
    field_provenance: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    missing_fields: List[str] = field(default_factory=list)
    identity_checks: List[Dict[str, Any]] = field(default_factory=list)
    cross_filing_differences: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    clean_tables: Dict[str, str] = field(default_factory=dict)
    citations: List[Dict[str, Any]] = field(default_factory=list)

    # ------------------------------------------------------------------
    @property
    def failed_identity_checks(self) -> List[Dict[str, Any]]:
        return [c for c in self.identity_checks if c.get("status") == "failed"]

    @property
    def is_usable(self) -> bool:
        """True when every required field exists for >= 2 contiguous years ending at the target year."""
        if len(self.years) < 2 or self.years[-1] != self.fiscal_year:
            return False
        latest_bs = self.balance.get(self.fiscal_year, {})
        if any(latest_bs.get(f) is None for f in REQUIRED_BALANCE_FIELDS):
            return False
        if all(latest_bs.get(f) is None for f in SHARE_COUNT_FIELDS):
            return False
        return not any(c.get("critical") for c in self.failed_identity_checks)

    @property
    def completeness_score(self) -> float:
        latest = self.annual.get(self.fiscal_year, {})
        bs = self.balance.get(self.fiscal_year, {})
        found = 0
        for f in EXPECTED_LATEST_FIELDS:
            if f == "gross_profit_or_cost_of_revenue":
                present = latest.get("gross_profit") is not None or latest.get("cost_of_revenue") is not None
            else:
                present = latest.get(f) is not None or bs.get(f) is not None
            found += 1 if present else 0
        return round(found / len(EXPECTED_LATEST_FIELDS), 3)

    # ------------------------------------------------------------------
    def to_annual_inputs(self) -> List[Dict[str, Any]]:
        """Builds `AnnualFinancialInput`-compatible dicts for the contiguous audit window."""
        return self._annual_records(self.years)

    def to_annual_inputs_all_years(self) -> List[Dict[str, Any]]:
        """Annual records for every year any parsed statement reported (used to gap-fill LLM inputs)."""
        return self._annual_records(sorted(self.annual))

    def _annual_records(self, years: List[int]) -> List[Dict[str, Any]]:
        out = []
        for yr in years:
            vals = dict(self.annual.get(yr, {}))
            bs = self.balance.get(yr, {})
            record: Dict[str, Any] = {"fiscal_year": yr}
            for key in [
                "revenue", "gross_profit", "cost_of_revenue", "operating_expenses", "operating_income",
                "interest_income", "pretax_income", "income_tax_expense", "net_income", "net_income_total",
                "operating_cash_flow", "capital_expenditures", "depreciation_amortization",
                "stock_based_compensation", "deferred_income_taxes", "diluted_weighted_shares",
            ]:
                if vals.get(key) is not None:
                    record[key] = vals[key]
            for key in ["accounts_receivable", "inventories", "accounts_payable", "total_assets"]:
                if bs.get(key) is not None:
                    record[key] = bs[key]
            if bs.get("period_end_shares_outstanding") is not None:
                record["period_end_shares_outstanding"] = bs["period_end_shares_outstanding"]
            out.append(record)
        return out

    def to_balance_sheet_input(self) -> Dict[str, Any]:
        bs = self.balance.get(self.fiscal_year, {})
        keys = [
            "cash_and_equivalents", "marketable_securities", "short_term_debt", "long_term_debt",
            "stockholders_equity", "weighted_diluted_shares", "period_end_shares_outstanding",
            "current_assets", "current_liabilities", "total_assets", "noncontrolling_interests",
        ]
        record: Dict[str, Any] = {"fiscal_year": self.fiscal_year}
        for k in keys:
            if bs.get(k) is not None:
                record[k] = bs[k]
        return record

    def to_prior_balance_sheet_input(self) -> Optional[Dict[str, Any]]:
        prior_year = self.fiscal_year - 1
        bs = self.balance.get(prior_year)
        if not bs or bs.get("cash_and_equivalents") is None or bs.get("stockholders_equity") is None:
            return None
        record: Dict[str, Any] = {"fiscal_year": prior_year}
        for k in [
            "cash_and_equivalents", "marketable_securities", "short_term_debt", "long_term_debt",
            "stockholders_equity", "total_assets",
        ]:
            if bs.get(k) is not None:
                record[k] = bs[k]
        return record

    def to_tool_kwargs(self) -> Dict[str, Any]:
        """Keyword arguments for `audit_financial_metrics` (validated into Pydantic models by the caller)."""
        return {
            "annual_financials": self.to_annual_inputs(),
            "balance_sheet": self.to_balance_sheet_input(),
            "prior_balance_sheet": self.to_prior_balance_sheet_input(),
        }

    def summary_for_prompt(self) -> Dict[str, Any]:
        return {
            "years": self.years,
            "completeness_score": self.completeness_score,
            "missing_fields": self.missing_fields,
            "failed_identity_checks": self.failed_identity_checks,
            "cross_filing_differences": self.cross_filing_differences,
            "warnings": self.warnings,
        }


# ==============================================================================
# 3. Row Matching Helpers
# ==============================================================================
def _find_row(
    grid: StatementGrid,
    patterns: List[str],
    year: Optional[int] = None,
    start: Optional[int] = None,
    end: Optional[int] = None,
) -> Optional[StatementRow]:
    """First value row (pattern order, then statement order) whose label matches and has a value for `year`."""
    for pat in patterns:
        rx = re.compile(pat)
        for row in grid.value_rows:
            if start is not None and row.index <= start:
                continue
            if end is not None and row.index >= end:
                continue
            if not rx.search(row.norm_label):
                continue
            if year is not None and year not in row.values:
                continue
            return row
    return None


def _scaled(grid: StatementGrid, value: float) -> float:
    return round(value / grid.scale_divisor, 4) if grid.scale_divisor else value


def _normalize_shares(value: float) -> float:
    """Normalizes a share count to millions (counts reported in thousands are divided by 1,000)."""
    return round(value / 1000.0, 4) if abs(value) > SHARE_THOUSANDS_THRESHOLD else value


def _find_diluted_shares_row(grid: StatementGrid, year: int) -> Optional[StatementRow]:
    for row in grid.value_rows:
        if year not in row.values:
            continue
        label = row.norm_label
        if re.search(r"weighted.average.*diluted|diluted.*weighted|^diluted .*shares", label):
            return row
        if re.fullmatch(r"diluted( shares)?", label) and row.section and DILUTED_SHARE_SECTION.search(row.section):
            return row
    return None


def _pct_diff(a: float, b: float) -> float:
    base = max(abs(a), abs(b), 1e-9)
    return abs(a - b) / base * 100.0


# ==============================================================================
# 4. Per-Statement Field Extraction
# ==============================================================================
def _extract_income_fields(grid: StatementGrid, year: int) -> Dict[str, Tuple[float, StatementRow]]:
    found: Dict[str, Tuple[float, StatementRow]] = {}
    for fname, patterns in INCOME_STATEMENT_SYNONYMS.items():
        row = _find_row(grid, patterns, year)
        if row is not None:
            found[fname] = (_scaled(grid, row.values[year]), row)
    shares_row = _find_diluted_shares_row(grid, year)
    if shares_row is not None:
        found["diluted_weighted_shares"] = (_normalize_shares(shares_row.values[year]), shares_row)
    return found


def _extract_cash_flow_fields(grid: StatementGrid, year: int) -> Dict[str, Tuple[float, StatementRow]]:
    found: Dict[str, Tuple[float, StatementRow]] = {}
    ocf_idx = grid.index_of(CASH_FLOW_SYNONYMS["operating_cash_flow"][0])
    for fname, patterns in CASH_FLOW_SYNONYMS.items():
        # Operating adjustments must come from the operating section (above the OCF subtotal).
        end = ocf_idx if fname in ("depreciation_amortization", "stock_based_compensation", "deferred_income_taxes") else None
        start = ocf_idx if fname == "capital_expenditures" else None
        row = _find_row(grid, patterns, year, start=start, end=end)
        if row is not None:
            value = _scaled(grid, row.values[year])
            if fname == "capital_expenditures":
                value = abs(value)
            found[fname] = (value, row)
    return found


def _extract_balance_fields(grid: StatementGrid, year: int) -> Dict[str, Tuple[float, Any]]:
    found: Dict[str, Tuple[float, Any]] = {}
    ta_idx = grid.index_of(BALANCE_SHEET_SYNONYMS["total_assets"][0])
    tcl_idx = grid.index_of(BALANCE_SHEET_SYNONYMS["current_liabilities"][0])
    tl_idx = grid.index_of(BALANCE_SHEET_SYNONYMS["total_liabilities"][0])

    for fname, patterns in BALANCE_SHEET_SYNONYMS.items():
        start = end = None
        if fname in ("cash_and_equivalents", "accounts_receivable", "inventories"):
            end = ta_idx
        elif fname in ("accounts_payable",):
            start = ta_idx
        row = _find_row(grid, patterns, year, start=start, end=end)
        if row is not None:
            found[fname] = (_scaled(grid, row.values[year]), row)

    # Liquid marketable securities: sum every current / non-current securities row in the asset section.
    sec_rows = []
    for row in grid.value_rows:
        if ta_idx is not None and row.index >= ta_idx:
            break
        if year in row.values and any(re.search(p, row.norm_label) for p in SECURITIES_PATTERNS):
            sec_rows.append(row)
    if sec_rows:
        found["marketable_securities"] = (
            round(sum(_scaled(grid, r.values[year]) for r in sec_rows), 4),
            sec_rows,
        )

    # Debt: classify by position relative to the "Total current liabilities" subtotal.
    liab_start = ta_idx if ta_idx is not None else -1
    liab_end = tl_idx
    if liab_end is None:
        # No "Total liabilities" subtotal: liabilities end at the equity / commitments section,
        # never at a combined "Liabilities and stockholders' equity" caption.
        search_from = tcl_idx if tcl_idx is not None else liab_start
        liab_end = next(
            (
                r.index for r in grid.rows
                if r.index > search_from
                and "liabilit" not in r.norm_label
                and re.search(r"equity|deficit|commitments and contingencies", r.norm_label)
            ),
            None,
        )
    st_rows, lt_rows = [], []
    for row in grid.value_rows:
        if row.index <= liab_start or (liab_end is not None and row.index >= liab_end):
            continue
        if year not in row.values:
            continue
        if not DEBT_ROW_PATTERN.search(row.norm_label) or DEBT_ROW_EXCLUDE.search(row.norm_label):
            continue
        if tcl_idx is not None and row.index < tcl_idx:
            st_rows.append(row)
        else:
            lt_rows.append(row)
    found["short_term_debt"] = (round(sum(abs(_scaled(grid, r.values[year])) for r in st_rows), 4), st_rows)
    found["long_term_debt"] = (round(sum(abs(_scaled(grid, r.values[year])) for r in lt_rows), 4), lt_rows)

    # Period-end shares outstanding from the common stock caption ("3,751 and 3,216 shares issued and outstanding").
    for row in grid.rows:
        m = SHARES_OUTSTANDING_CAPTION.search(row.norm_label)
        if m and len(grid.periods) >= 2:
            pair = [float(m.group(1).replace(",", "")), float(m.group(2).replace(",", ""))]
            share_map = dict(zip(grid.periods[:2], pair))
            if year in share_map:
                found["period_end_shares_outstanding"] = (_normalize_shares(share_map[year]), row)
            break
    return found


# ==============================================================================
# 5. Public Entry Point
# ==============================================================================
def _record(
    target: Dict[int, Dict[str, float]],
    provenance: Dict[str, Dict[str, Any]],
    differences: List[Dict[str, Any]],
    year: int,
    fname: str,
    value: float,
    source: Any,
    grid: StatementGrid,
    statement: str,
):
    """Stores a value under Latest Filing Precedence; logs older-filing disagreements."""
    bucket = target.setdefault(year, {})
    key = f"FY{year}.{fname}"
    if fname in bucket:
        prior = provenance.get(key, {})
        if _pct_diff(bucket[fname], value) > CROSS_FILING_TOLERANCE_PCT:
            differences.append({
                "statement": statement,
                "field": fname,
                "fiscal_year": year,
                "latest_filing_year": prior.get("filing_year"),
                "latest_value": bucket[fname],
                "older_filing_year": grid.filing_year,
                "older_value": value,
                "difference_pct": round(_pct_diff(bucket[fname], value), 2),
            })
        return
    bucket[fname] = value
    rows = source if isinstance(source, list) else [source]
    provenance[key] = {
        "statement": statement,
        "labels": [r.label for r in rows if r is not None],
        "chunk_id": grid.chunk_id,
        "filing_year": grid.filing_year,
    }


def extract_audit_inputs(
    statements: Dict[str, List[StatementGrid]],
    fiscal_year: int,
    target_years: Optional[List[int]] = None,
) -> ExtractionResult:
    """
    Extracts audit inputs from parsed statement grids.

    Args:
        statements: {"income_statement": [...], "balance_sheet": [...], "cash_flow": [...]},
                    each list ordered newest filing first.
        fiscal_year: Anchor (latest) fiscal year of the audit.
        target_years: Desired contiguous window (defaults to the three years ending at fiscal_year).
    """
    target_years = sorted(target_years or [fiscal_year - 2, fiscal_year - 1, fiscal_year])
    result = ExtractionResult(fiscal_year=fiscal_year)
    annual: Dict[int, Dict[str, float]] = {}
    balance: Dict[int, Dict[str, float]] = {}

    candidate_years = sorted(set(target_years) | {fiscal_year - 1})
    extractors = {
        "income_statement": (_extract_income_fields, annual),
        "cash_flow": (_extract_cash_flow_fields, annual),
        "balance_sheet": (_extract_balance_fields, balance),
    }
    for statement, (extract_fn, target) in extractors.items():
        for grid in statements.get(statement, []):
            for year in candidate_years:
                if year not in grid.periods:
                    continue
                for fname, (value, source) in extract_fn(grid, year).items():
                    _record(target, result.field_provenance, result.cross_filing_differences,
                            year, fname, value, source, grid, statement)

    # Derived per-year fields
    for year, vals in annual.items():
        ni = vals.get("net_income_attributable", vals.get("net_income_total"))
        if ni is not None:
            vals["net_income"] = ni
        if vals.get("net_income_total") is None and vals.get("net_income_cf") is not None:
            vals["net_income_total"] = vals["net_income_cf"]
        # Filers present the tax line either expense-positive or benefit-positive regardless of its label
        # ("Provision for income taxes" shown as (19,087)). Orient it with pretax - tax ~= net income.
        tax, pretax, ni_total = vals.get("income_tax_expense"), vals.get("pretax_income"), vals.get("net_income_total")
        if tax and pretax is not None and ni_total is not None:
            if abs(pretax + tax - ni_total) < abs(pretax - tax - ni_total):
                vals["income_tax_expense"] = -tax
                result.warnings.append(
                    f"FY{year}: income tax line is presented benefit-positive; normalized to expense-positive "
                    f"({-tax:,.1f})."
                )

    # Share counts on the balance sheet record: weighted diluted (income statement) + period-end (caption).
    for year, bs in balance.items():
        diluted = annual.get(year, {}).get("diluted_weighted_shares")
        if diluted is not None:
            bs["weighted_diluted_shares"] = diluted
        mezz = bs.get("redeemable_noncontrolling_interests") or 0.0
        if mezz:
            bs["noncontrolling_interests"] = round((bs.get("noncontrolling_interests") or 0.0) + mezz, 4)

    result.annual = annual
    result.balance = balance

    # Contiguous window ending at fiscal_year where all required annual fields exist.
    years: List[int] = []
    for yr in sorted(target_years, reverse=True):
        vals = annual.get(yr, {})
        if all(vals.get(f) is not None for f in REQUIRED_ANNUAL_FIELDS):
            years.insert(0, yr)
        else:
            missing = [f for f in REQUIRED_ANNUAL_FIELDS if vals.get(f) is None]
            if yr == fiscal_year or not years:
                result.missing_fields.extend(f"FY{yr}.{m}" for m in missing)
            else:
                result.warnings.append(f"FY{yr} excluded from the audit window (missing: {', '.join(missing)}).")
            break
    result.years = years

    latest_bs = balance.get(fiscal_year, {})
    for f in REQUIRED_BALANCE_FIELDS:
        if latest_bs.get(f) is None:
            result.missing_fields.append(f"FY{fiscal_year}.balance_sheet.{f}")
    if all(latest_bs.get(f) is None for f in SHARE_COUNT_FIELDS):
        result.missing_fields.append(f"FY{fiscal_year}.share_count (weighted diluted or period-end)")
    elif latest_bs.get("weighted_diluted_shares") is None:
        result.warnings.append(f"FY{fiscal_year}: weighted diluted shares not found; period-end count only.")
    for f in ["gross_profit", "pretax_income", "income_tax_expense", "depreciation_amortization",
              "stock_based_compensation"]:
        latest = annual.get(fiscal_year, {})
        if latest.get(f) is None and not (f == "gross_profit" and latest.get("cost_of_revenue") is not None):
            result.missing_fields.append(f"FY{fiscal_year}.{f}")
    for f in ["current_assets", "current_liabilities"]:
        if latest_bs.get(f) is None:
            result.missing_fields.append(f"FY{fiscal_year}.balance_sheet.{f}")
    if latest_bs and latest_bs.get("marketable_securities") is None:
        result.warnings.append(
            "No short-term investment / marketable securities row detected; liquid securities treated as 0."
        )

    result.identity_checks = _run_identity_checks(annual, balance, years, fiscal_year)

    # Clean presentation tables and citations from the newest filing of each statement.
    for statement, grids in statements.items():
        if grids:
            result.clean_tables[statement] = render_clean_statement(grids[0])
        for g in grids:
            if g.chunk_id:
                result.citations.append({
                    "chunk_id": g.chunk_id,
                    "item": "Item 8",
                    "fiscal_year": g.filing_year,
                    "sub_section": g.title,
                    "statement_type": statement,
                })
    return result


def _identity(name: str, lhs: float, rhs: float, critical: bool, detail: str) -> Dict[str, Any]:
    diff = _pct_diff(lhs, rhs)
    return {
        "name": name,
        "status": "passed" if diff <= IDENTITY_TOLERANCE_PCT else "failed",
        "critical": critical,
        "lhs": round(lhs, 2),
        "rhs": round(rhs, 2),
        "difference_pct": round(diff, 3),
        "detail": detail,
    }


def _run_identity_checks(
    annual: Dict[int, Dict[str, float]],
    balance: Dict[int, Dict[str, float]],
    years: List[int],
    fiscal_year: int,
) -> List[Dict[str, Any]]:
    checks: List[Dict[str, Any]] = []
    bs = balance.get(fiscal_year, {})
    ta = bs.get("total_assets")
    if ta is not None and bs.get("total_liabilities_and_equity") is not None:
        checks.append(_identity(
            f"FY{fiscal_year} balance sheet: total assets = total liabilities and equity",
            ta, bs["total_liabilities_and_equity"], True,
            "Validates that the balance sheet grid was parsed with correct period alignment.",
        ))
    if ta is not None and bs.get("total_liabilities") is not None and bs.get("stockholders_equity") is not None:
        rhs = bs["total_liabilities"] + bs["stockholders_equity"] + (bs.get("noncontrolling_interests") or 0.0)
        checks.append(_identity(
            f"FY{fiscal_year} balance sheet: assets = liabilities + mezzanine + equity (incl. NCI)",
            ta, rhs, False,
            "Validates the stockholders' equity and liabilities rows used for ROE / invested capital.",
        ))
    for yr in years:
        vals = annual.get(yr, {})
        if None not in (vals.get("revenue"), vals.get("cost_of_revenue"), vals.get("gross_profit")):
            checks.append(_identity(
                f"FY{yr} gross profit = revenue - cost of revenue",
                vals["gross_profit"], vals["revenue"] - vals["cost_of_revenue"], False,
                "Validates revenue and cost of revenue row selection.",
            ))
        if vals.get("net_income_total") is not None and vals.get("net_income_cf") is not None:
            checks.append(_identity(
                f"FY{yr} net income: income statement = cash flow statement",
                vals["net_income_total"], vals["net_income_cf"], False,
                "Validates that the income statement and cash flow statement describe the same periods.",
            ))
    return checks


def build_statement_grids(tables: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[StatementGrid]]:
    """Parses retrieved table payloads ({statement: [TableChunkResult dicts]}) into StatementGrids."""
    grids: Dict[str, List[StatementGrid]] = {}
    for statement, payloads in tables.items():
        parsed = []
        for p in payloads:
            md = p.get("table_markdown") or ""
            grid = parse_statement_table(md, chunk_id=p.get("chunk_id"), filing_year=p.get("fiscal_year"))
            if grid.periods and grid.value_rows:
                parsed.append(grid)
            else:
                logger.warning(
                    f"build_statement_grids: could not detect periods in {statement} chunk {p.get('chunk_id')}"
                )
        grids[statement] = parsed
    return grids
