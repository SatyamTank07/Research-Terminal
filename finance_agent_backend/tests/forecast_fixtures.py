"""Synthetic auditor outputs and MD&A chunks shared by the forecasting tests."""

from typing import Any, Dict, Iterable, List, Optional

from app.agents.specialized.financial_auditor.state_financial_auditor import FinancialAuditOutput
from app.agents.specialized.forecasting_analyst.assemble import (
    assemble_forecast,
    fallback_rationales,
    mdna_fade_reason,
)
from app.agents.specialized.forecasting_analyst.state_forecasting_analyst import ForecastOutput
from app.agents.tools.financial_math_tools import (
    AnnualFinancialInput,
    BalanceSheetInput,
    NormalizationAdjustment,
    audit_financial_metrics,
)
from app.agents.tools.forecast_baseline import build_forecast_baseline
from app.agents.tools.forecast_drivers import DEFAULT_TERMINAL_GROWTH, historical_fade_drivers
from app.agents.tools.forecast_model import build_forecast
from app.agents.tools.mdna_guidance import MdnaAssessment
from app.agents.shared.filing.narrative_search import NarrativeChunkResult

AAPL_ROWS: List[Dict[str, Any]] = [
    dict(fiscal_year=2023, revenue=383285.0, gross_profit=169148.0, operating_income=114301.0,
         pretax_income=113736.0, income_tax_expense=16741.0, net_income=96995.0,
         operating_cash_flow=110543.0, capital_expenditures=10959.0),
    dict(fiscal_year=2024, revenue=391035.0, gross_profit=180683.0, operating_income=123216.0,
         pretax_income=123485.0, income_tax_expense=29749.0, net_income=93736.0,
         operating_cash_flow=118254.0, capital_expenditures=9447.0),
    dict(fiscal_year=2025, revenue=416161.0, gross_profit=195201.0, operating_income=133050.0,
         pretax_income=132717.0, income_tax_expense=20707.0, net_income=112010.0,
         operating_cash_flow=111482.0, capital_expenditures=12715.0, depreciation_amortization=11445.0),
]
AAPL_BALANCE_SHEET: Dict[str, Any] = dict(
    fiscal_year=2025, cash_and_equivalents=35934.0, marketable_securities=96486.0,
    short_term_debt=10912.0, long_term_debt=87745.0, stockholders_equity=53736.0,
    weighted_diluted_shares=15004.7, current_assets=154388.0, current_liabilities=145308.0,
)


def make_audit(
    ticker: str,
    rows: List[Dict[str, Any]],
    balance_sheet: Dict[str, Any],
    normalization: Optional[List[NormalizationAdjustment]] = None,
) -> FinancialAuditOutput:
    res = audit_financial_metrics(
        [AnnualFinancialInput(**r) for r in rows],
        BalanceSheetInput(**balance_sheet),
        normalization_adjustments=normalization,
    )
    res.update(
        ticker=ticker,
        fiscal_year=max(r["fiscal_year"] for r in rows),
        auditor_summary=f"Synthetic audit for {ticker}.",
        data_quality={"extraction_mode": "deterministic"},
    )
    return FinancialAuditOutput.model_validate(res)


def make_aapl_audit(
    include_depreciation: bool = True,
    working_capital: bool = False,
    sbc: Optional[float] = None,
    years: Iterable[int] = (2023, 2024, 2025),
    normalization: Optional[List[NormalizationAdjustment]] = None,
) -> FinancialAuditOutput:
    wanted = set(years)
    rows = [dict(r) for r in AAPL_ROWS if r["fiscal_year"] in wanted]
    latest = rows[-1]
    if not include_depreciation:
        latest.pop("depreciation_amortization", None)
    if working_capital:
        # DSO 36.5, DIO 10.0, DPO 109.5 days on FY2025 revenue / cost of revenue.
        latest.update(accounts_receivable=41616.1, inventories=6053.7, accounts_payable=66288.0)
    if sbc is not None:
        latest["stock_based_compensation"] = sbc
    return make_audit("AAPL", rows, AAPL_BALANCE_SHEET, normalization)


def make_low_margin_audit() -> FinancialAuditOutput:
    """Distributor-style company: 0.7% operating margin, 0.8% effective tax rate."""
    rows = [
        dict(fiscal_year=2023, revenue=100000.0, gross_profit=5500.0, operating_income=700.0,
             pretax_income=700.0, income_tax_expense=5.6, net_income=694.4,
             operating_cash_flow=900.0, capital_expenditures=300.0),
        dict(fiscal_year=2024, revenue=105000.0, gross_profit=5775.0, operating_income=735.0,
             pretax_income=735.0, income_tax_expense=5.88, net_income=729.12,
             operating_cash_flow=950.0, capital_expenditures=320.0),
        dict(fiscal_year=2025, revenue=110000.0, gross_profit=6050.0, operating_income=770.0,
             pretax_income=770.0, income_tax_expense=6.16, net_income=763.84,
             operating_cash_flow=1000.0, capital_expenditures=340.0),
    ]
    balance = dict(fiscal_year=2025, cash_and_equivalents=2000.0, marketable_securities=0.0,
                   short_term_debt=500.0, long_term_debt=3000.0, stockholders_equity=6000.0,
                   weighted_diluted_shares=200.0, current_assets=30000.0, current_liabilities=28000.0)
    return make_audit("LOWM", rows, balance)


def make_loss_audit() -> FinancialAuditOutput:
    """Loss-making grower: negative EBIT and pre-tax income with small tax benefits."""
    rows = [
        dict(fiscal_year=2023, revenue=1000.0, gross_profit=400.0, operating_income=-100.0,
             pretax_income=-120.0, income_tax_expense=-10.0, net_income=-110.0,
             operating_cash_flow=20.0, capital_expenditures=50.0),
        dict(fiscal_year=2024, revenue=1100.0, gross_profit=450.0, operating_income=-80.0,
             pretax_income=-100.0, income_tax_expense=-8.0, net_income=-92.0,
             operating_cash_flow=30.0, capital_expenditures=55.0),
        dict(fiscal_year=2025, revenue=1200.0, gross_profit=500.0, operating_income=-60.0,
             pretax_income=-70.0, income_tax_expense=-5.0, net_income=-65.0,
             operating_cash_flow=40.0, capital_expenditures=60.0),
    ]
    balance = dict(fiscal_year=2025, cash_and_equivalents=500.0, marketable_securities=0.0,
                   short_term_debt=0.0, long_term_debt=200.0, stockholders_equity=800.0,
                   weighted_diluted_shares=100.0, current_assets=700.0, current_liabilities=300.0)
    return make_audit("LOSS", rows, balance)


def make_heavy_capex_audit() -> FinancialAuditOutput:
    """Investment step-up: CapEx 9% -> 11% -> 18% of revenue against 9% D&A; margins 6% -> 10% -> 11%."""
    rows = [
        dict(fiscal_year=2023, revenue=500000.0, gross_profit=230000.0, operating_income=30000.0,
             pretax_income=31000.0, income_tax_expense=5000.0, net_income=26000.0,
             operating_cash_flow=80000.0, capital_expenditures=45000.0),
        dict(fiscal_year=2024, revenue=560000.0, gross_profit=270000.0, operating_income=56000.0,
             pretax_income=58000.0, income_tax_expense=9000.0, net_income=49000.0,
             operating_cash_flow=110000.0, capital_expenditures=61600.0),
        dict(fiscal_year=2025, revenue=640000.0, gross_profit=320000.0, operating_income=70400.0,
             pretax_income=74000.0, income_tax_expense=12000.0, net_income=62000.0,
             operating_cash_flow=130000.0, capital_expenditures=115200.0, depreciation_amortization=57600.0),
    ]
    balance = dict(fiscal_year=2025, cash_and_equivalents=90000.0, marketable_securities=40000.0,
                   short_term_debt=10000.0, long_term_debt=60000.0, stockholders_equity=380000.0,
                   weighted_diluted_shares=10800.0, current_assets=200000.0, current_liabilities=190000.0)
    return make_audit("HCX", rows, balance)


def make_shrinking_audit() -> FinancialAuditOutput:
    """Declining revenue (-8% a year) with a thin, falling margin and net debt."""
    rows = [
        dict(fiscal_year=2023, revenue=10000.0, gross_profit=3000.0, operating_income=800.0,
             pretax_income=700.0, income_tax_expense=150.0, net_income=550.0,
             operating_cash_flow=900.0, capital_expenditures=400.0),
        dict(fiscal_year=2024, revenue=9200.0, gross_profit=2700.0, operating_income=600.0,
             pretax_income=500.0, income_tax_expense=105.0, net_income=395.0,
             operating_cash_flow=750.0, capital_expenditures=350.0),
        dict(fiscal_year=2025, revenue=8464.0, gross_profit=2400.0, operating_income=420.0,
             pretax_income=320.0, income_tax_expense=67.0, net_income=253.0,
             operating_cash_flow=600.0, capital_expenditures=300.0, depreciation_amortization=380.0),
    ]
    balance = dict(fiscal_year=2025, cash_and_equivalents=300.0, marketable_securities=0.0,
                   short_term_debt=500.0, long_term_debt=2500.0, stockholders_equity=2000.0,
                   weighted_diluted_shares=100.0, current_assets=2500.0, current_liabilities=2000.0)
    return make_audit("SHRK", rows, balance)


GUIDANCE_CHUNK = (
    "We expect capital expenditures of approximately $14 billion in fiscal 2026, primarily for data centers. "
    "We anticipate Services revenue will continue to grow faster than Products revenue."
)
HISTORY_CHUNK = "Net sales increased 6% in 2025 driven by iPhone and Services. Gross margin was 46.9% of net sales."
# Verbatim from AMZN's FY2025 10-K "Guidance" section: safe-harbor hedging, then single-quarter guidance and an
# FX-impact note. None of it is annual operating guidance.
AMZN_SAFE_HARBOR_SENTENCE = (
    "Our results are inherently unpredictable and may be materially affected by many factors, such as fluctuations "
    "in foreign exchange rates and energy prices, changes in global economic and geopolitical conditions, tariff and "
    "trade policies, resource and supply volatility, including for memory chips, and customer demand and spending "
    "(including the impact of recessionary fears), inflation, interest rates, regional labor market constraints, "
    "world events, the rate of growth of the internet, online commerce, cloud services, and new and emerging "
    "technologies, as well as those outlined in Item 1A of Part I, “Risk Factors.”"
)
AMZN_FX_SENTENCE = "This guidance anticipates a favorable impact of approximately 180 basis points from foreign exchange rates."
AMZN_GUIDANCE_CHUNK = (
    "Guidance\n\nWe provided guidance on February 5, 2026, in our earnings release furnished on Form 8-K as set forth "
    "below. These forward-looking statements reflect Amazon.com’s expectations as of February 5, 2026, and are "
    "subject to substantial uncertainty. " + AMZN_SAFE_HARBOR_SENTENCE + "\n\nFirst Quarter 2026 Guidance\n\n"
    "• Net sales are expected to be between $173.5 billion and $178.5 billion, or to grow between 11% and 15% "
    "compared with first quarter 2025. " + AMZN_FX_SENTENCE
)
MARGIN_OUTLOOK_CHUNK = (
    "We expect operating margins to decline in 2026 as depreciation from our data center build-out rises. "
    "We anticipate pricing pressure in our retail segment."
)
BOILERPLATE_CHUNK = (
    "This report contains forward-looking statements within the meaning of the Private Securities Litigation "
    "Reform Act of 1995. Actual results could differ materially from those anticipated."
)


def make_chunk(chunk_id: str, content: str, item: str = "Item 7") -> NarrativeChunkResult:
    return NarrativeChunkResult(
        chunk_id=chunk_id, document_id="doc-1", ticker="AAPL", fiscal_year=2025, item=item,
        breadcrumb=f"{item} > MD&A", content=content, chunk_index=0,
    )


def make_forecast_output(
    ticker: str = "AAPL", fiscal_year: int = 2025, audit: Optional[FinancialAuditOutput] = None
) -> ForecastOutput:
    """Deterministic forecast (historical fade) that never goes through ForecastingAnalystAgent.forecast."""
    baseline = build_forecast_baseline(audit or make_aapl_audit())
    drivers = historical_fade_drivers(baseline, DEFAULT_TERMINAL_GROWTH)
    mdna = MdnaAssessment()
    return assemble_forecast(
        ticker=ticker, fiscal_year=fiscal_year, baseline=baseline,
        build=build_forecast(baseline, drivers, DEFAULT_TERMINAL_GROWTH),
        guidance_source="historical_fade", mdna=mdna, terminal_growth=DEFAULT_TERMINAL_GROWTH,
        rationales=fallback_rationales(baseline, drivers, DEFAULT_TERMINAL_GROWTH, mdna_fade_reason(mdna)),
    )
