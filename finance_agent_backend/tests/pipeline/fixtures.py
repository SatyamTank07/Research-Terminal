"""Shared builders for pipeline tests."""

from datetime import date
from typing import Any

from app.agents.pipeline.contract import RunContext

RUN_DATE = date(2026, 10, 28)


def make_run_context(**changes: Any) -> RunContext:
    fields = dict(
        ticker="AAPL",
        company_name="Apple Inc.",
        fiscal_year=2025,
        available_fiscal_years=(2025, 2024),
        document_id="doc-aapl-2025",
        user_query="Write a full report on AAPL",
        route="full_10k_report",
        year_substituted=False,
        valuation_overrides={},
        terminal_growth_rate=0.025,
        valuation_date=RUN_DATE,
        callbacks=None,
    )
    fields.update(changes)
    return RunContext(**fields)
