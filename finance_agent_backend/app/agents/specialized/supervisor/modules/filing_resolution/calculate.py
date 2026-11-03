"""Fiscal-year anchor precedence: explicit year, session year on a follow-up, latest audited year."""

import logging
import re
from datetime import date
from typing import Any, Dict, List, Optional

from app.agents.specialized.supervisor.modules.filing_resolution.schema import FilingChoice, RequestedYear

logger = logging.getLogger("finance_agent.agents.supervisor")

_EXPLICIT_YEAR = re.compile(r"\b(?:fy|fiscal(?:\s+year)?)\s*'?(\d{4}|\d{2})\b", re.IGNORECASE)
_BARE_YEAR = re.compile(r"\b((?:19|20)\d{2})\b")
_COMPANY_STOPWORDS = {"inc", "corp", "corporation", "co", "company", "ltd", "plc", "holdings", "group", "the", "com"}


def extract_fiscal_year(text: str) -> Optional[int]:
    """Fiscal year the user names: 'FY2023' / 'fiscal 2023' / "FY'23" first, else a bare past year.

    A bare year later than the current calendar year ("forecast to 2030") is a horizon, not a filing.
    """
    explicit = _EXPLICIT_YEAR.search(text or "")
    if explicit:
        raw = explicit.group(1)
        return int(raw) if len(raw) == 4 else 2000 + int(raw)
    for match in _BARE_YEAR.finditer(text or ""):
        year = int(match.group(1))
        if year <= date.today().year:
            return year
    return None


def names_entity(text: str, ticker: Optional[str], company_name: Optional[str]) -> bool:
    """True when the message names the company (ticker or first distinctive word of its name)."""
    words = set(re.findall(r"[a-z0-9]+", (text or "").lower()))
    if ticker and ticker.lower() in words:
        return True
    name_words = [w for w in re.findall(r"[a-z0-9]+", (company_name or "").lower()) if w not in _COMPANY_STOPWORDS]
    return bool(name_words) and name_words[0] in words


def resolve_requested_year(
    explicit_year: Optional[int],
    extracted_year: Optional[int],
    user_query: str,
    ticker: str,
    session: Dict[str, Any],
) -> RequestedYear:
    """Explicit argument, then the judge's extraction; a follow-up about the active company that names no year
    and no company keeps the session's fiscal year instead of jumping to the latest filing."""
    year = explicit_year if explicit_year is not None else extracted_year
    session_ticker = (session.get("active_ticker") or "").upper() or None
    if (
        year is None
        and extract_fiscal_year(user_query) is None
        and ticker.upper() == session_ticker
        and session.get("active_fiscal_year")
        and not names_entity(user_query, session_ticker, session.get("active_company"))
    ):
        year = int(session["active_fiscal_year"])
        return RequestedYear(year=year, note=f"Continuing with the FY{year} filing from earlier in the conversation.")
    return RequestedYear(year=year, note=None)


def choose_filing_year(
    ticker: str, requested_year: Optional[int], user_query: Optional[str], available_years: List[int]
) -> FilingChoice:
    """The requested year when ingested, else the latest audited year (flagged as substituted); no request (or a
    year 0) falls back to a year named in the query, then to the latest."""
    year_requested = requested_year
    if not year_requested and user_query:
        year_requested = extract_fiscal_year(user_query)
    if not year_requested:
        return FilingChoice(year_requested=year_requested, fiscal_year=available_years[0], year_substituted=False)
    if year_requested in available_years:
        return FilingChoice(year_requested=year_requested, fiscal_year=year_requested, year_substituted=False)
    target_year = available_years[0]
    logger.warning(
        f"Requested filing FY{year_requested} for '{(ticker or '').strip().upper()}' not found in database. "
        f"Substituted with latest audited filing FY{target_year} (year_substituted=True)."
    )
    return FilingChoice(year_requested=year_requested, fiscal_year=target_year, year_substituted=True)


def explain_missing_filing(reason: str) -> str:
    """Reply when the catalog holds no usable 10-K; `reason` already lists the ingested tickers."""
    return (
        f"I am your Senior Equity Research Director. {reason} "
        "Please name an ingested company or upload the requested 10-K filing."
    )
