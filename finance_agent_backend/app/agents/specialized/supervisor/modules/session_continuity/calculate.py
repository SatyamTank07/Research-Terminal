"""The session record every turn writes back; the one place that decides which keys persist."""

from typing import Any, Dict, Optional

from app.agents.specialized.supervisor.modules.session_continuity.schema import SessionFiling
from app.agents.specialized.supervisor.modules.valuation_parameters import SESSION_KEY


def remember_turn(
    session: Dict[str, Any],
    overrides: Dict[str, Any],
    filing: Optional[SessionFiling] = None,
    query_type: Optional[str] = None,
) -> Dict[str, Any]:
    """A research turn starts a fresh record for its filing and report type; a chat turn keeps the active record.

    Both remember the valuation overrides (or forget them when there are none left).
    """
    if filing is None:
        remembered = dict(session)
    else:
        remembered = {
            "active_ticker": filing.ticker,
            "active_company": filing.company_name,
            "active_fiscal_year": filing.fiscal_year,
            "available_fiscal_years": filing.available_fiscal_years,
            "last_query_type": query_type,
        }
    if overrides:
        remembered[SESSION_KEY] = overrides
    else:
        remembered.pop(SESSION_KEY, None)
    return remembered
