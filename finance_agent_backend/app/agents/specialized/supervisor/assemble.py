"""Builds what the Supervisor hands on: the routing plan, and for a research turn the run's shared facts."""

import logging
from typing import Any, Dict, List, Optional, Sequence

from app.agents.pipeline.contract import Conversation, RunContext
from app.agents.specialized.supervisor.context import FilingRecord
from app.agents.specialized.supervisor.modules.filing_resolution import FilingChoice
from app.agents.specialized.supervisor.modules.query_intent import CONVERSATIONAL, QueryIntent
from app.agents.specialized.supervisor.modules.valuation_parameters import ValuationParameters, resolve_terminal_growth
from app.agents.specialized.supervisor.state_supervisor import RoutingPlan, TurnResolution

logger = logging.getLogger("finance_agent.agents.supervisor")


def assemble_research_plan(
    intent: QueryIntent,
    filing: FilingRecord,
    choice: FilingChoice,
    available_years: List[int],
    parse_notes: Sequence[str],
    filing_note: Optional[str],
    parameters: ValuationParameters,
    session: Dict[str, Any],
) -> RoutingPlan:
    """Plan for a research route; notes read: rejected inputs, the carried filing year, carried overrides."""
    notes = [*parse_notes, *(note for note in (filing_note, parameters.carry_note) if note)]
    return RoutingPlan(
        ticker=filing.ticker,
        company_name=filing.company_name,
        fiscal_year=filing.fiscal_year,
        available_fiscal_years=available_years,
        year_requested=choice.year_requested,
        year_substituted=choice.year_substituted,
        document_id=filing.document_id,
        query_type=intent.query_type,
        routing_provenance=intent.provenance,
        conversational_response=None,
        valuation_overrides=parameters.overrides,
        parameter_notes=notes,
        updated_session_state=session,
    )


def assemble_conversational_plan(
    session: Dict[str, Any],
    reply: str,
    provenance: str,
    parse_notes: Sequence[str],
    ticker: Optional[str] = None,
) -> RoutingPlan:
    """Plan for a direct reply; it keeps the remembered research context and runs no agent."""
    active_ticker = ticker or session.get("active_ticker") or ""
    return RoutingPlan(
        ticker=active_ticker,
        company_name=session.get("active_company") or active_ticker or None,
        fiscal_year=session.get("active_fiscal_year") or 0,
        available_fiscal_years=session.get("available_fiscal_years") or [],
        document_id=None,
        query_type=CONVERSATIONAL,
        routing_provenance=provenance,
        conversational_response=reply,
        parameter_notes=list(parse_notes),
        updated_session_state=session,
    )


def assemble_turn(conversation: Conversation, plan: RoutingPlan) -> TurnResolution:
    """The turn's route and session update; a research route also fixes the facts every agent will share."""
    logger.info(
        f"[supervisor] Resolved {plan.ticker} FY{plan.fiscal_year} "
        f"route={plan.query_type} (year_substituted={plan.year_substituted})"
    )
    run_context = None
    if plan.query_type != CONVERSATIONAL:
        run_context = RunContext(
            ticker=plan.ticker,
            company_name=plan.company_name or plan.ticker,
            fiscal_year=plan.fiscal_year,
            available_fiscal_years=tuple(plan.available_fiscal_years),
            document_id=plan.document_id,
            user_query=conversation.user_query,
            route=plan.query_type,
            year_substituted=plan.year_substituted,
            valuation_overrides=dict(plan.valuation_overrides),
            terminal_growth_rate=resolve_terminal_growth(plan.valuation_overrides),
            valuation_date=conversation.run_date,
            callbacks=conversation.callbacks,
        )
    return TurnResolution(
        route=plan.query_type,
        plan=plan,
        run_context=run_context,
        conversational_response=plan.conversational_response,
        session_update=plan.updated_session_state,
        parameter_notes=tuple(plan.parameter_notes),
    )
