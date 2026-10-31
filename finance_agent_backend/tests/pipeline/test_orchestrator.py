import asyncio
import unittest
from unittest.mock import patch

from app.agents import MultiAgentOrchestrator
from app.agents.specialized.financial_auditor import FinancialAuditorAgent
from app.agents.specialized.supervisor import RoutingPlan, TurnResolution
from tests.pipeline.fixtures import make_run_context


def _audit_turn(**ctx_changes) -> TurnResolution:
    ctx = make_run_context(route="financial_audit_only", **ctx_changes)
    plan = RoutingPlan(ticker="AAPL", fiscal_year=2025, query_type="financial_audit_only",
                       available_fiscal_years=[2025, 2024], active_agents=[])
    return TurnResolution(route="financial_audit_only", plan=plan, run_context=ctx, conversational_response=None,
                          session_update={"active_ticker": "AAPL"}, parameter_notes=())


async def _events(**kwargs):
    return [event async for event in MultiAgentOrchestrator().astream_run(user_query="audit AAPL", **kwargs)]


class TestOrchestratorStreaming(unittest.TestCase):
    def test_every_research_agent_failing_ends_with_an_error_event(self):
        with patch("app.agents.pipeline.node.resolve_turn", return_value=_audit_turn()), \
                patch.object(FinancialAuditorAgent, "audit", side_effect=RuntimeError("boom")):
            events = asyncio.run(_events())
        self.assertEqual(events[0], {"type": "status", "node": "supervisor",
                                     "message": "Analyzing research inquiry & context..."})
        self.assertIn({"type": "status", "node": "financial_auditor", "message": "Financial statement audit unavailable"}, events)
        self.assertEqual(events[-1]["type"], "error")
        self.assertEqual(
            events[-1]["message"],
            "Research pipeline failed: No research output could be produced for AAPL FY2025. "
            "Financial Auditor failed (RuntimeError: boom); its section is omitted from this report.",
        )

    def test_conversational_turn_returns_the_reply(self):
        turn = TurnResolution(route="conversational", plan=None, run_context=None,
                              conversational_response="A DCF discounts cash flows.",
                              session_update={"active_ticker": "AAPL"}, parameter_notes=())
        with patch("app.agents.pipeline.node.resolve_turn", return_value=turn):
            events = asyncio.run(_events())
        self.assertEqual(events[-1], {"type": "result", "response": "A DCF discounts cash flows.", "sources": [],
                                      "updated_session_state": {"active_ticker": "AAPL"}})

    def test_unknown_route_replies_with_guidance_instead_of_an_error(self):
        # RoutingPlan validation rejects unknown routes, so build one without validation to reach this branch.
        plan = RoutingPlan.model_construct(ticker="AAPL", fiscal_year=2025, query_type="no_such_route",
                                           year_substituted=False, valuation_overrides={})
        turn = TurnResolution(route="no_such_route", plan=plan, run_context=make_run_context(route="no_such_route"),
                              conversational_response=None, session_update={"active_ticker": "AAPL"},
                              parameter_notes=())
        with patch("app.agents.pipeline.node.resolve_turn", return_value=turn):
            events = asyncio.run(_events())
        self.assertEqual(events[-1], {
            "type": "result",
            "response": "I couldn't tell which analysis you want for AAPL. Try asking for a full report, "
                        "a DCF valuation, a financial audit, a moat analysis or a risk review.",
            "sources": [],
            "updated_session_state": {"active_ticker": "AAPL"},
        })

    def test_sync_run_raises_when_no_report_exists(self):
        with patch("app.agents.pipeline.node.resolve_turn", return_value=_audit_turn()), \
                patch.object(FinancialAuditorAgent, "audit", side_effect=RuntimeError("boom")):
            with self.assertRaises(RuntimeError):
                MultiAgentOrchestrator().run([{"role": "user", "content": "audit AAPL"}])
