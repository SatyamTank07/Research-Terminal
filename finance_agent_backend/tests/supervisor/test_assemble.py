"""assemble: the routing plan for research and chat turns, and the run facts a research turn fixes."""

import unittest
from datetime import date

from app.agents.pipeline.contract import Conversation
from app.agents.specialized.supervisor.assemble import (
    assemble_conversational_plan,
    assemble_research_plan,
    assemble_turn,
)
from app.agents.specialized.supervisor.context import FilingRecord
from app.agents.specialized.supervisor.modules.filing_resolution import FilingChoice
from app.agents.specialized.supervisor.modules.query_intent import QueryIntent
from app.agents.specialized.supervisor.modules.valuation_parameters import ValuationParameters

FILING = FilingRecord(ticker="AAPL", company_name="Apple Inc.", fiscal_year=2024, document_id="doc-aapl-2024")
INTENT = QueryIntent(query_type="dcf_valuation_only", provenance="llm_inferred", reply=None)


def _research_plan(parameters: ValuationParameters, filing_note=None):
    return assemble_research_plan(
        intent=INTENT, filing=FILING, choice=FilingChoice(2024, 2024, False), available_years=[2025, 2024],
        parse_notes=["Ignored beta 9.00: out of range."], filing_note=filing_note, parameters=parameters,
        session={"active_ticker": "AAPL"},
    )


class TestAssembleResearchPlan(unittest.TestCase):
    def test_notes_read_rejected_inputs_then_the_carried_year_then_carried_overrides(self):
        plan = _research_plan(
            ValuationParameters(overrides={"ticker": "AAPL", "wacc": 0.08}, carry_note="Applied wacc earlier."),
            filing_note="Continuing with the FY2024 filing.",
        )
        self.assertEqual(plan.parameter_notes, [
            "Ignored beta 9.00: out of range.", "Continuing with the FY2024 filing.", "Applied wacc earlier.",
        ])
        self.assertEqual((plan.ticker, plan.fiscal_year, plan.document_id), ("AAPL", 2024, "doc-aapl-2024"))
        self.assertIsNone(plan.conversational_response)

    def test_no_carried_notes_when_nothing_was_carried(self):
        plan = _research_plan(ValuationParameters(overrides={}, carry_note=None))
        self.assertEqual(plan.parameter_notes, ["Ignored beta 9.00: out of range."])


class TestAssembleConversationalPlan(unittest.TestCase):
    def test_reply_keeps_the_active_research_context(self):
        session = {"active_ticker": "AAPL", "active_company": "Apple Inc.", "active_fiscal_year": 2024,
                   "available_fiscal_years": [2025, 2024]}
        plan = assemble_conversational_plan(session, "Hi.", "llm_inferred", [])
        self.assertEqual((plan.ticker, plan.company_name, plan.fiscal_year), ("AAPL", "Apple Inc.", 2024))
        self.assertEqual((plan.query_type, plan.conversational_response), ("conversational", "Hi."))
        self.assertEqual(plan.updated_session_state, session)

    def test_empty_session_gives_an_empty_plan(self):
        plan = assemble_conversational_plan({}, "Hi.", "deterministic_rule", [])
        self.assertEqual((plan.ticker, plan.company_name, plan.fiscal_year, plan.available_fiscal_years),
                         ("", None, 0, []))


class TestAssembleTurn(unittest.TestCase):
    def test_research_turn_shares_terminal_growth_from_the_overrides(self):
        plan = _research_plan(ValuationParameters(overrides={"ticker": "AAPL", "terminal_growth": 0.03}, carry_note=None))
        conversation = Conversation(user_query="dcf", messages=None, session_state=None, ticker="", fiscal_year=0,
                                    run_date=date(2026, 11, 3))
        turn = assemble_turn(conversation, plan)
        self.assertAlmostEqual(turn.run_context.terminal_growth_rate, 0.03)
        self.assertEqual(turn.run_context.valuation_date, date(2026, 11, 3))
        self.assertEqual(turn.route, "dcf_valuation_only")


if __name__ == "__main__":
    unittest.main()
