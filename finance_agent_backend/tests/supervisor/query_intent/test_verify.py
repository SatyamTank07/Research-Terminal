"""query_intent verify: a known report type runs as asked; anything else becomes a conversational reply."""

import unittest
from typing import get_args

from app.agents.pipeline.routes import CONVERSATIONAL, ROUTES
from app.agents.specialized.supervisor.modules.query_intent import (
    DEFAULT_REPLY,
    QueryType,
    SupervisorExtraction,
    verify_query_intent,
)


class TestVerifyQueryIntent(unittest.TestCase):
    def test_every_report_type_the_judge_names_is_kept(self):
        for query_type in ROUTES:
            intent = verify_query_intent(SupervisorExtraction(query_type=query_type, extracted_ticker="AAPL"))
            self.assertEqual((intent.query_type, intent.provenance, intent.reply), (query_type, "llm_inferred", None))

    def test_report_types_are_exactly_the_routes_plus_conversational(self):
        self.assertEqual(set(get_args(QueryType)), set(ROUTES) | {CONVERSATIONAL})

    def test_conversational_turn_keeps_the_judges_reply(self):
        intent = verify_query_intent(SupervisorExtraction(query_type="conversational", conversational_response="Hi."))
        self.assertEqual((intent.query_type, intent.provenance, intent.reply), ("conversational", "llm_inferred", "Hi."))

    def test_conversational_turn_without_a_reply_gets_the_default(self):
        intent = verify_query_intent(SupervisorExtraction(query_type="conversational"))
        self.assertEqual(intent.reply, DEFAULT_REPLY)

    def test_failed_judge_call_becomes_a_conversational_turn(self):
        intent = verify_query_intent(None)
        self.assertEqual((intent.query_type, intent.provenance, intent.reply),
                         ("conversational", "deterministic_rule", DEFAULT_REPLY))

    def test_unknown_report_type_becomes_a_conversational_turn(self):
        intent = verify_query_intent(SupervisorExtraction.model_construct(query_type="price_target", conversational_response=None))
        self.assertEqual((intent.query_type, intent.provenance), ("conversational", "deterministic_rule"))


if __name__ == "__main__":
    unittest.main()
