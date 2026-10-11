"""Chat history sent back to the agents must not carry report citation markers."""

import unittest
from types import SimpleNamespace

from app.services.chat_service import _history_payload


def _msg(role: str, content: str) -> SimpleNamespace:
    return SimpleNamespace(role=role, content=content)


class TestHistoryPayload(unittest.TestCase):
    def test_assistant_markers_are_stripped_before_reaching_the_llm(self):
        past = [
            _msg("user", "Value AAPL"),
            _msg("assistant", "| Current Share Price[[2]](#cite-2) | $235.00 |\nNet cash of $33.8 billion[[6]](#cite-6)."),
            _msg("system", "ignored"),
        ]
        payload = _history_payload(past, "What drives the WACC?")
        self.assertEqual(payload, [
            {"role": "user", "content": "Value AAPL"},
            {"role": "assistant", "content": "| Current Share Price | $235.00 |\nNet cash of $33.8 billion."},
            {"role": "user", "content": "What drives the WACC?"},
        ])

    def test_user_text_is_passed_verbatim(self):
        payload = _history_payload([_msg("user", "literal [[1]](#cite-1) typed by a user")], "next")
        self.assertEqual(payload[0]["content"], "literal [[1]](#cite-1) typed by a user")
