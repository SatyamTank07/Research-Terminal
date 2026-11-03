"""The single tool-free LLM completion shared by the Strategist, Risk and Valuation agents, and its JSON parser."""

import unittest
from unittest.mock import MagicMock, patch

from langchain_core.messages import HumanMessage, SystemMessage

from app.agents.shared.llm_call import complete_once, parse_json_content

MODULE = "app.agents.shared.llm_call"


class TestCompleteOnce(unittest.TestCase):
    def _call(self, **kwargs):
        llm = MagicMock()
        llm.invoke.return_value.content = "reply"
        with patch(f"{MODULE}.ChatOpenAI", return_value=llm) as chat, patch(f"{MODULE}.load_dotenv"):
            out = complete_once("openai:gpt-4o-mini", "user text", **kwargs)
        return out, chat.call_args.kwargs, llm.invoke.call_args

    def test_minimal_call_has_no_system_prompt_cap_or_json_mode(self):
        out, llm_kwargs, invoke = self._call()
        self.assertEqual(out, "reply")
        self.assertEqual(llm_kwargs, {"model": "gpt-4o-mini", "temperature": 0, "max_retries": 5})
        self.assertEqual(invoke.args[0], [HumanMessage(content="user text")])
        self.assertEqual(invoke.kwargs["config"], {})

    def test_system_prompt_cap_json_mode_and_callbacks_pass_through(self):
        cb = object()
        _, llm_kwargs, invoke = self._call(system_prompt="sys", max_tokens=900, max_retries=3, json_mode=True, callbacks=[cb])
        self.assertEqual(llm_kwargs, {
            "model": "gpt-4o-mini", "temperature": 0, "max_retries": 3, "max_tokens": 900,
            "model_kwargs": {"response_format": {"type": "json_object"}},
        })
        self.assertEqual(invoke.args[0], [SystemMessage(content="sys"), HumanMessage(content="user text")])
        self.assertEqual(invoke.kwargs["config"], {"callbacks": [cb]})


class TestParseJsonContent(unittest.TestCase):
    def test_plain_json_object_is_parsed(self):
        self.assertEqual(parse_json_content('{"a": 1}'), {"a": 1})

    def test_code_fences_are_stripped(self):
        self.assertEqual(parse_json_content('```json\n{"a": 1}\n```'), {"a": 1})

    def test_outermost_object_is_recovered_from_surrounding_prose(self):
        self.assertEqual(parse_json_content('Here it is: {"a": {"b": 2}} done.'), {"a": {"b": 2}})

    def test_list_of_content_parts_is_joined(self):
        self.assertEqual(parse_json_content([{"text": '{"a": '}, "1}"]), {"a": 1})

    def test_anything_but_a_json_object_gives_an_empty_dict(self):
        for raw in ("[1, 2]", "not json {broken", "", None, 42):
            with self.subTest(raw=raw):
                self.assertEqual(parse_json_content(raw), {})


if __name__ == "__main__":
    unittest.main()
