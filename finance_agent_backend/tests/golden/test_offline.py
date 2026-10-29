"""The offline fakes must answer exactly like the services they replace, with fixed values."""

import asyncio
import importlib
import unittest

import yfinance
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import BaseModel

from tests.golden.offline import CLOCK_TARGETS, EMBEDDING_DIM, GOLDEN_DATE, FakeDate, offline


class _Answer(BaseModel):
    text: str


def _chat() -> ChatOpenAI:
    return ChatOpenAI(model="gpt-4o-mini", api_key="sk-offline")


class TestOfflineFakes(unittest.TestCase):
    def test_chat_model_replies_with_empty_json(self):
        with offline({}):
            reply = _chat().invoke("hello")
            async_reply = asyncio.run(_chat().ainvoke("hello"))
            streamed = "".join(chunk.content for chunk in _chat().stream("hello"))
        self.assertEqual(reply.content, "{}")
        self.assertEqual(async_reply.content, "{}")
        self.assertEqual(streamed, "{}")

    def test_structured_output_returns_the_canned_answer_for_its_schema(self):
        with offline({"_Answer": {"text": "canned"}}):
            answer = _chat().with_structured_output(_Answer).invoke("question")
        self.assertEqual(answer, _Answer(text="canned"))

    def test_structured_output_without_a_canned_answer_raises(self):
        with offline({}):
            with self.assertRaises(RuntimeError):
                _chat().with_structured_output(_Answer).invoke("question")

    def test_embeddings_are_deterministic_and_non_zero(self):
        with offline({}):
            client = OpenAIEmbeddings(model="text-embedding-3-small", api_key="sk-offline")
            first, again = client.embed_query("revenue"), client.embed_query("revenue")
            other = client.embed_documents(["margin"])[0]
        self.assertEqual(first, again)
        self.assertEqual(len(first), EMBEDDING_DIM)
        self.assertNotEqual(first, other)
        self.assertTrue(any(value != 0.0 for value in first))

    def test_quotes_are_fixed_per_ticker(self):
        with offline({}):
            apple = yfinance.Ticker("aapl")
            treasury = yfinance.Ticker("^TNX").history(period="5d")
            unknown = yfinance.Ticker("ZZZZ")
        self.assertEqual(apple.fast_info.last_price, 230.0)
        self.assertEqual(apple.fast_info.market_cap, 3.45e12)
        self.assertEqual(apple.info["beta"], 1.2)
        self.assertEqual(treasury["Close"].iloc[-1], 4.3)
        self.assertIsNone(unknown.fast_info)
        self.assertEqual(unknown.info, {})

    def test_clock_is_pinned_in_every_target_module(self):
        self.assertEqual(FakeDate.today(), GOLDEN_DATE)
        with offline({}):
            for target in CLOCK_TARGETS:
                module_path, _, attribute = target.rpartition(".")
                module = importlib.import_module(module_path)
                self.assertEqual(getattr(module, attribute).today(), GOLDEN_DATE, target)

    def test_patches_are_removed_on_exit(self):
        with offline({}):
            pass
        self.assertNotEqual(yfinance.Ticker.__name__, "FakeTicker")
        self.assertNotEqual(ChatOpenAI.with_structured_output.__name__, "<lambda>")
