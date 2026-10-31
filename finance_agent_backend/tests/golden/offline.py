"""Offline fakes for golden snapshot runs.

Every boundary that leaves the process is replaced: chat models, embeddings, market quotes and the clock
that dates the valuation. The fakes patch library classes (langchain_openai, yfinance) rather than app
modules, so snapshots keep working while app code moves between files. The one exception is
CLOCK_TARGETS: update it when a module that reads today's date moves.
"""

import contextlib
import hashlib
import random
from datetime import date
from typing import Any, AsyncIterator, Dict, Iterator, List, Mapping, Optional
from unittest.mock import patch

import pandas as pd
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

GOLDEN_DATE = date(2026, 10, 28)
EMBEDDING_DIM = 1536
TREASURY_10Y_PCT = 4.3
CHAT_REPLY = "{}"

MARKET: Dict[str, Dict[str, float]] = {
    "AAPL": {"last_price": 230.0, "market_cap": 3.45e12, "beta": 1.2},
    "AMZN": {"last_price": 220.0, "market_cap": 2.3e12, "beta": 1.3},
    "NVDA": {"last_price": 180.0, "market_cap": 4.4e12, "beta": 1.7},
    "TSLA": {"last_price": 250.0, "market_cap": 8.0e11, "beta": 2.0},
}

# Modules that read today's date: the valuation date (roll-forward) and the "is this year in the
# future?" checks on user-supplied years. Each module's `date` is replaced by FakeDate.
CLOCK_TARGETS = (
    "app.agents.pipeline.orchestrator.date",
    "app.agents.tools.valuation_model.date",
    "app.agents.query_parameters.date",
    "app.agents.specialized.supervisor.agent_supervisor.date",
)


class FakeDate(date):
    """`date` with a pinned today(), so the DCF roll-forward does not change from day to day."""

    @classmethod
    def today(cls) -> date:
        return GOLDEN_DATE


def _fake_generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
    return ChatResult(generations=[ChatGeneration(message=AIMessage(content=CHAT_REPLY))])


async def _fake_agenerate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
    return _fake_generate(self, messages, stop, run_manager, **kwargs)


def _fake_stream(self, messages, stop=None, run_manager=None, **kwargs) -> Iterator[ChatGenerationChunk]:
    yield ChatGenerationChunk(message=AIMessageChunk(content=CHAT_REPLY))


async def _fake_astream(self, messages, stop=None, run_manager=None, **kwargs) -> AsyncIterator[ChatGenerationChunk]:
    yield ChatGenerationChunk(message=AIMessageChunk(content=CHAT_REPLY))


class FakeStructuredModel:
    """Stands in for `ChatOpenAI.with_structured_output(schema)`: returns the canned answer for that schema."""

    def __init__(self, schema: Any, responses: Mapping[str, Dict[str, Any]]):
        self.schema = schema
        self.responses = responses

    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        canned = self.responses.get(self.schema.__name__)
        if canned is None:
            raise RuntimeError(f"No canned structured response for {self.schema.__name__}")
        return self.schema.model_validate(canned)

    async def ainvoke(self, *args: Any, **kwargs: Any) -> Any:
        return self.invoke(*args, **kwargs)


def fake_vector(text: str) -> List[float]:
    """Deterministic, non-zero pseudo-embedding (a zero vector makes cosine distance undefined)."""
    rng = random.Random(hashlib.sha256(text.encode("utf-8")).digest())
    return [rng.uniform(-1.0, 1.0) for _ in range(EMBEDDING_DIM)]


class _FastInfo:
    def __init__(self, quote: Dict[str, float]):
        self.last_price = quote["last_price"]
        self.market_cap = quote["market_cap"]


class FakeTicker:
    """yfinance.Ticker stand-in: fixed price, market cap and beta per ticker, fixed 10Y Treasury yield."""

    def __init__(self, symbol: str):
        self.symbol = symbol.strip().upper()

    @property
    def fast_info(self) -> Optional[_FastInfo]:
        quote = MARKET.get(self.symbol)
        return _FastInfo(quote) if quote else None

    @property
    def info(self) -> Dict[str, float]:
        quote = MARKET.get(self.symbol)
        return {"beta": quote["beta"]} if quote else {}

    def history(self, period: str = "5d") -> pd.DataFrame:
        if self.symbol == "^TNX":
            return pd.DataFrame({"Close": [TREASURY_10Y_PCT]})
        return pd.DataFrame({"Close": []})


@contextlib.contextmanager
def offline(structured_responses: Mapping[str, Dict[str, Any]]) -> Iterator[None]:
    """Runs the block with every external boundary faked; patches are removed on exit."""
    with contextlib.ExitStack() as stack:
        stack.enter_context(patch.object(ChatOpenAI, "_generate", _fake_generate))
        stack.enter_context(patch.object(ChatOpenAI, "_agenerate", _fake_agenerate))
        stack.enter_context(patch.object(ChatOpenAI, "_stream", _fake_stream))
        stack.enter_context(patch.object(ChatOpenAI, "_astream", _fake_astream))
        stack.enter_context(patch.object(
            ChatOpenAI, "with_structured_output",
            lambda self, schema, **kwargs: FakeStructuredModel(schema, structured_responses),
        ))
        stack.enter_context(patch.object(OpenAIEmbeddings, "embed_query", lambda self, text: fake_vector(text)))
        stack.enter_context(patch.object(
            OpenAIEmbeddings, "embed_documents",
            lambda self, texts, chunk_size=None, **kwargs: [fake_vector(text) for text in texts],
        ))
        stack.enter_context(patch("yfinance.Ticker", FakeTicker))
        for target in CLOCK_TARGETS:
            stack.enter_context(patch(target, FakeDate))
        yield
