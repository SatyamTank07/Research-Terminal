"""Research run entry point: build the turn, run the graph, stream milestones, shape the result."""

import asyncio
import concurrent.futures
import logging
from datetime import date
from typing import Any, AsyncIterator, Dict, List, Optional

from langgraph.graph.state import CompiledStateGraph

from app.agents.base import AgentOutput, BaseAgent
from app.agents.pipeline.contract import Conversation, StatusLine
from app.agents.pipeline.graph import build_graph
from app.agents.pipeline.routes import ROUTES, all_specs
from app.agents.pipeline.run_state import ResearchRunState
from app.agents.registry import AgentRegistry
from app.agents.specialized.lead_synthesizer import Final10KResearchReport
from app.agents.specialized.supervisor import supervisor_milestone

logger = logging.getLogger("finance_agent.agents.orchestrator")

SPECS_BY_NAME = {spec.name: spec for spec in all_specs()}
START_MESSAGE = "Analyzing research inquiry & context..."
NO_REPORT_MESSAGE = "The research pipeline finished without a report."
UNKNOWN_ROUTE_MESSAGE = (
    "I couldn't tell which analysis you want for {ticker}. Try asking for a full report, "
    "a DCF valuation, a financial audit, a moat analysis or a risk review."
)


@AgentRegistry.register("multi_agent")
@AgentRegistry.register("orchestrator")
class MultiAgentOrchestrator(BaseAgent):
    """Runs one chat turn through the research pipeline."""

    def __init__(self, recursion_limit: int = 50):
        self.recursion_limit = recursion_limit
        self._cached_graph: Optional[CompiledStateGraph] = None

    def get_graph(self) -> CompiledStateGraph:
        if self._cached_graph is None:
            self._cached_graph = build_graph()
        return self._cached_graph

    @staticmethod
    def _initial_state(user_query: str, ticker: Optional[str], fiscal_year: Optional[int],
                       messages: Optional[List[Dict[str, str]]], session_state: Optional[Dict[str, Any]],
                       callbacks: Optional[List[Any]]) -> ResearchRunState:
        conversation = Conversation(
            user_query=user_query,
            messages=messages,
            session_state=session_state,
            ticker=ticker.upper() if ticker else "",
            fiscal_year=fiscal_year or 0,
            run_date=date.today(),
            callbacks=callbacks,
        )
        return {"conversation": conversation}

    def _config(self, callbacks: Optional[List[Any]]) -> Dict[str, Any]:
        config: Dict[str, Any] = {"recursion_limit": self.recursion_limit}
        if callbacks:
            config["callbacks"] = callbacks
        return config

    async def arun(self, user_query: str, ticker: Optional[str] = None, fiscal_year: Optional[int] = None,
                   messages: Optional[List[Dict[str, str]]] = None, session_state: Optional[Dict[str, Any]] = None,
                   callbacks: Optional[List[Any]] = None) -> ResearchRunState:
        """Runs the graph and returns the final run state."""
        initial = self._initial_state(user_query, ticker, fiscal_year, messages, session_state, callbacks)
        return await self.get_graph().ainvoke(initial, config=self._config(callbacks))

    async def astream_run(self, user_query: str, ticker: Optional[str] = None, fiscal_year: Optional[int] = None,
                          messages: Optional[List[Dict[str, str]]] = None,
                          session_state: Optional[Dict[str, Any]] = None,
                          callbacks: Optional[List[Any]] = None) -> AsyncIterator[Dict[str, Any]]:
        """Runs the graph, streaming one status event per finished node, then the result or an error."""
        initial = self._initial_state(user_query, ticker, fiscal_year, messages, session_state, callbacks)
        yield {"type": "status", "node": "supervisor", "message": START_MESSAGE}
        final_state: Dict[str, Any] = {}
        try:
            async for mode, chunk in self.get_graph().astream(
                initial, config=self._config(callbacks), stream_mode=["updates", "values"]
            ):
                if mode == "values":
                    final_state = chunk
                    continue
                for node_name, update in chunk.items():
                    message, details = self._milestone(node_name, update)
                    if message:
                        event: Dict[str, Any] = {"type": "status", "node": node_name, "message": message}
                        if details:
                            event["details"] = details
                        yield event
                    for note in update.get("notes") or []:
                        yield {"type": "status", "node": node_name, "message": note.message,
                               "details": {"severity": "warning"}}
        except Exception as e:
            # Stream an error event instead of tearing the SSE connection down; the chat service persists it.
            logger.exception("astream_run: research pipeline failed")
            yield {"type": "error", "message": f"Research pipeline failed: {e}"}
            return
        yield self._result_event(final_state)

    @staticmethod
    def _milestone(node_name: str, update: Dict[str, Any]) -> StatusLine:
        if node_name == "supervisor":
            return supervisor_milestone(update.get("run"))
        spec = SPECS_BY_NAME.get(node_name)
        if spec is None:
            return None, None
        output = update.get("final") if spec.writes_final else (update.get("outputs") or {}).get(spec.output_key)
        return spec.milestone(output)

    @staticmethod
    def _result_event(state: Dict[str, Any]) -> Dict[str, Any]:
        final = state.get("final")
        turn = state.get("run")
        session_update = turn.session_update if turn is not None else None
        if isinstance(final, str) and final:
            return {"type": "result", "response": final, "sources": [], "updated_session_state": session_update}
        if isinstance(final, Final10KResearchReport):
            return {
                "type": "result",
                "response": final.full_markdown_report,
                "sources": [s.model_dump() for s in final.sources],
                "final_report": final.model_dump(),
                "updated_session_state": session_update,
            }
        if turn is not None and turn.run_context is not None and turn.route not in ROUTES:
            return {
                "type": "result",
                "response": UNKNOWN_ROUTE_MESSAGE.format(ticker=turn.run_context.ticker),
                "sources": [],
                "updated_session_state": session_update,
            }
        return {"type": "error", "message": NO_REPORT_MESSAGE}

    def run(self, messages: List[Dict[str, str]], session_state: Optional[Dict[str, Any]] = None,
            callbacks: Optional[List[Any]] = None) -> AgentOutput:
        """Synchronous BaseAgent interface; a run without a report raises."""
        query = messages[-1]["content"] if messages else ""
        coroutine = self.arun(user_query=query, messages=messages, session_state=session_state, callbacks=callbacks)
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop and loop.is_running():
            with concurrent.futures.ThreadPoolExecutor() as pool:
                state = pool.submit(asyncio.run, coroutine).result()
        else:
            state = asyncio.run(coroutine)
        event = self._result_event(state)
        if event["type"] == "error":
            raise RuntimeError(event["message"])
        return AgentOutput(content=event["response"], sources=event["sources"],
                           updated_session_state=event["updated_session_state"])
