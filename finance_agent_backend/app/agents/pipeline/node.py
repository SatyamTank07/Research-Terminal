"""The one node every routed agent runs through: check its inputs, run it, record a failure, report."""

import asyncio
import logging
from typing import Any, Awaitable, Callable, Dict, List

from app.agents.pipeline.contract import AgentSpec, RunNote, Upstream
from app.agents.pipeline.run_state import ResearchRunState

logger = logging.getLogger("finance_agent.agents.pipeline")


class RunAborted(RuntimeError):
    """No research output exists, so no report can be written; the run ends with an error event."""


def failure_note(label: str, exc: BaseException) -> str:
    detail = " ".join(str(exc).split())[:240] or type(exc).__name__
    return f"{label} failed ({type(exc).__name__}: {detail}); its section is omitted from this report."


def _update(spec: AgentSpec, output: Any, notes: List[RunNote]) -> Dict[str, Any]:
    update: Dict[str, Any] = {"final": output} if spec.writes_final else {"outputs": {spec.output_key: output}}
    if notes:
        update["notes"] = notes
    return update


def make_agent_node(spec: AgentSpec) -> Callable[[ResearchRunState], Awaitable[Dict[str, Any]]]:
    async def agent_node(state: ResearchRunState) -> Dict[str, Any]:
        run_ctx = state["run"].run_context
        outputs = state.get("outputs") or {}
        notes = tuple(state.get("notes") or [])
        if any(outputs.get(key) is None for key in spec.needs):
            return _update(spec, None, [RunNote(agent=spec.name, kind="skipped", message=spec.skip_message)])
        if spec.needs_any and all(outputs.get(key) is None for key in spec.needs_any):
            raise RunAborted(
                f"No research output could be produced for {run_ctx.ticker} FY{run_ctx.fiscal_year}. "
                + " ".join(note.message for note in notes)
            )
        try:
            output = await asyncio.to_thread(spec.run, run_ctx, Upstream(outputs, spec.declared, notes))
        except Exception as exc:
            if spec.writes_final:
                raise
            logger.exception(f"[{spec.name}] {run_ctx.ticker} FY{run_ctx.fiscal_year} failed")
            return _update(spec, None, [RunNote(agent=spec.name, kind="failed", message=failure_note(spec.label, exc))])
        return _update(spec, output, [])

    agent_node.__name__ = f"{spec.name}_node"
    return agent_node
