"""Runs a golden scenario through the orchestrator offline and turns the result into a JSON snapshot."""

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from app.agents import MultiAgentOrchestrator
from tests.golden.offline import offline
from tests.golden.scenarios import Scenario

SNAPSHOT_DIR = Path(__file__).resolve().parent / "snapshots"
UPDATE_ENV = "UPDATE_GOLDEN"
OUTPUT_KEYS = ("business_moat", "financial_audit", "risk_audit", "forecast", "dcf_valuation")
# SSE node names of the agents that run in parallel; their events arrive in completion order.
PARALLEL_NODES = ("business_strategist", "financial_auditor", "risk_analyst")


def _dump(value: Any) -> Any:
    return value.model_dump(mode="json") if isinstance(value, BaseModel) else value


def extract_outputs(state: Dict[str, Any]) -> Dict[str, Any]:
    """Agent outputs and turn results from the final run state.

    This is the only place that knows the state's shape; update it when the shape changes, never the snapshots.
    """
    return {
        "routing_plan": _dump(state.get("routing_plan")),
        **{key: _dump(state.get(key)) for key in OUTPUT_KEYS},
        "final_report": _dump(state.get("final_report")),
        "conversational_response": state.get("conversational_response"),
        "sources": [_dump(source) for source in state.get("sources") or []],
        "updated_session_state": state.get("updated_session_state"),
        "pipeline_notes": list(state.get("pipeline_notes") or []),
        "error_message": state.get("error_message"),
    }


def _order_parallel_events(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Phase-1 agents run in parallel and finish in any order; sort their events by node, in place.

    Each node's own events keep their relative order, and every other event keeps its position.
    """
    slots = [index for index, event in enumerate(events) if event.get("node") in PARALLEL_NODES]
    ordered = sorted((events[index] for index in slots), key=lambda event: PARALLEL_NODES.index(event["node"]))
    result = list(events)
    for index, event in zip(slots, ordered):
        result[index] = event
    return result


def _messages(scenario: Scenario) -> List[Dict[str, str]]:
    return [{"role": "user", "content": scenario.query}]


async def _collect(scenario: Scenario) -> Dict[str, Any]:
    orchestrator = MultiAgentOrchestrator()
    turn = dict(user_query=scenario.query, messages=_messages(scenario), session_state=scenario.session_state)
    state = await orchestrator.arun(**turn)
    # The result event's final_report duplicates outputs.final_report, so it is left out.
    events = [
        {key: value for key, value in event.items() if key != "final_report"}
        async for event in orchestrator.astream_run(**turn)
    ]
    return {"scenario": scenario.name, "outputs": extract_outputs(state), "events": _order_parallel_events(events)}


def run_scenario(scenario: Scenario) -> Dict[str, Any]:
    with offline(scenario.structured_responses()):
        snapshot = asyncio.run(_collect(scenario))
    return json.loads(json.dumps(snapshot, sort_keys=True, default=str))  # plain JSON types only


def snapshot_path(name: str) -> Path:
    return SNAPSHOT_DIR / f"{name}.json"


def write_snapshot(name: str, snapshot: Dict[str, Any]) -> None:
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    text = json.dumps(snapshot, indent=1, sort_keys=True, ensure_ascii=False) + "\n"
    snapshot_path(name).write_text(text, encoding="utf-8")


def read_snapshot(name: str) -> Dict[str, Any]:
    return json.loads(snapshot_path(name).read_text(encoding="utf-8"))


def first_difference(expected: Any, actual: Any, path: str = "$") -> Optional[str]:
    """Path and values of the first mismatch between two JSON values, or None when they are equal."""
    if type(expected) is not type(actual):
        return f"{path}: type {type(expected).__name__} != {type(actual).__name__} ({expected!r:.200} vs {actual!r:.200})"
    if isinstance(expected, dict):
        for key in sorted(set(expected) | set(actual)):
            if key not in expected:
                return f"{path}.{key}: unexpected key (value {actual[key]!r:.200})"
            if key not in actual:
                return f"{path}.{key}: missing key (expected {expected[key]!r:.200})"
            diff = first_difference(expected[key], actual[key], f"{path}.{key}")
            if diff:
                return diff
        return None
    if isinstance(expected, list):
        for index, (want, got) in enumerate(zip(expected, actual)):
            diff = first_difference(want, got, f"{path}[{index}]")
            if diff:
                return diff
        if len(expected) != len(actual):
            return f"{path}: length {len(expected)} != {len(actual)}"
        return None
    if expected != actual:
        return f"{path}: {expected!r:.300} != {actual!r:.300}"
    return None
