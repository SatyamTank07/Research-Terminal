import asyncio
import unittest
from datetime import date
from types import SimpleNamespace
from unittest.mock import patch

from app.agents.pipeline.contract import AgentSpec, Conversation
from app.agents.pipeline.graph import build_graph
from tests.pipeline.fixtures import make_run_context

CONVERSATION = Conversation(user_query="q", messages=None, session_state=None, ticker="", fiscal_year=0,
                            run_date=date(2026, 10, 28))


def _milestone(output):
    return None, None


def _spec(name, key, run, **fields):
    return AgentSpec(name=name, label=name, output_key=key, run=run, milestone=_milestone, **fields)


def _turn(route, run_context, notes=(), reply=None):
    return SimpleNamespace(route=route, run_context=run_context, parameter_notes=notes, conversational_response=reply)


def _invoke(routes, turn):
    with patch("app.agents.pipeline.node.resolve_turn", return_value=turn):
        return asyncio.run(build_graph(routes).ainvoke({"conversation": CONVERSATION}))


class TestGraph(unittest.TestCase):
    def test_parallel_agents_all_reach_the_final_agent(self):
        alpha = _spec("alpha", "a", lambda ctx, up: "A")
        beta = _spec("beta", "b", lambda ctx, up: "B")
        final = _spec("omega", "report", lambda ctx, up: f"{up.get('a')}+{up.get('b')}",
                      uses=frozenset({"a", "b"}), writes_final=True)
        state = _invoke({"r": [[alpha, beta], final]}, _turn("r", make_run_context(route="r")))
        self.assertEqual(state["outputs"], {"a": "A", "b": "B"})
        self.assertEqual(state["final"], "A+B")

    def test_conversational_turn_runs_no_agent(self):
        def explode(ctx, up):
            raise AssertionError("no agent may run on a conversational turn")

        routes = {"r": [_spec("omega", "report", explode, writes_final=True)]}
        state = _invoke(routes, _turn("conversational", None, notes=("Remembered beta 1.4.",), reply="Hi."))
        self.assertEqual(state["final"], "Hi.")
        self.assertEqual([note.message for note in state["notes"]], ["Remembered beta 1.4."])
        self.assertEqual(state.get("outputs") or {}, {})

    def test_real_graph_compiles_with_todays_node_names(self):
        nodes = set(build_graph().get_graph().nodes)
        self.assertLessEqual({"supervisor", "business_strategist", "financial_auditor", "risk_analyst",
                              "forecasting_analyst", "valuation_specialist", "lead_synthesizer"}, nodes)
