import dataclasses
import unittest

from app.agents.pipeline.contract import AgentSpec, RunNote, UndeclaredUpstreamError, Upstream
from tests.pipeline.fixtures import make_run_context


def _noop_run(run, upstream):
    return None


def _noop_milestone(output):
    return None, None


class TestUpstream(unittest.TestCase):
    def test_declared_output_is_readable_and_missing_one_is_none(self):
        upstream = Upstream({"financial_audit": "audit"}, frozenset({"financial_audit", "forecast"}))
        self.assertEqual(upstream.get("financial_audit"), "audit")
        self.assertIsNone(upstream.get("forecast"))

    def test_reading_an_undeclared_output_raises(self):
        upstream = Upstream({"financial_audit": "audit", "risk_audit": "risk"}, frozenset({"financial_audit"}))
        with self.assertRaises(UndeclaredUpstreamError):
            upstream.get("risk_audit")

    def test_notes_are_carried(self):
        note = RunNote(agent="supervisor", kind="parameter", message="WACC override 9% applied")
        self.assertEqual(Upstream({}, frozenset(), (note,)).notes, (note,))


class TestAgentSpec(unittest.TestCase):
    def test_declared_is_the_union_of_needs_uses_and_needs_any(self):
        spec = AgentSpec(
            name="lead_synthesizer", label="Lead Synthesizer", output_key="final_report",
            run=_noop_run, milestone=_noop_milestone,
            needs=frozenset({"a"}), uses=frozenset({"b"}), needs_any=frozenset({"c"}),
        )
        self.assertEqual(spec.declared, frozenset({"a", "b", "c"}))


class TestRunContext(unittest.TestCase):
    def test_run_context_is_immutable(self):
        ctx = make_run_context()
        with self.assertRaises(dataclasses.FrozenInstanceError):
            ctx.ticker = "MSFT"
