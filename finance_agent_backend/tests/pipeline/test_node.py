import asyncio
import unittest
from types import SimpleNamespace

from app.agents.pipeline.contract import AgentSpec, RunNote
from app.agents.pipeline.node import RunAborted, failure_note, make_agent_node
from app.agents.pipeline.run_state import merge_outputs
from tests.pipeline.fixtures import make_run_context


def _milestone(output):
    return None, None


def _spec(run, **fields) -> AgentSpec:
    base = dict(name="forecasting_analyst", label="Forecasting Analyst", output_key="forecast",
                run=run, milestone=_milestone)
    base.update(fields)
    return AgentSpec(**base)


def _state(outputs=None, notes=None):
    return {"run": SimpleNamespace(run_context=make_run_context()), "outputs": outputs or {}, "notes": notes or []}


def _call(spec, state):
    return asyncio.run(make_agent_node(spec)(state))


class TestMergeOutputs(unittest.TestCase):
    def test_parallel_updates_merge_instead_of_overwriting(self):
        merged = merge_outputs(merge_outputs({}, {"business_moat": "m"}), {"financial_audit": "a"})
        self.assertEqual(merged, {"business_moat": "m", "financial_audit": "a"})

    def test_none_current_is_treated_as_empty(self):
        self.assertEqual(merge_outputs(None, {"risk_audit": "r"}), {"risk_audit": "r"})


class TestAgentNode(unittest.TestCase):
    def test_runs_the_agent_with_context_and_declared_upstream(self):
        seen = {}

        def run(ctx, upstream):
            seen["ticker"], seen["audit"] = ctx.ticker, upstream.get("financial_audit")
            return "forecast"

        update = _call(_spec(run, needs=frozenset({"financial_audit"})), _state({"financial_audit": "audit"}))
        self.assertEqual(update, {"outputs": {"forecast": "forecast"}})
        self.assertEqual(seen, {"ticker": "AAPL", "audit": "audit"})

    def test_missing_need_skips_with_the_agents_note(self):
        def run(ctx, upstream):
            raise AssertionError("must not run")

        spec = _spec(run, needs=frozenset({"financial_audit"}),
                     skip_message="Forecast skipped: no audited financial statements were available.")
        update = _call(spec, _state({"financial_audit": None}))
        self.assertEqual(update["outputs"], {"forecast": None})
        self.assertEqual(update["notes"], [RunNote(agent="forecasting_analyst", kind="skipped",
                                                   message="Forecast skipped: no audited financial statements were available.")])

    def test_failure_is_recorded_not_raised(self):
        def run(ctx, upstream):
            raise ValueError("base_revenue must be positive")

        update = _call(_spec(run), _state())
        self.assertEqual(update["outputs"], {"forecast": None})
        self.assertEqual(update["notes"][0].kind, "failed")
        self.assertEqual(
            update["notes"][0].message,
            "Forecasting Analyst failed (ValueError: base_revenue must be positive); its section is omitted from this report.",
        )

    def test_final_writer_puts_its_output_in_final(self):
        update = _call(_spec(lambda ctx, up: "report", writes_final=True), _state())
        self.assertEqual(update, {"final": "report"})

    def test_final_writer_failure_ends_the_run(self):
        def run(ctx, upstream):
            raise RuntimeError("template error")

        with self.assertRaises(RuntimeError):
            _call(_spec(run, writes_final=True), _state())

    def test_unmet_needs_any_aborts_with_todays_message(self):
        notes = [RunNote(agent="financial_auditor", kind="failed", message="Financial Auditor failed (X: y).")]
        spec = _spec(lambda ctx, up: "report", writes_final=True,
                     needs_any=frozenset({"business_moat", "financial_audit", "risk_audit"}))
        with self.assertRaises(RunAborted) as caught:
            _call(spec, _state({"financial_audit": None}, notes))
        self.assertEqual(str(caught.exception),
                         "No research output could be produced for AAPL FY2025. Financial Auditor failed (X: y).")


class TestFailureNote(unittest.TestCase):
    def test_long_messages_are_collapsed_and_cut(self):
        note = failure_note("Risk Analyst", ValueError("a\n  b " + "x" * 300))
        self.assertTrue(note.startswith("Risk Analyst failed (ValueError: a b xxx"))
        self.assertTrue(note.endswith("; its section is omitted from this report."))

    def test_empty_message_falls_back_to_the_exception_type(self):
        self.assertEqual(failure_note("Risk Analyst", KeyError()),
                         "Risk Analyst failed (KeyError: KeyError); its section is omitted from this report.")
