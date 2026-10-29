"""Golden snapshots: the whole pipeline, run offline, must reproduce the committed JSON exactly.

Regenerate only when an output change is intended, in its own commit:
    docker exec -e UPDATE_GOLDEN=1 finance_agent pytest tests/golden -q
Regenerate on main after re-ingesting any filing, before starting a PR.
"""

import os
import unittest

from tests.golden.harness import (
    OUTPUT_KEYS,
    UPDATE_ENV,
    first_difference,
    read_snapshot,
    run_scenario,
    snapshot_path,
    write_snapshot,
)
from tests.golden.scenarios import SCENARIOS


class TestGoldenPipeline(unittest.TestCase):
    def _check(self, name: str) -> None:
        actual = run_scenario(SCENARIOS[name])
        if os.environ.get(UPDATE_ENV) == "1":
            write_snapshot(name, actual)
            return
        self.assertTrue(snapshot_path(name).exists(), f"No snapshot for {name}; run with {UPDATE_ENV}=1 first")
        diff = first_difference(read_snapshot(name), actual)
        self.assertIsNone(diff, f"Golden mismatch in {name}: {diff}")

    def test_full_aapl_2025(self):
        self._check("full_aapl_2025")

    def test_full_amzn_2025(self):
        self._check("full_amzn_2025")

    def test_full_nvda_2026(self):
        self._check("full_nvda_2026")

    def test_full_tsla_2025(self):
        self._check("full_tsla_2025")

    def test_dcf_aapl_2025_overrides(self):
        self._check("dcf_aapl_2025_overrides")

    def test_audit_aapl_2025(self):
        self._check("audit_aapl_2025")

    def test_moat_nvda_2026(self):
        self._check("moat_nvda_2026")

    def test_risk_tsla_2025(self):
        self._check("risk_tsla_2025")

    def test_followup_dcf_aapl_session(self):
        self._check("followup_dcf_aapl_session")

    def test_chat_what_is_dcf(self):
        self._check("chat_what_is_dcf")

    def test_every_scenario_has_a_test(self):
        tested = {name[len("test_"):] for name in dir(self) if name.startswith("test_")}
        self.assertEqual(set(SCENARIOS) - tested, set())


@unittest.skipIf(os.environ.get(UPDATE_ENV) == "1", "snapshots are being regenerated")
class TestGoldenSnapshotContent(unittest.TestCase):
    """Guards against snapshots that are byte-stable but meaningless (e.g. a broken fake nulls every output)."""

    def test_full_reports_have_every_section(self):
        for name in ("full_aapl_2025", "full_amzn_2025", "full_nvda_2026", "full_tsla_2025"):
            outputs = read_snapshot(name)["outputs"]
            for key in OUTPUT_KEYS + ("final_report",):
                self.assertIsNotNone(outputs[key], f"{name}: {key}")

    def test_research_turns_start_with_the_supervisor_and_end_with_a_result(self):
        for name in ("full_aapl_2025", "audit_aapl_2025", "risk_tsla_2025"):
            events = read_snapshot(name)["events"]
            self.assertEqual((events[0]["type"], events[0]["node"]), ("status", "supervisor"), name)
            self.assertEqual(events[-1]["type"], "result", name)

    def test_dcf_route_skips_qualitative_agents_and_applies_the_wacc_override(self):
        outputs = read_snapshot("dcf_aapl_2025_overrides")["outputs"]
        self.assertIsNone(outputs["business_moat"])
        self.assertIsNone(outputs["risk_audit"])
        self.assertEqual(outputs["dcf_valuation"]["wacc_audit"]["wacc_source"], "user_override")
        self.assertAlmostEqual(outputs["dcf_valuation"]["wacc_audit"]["wacc_pct"], 9.0)

    def test_follow_up_carries_session_overrides(self):
        outputs = read_snapshot("followup_dcf_aapl_session")["outputs"]
        self.assertIn("Applied terminal_growth specified earlier in the conversation.", outputs["pipeline_notes"])
        self.assertEqual(outputs["updated_session_state"]["valuation_overrides"]["beta"], 1.4)

    def test_fabricated_figure_is_screened_out_of_the_report(self):
        report = read_snapshot("full_aapl_2025")["outputs"]["final_report"]["full_markdown_report"]
        self.assertNotIn("999,999", report)

    def test_conversational_turn_runs_no_agent(self):
        outputs = read_snapshot("chat_what_is_dcf")["outputs"]
        self.assertTrue(outputs["conversational_response"].startswith("A DCF values a business"))
        for key in OUTPUT_KEYS + ("final_report",):
            self.assertIsNone(outputs[key], key)
