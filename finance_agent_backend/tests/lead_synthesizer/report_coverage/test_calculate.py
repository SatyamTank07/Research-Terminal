"""report_coverage rules: which layout the report uses and which run notes it discloses."""

import unittest

from app.agents.specialized.lead_synthesizer.modules.report_coverage import REPORT_LAYOUTS, assess_report_coverage

ALL_OUTPUTS = dict(business_moat="moat", financial_audit="audit", forecast="fc", dcf_valuation="val", risk_audit="risk")


class TestReportCoverage(unittest.TestCase):
    def test_each_route_with_its_outputs_uses_its_own_layout(self):
        for route in REPORT_LAYOUTS:
            coverage = assess_report_coverage(route, ALL_OUTPUTS, [])
            self.assertEqual((coverage.route, coverage.layout), (route, route))

    def test_a_single_topic_route_without_its_topic_uses_the_full_layout(self):
        for route, missing in (("financial_audit_only", "financial_audit"), ("business_moat_only", "business_moat"),
                               ("risk_factors_only", "risk_audit")):
            with self.assertLogs("finance_agent.agents.lead_synthesizer", "WARNING"):
                coverage = assess_report_coverage(route, {**ALL_OUTPUTS, missing: None}, [])
            self.assertEqual((coverage.route, coverage.layout), ("full_10k_report", "full_10k_report"))

    def test_the_dcf_route_keeps_its_layout_without_a_valuation(self):
        coverage = assess_report_coverage("dcf_valuation_only", {**ALL_OUTPUTS, "dcf_valuation": None}, [])
        self.assertEqual((coverage.route, coverage.layout), ("dcf_valuation_only", "dcf_valuation_only"))

    def test_an_unknown_route_uses_the_full_layout_but_keeps_its_name(self):
        coverage = assess_report_coverage("peer_comparison", ALL_OUTPUTS, [])
        self.assertEqual((coverage.route, coverage.layout), ("peer_comparison", "full_10k_report"))

    def test_duplicate_notes_are_kept_once_in_first_seen_order(self):
        coverage = assess_report_coverage("full_10k_report", ALL_OUTPUTS, ["b skipped", "a failed", "b skipped"])
        self.assertEqual(coverage.notes, ("b skipped", "a failed"))
