import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.agents.pipeline.contract import Upstream
from app.agents.specialized.business_strategist import AGENT as STRATEGIST
from app.agents.specialized.business_strategist import BusinessStrategistAgent
from app.agents.specialized.financial_auditor import AGENT as AUDITOR
from app.agents.specialized.financial_auditor import FinancialAuditorAgent
from app.agents.specialized.risk_analyst import AGENT as RISK
from app.agents.specialized.risk_analyst import RiskAnalystAgent
from tests.forecast_fixtures import make_aapl_audit
from tests.pipeline.fixtures import make_run_context


def _empty(spec):
    return Upstream({}, spec.declared)


class TestPhaseOneSpecs(unittest.TestCase):
    def test_declarations(self):
        for spec, name, key in ((STRATEGIST, "business_strategist", "business_moat"),
                                (AUDITOR, "financial_auditor", "financial_audit"),
                                (RISK, "risk_analyst", "risk_audit")):
            self.assertEqual((spec.name, spec.output_key), (name, key))
            self.assertEqual(spec.declared, frozenset())
            self.assertFalse(spec.writes_final)

    def test_strategist_runs_on_the_filing(self):
        ctx = make_run_context(callbacks=["cb"])
        with patch.object(BusinessStrategistAgent, "analyze", return_value="moat") as analyze:
            self.assertEqual(STRATEGIST.run(ctx, _empty(STRATEGIST)), "moat")
        self.assertEqual(analyze.call_args.kwargs, {"ticker": "AAPL", "fiscal_year": 2025, "callbacks": ["cb"]})

    def test_auditor_gets_the_available_years_as_a_list(self):
        with patch.object(FinancialAuditorAgent, "audit", return_value="audit") as audit:
            AUDITOR.run(make_run_context(), _empty(AUDITOR))
        self.assertEqual(audit.call_args.kwargs["available_fiscal_years"], [2025, 2024])

    def test_risk_analyst_gets_the_available_years_as_a_list(self):
        with patch.object(RiskAnalystAgent, "analyze", return_value="risk") as analyze:
            RISK.run(make_run_context(), _empty(RISK))
        self.assertEqual(analyze.call_args.kwargs["available_fiscal_years"], [2025, 2024])

    def test_milestones_when_unavailable(self):
        self.assertEqual(STRATEGIST.milestone(None), ("Economic moat analysis unavailable", None))
        self.assertEqual(AUDITOR.milestone(None), ("Financial statement audit unavailable", None))
        self.assertEqual(RISK.milestone(None), ("Analyzed Item 1A risks & threats (no risk output)", None))

    def test_strategist_milestone_names_the_moat(self):
        self.assertEqual(STRATEGIST.milestone(SimpleNamespace(economic_moat_type="Network Effects"))[0],
                         "Completed economic moat analysis (Moat: Network Effects)")

    def test_auditor_milestone_counts_high_and_medium_findings(self):
        message, _ = AUDITOR.milestone(make_aapl_audit())
        self.assertRegex(message, r"^Audited financial statements & ratios \(\d+ high/medium forensic findings, \w+ extraction\)$")

    def test_risk_milestone_for_a_filing_without_risk_factors(self):
        risk = SimpleNamespace(data_quality=SimpleNamespace(coverage_mode="not_provided"))
        self.assertEqual(RISK.milestone(risk)[0],
                         "Analyzed Item 1A risks & threats (filing provides no risk factors; nothing rated)")


from app.agents.pipeline.contract import RunNote
from app.agents.specialized.lead_synthesizer import AGENT as SYNTHESIZER
from app.agents.specialized.lead_synthesizer import LeadSynthesizerAgent


class TestSynthesizerSpec(unittest.TestCase):
    def test_declarations(self):
        self.assertTrue(SYNTHESIZER.writes_final)
        self.assertEqual(SYNTHESIZER.uses, frozenset({"business_moat", "financial_audit", "forecast",
                                                       "dcf_valuation", "risk_audit"}))
        self.assertEqual(SYNTHESIZER.needs_any, frozenset({"business_moat", "financial_audit", "risk_audit"}))

    def test_passes_route_notes_and_every_upstream_output(self):
        notes = (RunNote(agent="supervisor", kind="parameter", message="WACC override 9% applied"),)
        upstream = Upstream({"financial_audit": "audit", "risk_audit": "risk"}, SYNTHESIZER.declared, notes)
        ctx = make_run_context(route="financial_audit_only", year_substituted=True, company_name="Apple Inc.")
        with patch.object(LeadSynthesizerAgent, "synthesize", return_value="report") as synthesize:
            self.assertEqual(SYNTHESIZER.run(ctx, upstream), "report")
        kwargs = synthesize.call_args.kwargs
        self.assertEqual(kwargs["query_type"], "financial_audit_only")
        self.assertEqual(kwargs["pipeline_notes"], ["WACC override 9% applied"])
        self.assertTrue(kwargs["year_substituted"])
        self.assertEqual((kwargs["financial_audit"], kwargs["risk_audit"], kwargs["business_moat"]), ("audit", "risk", None))
        self.assertEqual(kwargs["available_fiscal_years"], [2025, 2024])
        self.assertEqual(kwargs["company_name"], "Apple Inc.")

    def test_milestone(self):
        self.assertEqual(SYNTHESIZER.milestone("report"),
                         ("Synthesized 3-Pillar Thesis and compiled final publication report.", None))
