"""Artifact -> filing source mapping for the code-assembled report (lead_synthesizer/report_citations.py)."""

import re
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from app.agents.specialized.business_strategist.state_business_strategist import StrategyEvidence
from app.agents.specialized.lead_synthesizer.report_citations import (
    ROW_STATEMENTS,
    NullCiter,
    ReportCiter,
)
from app.agents.specialized.risk_analyst import RiskEvidence
from app.agents.specialized.business_strategist.modules.revenue_mix import SegmentFigure
from app.agents.shared.citations import CitationRegistry, marker
from tests.test_agent_lead_synthesizer import _create_mock_audit, _create_mock_moat, _create_mock_risk
from tests.valuation_fixtures import make_valuation_output

IS, BS, CF = "income_statement", "balance_sheet", "cash_flow"
TEMPLATES = Path(__file__).resolve().parents[1] / "app" / "agents" / "specialized" / "lead_synthesizer"


def provenance():
    out = {}
    for year in (2023, 2024, 2025):
        for field, statement in (("revenue", IS), ("operating_income", IS), ("operating_cash_flow", CF)):
            out[f"FY{year}.{field}"] = {"statement": statement, "chunk_id": f"{statement}-fy2025", "filing_year": 2025}
    out["FY2025.cash_and_equivalents"] = {"statement": BS, "chunk_id": "balance_sheet-fy2025", "filing_year": 2025}
    out["FY2024.cash_and_equivalents"] = {"statement": BS, "chunk_id": "balance_sheet-fy2024", "filing_year": 2024}
    return out


def audit_with_provenance():
    audit = _create_mock_audit()
    audit.data_quality.field_provenance = provenance()
    return audit


def moat_with_evidence():
    return _create_mock_moat().model_copy(update={
        "competitive_threats": ["Android OEMs"],
        "evidence": [
            StrategyEvidence(claim="primary_moat", subject="High Switching Costs", chunk_id="item1-a",
                             quote="Customers who own multiple Apple devices rely on iCloud continuity"),
            StrategyEvidence(claim="competitive_threat", subject="Android OEMs", chunk_id="item1-b",
                             quote="The Company faces substantial competition from Android OEMs"),
            StrategyEvidence(claim="competitive_threat", subject="Other", chunk_id="item1-c", quote="Other rivals"),
        ],
        "segment_financials": [
            SegmentFigure(segment_name="Americas", fiscal_year=2025, revenue=178353.0, revenue_share_pct=42.86,
                          source_chunk_id="seg-note"),
        ],
    })


def citer(**kwargs):
    defaults = dict(financial_audit=audit_with_provenance(), business_moat=moat_with_evidence(),
                    risk_audit=_create_mock_risk(), dcf_valuation=make_valuation_output())
    return ReportCiter(CitationRegistry(resolver=MagicMock(return_value={})), fiscal_year=2025, **{**defaults, **kwargs})


class TestReportCiter(unittest.TestCase):
    def test_every_audit_table_attribute_is_mapped(self):
        text = (TEMPLATES / "audit_tables.j2").read_text(encoding="utf-8")
        attrs = set(re.findall(r'row\("[^"]+", fa\.\w+, "(\w+)"', text)) | set(re.findall(r'fe\("(\w+)"', text))
        self.assertGreater(len(attrs), 40)
        self.assertEqual(attrs - set(ROW_STATEMENTS), set())

    def test_rows_on_the_same_statements_share_one_source(self):
        c = citer()
        items = audit_with_provenance().multi_year_history
        revenue = c.row("revenue", items)
        self.assertEqual(revenue, marker(1))
        self.assertEqual(c.row("operating_income", items), revenue)
        self.assertEqual(c.row("operating_cash_flow", items), marker(2))
        spec = c.registry._specs[0]
        self.assertEqual(spec.kind, "derived")
        self.assertEqual(spec.chunk_ids, ("income_statement-fy2025",))
        self.assertEqual(spec.derived_from, "Computed from the audited income statement")

    def test_unmapped_or_unprovenanced_rows_get_no_marker(self):
        c = citer()
        self.assertEqual(c.row("not_a_metric", audit_with_provenance().multi_year_history), "")
        self.assertEqual(citer(financial_audit=_create_mock_audit()).row("revenue", _create_mock_audit().multi_year_history), "")

    def test_field_uses_anchor_year_by_default(self):
        c = citer()
        c.field("net_debt")
        self.assertEqual(c.registry._specs[0].chunk_ids, ("balance_sheet-fy2025",))
        c.field("net_debt", 2024)
        self.assertEqual(c.registry._specs[1].chunk_ids, ("balance_sheet-fy2024",))

    def test_statements_spans_all_three_statements(self):
        c = citer()
        c.statements()
        self.assertEqual(
            c.registry._specs[0].chunk_ids,
            ("income_statement-fy2025", "balance_sheet-fy2025", "cash_flow-fy2025"),
        )

    def test_evidence_filters_by_claim_and_subject(self):
        c = citer()
        self.assertEqual(c.evidence("competitive_threat", "Android OEMs"), marker(1))
        self.assertEqual(c.registry._specs[0].quote, "The Company faces substantial competition from Android OEMs")
        self.assertEqual(c.evidence("pricing_power"), "")
        self.assertEqual(c.evidence("competitive_threat"), marker(1) + marker(2))

    def test_risk_cites_only_verified_quotes(self):
        c = citer()
        risk = _create_mock_risk().identified_risks[0].model_copy(update={"evidence": [
            RiskEvidence(chunk_id="r1", quote="verified quote", verified=True),
            RiskEvidence(chunk_id="r2", quote="unverified quote", verified=False),
        ]})
        self.assertEqual(c.risk(risk), marker(1))
        self.assertEqual(c.mitigation(risk), "")

    def test_market_inputs_carry_provider(self):
        c = citer()
        self.assertEqual(c.market("rf"), marker(1))
        spec = c.registry._specs[0]
        self.assertEqual((spec.kind, spec.label), ("market", "Risk-free rate"))
        self.assertTrue(spec.provider)
        self.assertEqual(c.market("unknown"), "")
        self.assertEqual(citer(dcf_valuation=None).market("price"), "")

    def test_segments_and_failures_never_raise(self):
        c = citer()
        self.assertEqual(c.segments(), marker(1))
        self.assertEqual(c.revenue_mix(), "")
        self.assertEqual(c.forecast_drivers(), "")
        self.assertEqual(c.risk(None), "")

    def test_figure_entries_priority(self):
        entries = citer().figure_entries()
        kinds = [spec.kind for _, spec in entries if spec is not None]
        self.assertEqual(kinds[0], "derived")
        self.assertIn("market", kinds)
        self.assertEqual(kinds[-1], "verbatim")
        texts = [text for text, _ in entries]
        segment_revenue = texts.index(f"{178353.0:.6f}")
        self.assertLess(texts.index(f"{416161.0:.6f}"), segment_revenue)

    def test_computed_figures_are_unciteable_and_rank_before_audit_data(self):
        val = make_valuation_output()
        entries = citer().figure_entries()
        first_audit_revenue = [text for text, _ in entries].index(f"{416161.0:.6f}")
        blockers = [text for text, spec in entries[:first_audit_revenue] if spec is None]
        self.assertIn(f"{val.wacc_audit.wacc_pct:.6f}%", blockers)
        self.assertIn(f"{val.terminal_growth_rate * 100:.6f}%", blockers)
        self.assertIn(f"{val.upside_downside_pct:.6f}%", blockers)

    def test_percentage_fields_are_indexed_as_percentages(self):
        texts = {text for text, spec in citer().figure_entries() if spec is not None}
        self.assertIn(f"{416161.0:.6f}", texts)
        self.assertTrue(any(t.startswith("31.97") and t.endswith("%") for t in texts))
        self.assertFalse(any(t.startswith("31.97") and not t.endswith("%") for t in texts))

    def test_null_citer_is_silent(self):
        self.assertEqual(NullCiter().row("revenue", []), "")
        self.assertEqual(NullCiter().market("price"), "")
