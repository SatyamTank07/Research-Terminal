"""Unit and integration tests for the deterministic-first Business & Moat Strategist.

The Item 1 pre-fetch, the filing loader and the single LLM call (`_run_judgement_llm`) are
patched, so every test is hermetic. Covered:
1. Registration and prompt rendering.
2. Quote verification (near-verbatim, re-attribution, rejection), moat downgrade / promotion,
   the Wide-durability evidence gate, segment-moat name reconciliation.
3. Product lines reconciled against the disaggregation table (no roll-ups), deterministic
   fill-in for lines the model skipped.
4. Pricing / customer-concentration grounding and the deterministic concentration fallback.
5. Graceful degradation: LLM exception, malformed JSON, Item 1 missing (retrieval fallback),
   no text at all, filing load failure.
6. A DB-backed deterministic smoke when the AAPL FY2025 filing is ingested.
"""

import json
import unittest
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

from app.agents.registry import AgentRegistry
from app.agents.specialized.business_strategist import BusinessMoatOutput, BusinessStrategistAgent
from app.agents.specialized.prompts import render_prompt
from app.agents.tools.business_disclosure_tools import FilingChunk
from app.agents.shared.filing.narrative_search import NarrativeChunkResult
from tests.test_tool_business_disclosure import DISAGGREGATION, INCOME_STATEMENT, STACKED_SEGMENTS

MODULE = "app.agents.specialized.business_strategist.agent_business_strategist"

ITEM_1 = {
    "i1": (
        "Our marketplace connects millions of buyers and sellers, and each additional seller attracts more "
        "buyers to the platform. We operate a global logistics network that lowers our cost to serve each order."
    ),
    "i2": (
        "Competition. Our businesses face intense competition from physical retailers, online marketplaces and "
        "cloud service providers, some of which have greater resources. We believe the principal competitive "
        "factors include selection, price and convenience."
    ),
    "i3": (
        "Our cloud customers build long-running workloads on our infrastructure, and migrating those workloads "
        "to another provider requires significant engineering effort."
    ),
}
MDNA = "No single customer accounted for 10% or more of our net sales in 2025, 2024 or 2023."
PRICING = "We reduced prices on our entry-level subscription plans in 2025 in response to competitive pressure."


def _item1_chunks() -> List[NarrativeChunkResult]:
    return [
        NarrativeChunkResult(
            chunk_id=cid,
            document_id="doc",
            ticker="TEST",
            fiscal_year=2025,
            item="Item 1",
            breadcrumb=f"TEST > 10-K FY2025 > PART I > Item 1: Business ({cid})",
            sub_section="Business",
            content=content,
            chunk_index=i,
        )
        for i, (cid, content) in enumerate(ITEM_1.items())
    ]


def _filing() -> List[FilingChunk]:
    chunks = [
        FilingChunk(chunk_id=cid, chunk_index=i, chunk_type="narrative", item="Item 1", content=content)
        for i, (cid, content) in enumerate(ITEM_1.items())
    ]
    chunks += [
        FilingChunk(chunk_id="m1", chunk_index=10, chunk_type="narrative", item="Item 7", breadcrumb="MD&A", content=MDNA),
        FilingChunk(chunk_id="m2", chunk_index=11, chunk_type="narrative", item="Item 7", breadcrumb="MD&A", content=PRICING),
        FilingChunk(chunk_id="t1", chunk_index=20, chunk_type="table", item="Item 8", content=INCOME_STATEMENT),
        FilingChunk(chunk_id="n2", chunk_index=21, chunk_type="narrative", item="Item 8", content="Segment information is as follows:"),
        FilingChunk(chunk_id="t3", chunk_index=22, chunk_type="table", item="Item 8", breadcrumb="Note 10", content=STACKED_SEGMENTS),
        FilingChunk(chunk_id="n4", chunk_index=23, chunk_type="narrative", item="Item 8", content="Net sales by groups of similar products and services:"),
        FilingChunk(chunk_id="t5", chunk_index=24, chunk_type="table", item="Item 8", breadcrumb="Note 2", content=DISAGGREGATION),
    ]
    return chunks


def _ev(chunk_id: str, quote: str) -> List[Dict[str, str]]:
    return [{"chunk_id": chunk_id, "quote": quote}]


def _llm_payload(**overrides: Any) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "business_summary": "A marketplace and cloud operator.",
        "revenue_architecture": "Product sales, seller fees and cloud usage fees.",
        "segment_details": [
            {"name": "Online stores", "description": "First-party retail sales.", "growth_drivers": ["Selection"]},
            {"name": "Retail", "description": "Rolled-up retail bucket.", "growth_drivers": []},
        ],
        "primary_moat": {
            "type": "Network Effects",
            "rationale": "Each seller attracts more buyers.",
            "evidence": _ev("i1", "each additional seller attracts more buyers to the platform"),
        },
        "secondary_moats": [
            {"type": "Cost Advantage", "rationale": "Logistics scale.",
             "evidence": _ev("i1", "We operate a global logistics network that lowers our cost to serve each order.")},
        ],
        "segment_moats": [
            {"segment_name": "Cloud segment", "type": "High Switching Costs", "rationale": "Workload migration is costly.",
             "evidence": _ev("i3", "migrating those workloads to another provider requires significant engineering effort")},
            {"segment_name": "Advertising", "type": "Network Effects", "rationale": "Invented segment.",
             "evidence": _ev("i1", "each additional seller attracts more buyers to the platform")},
        ],
        "moat_durability": "Wide",
        "moat_trajectory": "Stable",
        "competitive_threats": [
            {"threat": "Physical and online retailers compete on selection and price.",
             "evidence": _ev("i2", "intense competition from physical retailers, online marketplaces and cloud service providers")},
            {"threat": "Fabricated threat.", "evidence": _ev("i2", "a regulator has banned our cloud business entirely")},
        ],
        "pricing_power": {
            "assessment": "The company cut entry-level subscription prices under competitive pressure.",
            "evidence": _ev("m2", "We reduced prices on our entry-level subscription plans in 2025"),
        },
        "customer_concentration": {
            "disclosure": "No single customer accounted for 10% or more of net sales.",
            "evidence": _ev("m1", "No single customer accounted for 10% or more of our net sales"),
        },
    }
    payload.update(overrides)
    return payload


def _run(
    payload: Any = None,
    side_effect: Optional[Exception] = None,
    located_by: str = "item_label",
    item1: Optional[List[NarrativeChunkResult]] = None,
    filing: Optional[List[FilingChunk]] = None,
    rag: Optional[List[NarrativeChunkResult]] = None,
):
    """Runs analyze() with the pre-fetch, loader and LLM patched. Returns (output, llm_mock, rag_mock)."""
    item1 = _item1_chunks() if item1 is None else item1
    filing = _filing() if filing is None else filing
    if side_effect is not None:
        llm = MagicMock(side_effect=side_effect)
    else:
        content = payload if isinstance(payload, str) else json.dumps(payload if payload is not None else _llm_payload())
        llm = MagicMock(return_value=content)
    rag_mock = MagicMock(return_value=rag or [])
    with patch(f"{MODULE}.locate_section_chunks", return_value=(item1, located_by if item1 else "not_found")), \
            patch(f"{MODULE}.load_filing_chunks", return_value=filing), \
            patch(f"{MODULE}.retrieve_10k_narrative", rag_mock), \
            patch.object(BusinessStrategistAgent, "_run_judgement_llm", llm):
        out = BusinessStrategistAgent().analyze("test", 2025)
    return out, llm, rag_mock


class TestRegistrationAndPrompts(unittest.TestCase):
    def test_registry_and_tool_free(self):
        agent = AgentRegistry.get("business_strategist")
        self.assertIsInstance(agent, BusinessStrategistAgent)
        self.assertEqual(agent.get_tools(), [])

    def test_prompts_render(self):
        system = render_prompt("business_strategist")
        self.assertIn("Senior Equity Research Analyst", system)
        self.assertIn('"primary_moat"', system)
        self.assertIn("Wide", system)
        query = render_prompt(
            "prompt_business_strategist_query.j2",
            ticker="test", fiscal_year=2025, coverage_mode="full_section", section_context="[chunk:i1] text",
            disclosures=[], segment_basis="business", segment_names=["Retail"], line_names=["Online stores"],
            segment_table="SEG", revenue_mix="MIX",
        )
        self.assertIn("TEST", query)
        self.assertIn("[chunk:i1] text", query)
        self.assertIn("Online stores", query)


class TestEvidenceEnforcement(unittest.TestCase):
    def test_verified_run(self):
        out, llm, _ = _run()
        llm.assert_called_once()
        self.assertIsInstance(out, BusinessMoatOutput)
        self.assertEqual(out.ticker, "TEST")
        self.assertEqual(out.economic_moat_type, "Network Effects")
        self.assertEqual(out.moat_assessment, "verified")
        self.assertEqual(out.secondary_moat_types, ["Cost Advantage"])
        self.assertEqual(out.moat_durability, "Wide")  # two distinct verified moat quotes
        self.assertEqual([m.segment_name for m in out.segment_specific_moats], ["Cloud"])  # canonical name
        self.assertEqual(out.competitive_threats, ["Physical and online retailers compete on selection and price."])
        self.assertTrue(out.pricing_power_assessment.startswith("The company cut"))
        self.assertEqual(out.customer_concentration, "No single customer accounted for 10% or more of net sales.")
        dq = out.data_quality
        self.assertEqual(dq.llm_status, "ok")
        self.assertEqual(dq.quotes_rejected, 1)  # the fabricated threat (the invented segment is dropped by name first)
        self.assertEqual(dq.claims_dropped, 2)
        self.assertTrue(any("'Advertising' dropped" in n for n in dq.notes))
        self.assertEqual(dq.section_located_by, "item_label")
        self.assertEqual(dq.coverage_mode, "full_section")
        self.assertEqual(dq.supplementary_passages, 2)
        # Citations: evidence chunks with true items, plus the segment and revenue tables.
        items = {c["chunk_id"]: c.get("item") for c in out.citations}
        self.assertEqual(items["i1"], "Item 1")
        self.assertEqual(items["m1"], "Item 7")
        self.assertEqual(items["t3"], "Item 8")
        self.assertEqual(items["t5"], "Item 8")
        self.assertTrue(all(e.item for e in out.evidence))

    def test_reconciled_product_lines_and_segments(self):
        out, _, _ = _run()
        self.assertEqual(out.reportable_segments, ["Retail", "Cloud"])
        self.assertEqual(out.primary_product_segments, ["Online stores", "Seller services", "Cloud services"])
        details = {d.name: d for d in out.segment_details}
        self.assertEqual(list(details), ["Online stores", "Seller services", "Cloud services"])
        self.assertEqual(details["Online stores"].description, "First-party retail sales.")
        self.assertIn("$300M", details["Seller services"].description)  # deterministic fill-in
        self.assertIn("30.0% of revenue", details["Seller services"].description)
        self.assertTrue(any("not in the extracted revenue table" in n for n in out.data_quality.notes))
        self.assertIn("| Cloud | 200 |", out.segment_table_markdown)
        self.assertIn("| Online stores | 500 |", out.revenue_mix_markdown)
        self.assertEqual(len(out.segment_financials), 4)
        self.assertEqual(len(out.revenue_lines), 6)

    def test_near_verbatim_and_reattributed_quotes(self):
        payload = _llm_payload(
            primary_moat={
                "type": "Network Effects",
                "rationale": "x",
                # curly apostrophe-free variant with extra whitespace on the right chunk
                "evidence": _ev("i1", "each  additional   seller attracts more buyers to the platform"),
            },
            secondary_moats=[{
                "type": "High Switching Costs", "rationale": "y",
                # right quote, wrong chunk id: re-attributed to i3
                "evidence": _ev("i2", "migrating those workloads to another provider requires significant engineering effort"),
            }],
        )
        out, _, _ = _run(payload)
        self.assertEqual(out.economic_moat_type, "Network Effects")
        sec = [e for e in out.evidence if e.claim == "secondary_moat"]
        self.assertEqual(sec[0].chunk_id, "i3")

    def test_wide_capped_without_two_quotes(self):
        out, _, _ = _run(_llm_payload(secondary_moats=[]))
        self.assertEqual(out.moat_durability, "Narrow")
        self.assertTrue(any("capped at Narrow" in n for n in out.data_quality.notes))

    def test_unsupported_primary_promotes_verified_secondary(self):
        payload = _llm_payload(primary_moat={"type": "Intangible Assets / Brand", "rationale": "brand",
                                             "evidence": _ev("i1", "our brand is the most recognized in the world")})
        out, _, _ = _run(payload)
        self.assertEqual(out.economic_moat_type, "Cost Advantage")
        self.assertEqual(out.secondary_moat_types, [])
        self.assertTrue(any("promoted to primary" in n for n in out.data_quality.notes))

    def test_unsupported_moat_downgraded(self):
        payload = _llm_payload(
            primary_moat={"type": "Intangible Assets / Brand", "rationale": "brand recognition",
                          "evidence": _ev("i1", "our brand is the most recognized in the world")},
            secondary_moats=[{"type": "Efficient Scale", "rationale": "z", "evidence": []}],
            moat_durability="Wide",
            moat_trajectory="Expanding",
        )
        out, _, _ = _run(payload)
        self.assertEqual(out.economic_moat_type, "None")
        self.assertEqual(out.moat_durability, "Indeterminate")
        self.assertEqual(out.moat_trajectory, "Indeterminate")
        self.assertEqual(out.moat_assessment, "unsupported")
        self.assertTrue(out.moat_rationale.startswith("Not assessed"))
        self.assertFalse([e for e in out.evidence if e.claim in ("primary_moat", "secondary_moat")])

    def test_explicit_no_moat(self):
        payload = _llm_payload(primary_moat={"type": "None", "rationale": "Commodity provider.", "evidence": []},
                               secondary_moats=[], moat_durability="Wide")
        out, _, _ = _run(payload)
        self.assertEqual(out.economic_moat_type, "None")
        self.assertEqual(out.moat_assessment, "no_moat")
        self.assertEqual(out.moat_durability, "None")

    def test_real_but_off_topic_quotes_do_not_support_claims(self):
        payload = _llm_payload(
            primary_moat={"type": "Network Effects", "rationale": "Competitors everywhere.",
                          "evidence": _ev("i2", "intense competition from physical retailers, online marketplaces and cloud service providers")},
            secondary_moats=[],
            pricing_power={"assessment": "The company maintains pricing power.",
                           "evidence": _ev("m2", "We reduced prices on our entry-level subscription plans in 2025")},
        )
        out, _, _ = _run(payload)
        self.assertEqual(out.moat_assessment, "unsupported")
        self.assertEqual(out.economic_moat_type, "None")
        self.assertGreaterEqual(out.data_quality.quotes_off_topic, 1)
        self.assertTrue(out.pricing_power_assessment.startswith('Filing states: "We reduced prices'))
        self.assertTrue(any("not evidenced by its quotes" in n for n in out.data_quality.notes))

    def test_pricing_figures_must_be_in_quotes(self):
        payload = _llm_payload(pricing_power={
            "assessment": "Prices were cut by 15% across all plans.",
            "evidence": _ev("m2", "We reduced prices on our entry-level subscription plans in 2025"),
        })
        out, _, _ = _run(payload)
        self.assertTrue(out.pricing_power_assessment.startswith('Filing states: "We reduced prices'))

    def test_unverified_pricing_becomes_not_disclosed(self):
        payload = _llm_payload(pricing_power={"assessment": "Strong pricing power.", "evidence": _ev("i2", "we raise prices every year at will")})
        out, _, _ = _run(payload)
        self.assertEqual(out.pricing_power_assessment, "Not explicitly disclosed in 10-K")

    def test_concentration_deterministic_override(self):
        payload = _llm_payload(customer_concentration={"disclosure": "Not explicitly disclosed in 10-K", "evidence": []})
        out, _, _ = _run(payload)
        self.assertEqual(out.customer_concentration, f'Filing states: "{MDNA}"')
        self.assertTrue(any(e.claim == "customer_concentration" and e.chunk_id == "m1" for e in out.evidence))

    def test_malformed_fields_tolerated(self):
        payload = _llm_payload(
            secondary_moats="Network Effects",
            segment_moats=[None, {"segment_name": "Cloud", "type": "Moat-ish", "evidence": "x"}],
            competitive_threats=[{"threat": "", "evidence": None}, "string"],
            segment_details="none",
            primary_moat={"type": "network effect", "rationale": None, "evidence": {"chunk_id": "i1", "quote": "each additional seller attracts more buyers to the platform"}},
        )
        out, _, _ = _run(payload)
        self.assertEqual(out.economic_moat_type, "Network Effects")
        self.assertEqual(out.segment_specific_moats, [])
        self.assertEqual(out.competitive_threats, [])
        self.assertEqual(len(out.segment_details), 3)  # filled deterministically from revenue lines


class TestGracefulDegradation(unittest.TestCase):
    def test_llm_exception(self):
        out, _, _ = _run(side_effect=RuntimeError("rate limit"))
        dq = out.data_quality
        self.assertEqual(dq.llm_status, "failed")
        self.assertEqual(out.moat_assessment, "not_assessed")
        self.assertEqual(out.economic_moat_type, "None")
        self.assertEqual(out.moat_durability, "Indeterminate")
        self.assertTrue(out.business_summary.startswith("Not assessed"))
        self.assertTrue(out.pricing_power_assessment.startswith("Not assessed"))
        self.assertEqual(out.customer_concentration, f'Filing states: "{MDNA}"')  # deterministic screen still applies
        self.assertIn("Online stores 50.0%", out.revenue_architecture)
        self.assertEqual(out.primary_product_segments, ["Online stores", "Seller services", "Cloud services"])
        self.assertEqual(out.reportable_segments, ["Retail", "Cloud"])
        self.assertTrue(out.segment_financials and out.revenue_lines)
        self.assertTrue(any("LLM call failed" in n for n in dq.notes))

    def test_malformed_json(self):
        out, _, _ = _run(payload="I cannot comply with this request.")
        self.assertEqual(out.data_quality.llm_status, "malformed")
        self.assertEqual(out.moat_assessment, "not_assessed")
        self.assertTrue(out.segment_financials)

    def test_item1_missing_uses_retrieval_fallback(self):
        rag_chunk = _item1_chunks()[0].model_copy(update={"item": "Cover Page"})
        out, llm, rag = _run(item1=[], rag=[rag_chunk])
        self.assertTrue(rag.called)
        self.assertEqual(out.data_quality.coverage_mode, "rag_fallback")
        self.assertEqual(out.data_quality.section_located_by, "not_found")
        llm.assert_called_once()

    def test_no_text_at_all_skips_llm(self):
        tables_only = [c for c in _filing() if c.chunk_type == "table"]
        out, llm, _ = _run(item1=[], filing=tables_only, rag=[])
        llm.assert_not_called()
        self.assertEqual(out.data_quality.coverage_mode, "no_context")
        self.assertEqual(out.data_quality.llm_status, "skipped")
        self.assertTrue(out.segment_financials)
        self.assertEqual(out.customer_concentration, "Not assessed: no filing text was available.")

    def test_filing_load_failure_still_returns(self):
        with patch(f"{MODULE}.locate_section_chunks", return_value=(_item1_chunks(), "item_label")), \
                patch(f"{MODULE}.load_filing_chunks", side_effect=RuntimeError("db down")), \
                patch.object(BusinessStrategistAgent, "_run_judgement_llm", return_value=json.dumps(_llm_payload())):
            out = BusinessStrategistAgent().analyze("TEST", 2025)
        self.assertEqual(out.segment_financials, [])
        self.assertEqual(out.economic_moat_type, "Network Effects")
        self.assertTrue(any("could not be loaded" in n for n in out.data_quality.notes))

    def test_heading_boundary_uses_inferred_items(self):
        filing = [
            FilingChunk(chunk_id="h1", chunk_index=0, chunk_type="table", item="Cover Page", content="| Item 1. | Business |"),
        ] + [c.model_copy(update={"item": "Cover Page"}) for c in _filing() if c.chunk_type == "narrative" and c.chunk_id.startswith("i")] + [
            FilingChunk(chunk_id="h7", chunk_index=9, chunk_type="table", item="Cover Page", content="| Item 7. | Management's Discussion and Analysis |"),
            FilingChunk(chunk_id="m1", chunk_index=10, chunk_type="narrative", item="Cover Page", content=MDNA),
        ]
        payload = _llm_payload(pricing_power={"assessment": "Not explicitly disclosed in 10-K", "evidence": []})
        out, _, _ = _run(payload, located_by="heading_boundary", filing=filing)
        items = {c["chunk_id"]: c.get("item") for c in out.citations}
        self.assertEqual(items["m1"], "Item 7")  # true item, not the stored "Cover Page"
        self.assertTrue(any("section headings" in n for n in out.data_quality.notes))


class TestRealFilingSmoke(unittest.TestCase):
    """Deterministic (no LLM) run on the ingested AAPL FY2025 10-K, skipped when not ingested."""

    def test_aapl_segments_and_revenue_lines(self):
        from app.agents.tools.business_disclosure_tools import load_filing_chunks

        try:
            chunks = load_filing_chunks("AAPL", 2025)
        except Exception as e:  # DB not reachable
            self.skipTest(f"database unavailable: {e}")
        if not chunks:
            self.skipTest("AAPL FY2025 not ingested")
        with patch.object(BusinessStrategistAgent, "_run_judgement_llm", side_effect=RuntimeError("no LLM in tests")):
            out = BusinessStrategistAgent().analyze("AAPL", 2025)
        self.assertEqual(out.data_quality.section_located_by, "item_label")
        self.assertEqual(out.data_quality.segment_basis, "geographic")
        self.assertIn("iPhone", out.primary_product_segments)
        self.assertIn("Services", out.primary_product_segments)
        self.assertGreaterEqual(len(out.reportable_segments), 2)


if __name__ == "__main__":
    unittest.main()
