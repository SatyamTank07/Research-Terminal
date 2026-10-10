"""Unit and Integration Tests for the evidence-first Risk & Red Flag Analyst Agent.

Validates (with the LLM mocked and Item 1A pre-fetch patched):
1. Registration, prompt rendering, and the full-section / retrieval-fallback query templates.
2. Deterministic enforcement on an NVDA-shaped regression run:
   - quote verification against chunk text, dropping risks with no verified quote;
   - citations carrying the true filing section of each chunk (Item 15, not "Item 3");
   - the materiality gate demoting a matter management calls immaterial;
   - primary-threat re-pointing to the most severe verified risk;
   - duplicate merging, unsupported figures / mitigations removed, boilerplate capped;
   - likelihood x impact severity, ranking, R1..Rn renumbering and the overall profile.
3. Year-over-year disclosure-change classification against a prior-year Item 1A.
4. The 8-risk cap, the <5 note, and the empty-output failure path.
5. DB-backed smoke tests when the AAPL filings are ingested.
"""

import json
import unittest
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, ToolMessage

from app.agents.registry import AgentRegistry
from app.agents.specialized.prompts import render_prompt
from app.agents.specialized.risk_analyst import RiskAnalystAgent
from app.agents.state import RiskAuditOutput, RiskEvidence, RiskItem
from app.agents.tools.rag_narrative_tools import NarrativeChunkResult, retrieve_10k_narrative_tool
from app.agents.tools.risk_disclosure_tools import diff_risk_disclosures, fetch_section_chunks, locate_section_chunks

LOCATE_PATH = "app.agents.specialized.risk_analyst.agent_risk_analyst.locate_section_chunks"


def _as_locator(fetch, located_by: str = "item_label"):
    """Adapts a (ticker, fiscal_year, item) -> chunks stub to locate_section_chunks' (chunks, located_by)."""
    def locate(ticker, fiscal_year, item="Item 1A"):
        chunks = fetch(ticker, fiscal_year, item)
        return chunks, (located_by if chunks else "not_found")
    return locate

ITEM_1A = {
    "a1": (
        "We receive a significant amount of our revenue from a limited number of customers. For fiscal year 2026, "
        "sales to one direct customer represented 22% of total revenue and sales to another direct customer "
        "represented 14% of total revenue."
    ),
    "a2": (
        "New export licensing requirements restrict sales of our products to China. We incurred a $4.5 billion "
        "charge in the first quarter of fiscal year 2026 associated with H20 for excess inventory and purchase obligations."
    ),
    "a3": (
        "Data privacy laws are evolving rapidly across jurisdictions and non-compliance could result in penalties of "
        "up to 4% of worldwide revenue. Cybersecurity incidents could compromise our systems and the data of our customers."
    ),
    "a4": (
        "We rely on third-party foundries, primarily located in Taiwan, to manufacture our semiconductor wafers. "
        "We have entered into prepaid manufacturing and capacity agreements to secure supply."
    ),
    "a5": (
        "Our target markets remain competitive, and competition may intensify as customers develop their own "
        "in-house accelerators and custom silicon."
    ),
    "a6": "Adverse economic conditions, including inflation and market volatility, may harm our business and results.",
}
LEGAL_ITEM_15 = {
    "chunk_id": "l1",
    "ticker": "TEST",
    "fiscal_year": 2026,
    "item": "Item 15",
    "breadcrumb": "TEST CORP (TEST) > 10-K FY2026 > PART IV > Item 15: Exhibits and Financial Statement Schedules",
    "sub_section": "Exhibits and Financial Statement Schedules",
    "content": (
        "A putative securities class action lawsuit alleges misleading statements regarding channel inventory and the "
        "impact of cryptocurrency mining. We believe the ultimate outcome will not have a material adverse effect on "
        "our operating results, liquidity, or financial position."
    ),
    "chunk_index": 237,
}


def _chunk(chunk_id: str, content: str, idx: int, fy: int = 2026, item: str = "Item 1A") -> NarrativeChunkResult:
    return NarrativeChunkResult(
        chunk_id=chunk_id,
        document_id=f"doc-{fy}",
        ticker="TEST",
        fiscal_year=fy,
        item=item,
        breadcrumb=f"TEST CORP (TEST) > 10-K FY{fy} > PART I > {item}: Risk Factors",
        sub_section="Risk Factors",
        content=content,
        chunk_index=idx,
    )


def _item1a(fy: int = 2026) -> List[NarrativeChunkResult]:
    return [_chunk(cid, text, i, fy) for i, (cid, text) in enumerate(ITEM_1A.items())]


def _risk(rid: str, category: str, title: str, likelihood: str, impact: str, evidence: List[Dict[str, str]], **extra) -> Dict[str, Any]:
    risk = {
        "risk_id": rid,
        "risk_category": category,
        "risk_title": title,
        "risk_summary": f"{title} as disclosed in the filing, with a company-specific effect on results.",
        "likelihood": likelihood,
        "impact": impact,
        "financial_transmission": ["Revenue"],
        "evidence": evidence,
        "monitoring_signposts": ["Track the next filing"],
    }
    risk.update(extra)
    return risk


def _nvda_shaped_payload() -> Dict[str, Any]:
    return {
        "ticker": "TEST",
        "fiscal_year": 2026,
        "identified_risks": [
            _risk("R1", "Regulatory & Legal", "Ongoing Securities Class Action and Derivative Lawsuits", "High", "High",
                  [{"chunk_id": "l1", "quote": "A putative securities class action lawsuit alleges misleading statements regarding channel inventory"}],
                  mitigating_factors="Management believes the outcome will not be material.",
                  mitigation_evidence={"chunk_id": "l1", "quote": "We believe the ultimate outcome will not have a material adverse effect on our operating results"}),
            _risk("R2", "Supply Chain & Concentration", "Direct Customer Revenue Concentration", "High", "High",
                  [{"chunk_id": "a1", "quote": "sales to one direct customer represented 22% of total revenue and sales to another direct customer represented 14% of total revenue"}],
                  quantified_exposure="Two direct customers at 22% and 14% of total revenue"),
            _risk("R3", "Regulatory & Legal", "Export Licensing Restrictions on China Sales", "High", "High",
                  [{"chunk_id": "a2", "quote": "We incurred a $4.5 billion charge in the first quarter of fiscal year 2026 associated with H20"}],
                  quantified_exposure="$4.5 billion H20 charge"),
            _risk("R4", "Technological & Cybersecurity", "Cybersecurity Threats and Data Privacy Compliance Risks", "Medium", "Medium",
                  [{"chunk_id": "a3", "quote": "Cybersecurity incidents could compromise our systems and the data of our customers."}],
                  mitigating_factors="The company has implemented comprehensive cybersecurity measures."),
            _risk("R5", "Regulatory & Legal", "Changing Data Privacy Laws and Compliance Costs", "Medium", "Medium",
                  [{"chunk_id": "a3", "quote": "Data privacy laws are evolving rapidly across jurisdictions"}],
                  quantified_exposure="Penalties of up to 4% of worldwide revenue"),
            _risk("R6", "Supply Chain & Concentration", "Third-Party Foundry Dependence in Taiwan", "High", "Medium",
                  [{"chunk_id": "a4", "quote": "We rely on third-party foundries, primarily located in Taiwan, to manufacture our semiconductor wafers."}],
                  quantified_exposure="35% of wafers from one foundry",
                  mitigating_factors="Prepaid manufacturing and capacity agreements.",
                  mitigation_evidence={"chunk_id": "a4", "quote": "We have entered into prepaid manufacturing and capacity agreements to secure supply."}),
            _risk("R7", "Macroeconomic & Geopolitical", "Economic Conditions and Market Volatility", "Medium", "Medium",
                  [{"chunk_id": "a6", "quote": "Adverse economic conditions, including inflation and market volatility, may harm our business"}]),
            _risk("R8", "Competitive & Demand", "Hyperscaler Custom Silicon Displacement", "Medium", "High",
                  [{"chunk_id": "a5", "quote": "customers are expected to replace all of our accelerators with custom silicon by 2027"}]),
        ],
        "primary_threat_risk_id": "R1",
        "primary_existential_threat": "Securities litigation could create substantial liabilities.",
    }


def _llm_result(payload: Optional[Dict[str, Any]], tool_chunks: Optional[List[Dict[str, Any]]] = None, raw: Optional[str] = None) -> Dict[str, Any]:
    messages: List[Any] = []
    if tool_chunks:
        messages.append(ToolMessage(name="retrieve_10k_narrative_tool", content=json.dumps(tool_chunks), tool_call_id="call_legal_1"))
    messages.append(AIMessage(content=raw if raw is not None else json.dumps(payload)))
    return {"messages": messages}


EMPTY_REPAIR = '{"identified_risks": []}'


def _run_agent(result: Dict[str, Any], fetch_side_effect, available_fiscal_years=None, fiscal_year: int = 2026,
               repair_response: Any = EMPTY_REPAIR, return_repair_mock: bool = False, located_by: str = "item_label"):
    """Runs analyze() with section location, the ReAct agent and the coverage-repair completion mocked."""
    agent = RiskAnalystAgent()
    mock_active = MagicMock()
    mock_active.invoke.return_value = result
    repair_kwargs = {"side_effect": repair_response} if isinstance(repair_response, Exception) else {"return_value": repair_response}
    with patch(LOCATE_PATH, side_effect=_as_locator(fetch_side_effect, located_by)), \
            patch.object(RiskAnalystAgent, "_get_or_create_agent", return_value=mock_active), \
            patch.object(RiskAnalystAgent, "_run_repair_llm", **repair_kwargs) as repair_mock:
        output = agent.analyze("test", fiscal_year, available_fiscal_years=available_fiscal_years)
    query = mock_active.invoke.call_args[0][0]["messages"][0]["content"]
    if return_repair_mock:
        return output, query, repair_mock
    return output, query


def _current_only(ticker, fiscal_year, item="Item 1A"):
    return _item1a(fiscal_year) if fiscal_year == 2026 else []


class TestRiskAnalystSetup(unittest.TestCase):
    def test_registration_and_system_prompt(self):
        agent = AgentRegistry.get("risk_analyst")
        self.assertIsInstance(agent, RiskAnalystAgent)
        self.assertEqual(agent.model_name, "openai:gpt-4o-mini")

        prompt = render_prompt("risk_analyst")
        for needle in ("Chief Risk Officer", "retrieve_10k_narrative_tool", "RiskAuditOutput", "BOILERPLATE FILTER",
                       "character-for-character", "primary_threat_risk_id", "Competitive & Demand"):
            self.assertIn(needle, prompt)
        self.assertNotIn("digital markets act", prompt.lower())

    def test_full_section_query_embeds_item_1a(self):
        output, query = _run_agent(_llm_result(_nvda_shaped_payload(), [LEGAL_ITEM_15]), _current_only, [2026])
        self.assertIn("ITEM 1A — RISK FACTORS (FY2026, complete section)", query)
        self.assertIn("[chunk:a1] We receive a significant amount of our revenue", query)
        self.assertIn("no prior-year Item 1A available", query)
        self.assertNotIn("RETRIEVAL FALLBACK", query)
        self.assertEqual(output.data_quality.coverage_mode, "full_section")
        self.assertEqual(output.data_quality.item1a_chunks, 6)


class TestRiskAnalystEnforcement(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.output, _ = _run_agent(_llm_result(_nvda_shaped_payload(), [LEGAL_ITEM_15]), _current_only, [2026])
        cls.by_title = {r.risk_title: r for r in cls.output.identified_risks}

    def test_unverified_risk_dropped_and_counted(self):
        self.assertNotIn("Hyperscaler Custom Silicon Displacement", self.by_title)
        dq = self.output.data_quality
        self.assertEqual(dq.risks_proposed, 8)
        self.assertEqual(dq.risks_dropped_unverified, 1)
        self.assertEqual(dq.quotes_rejected, 1)
        for risk in self.output.identified_risks:
            self.assertTrue(risk.evidence)
            self.assertTrue(all(e.verified for e in risk.evidence))

    def test_materiality_gate_demotes_lawsuit(self):
        lawsuit = self.by_title["Ongoing Securities Class Action and Derivative Lawsuits"]
        self.assertEqual(lawsuit.impact, "Medium")
        self.assertEqual(lawsuit.severity, "Moderate")
        self.assertTrue(any("not expected to be material" in a for a in lawsuit.adjustments))
        self.assertEqual(lawsuit.evidence[0].item, "Item 15")
        self.assertIsNotNone(lawsuit.mitigation_evidence)

    def test_primary_threat_repointed_to_most_severe(self):
        out = self.output
        self.assertEqual(out.primary_threat_risk_id, "R1")
        primary = out.identified_risks[0]
        self.assertEqual(primary.risk_title, "Direct Customer Revenue Concentration")
        self.assertEqual(primary.severity, "Severe")
        self.assertIn("Direct Customer Revenue Concentration", out.primary_existential_threat)
        self.assertTrue(any("re-pointed" in n for n in out.data_quality.notes))

    def test_duplicates_merged(self):
        self.assertEqual(self.output.data_quality.risks_merged, 1)
        titles = [r.risk_title for r in self.output.identified_risks]
        self.assertEqual(sum("Data Privacy" in t for t in titles), 1)
        merged = next(r for r in self.output.identified_risks if "Data Privacy" in r.risk_title)
        self.assertEqual(len(merged.evidence), 2)
        self.assertTrue(any("Merged duplicate risk" in a for a in merged.adjustments))
        self.assertEqual(merged.quantified_exposure, "Penalties of up to 4% of worldwide revenue")

    def test_unsupported_mitigation_and_exposure_removed(self):
        merged = next(r for r in self.output.identified_risks if "Data Privacy" in r.risk_title)
        self.assertIsNone(merged.mitigating_factors)
        foundry = self.by_title["Third-Party Foundry Dependence in Taiwan"]
        self.assertIsNone(foundry.quantified_exposure)
        self.assertEqual(foundry.mitigating_factors, "Prepaid manufacturing and capacity agreements.")
        self.assertTrue(foundry.mitigation_evidence.verified)
        concentration = self.by_title["Direct Customer Revenue Concentration"]
        self.assertEqual(concentration.quantified_exposure, "Two direct customers at 22% and 14% of total revenue")

    def test_boilerplate_capped(self):
        macro = self.by_title["Economic Conditions and Market Volatility"]
        self.assertEqual(macro.impact, "Low")
        self.assertEqual(macro.severity, "Low")

    def test_ranking_numbering_and_profile(self):
        risks = self.output.identified_risks
        self.assertEqual(len(risks), 6)
        self.assertEqual([r.risk_id for r in risks], [f"R{i}" for i in range(1, 7)])
        ranks = [{"Severe": 0, "Moderate": 1, "Low": 2}[r.severity] for r in risks]
        self.assertEqual(ranks, sorted(ranks))
        self.assertEqual([r.severity for r in risks[:2]], ["Severe", "Severe"])
        self.assertEqual(self.output.overall_risk_profile, "High")
        self.assertTrue(all(r.disclosure_change == "Not Assessed" for r in risks))
        self.assertIsNone(self.output.disclosure_changes)

    def test_quantified_checklist_fully_covered(self):
        dq = self.output.data_quality
        self.assertEqual(dq.quantified_disclosures, 3)
        self.assertEqual(dq.quantified_uncovered, [])
        self.assertFalse(any("quantified Item 1A disclosures" in n for n in dq.notes))

    def test_citations_carry_true_items(self):
        items = {c["chunk_id"]: c["item"] for c in self.output.citations}
        self.assertEqual(items["l1"], "Item 15")
        self.assertEqual(items["a1"], "Item 1A")
        self.assertNotIn("a5", items)
        self.assertIn("| `l1` | Item 15 |", self.output.evidence_table_markdown)

    def test_risk_matrix_rendered(self):
        matrix = self.output.risk_matrix_markdown
        self.assertIn("| R1 | Direct Customer Revenue Concentration | Supply Chain & Concentration | High | High | **Severe** |", matrix)
        self.assertEqual(matrix.count("\n| R"), 6)


class TestRiskAnalystDisclosureChange(unittest.TestCase):
    def test_new_expanded_and_unchanged_against_prior_year(self):
        prior = [
            _chunk("p1", ITEM_1A["a1"], 0, 2025),
            _chunk("p2", ITEM_1A["a4"].split(". We have")[0] + ".", 1, 2025),
        ]

        def fetch(ticker, fiscal_year, item="Item 1A"):
            return {2026: _item1a(2026), 2025: prior}.get(fiscal_year, [])

        payload = {
            "identified_risks": [
                _risk("R1", "Supply Chain & Concentration", "Direct Customer Revenue Concentration", "High", "High",
                      [{"chunk_id": "a1", "quote": "sales to one direct customer represented 22% of total revenue"}]),
                _risk("R2", "Regulatory & Legal", "Export Licensing Restrictions on China Sales", "High", "High",
                      [{"chunk_id": "a2", "quote": "We incurred a $4.5 billion charge in the first quarter of fiscal year 2026 associated with H20"}]),
                _risk("R3", "Supply Chain & Concentration", "Third-Party Foundry Dependence in Taiwan", "Medium", "Medium",
                      [{"chunk_id": "a4", "quote": "We rely on third-party foundries, primarily located in Taiwan, to manufacture our semiconductor wafers."},
                       {"chunk_id": "a4", "quote": "We have entered into prepaid manufacturing and capacity agreements to secure supply."}]),
            ],
            "primary_threat_risk_id": "R2",
            "primary_existential_threat": "Export licensing could close the China market.",
        }
        output, query = _run_agent(_llm_result(payload), fetch, [2026, 2025])

        self.assertIn("DISCLOSURE CHANGES vs FY2025 Item 1A", query)
        changes = {r.risk_title: r.disclosure_change for r in output.identified_risks}
        self.assertEqual(changes["Direct Customer Revenue Concentration"], "Unchanged")
        self.assertEqual(changes["Export Licensing Restrictions on China Sales"], "New")
        self.assertEqual(changes["Third-Party Foundry Dependence in Taiwan"], "Expanded")
        self.assertEqual(output.disclosure_changes.prior_fiscal_year, 2025)
        self.assertEqual(output.data_quality.prior_fiscal_year, 2025)
        # A valid choice among equally severe risks is respected.
        self.assertEqual(output.identified_risks[0].risk_title, "Direct Customer Revenue Concentration")
        chosen = next(r for r in output.identified_risks if r.risk_id == output.primary_threat_risk_id)
        self.assertEqual(chosen.risk_title, "Export Licensing Restrictions on China Sales")
        self.assertEqual(output.primary_existential_threat, "Export licensing could close the China market.")
        self.assertTrue(any("Only 3 risks" in n for n in output.data_quality.notes))
        # The data-privacy penalty figure (a3) is on the checklist but no risk cites it.
        self.assertEqual(output.data_quality.quantified_disclosures, 3)
        self.assertEqual(len(output.data_quality.quantified_uncovered), 1)
        self.assertIn("4% of worldwide revenue", output.data_quality.quantified_uncovered[0])
        self.assertTrue(any("1 of 3 quantified Item 1A disclosures" in n for n in output.data_quality.notes))

    def test_missing_prior_year_noted(self):
        output, _ = _run_agent(_llm_result({"identified_risks": []}), _current_only, [2026, 2025])
        self.assertIsNone(output.disclosure_changes)
        self.assertTrue(any("FY2025" in n and "not assessed" in n for n in output.data_quality.notes))


class TestRiskAnalystCoverageRepair(unittest.TestCase):
    """The first pass omits the customer-concentration figure; a focused repair call adds it."""

    FIRST_PASS = {
        "identified_risks": [
            _risk("R1", "Regulatory & Legal", "Export Licensing Restrictions on China Sales", "High", "High",
                  [{"chunk_id": "a2", "quote": "We incurred a $4.5 billion charge in the first quarter of fiscal year 2026 associated with H20"}],
                  quantified_exposure="$4.5 billion charge"),
            _risk("R2", "Technological & Cybersecurity", "Cybersecurity Incidents", "High", "High",
                  [{"chunk_id": "a3", "quote": "Cybersecurity incidents could compromise our systems and the data of our customers."}]),
        ],
        "primary_threat_risk_id": "R1",
        "primary_existential_threat": "Export licensing restricts access to China.",
    }
    REPAIR = json.dumps({"identified_risks": [
        _risk("X1", "Supply Chain & Concentration", "Direct Customer Revenue Concentration", "High", "High",
              [{"chunk_id": "a1", "quote": "sales to one direct customer represented 22% of total revenue and sales to another direct customer represented 14% of total revenue"}],
              quantified_exposure="22% and 14% of total revenue"),
    ]})

    def test_repair_adds_missing_concentration_risk(self):
        output, _, repair = _run_agent(_llm_result(self.FIRST_PASS), _current_only, [2026],
                                       repair_response=self.REPAIR, return_repair_mock=True)
        repair.assert_called_once()
        prompt = repair.call_args[0][0]
        self.assertIn("[chunk:a1] For fiscal year 2026, sales to one direct customer represented 22%", prompt)
        self.assertIn("- Export Licensing Restrictions on China Sales", prompt)
        self.assertNotIn("penalties of up to 4%", prompt.split("Source chunks:")[0])  # statutory maximum is not a target

        titles = [r.risk_title for r in output.identified_risks]
        self.assertIn("Direct Customer Revenue Concentration", titles)
        concentration = next(r for r in output.identified_risks if r.risk_title == "Direct Customer Revenue Concentration")
        self.assertEqual(concentration.severity, "Severe")
        self.assertTrue(concentration.evidence[0].verified)
        self.assertEqual([r.risk_id for r in output.identified_risks], ["R1", "R2", "R3"])
        cyber = next(r for r in output.identified_risks if r.risk_title == "Cybersecurity Incidents")
        self.assertEqual(cyber.adjustments.count(
            "Impact capped at Medium: no filing-stated magnitude or realized effect supports a High impact."), 1)

        dq = output.data_quality
        self.assertEqual(dq.quantified_uncovered, [d for d in dq.quantified_uncovered if "4%" in d])
        self.assertEqual(dq.risks_proposed, 3)
        self.assertTrue(any("Coverage repair pass proposed 1 risk(s) for 1 uncited" in n for n in dq.notes))
        self.assertEqual(sum("re-pointed" in n for n in dq.notes), 0)
        self.assertEqual(output.primary_threat_risk_id,
                         next(r.risk_id for r in output.identified_risks if r.risk_title.startswith("Export")))
        self.assertTrue(any(c["chunk_id"] == "a1" for c in output.citations))

    def test_repair_skipped_when_material_figures_are_covered(self):
        _, _, repair = _run_agent(_llm_result(_nvda_shaped_payload(), [LEGAL_ITEM_15]), _current_only, [2026],
                                  repair_response=self.REPAIR, return_repair_mock=True)
        repair.assert_not_called()

    def test_repair_failure_keeps_first_pass(self):
        output, _ = _run_agent(_llm_result(self.FIRST_PASS), _current_only, [2026], repair_response=RuntimeError("timeout"))
        self.assertEqual(len(output.identified_risks), 2)
        self.assertTrue(any("Coverage repair pass failed" in n for n in output.data_quality.notes))
        self.assertTrue(any("22% of total revenue" in t for t in output.data_quality.quantified_uncovered))

    def test_repair_with_unverifiable_quote_is_dropped(self):
        bad = json.dumps({"identified_risks": [
            _risk("X1", "Supply Chain & Concentration", "Customer Concentration", "High", "High",
                  [{"chunk_id": "a1", "quote": "one customer represented 45% of total revenue in fiscal 2026"}],
                  quantified_exposure="45%"),
        ]})
        output, _ = _run_agent(_llm_result(self.FIRST_PASS), _current_only, [2026], repair_response=bad)
        self.assertNotIn("Customer Concentration", [r.risk_title for r in output.identified_risks])
        self.assertEqual(output.data_quality.risks_dropped_unverified, 1)
        self.assertTrue(any("22% of total revenue" in t for t in output.data_quality.quantified_uncovered))


class TestRiskAnalystSectionLocation(unittest.TestCase):
    def test_item_1a_not_provided_skips_llm(self):
        statement = _chunk("n1", "Item 1A. Risk Factors. Not applicable. As a smaller reporting company, we are not required to provide this information.", 0)
        agent = RiskAnalystAgent()
        mock_active = MagicMock()
        with patch(LOCATE_PATH, return_value=([statement], "heading_boundary")), \
                patch.object(RiskAnalystAgent, "_get_or_create_agent", return_value=mock_active), \
                patch.object(RiskAnalystAgent, "_run_repair_llm") as repair:
            output = agent.analyze("test", 2026, available_fiscal_years=[2026])
        mock_active.invoke.assert_not_called()
        repair.assert_not_called()
        self.assertEqual(output.identified_risks, [])
        self.assertEqual(output.overall_risk_profile, "Not Assessed")
        self.assertIsNone(output.primary_threat_risk_id)
        dq = output.data_quality
        self.assertEqual(dq.coverage_mode, "not_provided")
        self.assertEqual(dq.section_located_by, "heading_boundary")
        self.assertTrue(any("provides no Item 1A risk factors" in n and "Not applicable" in n for n in dq.notes))
        self.assertIn("no risks were rated", output.risk_matrix_markdown)
        self.assertEqual(output.citations[0]["chunk_id"], "n1")

    def test_heading_boundary_location_is_noted(self):
        output, query = _run_agent(_llm_result(_nvda_shaped_payload(), [LEGAL_ITEM_15]), _current_only, [2026],
                                   located_by="heading_boundary")
        self.assertIn("[chunk:a1]", query)
        self.assertEqual(output.data_quality.section_located_by, "heading_boundary")
        self.assertEqual(output.data_quality.coverage_mode, "full_section")
        self.assertTrue(any("stored section labels were wrong" in n for n in output.data_quality.notes))
        self.assertTrue(output.identified_risks)


class TestRiskAnalystEdgeCases(unittest.TestCase):
    def test_exposure_must_be_percentage_or_currency(self):
        chunks = [_chunk("m1", "Our manufacturing lead times can exceed 12 months, which requires us to commit to supply early.", 0)]
        payload = {"identified_risks": [
            _risk("R1", "Operational", "Long Manufacturing Lead Times", "High", "Medium",
                  [{"chunk_id": "m1", "quote": "Our manufacturing lead times can exceed 12 months"}],
                  quantified_exposure="12 months"),
        ]}
        output, _ = _run_agent(_llm_result(payload), lambda t, fy, item="Item 1A": chunks, [2026])
        risk = output.identified_risks[0]
        self.assertIsNone(risk.quantified_exposure)
        self.assertIn("Quantified exposure removed: not stated as a percentage or currency amount.", risk.adjustments)

    def test_rag_fallback_when_item_1a_missing(self):
        tool_chunk = dict(LEGAL_ITEM_15, chunk_id="t1", item="Item 1A",
                          content="Competition may intensify as customers design their own custom silicon accelerators for data centers.")
        payload = {"identified_risks": [_risk("R1", "competition", "Custom Silicon Competition", "Medium", "High",
                                              [{"chunk_id": "t1", "quote": "customers design their own custom silicon accelerators"}])]}
        output, query = _run_agent(_llm_result(payload, [tool_chunk]), lambda *a, **k: [], [2026])
        self.assertIn("RETRIEVAL FALLBACK", query)
        self.assertEqual(output.data_quality.coverage_mode, "rag_fallback")
        self.assertEqual(output.data_quality.section_located_by, "not_found")
        self.assertTrue(any("could not be located" in n for n in output.data_quality.notes))
        self.assertEqual(len(output.identified_risks), 1)
        self.assertEqual(output.identified_risks[0].risk_category, "Competitive & Demand")
        # Hypothetical language with no stated magnitude cannot carry a High impact.
        self.assertEqual(output.identified_risks[0].impact, "Medium")
        self.assertEqual(output.identified_risks[0].severity, "Moderate")
        self.assertTrue(any("no filing-stated magnitude" in a for a in output.identified_risks[0].adjustments))

    def test_high_impact_requires_magnitude_or_realized_effect(self):
        chunks = [
            _chunk("h1", "A cybersecurity breach could disrupt our operations and could harm our reputation with customers.", 0),
            _chunk("h2", "We have experienced supply constraints in the past, which resulted in delayed shipments to our customers.", 1),
        ]
        payload = {"identified_risks": [
            _risk("R1", "Technological & Cybersecurity", "Cyber Breach Disruption", "High", "High",
                  [{"chunk_id": "h1", "quote": "A cybersecurity breach could disrupt our operations"}]),
            _risk("R2", "Supply Chain & Concentration", "Recurring Supply Constraints", "High", "High",
                  [{"chunk_id": "h2", "quote": "We have experienced supply constraints in the past, which resulted in delayed shipments"}]),
        ]}
        output, query = _run_agent(_llm_result(payload), lambda t, fy, item="Item 1A": chunks, [2026])
        by_title = {r.risk_title: r for r in output.identified_risks}
        self.assertEqual(by_title["Cyber Breach Disruption"].severity, "Moderate")
        self.assertEqual(by_title["Recurring Supply Constraints"].severity, "Severe")
        self.assertEqual(output.identified_risks[0].risk_title, "Recurring Supply Constraints")
        self.assertNotIn("QUANTIFIED DISCLOSURES CHECKLIST", query)

    def test_query_lists_quantified_checklist(self):
        _, query = _run_agent(_llm_result({"identified_risks": []}), _current_only, [2026])
        self.assertIn("QUANTIFIED DISCLOSURES CHECKLIST", query)
        self.assertIn("- [chunk:a1] For fiscal year 2026, sales to one direct customer represented 22% of total revenue", query)
        self.assertIn("- [chunk:a2] We incurred a $4.5 billion charge", query)
        self.assertNotIn("- [chunk:a5]", query)

    def test_empty_llm_output_reports_failure(self):
        output, _ = _run_agent(_llm_result(None, raw="I was unable to complete the analysis."), _current_only, [2026])
        self.assertEqual(output.identified_risks, [])
        self.assertIsNone(output.primary_threat_risk_id)
        self.assertEqual(output.overall_risk_profile, "Not Assessed")
        self.assertTrue(any("llm_output_missing" in n for n in output.data_quality.notes))
        self.assertIn("No risks survived", output.risk_matrix_markdown)

    def test_cap_of_eight_risks(self):
        many = {f"z{i}": f"Distinct disclosure number {i} describes a specific operational exposure for product line {i} in detail." for i in range(10)}
        chunks = [_chunk(cid, text, i) for i, (cid, text) in enumerate(many.items())]
        titles = ["Warehouse fire", "Pilot program delays", "Battery recall costs", "Union labor dispute",
                  "Software defect liability", "Key executive departure", "Data center outage", "Rare earth shortage",
                  "Franchise partner failure", "Patent expiry cliff"]
        payload = {"identified_risks": [
            _risk(f"R{i + 1}", "Operational", titles[i], "Medium", "Medium", [{"chunk_id": cid, "quote": text[:80]}])
            for i, (cid, text) in enumerate(many.items())
        ]}
        output, _ = _run_agent(_llm_result(payload), lambda t, fy, item="Item 1A": chunks, [2026])
        self.assertEqual(len(output.identified_risks), 8)
        self.assertTrue(any("2 lower-ranked verified risks omitted" in n for n in output.data_quality.notes))

    def test_malformed_ratings_and_entries_sanitized(self):
        payload = {"identified_risks": [
            _risk("R1", "Unknown Bucket", "Direct Customer Revenue Concentration", "very high", None,
                  [{"chunk_id": "a1", "quote": "sales to one direct customer represented 22% of total revenue"}],
                  financial_transmission=["revenue", "Not A Line"]),
            {"risk_title": "", "risk_summary": "missing title"},
            "not a dict",
        ]}
        output, _ = _run_agent(_llm_result(payload), _current_only, [2026])
        risk = output.identified_risks[0]
        self.assertEqual(risk.likelihood, "High")
        self.assertEqual(risk.impact, "Medium")
        self.assertEqual(risk.risk_category, "Supply Chain & Concentration")
        self.assertEqual(risk.financial_transmission, ["Revenue"])
        self.assertTrue(any("Impact not rated" in a for a in risk.adjustments))
        self.assertTrue(any("2 malformed risk entries" in n for n in output.data_quality.notes))

    def test_structured_response_path_is_enforced(self):
        structured = RiskAuditOutput(
            ticker="TEST",
            fiscal_year=2026,
            identified_risks=[RiskItem(
                risk_category="Operational", risk_title="Invented Risk", risk_summary="Not in the filing at all.",
                likelihood="High", impact="High", severity="Severe",
                evidence=[RiskEvidence(chunk_id="a1", quote="this sentence does not exist anywhere in the filing")],
            )],
            primary_threat_risk_id="R1",
            primary_existential_threat="Invented.",
        )
        output, _ = _run_agent({"messages": [], "structured_response": structured}, _current_only, [2026])
        self.assertEqual(output.identified_risks, [])
        self.assertEqual(output.data_quality.risks_dropped_unverified, 1)
        self.assertIsNone(output.primary_threat_risk_id)


class TestRiskAnalystDatabase(unittest.TestCase):
    """Smoke tests against ingested filings; skipped when the filings are absent."""

    @patch("app.agents.tools.rag_narrative_tools._get_embedder")
    def test_item1a_narrative_retrieval(self, mock_get_embedder):
        mock_embedder = MagicMock()
        mock_embedder.embed_query.return_value = [0.01] * 1536
        mock_get_embedder.return_value = mock_embedder
        chunks = retrieve_10k_narrative_tool.invoke({
            "ticker": "AAPL", "fiscal_year": 2025, "query": "supplier reliance single source components",
            "section_item": "Item 1A", "limit": 3,
        })
        if not chunks:
            self.skipTest("AAPL FY2025 not ingested")
        for c in chunks:
            self.assertEqual(c["ticker"], "AAPL")
            self.assertTrue(c["chunk_id"])

    def test_heading_boundary_recovers_mislabeled_item_1a(self):
        chunks, located_by = locate_section_chunks("AMZN", 2025)
        if located_by == "not_found":
            self.skipTest("AMZN FY2025 not ingested")
        if located_by == "item_label":
            self.skipTest("AMZN FY2025 labels are already correct (re-ingested)")
        self.assertEqual(located_by, "heading_boundary")
        self.assertGreater(sum(len(c.content) for c in chunks), 20_000)
        self.assertTrue(all(c.item == "Item 1A" for c in chunks))
        self.assertNotIn("Unresolved Staff Comments", chunks[-1].content[:200])

    def test_full_section_fetch_and_diff_on_real_filings(self):
        current = fetch_section_chunks("AAPL", 2025)
        prior = fetch_section_chunks("AAPL", 2024)
        if not current or not prior:
            self.skipTest("AAPL FY2024/FY2025 Item 1A not ingested")
        self.assertTrue(all(c.item == "Item 1A" for c in current))
        self.assertEqual([c.chunk_index for c in current], sorted(c.chunk_index for c in current))
        diff = diff_risk_disclosures(current, prior, 2024)
        self.assertGreater(diff.unchanged_sentences, diff.new_sentences)
        self.assertGreater(diff.new_sentences, 0)
        self.assertTrue(diff.new_passages)


if __name__ == "__main__":
    unittest.main()
