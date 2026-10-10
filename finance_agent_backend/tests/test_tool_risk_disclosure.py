"""Unit Tests for the deterministic Risk-Disclosure toolkit (Item 1A).

Validates quote verification, numeric-claim support, the likelihood x impact severity matrix,
aggregate risk profile, materiality / boilerplate screens, context budgeting, the
year-over-year disclosure diff, and the markdown renderers. No database or network access.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from app.agents.tools.rag_narrative_tools import NarrativeChunkResult
from app.agents.tools.risk_disclosure_tools import (
    build_prior_index,
    build_section_context,
    cap_level,
    classify_disclosure_change,
    derive_overall_profile,
    derive_severity,
    diff_risk_disclosures,
    disclosure_covered,
    extract_quantified_disclosures,
    heading_item,
    is_boilerplate,
    locate_section_chunks,
    section_not_provided,
    is_material_disclosure,
    normalize_level,
    numbers_supported,
    render_risk_evidence_table,
    render_risk_matrix_markdown,
    states_immaterial,
    substantiates_high_impact,
    verify_quote,
)

CONCENTRATION = (
    "We receive a significant amount of our revenue from a limited number of customers. "
    "For fiscal year 2026, sales to one direct customer represented 22% of total revenue and sales to "
    "another direct customer represented 14% of total revenue, all of which were primarily attributable "
    "to the Compute & Networking segment."
)


def _chunk(chunk_id: str, content: str, idx: int = 0, item: str = "Item 1A", fy: int = 2026) -> NarrativeChunkResult:
    return NarrativeChunkResult(
        chunk_id=chunk_id,
        document_id="doc-1",
        ticker="TEST",
        fiscal_year=fy,
        item=item,
        breadcrumb=f"TEST > 10-K FY{fy} > PART I > {item}: Risk Factors",
        sub_section="Risk Factors",
        content=content,
        chunk_index=idx,
    )


class TestQuoteVerification(unittest.TestCase):
    def test_exact_and_normalized_quotes_pass(self):
        quote = "sales to one direct customer represented 22% of total revenue"
        self.assertTrue(verify_quote(quote, CONCENTRATION))
        self.assertTrue(verify_quote(quote.upper().replace(" ", "   "), CONCENTRATION))

    def test_curly_quotes_and_dashes_are_normalized(self):
        text = "The Company’s products — including chips — face export licensing requirements in China."
        self.assertTrue(verify_quote("The Company's products - including chips - face export licensing requirements", text))

    def test_ellipsis_segments_each_must_match(self):
        quote = "sales to one direct customer represented 22% ... another direct customer represented 14% of total revenue"
        self.assertTrue(verify_quote(quote, CONCENTRATION))
        bad = "sales to one direct customer represented 22% ... another customer represented 40% of total revenue"
        self.assertFalse(verify_quote(bad, CONCENTRATION))

    def test_near_verbatim_passes_only_with_fuzzy(self):
        quote = "sales to one customer represented 22% of total revenue and sales to another direct customer represented 14%"
        self.assertTrue(verify_quote(quote, CONCENTRATION))
        self.assertFalse(verify_quote(quote, CONCENTRATION, allow_fuzzy=False))

    def test_fabricated_and_degenerate_quotes_fail(self):
        self.assertFalse(verify_quote("management believes the outcome will not have a material adverse effect", CONCENTRATION))
        self.assertFalse(verify_quote("22% of", CONCENTRATION))  # too short to be evidence
        self.assertFalse(verify_quote(None, CONCENTRATION))
        self.assertFalse(verify_quote("sales to one direct customer represented 22%", None))


class TestNumericSupport(unittest.TestCase):
    def test_numbers_must_appear_in_evidence(self):
        self.assertTrue(numbers_supported("Two direct customers at 22% and 14% of revenue", [CONCENTRATION]))
        self.assertFalse(numbers_supported("One customer is 35% of revenue", [CONCENTRATION]))
        self.assertTrue(numbers_supported("A $4,500.0 million charge", ["we incurred a $4500 million charge"]))
        self.assertTrue(numbers_supported(None, [CONCENTRATION]))
        self.assertTrue(numbers_supported("Most revenue comes from few customers", [CONCENTRATION]))


class TestSeverityAndProfile(unittest.TestCase):
    def test_severity_matrix(self):
        self.assertEqual(derive_severity("High", "High"), "Severe")
        self.assertEqual(derive_severity("Medium", "High"), "Severe")
        self.assertEqual(derive_severity("Low", "High"), "Moderate")
        self.assertEqual(derive_severity("High", "Medium"), "Moderate")
        self.assertEqual(derive_severity("Low", "Medium"), "Low")
        self.assertEqual(derive_severity("High", "Low"), "Low")

    def test_overall_profile(self):
        self.assertEqual(derive_overall_profile(["Severe", "Severe", "Low"]), "High")
        self.assertEqual(derive_overall_profile(["Severe", "Low", "Low"]), "Moderate")
        self.assertEqual(derive_overall_profile(["Moderate"] * 3), "Moderate")
        self.assertEqual(derive_overall_profile(["Moderate", "Low"]), "Low")
        self.assertEqual(derive_overall_profile([]), "Not Assessed")

    def test_level_normalization_and_caps(self):
        self.assertEqual(normalize_level("severe"), "High")
        self.assertEqual(normalize_level(" Moderate "), "Medium")
        self.assertEqual(normalize_level("low"), "Low")
        self.assertIsNone(normalize_level("unknown"))
        self.assertIsNone(normalize_level(3))
        self.assertEqual(cap_level("High", "Medium"), "Medium")
        self.assertEqual(cap_level("Low", "Medium"), "Low")


class TestScreens(unittest.TestCase):
    def test_immateriality_language(self):
        self.assertTrue(states_immaterial(
            "We believe the ultimate outcome will not have a material adverse effect on our operating results."
        ))
        self.assertTrue(states_immaterial("The matter is not expected to have a material impact on our results."))
        self.assertTrue(states_immaterial("We do not believe these claims will be material to our financial position."))
        self.assertFalse(states_immaterial(CONCENTRATION))

    def test_boilerplate_language(self):
        self.assertTrue(is_boilerplate("Economic Conditions and Market Volatility"))
        self.assertTrue(is_boilerplate("Adverse Economic Conditions could reduce demand"))
        self.assertTrue(is_boilerplate("Economic downturns may harm results"))
        self.assertTrue(is_boilerplate("Our stock price may be volatile and could decline"))
        self.assertFalse(is_boilerplate("Dependence on two direct customers for 36% of revenue"))


class TestQuantifiedDisclosures(unittest.TestCase):
    def test_extracts_figures_about_revenue_charges_and_trade(self):
        chunks = [
            _chunk("q1", CONCENTRATION, 0),
            _chunk("q2", "We incurred a $4.5 billion charge associated with excess inventory and purchase obligations. "
                         "Our employees are located in more than 30 countries around the world today.", 1),
        ]
        found = extract_quantified_disclosures(chunks)
        texts = [d.text for d in found]
        self.assertEqual([d.chunk_id for d in found], ["q1", "q2"])
        self.assertTrue(any("22% of total revenue" in t for t in texts))
        self.assertTrue(any("$4.5 billion charge" in t for t in texts))
        self.assertFalse(any("30 countries" in t for t in texts))  # a figure, but not an exposure topic
        self.assertEqual(len(extract_quantified_disclosures(chunks, max_items=1)), 1)

    def test_material_disclosures_exclude_statutory_maximums(self):
        self.assertTrue(is_material_disclosure(CONCENTRATION))
        self.assertTrue(is_material_disclosure("We incurred a $4.5 billion charge for excess inventory."))
        self.assertFalse(is_material_disclosure(
            "We could be subject to penalties of up to €20 million or 4% of worldwide revenue, whichever is greater."))
        self.assertFalse(is_material_disclosure("Our employees work in more than 30 countries."))

    def test_coverage_by_partial_quote(self):
        sentence = ("For fiscal year 2026, sales to one direct customer represented 22% of total revenue and sales to "
                    "another direct customer represented 14% of total revenue.")
        self.assertTrue(disclosure_covered(sentence, ["sales to one direct customer represented 22% of total revenue"]))
        self.assertFalse(disclosure_covered(sentence, ["We rely on third-party foundries located in Taiwan."]))
        self.assertFalse(disclosure_covered(sentence, []))

    def test_high_impact_substantiation(self):
        self.assertTrue(substantiates_high_impact(["anything"], "22% of revenue"))
        self.assertTrue(substantiates_high_impact(["sales to one customer represented 22% of total revenue"], None))
        self.assertTrue(substantiates_high_impact(["We incurred significant costs to remediate the defect"], None))
        self.assertTrue(substantiates_high_impact(["Export restrictions have adversely affected our China revenue"], None))
        self.assertFalse(substantiates_high_impact(["A cyberattack could disrupt our operations and may harm results"], None))
        self.assertFalse(substantiates_high_impact([], None))


def _row(idx: int, content: str, chunk_type: str = "narrative", item: str = "Cover Page"):
    chunk = SimpleNamespace(id=f"k{idx}", item=item, breadcrumb=f"X > {item}", sub_section=None,
                            content=content, chunk_index=idx, chunk_type=chunk_type)
    doc = SimpleNamespace(id="doc-1", ticker="TEST", fiscal_year=2026)
    return chunk, doc


class TestSectionLocation(unittest.TestCase):
    TOC = ("### Page | PART I | | Item 1. | Business | 3 | | Item 1A. | Risk Factors | 6 | "
           "| Item 1B. | Unresolved Staff Comments | 16 | | Item 2. | Properties | 17 |")
    LONG_BODY = "Our business depends on a small number of suppliers and customers in several regions. " * 6

    def test_heading_detection(self):
        self.assertEqual(heading_item("| Item 1A. | Risk Factors |\n| :--- | :--- |"), "1A")
        self.assertEqual(heading_item("| Item 6. | Reserved |"), "6")
        self.assertEqual(heading_item("ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS"), "7")
        self.assertIsNone(heading_item(self.TOC))
        self.assertEqual(heading_item("Item 1A. Risk Factors " + self.LONG_BODY), "1A")
        self.assertIsNone(heading_item("Item 1A of Part I describes factors that " + self.LONG_BODY))
        self.assertIsNone(heading_item("For more information, see Item 1A. Risk Factors."))
        self.assertIsNone(heading_item("| Item 99. | Nothing |"))

    def _locate(self, rows, labeled=None):
        query = MagicMock()
        query.order_by.return_value.all.return_value = rows
        with patch("app.agents.tools.risk_disclosure_tools.fetch_section_chunks", return_value=labeled or []), \
                patch("app.agents.tools.risk_disclosure_tools._filing_chunks_query", return_value=query):
            return locate_section_chunks("TEST", 2026, "Item 1A", db=MagicMock())

    def test_boundary_location_when_labels_are_wrong(self):
        rows = [
            _row(0, self.TOC, "table"),
            _row(1, "| Item 1. | Business |", "table"),
            _row(2, "We design products. " + self.LONG_BODY),
            _row(3, "| Item 1A. | Risk Factors |\n| :--- | :--- |", "table"),
            _row(4, "Risk one narrative. " + self.LONG_BODY),
            _row(5, "| Segment | Share |\n| :--- | :--- |\n| A | 40% |", "table"),
            _row(6, "Risk two narrative; see Item 7 for details. " + self.LONG_BODY),
            _row(7, "| Item 1B. | Unresolved Staff Comments |", "table"),
            _row(8, "None."),
            _row(9, "| Item 1C. | Cybersecurity |", "table"),
        ]
        chunks, located_by = self._locate(rows)
        self.assertEqual(located_by, "heading_boundary")
        self.assertEqual([c.chunk_index for c in chunks], [4, 6])
        self.assertTrue(all(c.item == "Item 1A" for c in chunks))

    def test_heading_inside_long_chunk_and_label_precedence(self):
        rows = [_row(0, "Item 1A. Risk Factors " + self.LONG_BODY), _row(1, "More risk text. " + self.LONG_BODY),
                _row(2, "Item 2. Properties " + self.LONG_BODY)]
        chunks, located_by = self._locate(rows)
        self.assertEqual(([c.chunk_index for c in chunks], located_by), ([0, 1], "heading_boundary"))

        labeled = [_chunk("L1", "Labeled risk text " + self.LONG_BODY)]
        chunks, located_by = self._locate(rows, labeled=labeled)
        self.assertEqual((chunks, located_by), (labeled, "item_label"))

        self.assertEqual(self._locate([_row(0, self.LONG_BODY)]), ([], "not_found"))

    def test_not_provided_detection(self):
        self.assertTrue(section_not_provided([_chunk("n", "Not applicable.")]))
        self.assertTrue(section_not_provided([_chunk("n", "As a smaller reporting company, we are not required to provide the information required by this Item.")]))
        self.assertTrue(section_not_provided([]))
        self.assertFalse(section_not_provided([_chunk("r", self.LONG_BODY * 10 + " Not applicable to our Japanese subsidiaries.")]))
        self.assertFalse(section_not_provided([_chunk("r", CONCENTRATION)]))

    def test_section_queries_are_scoped_to_annual_reports(self):
        from app.database import SessionLocal
        from app.agents.tools.risk_disclosure_tools import _filing_chunks_query
        db = SessionLocal()
        try:
            sql = str(_filing_chunks_query(db, "TEST", 2026).statement)
        finally:
            db.close()
        self.assertIn("documents.filing_type", sql)


class TestSectionContext(unittest.TestCase):
    def test_full_section_within_budget(self):
        chunks = [_chunk("c1", "Summary bullet text " * 10, 0), _chunk("c2", CONCENTRATION, 1)]
        context, mode, chars = build_section_context(chunks, char_budget=10_000)
        self.assertEqual(mode, "full_section")
        self.assertIn("[chunk:c1]", context)
        self.assertIn("[chunk:c2] We receive a significant amount", context)
        self.assertGreater(chars, len(CONCENTRATION))

    def test_budgeted_section_keeps_summary_and_truncates_rest(self):
        summary = "Summary of Risk Factors • Risk one is here. • Risk two is here. • Risk three is here. " * 5
        body = "Detailed risk heading sentence. " + ("Long detailed paragraph text. " * 200)
        chunks = [_chunk("s1", summary, 0)] + [_chunk(f"b{i}", body, i + 1) for i in range(5)]
        context, mode, _ = build_section_context(chunks, char_budget=5_000)
        self.assertEqual(mode, "budgeted_section")
        self.assertIn(summary.strip(), context)
        self.assertIn("[chunk:b0] Detailed risk heading sentence.", context)
        self.assertIn("[...]", context)
        self.assertLessEqual(len(context), 5_000 + 200)


class TestDisclosureDiff(unittest.TestCase):
    PRIOR = [
        _chunk("p1", "Our manufacturing is performed by outsourcing partners located primarily in a small number of Asian countries. "
                     "Global supply chains can be highly concentrated and an escalation of tensions could result in disruptions.", 0, fy=2024),
        _chunk("p2", "We previously relied on a single legacy distribution partner for all of our retail channel shipments worldwide.", 1, fy=2024),
    ]
    CURRENT = [
        _chunk("c1", "Our manufacturing is performed by outsourcing partners located primarily in a small number of Asian countries. "
                     "Beginning in the second quarter, new tariffs were announced on imports into the United States from several countries.", 0, fy=2025),
    ]

    def test_new_unchanged_and_removed_sentences(self):
        diff = diff_risk_disclosures(self.CURRENT, self.PRIOR, 2024)
        self.assertEqual(diff.prior_fiscal_year, 2024)
        self.assertEqual(diff.current_sentences, 2)
        self.assertEqual(diff.unchanged_sentences, 1)
        self.assertEqual(diff.new_sentences, 1)
        self.assertGreaterEqual(diff.removed_sentences, 2)
        self.assertEqual(len(diff.new_passages), 1)
        self.assertEqual(diff.new_passages[0].chunk_id, "c1")
        self.assertIn("new tariffs", diff.new_passages[0].text)
        self.assertTrue(any("legacy distribution partner" in p.text for p in diff.removed_passages))
        self.assertAlmostEqual(diff.change_ratio_pct, 50.0)

    def test_identical_filings_have_no_changes(self):
        diff = diff_risk_disclosures(self.PRIOR, self.PRIOR, 2024)
        self.assertEqual(diff.new_sentences, 0)
        self.assertEqual(diff.removed_sentences, 0)
        self.assertEqual(diff.new_passages, [])

    def test_preamble_sentences_are_ignored(self):
        preamble = _chunk("x", "The risks described below are not exhaustive and should be read with the forward-looking statements.", fy=2025)
        diff = diff_risk_disclosures([preamble], self.PRIOR, 2024)
        self.assertEqual(diff.current_sentences, 0)

    def test_quote_level_disclosure_change(self):
        index = build_prior_index(self.PRIOR)
        self.assertEqual(classify_disclosure_change(
            ["new tariffs were announced on imports into the United States from several countries"], index), "New")
        self.assertEqual(classify_disclosure_change(
            ["Our manufacturing is performed by outsourcing partners located primarily in a small number of Asian countries."], index),
            "Unchanged")
        self.assertEqual(classify_disclosure_change(
            ["Our manufacturing is performed by outsourcing partners located primarily in a small number of Asian countries.",
             "new tariffs were announced on imports into the United States from several countries"], index), "Expanded")
        self.assertEqual(classify_disclosure_change(["anything"], None), "Not Assessed")
        self.assertEqual(classify_disclosure_change([], index), "Not Assessed")


class TestRenderers(unittest.TestCase):
    def test_risk_matrix_and_evidence_table(self):
        risks = [{
            "risk_id": "R1", "risk_title": "Customer | concentration", "risk_category": "Supply Chain & Concentration",
            "likelihood": "High", "impact": "High", "severity": "Severe",
            "financial_transmission": ["Revenue", "Gross Margin"], "disclosure_change": "Unchanged",
        }]
        matrix = render_risk_matrix_markdown(risks)
        self.assertIn("| # | Risk | Category | Likelihood | Impact | Severity | Model line(s) hit | vs prior year |", matrix)
        self.assertIn("| R1 | Customer / concentration | Supply Chain & Concentration | High | High | **Severe** | Revenue, Gross Margin | Unchanged |", matrix)
        self.assertIn("No risks survived", render_risk_matrix_markdown([]))

        table = render_risk_evidence_table([{"chunk_id": "c1", "item": "Item 15", "breadcrumb": "TEST > Item 15"}])
        self.assertIn("| `c1` | Item 15 | TEST > Item 15 |", table)
        self.assertIn("No verified filing evidence", render_risk_evidence_table([]))


if __name__ == "__main__":
    unittest.main()
