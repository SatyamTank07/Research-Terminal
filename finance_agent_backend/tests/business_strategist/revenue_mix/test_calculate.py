"""Segment note and revenue disaggregation parsed from the filing tables, units and label vocabulary.

Synthetic markdown tables mirror the layouts found in real filings; no DB or network access.
"""

import unittest

from app.agents.specialized.business_strategist.context import FilingChunk
from app.agents.specialized.business_strategist.modules.revenue_mix.calculate import (
    extract_segment_disclosures,
    is_geographic,
    latest_year_figures,
    match_name,
    name_key,
    profit_rank,
    reportable_segment_names,
    scale_from_caption,
    segment_basis,
    top_level_line_names,
)
from tests.business_strategist.filing_fixtures import (
    DISAGGREGATION,
    INCOME_STATEMENT,
    STACKED_SEGMENTS,
    filing,
    md,
    table,
    text,
)


class TestSegmentExtraction(unittest.TestCase):
    def test_stacked_by_segment_ascending_columns(self):
        ex = extract_segment_disclosures(filing(), 2025)
        self.assertEqual(ex.segment_layout, "stacked_by_segment")
        self.assertEqual(ex.segment_reconciliation, "in_table_total")
        latest = {f.segment_name: f for f in ex.segments if f.fiscal_year == 2025}
        self.assertEqual(set(latest), {"Retail", "Cloud"})
        cloud = latest["Cloud"]
        self.assertEqual(cloud.revenue, 200.0)
        self.assertEqual(cloud.segment_profit, 70.0)
        self.assertEqual(cloud.profit_measure, "Operating income")  # "(loss)" suffix normalized
        self.assertEqual(cloud.revenue_share_pct, 20.0)
        self.assertEqual(cloud.profit_margin_pct, 35.0)
        self.assertEqual(cloud.profit_share_pct, 63.6)
        self.assertEqual(cloud.revenue_growth_pct, 33.3)
        self.assertEqual(ex.consolidated_revenue, 1000.0)
        self.assertEqual(sorted({f.fiscal_year for f in ex.segments}), [2024, 2025])
        self.assertEqual(ex.segment_basis, "business")

    def test_segments_as_columns_merged_across_year_tables_geographic(self):
        def year_table(year: str, a: str, b: str, total: str, oa: str, ob: str, ot: str) -> str:
            return md(
                [("Net sales", a, b, "—", total), ("Operating income/(loss)", oa, ob, "(20)", ot)],
                header=("", "Americas", "Europe", "Corporate", "Total"),
                title=year,
            )

        segs_2025 = year_table("2025", "600", "400", "1,000", "80", "50", "110")
        segs_2024 = year_table("2024", "500", "350", "850", "60", "40", "80")
        chunks = filing(segments=segs_2025) + [table(9, segs_2024)]
        ex = extract_segment_disclosures(chunks, 2025)
        self.assertEqual(ex.segment_layout, "segments_as_columns")
        self.assertEqual(ex.segment_basis, "geographic")
        self.assertEqual(sorted({f.fiscal_year for f in ex.segments}), [2024, 2025])
        corporate = [f for f in ex.segments if f.segment_name == "Corporate"]
        self.assertTrue(corporate and not corporate[0].reportable)
        self.assertTrue(any("geographic" in n for n in ex.notes))
        americas = next(f for f in ex.segments if f.segment_name == "Americas" and f.fiscal_year == 2025)
        self.assertEqual(americas.revenue_growth_pct, 20.0)

    def test_segments_as_columns_with_year_rows_and_unit_caption(self):
        segs = md(
            [
                ("", "(In millions)", "", ""),
                ("Year Ended Jan 26, 2025", "", "", ""),
                ("Revenue", "800", "200", "1,000"),
                ("Other segment items", "760", "130", "890"),
                ("Segment operating income", "40", "70", "110"),
                ("Year Ended Jan 28, 2024", "", "", ""),
                ("Revenue", "700", "150", "850"),
                ("Segment operating income", "30", "50", "80"),
            ],
            header=("", "Compute", "Graphics", "Total"),
        )
        ex = extract_segment_disclosures(filing(segments=segs), 2025)
        self.assertEqual(ex.segment_layout, "segments_as_columns")
        graphics = next(f for f in ex.segments if f.segment_name == "Graphics" and f.fiscal_year == 2025)
        self.assertEqual((graphics.revenue, graphics.segment_profit), (200.0, 70.0))
        self.assertEqual(graphics.profit_measure, "Segment operating income")

    def test_stacked_by_metric_with_contribution_measure(self):
        segs = md([
            ("Revenues:", "", ""),
            ("Retail", "800", "700"),
            ("Cloud", "200", "150"),
            ("Total revenues", "1,000", "850"),
            ("Segment contribution:", "", ""),
            ("Retail", "60", "45"),
            ("Cloud", "90", "70"),
        ])
        ex = extract_segment_disclosures(filing(segments=segs), 2025)
        self.assertEqual(ex.segment_layout, "stacked_by_metric")
        retail = next(f for f in ex.segments if f.segment_name == "Retail" and f.fiscal_year == 2025)
        self.assertEqual(retail.segment_profit, 60.0)
        self.assertEqual(retail.profit_measure, "Segment contribution")

    def test_gross_profit_segments_without_total_reconcile_to_income_statement(self):
        segs = md([
            ("Automotive segment", "", ""),
            ("Revenues", "850", "720"),
            ("Cost of revenues", "700", "600"),
            ("Gross profit", "150", "120"),
            ("Energy segment", "", ""),
            ("Revenues", "150", "130"),
            ("Gross profit", "40", "30"),
        ])
        ex = extract_segment_disclosures(filing(segments=segs), 2025)
        self.assertEqual(ex.segment_reconciliation, "income_statement")
        self.assertEqual({f.segment_name for f in ex.segments}, {"Automotive", "Energy"})
        self.assertTrue(any("gross profit" in n for n in ex.notes))

    def test_unreconciled_segment_table_rejected(self):
        bad = STACKED_SEGMENTS.replace("| Net sales | 850 | 1,000 |", "| Net sales | 850 | 1,400 |")
        ex = extract_segment_disclosures(filing(segments=bad), 2025)
        self.assertEqual(ex.segments, [])
        self.assertTrue(any("does not reconcile" in n for n in ex.notes))

    def test_single_segment_company_returns_empty_with_note(self):
        single = md([("Net sales", "1,000", "850"), ("Cost of sales", "600", "520"), ("Segment net income", "90", "60")])
        chunks = [
            table(1, INCOME_STATEMENT),
            text(2, "The Company operates as one reportable segment and its CODM reviews consolidated results."),
            table(3, single),
        ]
        ex = extract_segment_disclosures(chunks, 2025)
        self.assertEqual(ex.segments, [])
        self.assertEqual(ex.segment_basis, "single_segment")
        self.assertTrue(any("single reportable segment" in n for n in ex.notes))

    def test_thousands_caption_scaled_to_millions(self):
        income = INCOME_STATEMENT.replace("| Total net sales | 1,000 | 850 |", "| Total net sales | 1,000,000 | 850,000 |")
        income = "(In thousands, except per share data)\n" + income
        segs = md([
            ("Retail", "", ""), ("Net sales", "800,000", "700,000"), ("Operating income", "40,000", "30,000"),
            ("Cloud", "", ""), ("Net sales", "200,000", "150,000"), ("Operating income", "70,000", "50,000"),
        ])
        chunks = [table(1, income), text(2, "Segment information (in thousands):"), table(3, segs)]
        ex = extract_segment_disclosures(chunks, 2025)
        self.assertEqual(ex.consolidated_revenue, 1000.0)
        cloud = next(f for f in ex.segments if f.segment_name == "Cloud" and f.fiscal_year == 2025)
        self.assertEqual(cloud.revenue, 200.0)
        self.assertEqual(cloud.segment_profit, 70.0)

    def test_uncaptioned_segment_table_aligned_to_income_statement_scale(self):
        income = "(Amounts in millions)\n" + INCOME_STATEMENT
        segs = md([
            ("Retail", "", ""), ("Net sales", "800,000", "700,000"), ("Operating income", "40,000", "30,000"),
            ("Cloud", "", ""), ("Net sales", "200,000", "150,000"), ("Operating income", "70,000", "50,000"),
        ])
        ex = extract_segment_disclosures([table(1, income), text(2, "Segment information:"), table(3, segs)], 2025)
        cloud = next(f for f in ex.segments if f.segment_name == "Cloud" and f.fiscal_year == 2025)
        self.assertEqual(cloud.revenue, 200.0)

    def test_missing_unit_caption_noted(self):
        ex = extract_segment_disclosures(filing(), 2025)
        self.assertIn("assumed", ex.units)

    def test_no_tables_at_all(self):
        ex = extract_segment_disclosures([text(1, "Narrative only.")], 2025)
        self.assertEqual((ex.segments, ex.revenue_lines), ([], []))
        self.assertTrue(ex.notes)

    def test_extraction_never_raises(self):
        broken = FilingChunk(chunk_id="x", chunk_index=1, chunk_type="table", content="| a | b |\n|||||\n| (( | )) |")
        ex = extract_segment_disclosures([broken, table(2, "not a table at all")], 2025)
        self.assertEqual(ex.segments, [])


class TestRevenueLines(unittest.TestCase):
    def test_disaggregation_lines_reconciled_and_preferred(self):
        ex = extract_segment_disclosures(filing(), 2025)
        latest = [f for f in ex.revenue_lines if f.fiscal_year == 2025]
        self.assertEqual([f.name for f in latest], ["Online stores", "Seller services", "Cloud services"])
        self.assertEqual(latest[0].revenue_share_pct, 50.0)
        self.assertEqual(ex.revenue_lines_reconciliation, "income_statement")
        self.assertEqual(ex.revenue_lines_source_chunk_id, "t5")

    def test_nested_sub_lines_subtotals_and_trailing_rows(self):
        disagg = md([
            ("Revenue by End Market:", "", ""),
            ("Data Center", "700", "600"),
            ("Compute", "500", "450"),
            ("Networking", "200", "150"),
            ("Gaming", "200", "150"),
            ("Total revenues from product sales", "900", "750"),
            ("Licensing", "100", "100"),
            ("Total revenue", "1,000", "850"),
            ("Portion of revenue included in deferred revenue", "40", "30"),
        ])
        ex = extract_segment_disclosures(filing(disaggregation=disagg), 2025)
        latest = [f for f in ex.revenue_lines if f.fiscal_year == 2025]
        names = [(f.name, f.parent) for f in latest]
        self.assertIn(("Compute", "Data Center"), names)
        self.assertIn(("Networking", "Data Center"), names)
        self.assertEqual([n for n, p in names if p is None], ["Data Center", "Gaming", "Licensing"])
        self.assertNotIn("Portion of revenue included in deferred revenue", [n for n, _ in names])

    def test_geographic_revenue_table_is_not_a_product_mix(self):
        geo = md([("United States", "600", "500"), ("Germany", "250", "220"), ("Rest of world", "150", "130"), ("Consolidated", "1,000", "850")])
        ex = extract_segment_disclosures(filing(disaggregation=geo), 2025)
        self.assertEqual(ex.revenue_lines, [])
        self.assertTrue(any("disaggregation" in n for n in ex.notes))

    def test_change_columns_rejected(self):
        mdna = md(
            [("Online stores", "500", "11", "%", "450"), ("Seller services", "300", "20", "%", "250"),
             ("Cloud services", "200", "33", "%", "150"), ("Total net sales", "1,000", "18", "%", "850")],
            header=("", "2025", "Change", "", "2024"),
        )
        ex = extract_segment_disclosures(filing(disaggregation=mdna), 2025)
        self.assertEqual(ex.revenue_lines, [])

    def test_partial_breakdown_not_matching_revenue_rejected(self):
        inventory = md([("Automotive", "90", "80"), ("Energy", "30", "20"), ("Total", "120", "100")])
        ex = extract_segment_disclosures(filing(disaggregation=inventory), 2025)
        self.assertEqual(ex.revenue_lines, [])

    def test_income_statement_used_only_as_fallback(self):
        income = md([
            ("Net sales:", "", ""), ("Products", "700", "600"), ("Services", "300", "250"), ("Total net sales", "1,000", "850"),
            ("Cost of sales:", "", ""), ("Products", "500", "450"),
            ("Operating income", "110", "80"), ("Net income", "90", "60"),
        ])
        ex = extract_segment_disclosures([table(1, income), table(3, STACKED_SEGMENTS)], 2025)
        self.assertEqual([f.name for f in ex.revenue_lines if f.fiscal_year == 2025], ["Products", "Services"])
        ex2 = extract_segment_disclosures([table(1, income), table(3, STACKED_SEGMENTS), table(5, DISAGGREGATION)], 2025)
        self.assertEqual(ex2.revenue_lines_source_chunk_id, "t5")


class TestHelpers(unittest.TestCase):
    def test_vocabulary(self):
        self.assertEqual(profit_rank("Operating income (loss)"), 0)
        self.assertEqual(profit_rank("Segment profit"), 0)
        self.assertEqual(profit_rank("Gross profit"), 2)
        self.assertIsNone(profit_rank("Total non-operating income"))
        self.assertTrue(is_geographic("Rest of Asia Pacific"))
        self.assertTrue(is_geographic("U.S."))
        self.assertFalse(is_geographic("Business Services"))
        self.assertEqual(segment_basis(["North America", "International", "Cloud"]), "mixed")
        self.assertEqual(name_key("Compute & Networking segment"), name_key("compute and networking"))

    def test_scale_caption_earliest_wins(self):
        self.assertEqual(scale_from_caption("(In millions, except shares, which are reflected in thousands)"), (1.0, True))
        self.assertEqual(scale_from_caption("(in thousands)"), (1000.0, True))
        self.assertEqual(scale_from_caption("Deliveries (in thousands of units)"), (1.0, False))

    def test_extracted_names_are_the_latest_year(self):
        ex = extract_segment_disclosures(filing(), 2025)
        self.assertEqual({f.fiscal_year for f in latest_year_figures(ex.segments)}, {2025})
        self.assertEqual(latest_year_figures([]), [])
        self.assertEqual(reportable_segment_names(ex), ["Retail", "Cloud"])
        self.assertEqual(top_level_line_names(ex), ["Online stores", "Seller services", "Cloud services"])

    def test_match_name(self):
        lookup = {name_key(n): n for n in ["Wearables, Home and Accessories", "Services", "Mac"]}
        self.assertEqual(match_name("services", lookup), "Services")
        self.assertEqual(match_name("Wearables", lookup), "Wearables, Home and Accessories")
        self.assertIsNone(match_name("Vision Pro", lookup))
        both = {name_key(n): n for n in ["Cloud services", "Cloud storage"]}
        self.assertIsNone(match_name("Cloud", both))  # ambiguous: two names contain it



if __name__ == "__main__":
    unittest.main()
