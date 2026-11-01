"""The judge's business description reconciled to the extracted revenue lines and segments."""

import unittest

from app.agents.specialized.business_strategist.modules.business_model.verify import (
    deterministic_architecture,
    growth_drivers,
    verify_business_model,
)
from app.agents.specialized.business_strategist.modules.revenue_mix import (
    SegmentExtraction,
    extract_segment_disclosures,
)
from tests.business_strategist.filing_fixtures import filing

POOL = {"i1": {"chunk_id": "i1", "content": "We sell the Widget Pro and offer Cloud Backup subscriptions."}}


def _extraction() -> SegmentExtraction:
    return extract_segment_disclosures(filing(), 2025)


def _detail(name, description="Described by the judge.", drivers=None):
    return {"name": name, "description": description, "growth_drivers": drivers or []}


class TestProductLines(unittest.TestCase):
    def test_lines_follow_the_revenue_table_with_fuzzy_names(self):
        data = {"segment_details": [_detail("online store", drivers=["Selection"]), _detail("Retail bucket")]}
        out = verify_business_model(data, POOL, _extraction(), "unused")
        self.assertEqual(out.primary_product_segments, ["Online stores", "Seller services", "Cloud services"])
        details = {d.name: d for d in out.segment_details}
        self.assertEqual(details["Online stores"].description, "Described by the judge.")
        self.assertEqual(details["Online stores"].growth_drivers, ["Selection"])
        self.assertEqual(details["Seller services"].description,
                         "Seller services: FY2025 revenue $300M, 30.0% of revenue (from the 10-K tables).")
        self.assertEqual(out.tally.notes,
                         ["1 model-proposed product line(s) not in the extracted revenue table were dropped (Retail bucket)."])

    def test_segments_are_used_when_no_revenue_lines_exist(self):
        ex = _extraction().model_copy(update={"revenue_lines": []})
        out = verify_business_model({}, POOL, ex, "unused")
        self.assertEqual(out.primary_product_segments, ["Retail", "Cloud"])

    def test_geographic_segments_are_not_product_lines(self):
        ex = _extraction().model_copy(update={"revenue_lines": [], "segment_basis": "geographic"})
        data = {"segment_details": [_detail("Widget Pro"), _detail("Hover Car")]}
        out = verify_business_model(data, POOL, ex, "unused")
        self.assertEqual(out.primary_product_segments, ["Widget Pro"])  # named in the filing text
        self.assertEqual(out.tally.notes, ["Product line 'Hover Car' dropped: not named in the filing text."])

    def test_growth_drivers(self):
        self.assertEqual(growth_drivers(" AI "), ["AI"])
        self.assertEqual(growth_drivers(["a", "", 3, "b", "c", "d", "e"]), ["a", "b", "c", "d"])
        self.assertEqual(growth_drivers(None), [])


class TestProse(unittest.TestCase):
    def test_judge_prose_is_kept(self):
        out = verify_business_model({"business_summary": " A retailer. ", "revenue_architecture": "Sales."}, POOL,
                                    _extraction(), "unused")
        self.assertEqual((out.business_summary, out.revenue_architecture), ("A retailer.", "Sales."))

    def test_missing_prose_falls_back(self):
        out = verify_business_model({}, POOL, _extraction(), "the qualitative LLM call failed")
        self.assertEqual(out.business_summary, "Not assessed: the qualitative LLM call failed.")
        self.assertEqual(
            out.revenue_architecture,
            "Revenue mix by product / service line (FY2025, from the 10-K revenue disaggregation): "
            "Online stores 50.0%, Seller services 30.0%, Cloud services 20.0%.",
        )
        self.assertEqual(deterministic_architecture(SegmentExtraction(), "no filing text was available"),
                         "Not assessed: no filing text was available.")


if __name__ == "__main__":
    unittest.main()
