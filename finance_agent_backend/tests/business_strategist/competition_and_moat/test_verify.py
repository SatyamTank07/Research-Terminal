"""Moat and competition claims kept only where verified filing quotes name their mechanism."""

import unittest

from app.agents.specialized.business_strategist.modules.competition_and_moat.verify import (
    coerce_moat,
    not_assessed_competition,
    verify_competition_and_moat,
)
from app.agents.specialized.business_strategist.modules.revenue_mix import SegmentExtraction, extract_segment_disclosures
from tests.business_strategist.filing_fixtures import filing

POOL = {
    "i1": {"chunk_id": "i1", "item": "Item 1", "content": (
        "Our marketplace connects millions of buyers and sellers, and each additional seller attracts more buyers to "
        "the platform. We operate a global logistics network that lowers our cost to serve each order.")},
    "i2": {"chunk_id": "i2", "item": "Item 1", "content": (
        "Competition. Our businesses face intense competition from physical retailers, online marketplaces and cloud "
        "service providers. We believe the principal competitive factors include selection, price and convenience.")},
    "i3": {"chunk_id": "i3", "item": "Item 1", "content": (
        "Our cloud customers build long-running workloads on our infrastructure, and migrating those workloads to "
        "another provider requires significant engineering effort.")},
    "i4": {"chunk_id": "i4", "item": "Item 1", "content": (
        "High capital requirements and barriers to entry limit the number of operators in our regional markets.")},
}
NETWORK = "each additional seller attracts more buyers to the platform"
COST = "We operate a global logistics network that lowers our cost to serve each order."
OFF_TOPIC = "the principal competitive factors include selection, price and convenience"


def _ev(chunk_id, quote):
    return [{"chunk_id": chunk_id, "quote": quote}]


def _moat(moat_type, chunk_id, quote, rationale="r"):
    return {"type": moat_type, "rationale": rationale, "evidence": _ev(chunk_id, quote)}


def _run(**data):
    return verify_competition_and_moat(data, POOL, extract_segment_disclosures(filing(), 2025))


class TestMoats(unittest.TestCase):
    def test_verified_primary_and_secondary_keep_wide(self):
        out = _run(primary_moat=_moat("Network Effects", "i1", NETWORK),
                   secondary_moats=[_moat("Cost Advantage", "i1", COST)], moat_durability="wide", moat_trajectory="Stable")
        self.assertEqual((out.economic_moat_type, out.secondary_moat_types), ("Network Effects", ["Cost Advantage"]))
        self.assertEqual((out.moat_durability, out.moat_trajectory, out.moat_assessment), ("Wide", "Stable", "verified"))
        self.assertEqual([e.claim for e in out.evidence], ["primary_moat", "secondary_moat"])

    def test_wide_is_capped_with_one_quote(self):
        out = _run(primary_moat=_moat("Network Effects", "i1", NETWORK), moat_durability="Wide")
        self.assertEqual(out.moat_durability, "Narrow")
        self.assertIn("capped at Narrow", out.tally.notes[0])

    def test_none_durability_contradicts_a_verified_moat(self):
        out = _run(primary_moat=_moat("Network Effects", "i1", NETWORK), moat_durability="None")
        self.assertEqual(out.moat_durability, "Indeterminate")

    def test_off_topic_primary_promotes_the_verified_secondary(self):
        out = _run(primary_moat=_moat("Network Effects", "i2", OFF_TOPIC), secondary_moats=[_moat("Cost Advantage", "i1", COST)])
        self.assertEqual((out.economic_moat_type, out.secondary_moat_types), ("Cost Advantage", []))
        self.assertEqual(out.evidence[0].claim, "primary_moat")
        self.assertEqual((out.tally.quotes_off_topic, out.tally.claims_dropped), (1, 1))

    def test_unsupported_moat_is_downgraded(self):
        out = _run(primary_moat=_moat("Network Effects", "i2", OFF_TOPIC), moat_durability="Wide", moat_trajectory="Expanding")
        self.assertEqual((out.economic_moat_type, out.moat_durability, out.moat_trajectory, out.moat_assessment),
                         ("None", "Indeterminate", "Indeterminate", "unsupported"))
        self.assertTrue(out.moat_rationale.startswith("Not assessed"))

    def test_explicit_no_moat(self):
        out = _run(primary_moat={"type": "none", "rationale": "", "evidence": []}, moat_durability="Narrow",
                   moat_trajectory="Expanding")
        self.assertEqual((out.moat_assessment, out.moat_durability, out.moat_trajectory), ("no_moat", "None", "Indeterminate"))
        self.assertEqual(out.moat_rationale, "The filing evidences no structural competitive advantage.")

    def test_segment_moats_match_extracted_segments(self):
        out = _run(primary_moat=_moat("Network Effects", "i1", NETWORK), segment_moats=[
            _moat("High Switching Costs", "i3", "migrating those workloads to another provider requires significant engineering effort")
            | {"segment_name": "Cloud segment"},
            _moat("Network Effects", "i1", NETWORK) | {"segment_name": "Advertising"},
        ])
        self.assertEqual([(m.segment_name, m.moat_type) for m in out.segment_specific_moats], [("Cloud", "High Switching Costs")])
        self.assertIn("'Advertising' dropped", out.tally.notes[0])

    def test_secondaries_skip_the_primary_type_and_stop_at_two(self):
        brand = "Each additional seller attracts more buyers to the platform"
        out = _run(primary_moat=_moat("Network Effects", "i1", NETWORK), secondary_moats=[
            _moat("Network Effects", "i1", brand),
            _moat("Cost Advantage", "i1", COST),
            _moat("High Switching Costs", "i3", "migrating those workloads to another provider"),
            _moat("Efficient Scale", "i4", "barriers to entry limit the number of operators"),
        ])
        self.assertEqual(out.secondary_moat_types, ["Cost Advantage", "High Switching Costs"])  # third one past the cap
        self.assertEqual(out.tally.quotes_verified, 5)

    def test_a_segment_moat_is_kept_once(self):
        quote = "migrating those workloads to another provider requires significant engineering effort"
        out = _run(primary_moat=_moat("Network Effects", "i1", NETWORK), segment_moats=[
            _moat("High Switching Costs", "i3", quote) | {"segment_name": "Cloud"},
            _moat("High Switching Costs", "i3", quote) | {"segment_name": "cloud segment"},
        ])
        self.assertEqual([m.segment_name for m in out.segment_specific_moats], ["Cloud"])

    def test_without_extracted_names_the_judges_segment_name_stands(self):
        quote = "migrating those workloads to another provider requires significant engineering effort"
        data = {"primary_moat": _moat("Network Effects", "i1", NETWORK),
                "segment_moats": [_moat("High Switching Costs", "i3", quote) | {"segment_name": "Cloud workloads"}]}
        out = verify_competition_and_moat(data, POOL, SegmentExtraction())
        self.assertEqual([m.segment_name for m in out.segment_specific_moats], ["Cloud workloads"])

    def test_coerce_moat(self):
        self.assertEqual(coerce_moat("network effect"), "Network Effects")
        self.assertEqual(coerce_moat("Strong BRAND"), "Intangible Assets / Brand")
        self.assertEqual(coerce_moat("n/a"), "None")
        self.assertIsNone(coerce_moat("Moat-ish"))
        self.assertIsNone(coerce_moat(3))


class TestThreats(unittest.TestCase):
    def test_threats_need_a_competition_quote_and_are_capped(self):
        good = {"threat": "Retail rivals.", "evidence": _ev("i2", "intense competition from physical retailers")}
        fake = {"threat": "Fake.", "evidence": _ev("i2", "a regulator has banned our cloud business entirely")}
        off = {"threat": "Logistics.", "evidence": _ev("i1", COST)}
        out = _run(competitive_threats=[good, fake, off] + [dict(good, threat=f"Rival {i}.") for i in range(4)])
        self.assertEqual(out.competitive_threats, ["Retail rivals.", "Rival 0.", "Rival 1.", "Rival 2."])
        self.assertEqual(out.tally.claims_dropped, 3)

    def test_no_supported_threat_is_noted(self):
        out = _run(competitive_threats=[{"threat": "Fake.", "evidence": _ev("i2", "regulators banned us")}])
        self.assertEqual(out.tally.notes[-1], "No proposed competitive threat was supported by a verified filing quote.")


class TestNotAssessed(unittest.TestCase):
    def test_no_judgement_means_no_moat_and_no_threats(self):
        out = not_assessed_competition("the qualitative LLM call failed")
        self.assertEqual(out.moat_fields()["moat_assessment"], "not_assessed")
        self.assertEqual(out.moat_rationale, "Not assessed: the qualitative LLM call failed.")
        self.assertEqual((out.competitive_threats, out.evidence, out.tally.notes), ([], [], []))


if __name__ == "__main__":
    unittest.main()
