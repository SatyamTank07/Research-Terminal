"""Finding ranking: severity, then newest year, then the screen order; equal findings keep their order."""

import unittest

from app.agents.specialized.financial_auditor.findings import FINDING_ORDER, ForensicFinding, rank_findings


def _f(code, severity="medium", year=2025, metric="m"):
    return ForensicFinding(code=code, severity=severity, fiscal_year=year, metric=metric, message=code)


class TestRankFindings(unittest.TestCase):
    def test_high_severity_ranks_before_newer_years(self):
        ranked = rank_findings([_f("SBC_INTENSITY", "medium", 2025), _f("TAX_ANOMALY", "high", 2023)])
        self.assertEqual([f.code for f in ranked], ["TAX_ANOMALY", "SBC_INTENSITY"])

    def test_newer_years_rank_first_within_a_severity(self):
        ranked = rank_findings([_f("SBC_INTENSITY", year=2023), _f("SBC_INTENSITY", year=2025)])
        self.assertEqual([f.fiscal_year for f in ranked], [2025, 2023])

    def test_ties_follow_the_screen_order_whatever_order_modules_emit_them(self):
        tied = [_f("WORKING_CAPITAL_DETERIORATION"), _f("NEGATIVE_OPERATING_LEVERAGE"), _f("RECEIVABLES_DIVERGENCE"),
                _f("SBC_INTENSITY"), _f("SHARE_DILUTION")]
        ranked = rank_findings(tied)
        self.assertEqual([f.code for f in ranked], sorted((f.code for f in tied), key=FINDING_ORDER.index))

    def test_equal_findings_keep_their_emission_order(self):
        ranked = rank_findings([_f("WORKING_CAPITAL_DETERIORATION", metric="dso_change_pct"),
                                _f("WORKING_CAPITAL_DETERIORATION", metric="dio_change_pct")])
        self.assertEqual([f.metric for f in ranked], ["dso_change_pct", "dio_change_pct"])

    def test_a_finding_without_a_year_ranks_after_dated_peers(self):
        ranked = rank_findings([_f("DATA_QUALITY", year=None), _f("DATA_QUALITY", year=2025)])
        self.assertEqual([f.fiscal_year for f in ranked], [2025, None])


if __name__ == "__main__":
    unittest.main()
