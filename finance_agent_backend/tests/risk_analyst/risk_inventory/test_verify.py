"""Risk inventory screens: immaterial language, generic boilerplate, and what a High impact needs."""

import unittest

from app.agents.specialized.risk_analyst.modules.risk_inventory import (
    is_boilerplate,
    states_immaterial,
    substantiates_high_impact,
)
from tests.risk_analyst.item1a_fixtures import CONCENTRATION


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

    def test_high_impact_substantiation(self):
        self.assertTrue(substantiates_high_impact(["anything"], "22% of revenue"))
        self.assertTrue(substantiates_high_impact(["sales to one customer represented 22% of total revenue"], None))
        self.assertTrue(substantiates_high_impact(["We incurred significant costs to remediate the defect"], None))
        self.assertTrue(substantiates_high_impact(["Export restrictions have adversely affected our China revenue"], None))
        self.assertFalse(substantiates_high_impact(["A cyberattack could disrupt our operations and may harm results"], None))
        self.assertFalse(substantiates_high_impact([], None))


if __name__ == "__main__":
    unittest.main()
