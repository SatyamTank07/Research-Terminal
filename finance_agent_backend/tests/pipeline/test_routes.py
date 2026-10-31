import unittest

from langgraph.graph import END

from app.agents.pipeline.graph import first_step, next_step
from app.agents.pipeline.routes import ROUTES, all_specs, step_specs

# SSE `node` names are part of the API contract.
NODE_NAMES = {"business_strategist", "financial_auditor", "risk_analyst", "forecasting_analyst",
              "valuation_specialist", "lead_synthesizer"}


class TestRouteTable(unittest.TestCase):
    def test_routes_match_the_supervisors_report_types(self):
        self.assertEqual(set(ROUTES), {"full_10k_report", "dcf_valuation_only", "financial_audit_only",
                                       "business_moat_only", "risk_factors_only"})

    def test_every_need_is_produced_earlier_in_its_route(self):
        for route, steps in ROUTES.items():
            produced = set()
            for step in steps:
                for spec in step_specs(step):
                    self.assertLessEqual(spec.needs, produced, f"{route}: {spec.name} needs {spec.needs - produced}")
                produced |= {spec.output_key for spec in step_specs(step)}

    def test_every_route_ends_with_the_synthesizer_and_can_satisfy_it(self):
        for route, steps in ROUTES.items():
            last = step_specs(steps[-1])
            self.assertEqual([spec.name for spec in last], ["lead_synthesizer"], route)
            earlier = {spec.output_key for step in steps[:-1] for spec in step_specs(step)}
            self.assertTrue(last[0].needs_any & earlier, route)

    def test_node_names_are_unchanged(self):
        self.assertEqual({spec.name for spec in all_specs()}, NODE_NAMES)


class TestRouting(unittest.TestCase):
    def test_full_report_starts_with_three_agents_in_parallel(self):
        self.assertEqual(first_step("full_10k_report"), ["business_strategist", "financial_auditor", "risk_analyst"])

    def test_auditor_successor_depends_on_the_route(self):
        self.assertEqual(next_step("financial_auditor", "financial_audit_only"), ["lead_synthesizer"])
        self.assertEqual(next_step("financial_auditor", "dcf_valuation_only"), ["forecasting_analyst"])
        self.assertEqual(next_step("financial_auditor", "full_10k_report"), ["forecasting_analyst"])

    def test_synthesizer_ends_every_route(self):
        for route in ROUTES:
            self.assertEqual(next_step("lead_synthesizer", route), [END])

    def test_unknown_route_ends(self):
        self.assertEqual(first_step("no_such_route"), [END])
