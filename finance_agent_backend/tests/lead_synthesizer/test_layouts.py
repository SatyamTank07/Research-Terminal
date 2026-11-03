"""Every report type has exactly one layout, named after its route."""

import unittest
from pathlib import Path

from app.agents.pipeline.routes import ROUTES
from app.agents.specialized.lead_synthesizer.modules.report_coverage import REPORT_LAYOUTS

REPORT_DIR = Path(__file__).resolve().parents[2] / "app" / "agents" / "specialized" / "lead_synthesizer" / "report"
SHARED_BLOCKS = {"report_blocks.j2", "audit_tables.j2"}


class TestReportLayouts(unittest.TestCase):
    def test_every_route_has_exactly_one_layout_and_no_layout_lacks_a_route(self):
        layouts = {path.name for path in REPORT_DIR.glob("*.j2")} - SHARED_BLOCKS
        self.assertEqual(layouts, {f"{route}.j2" for route in ROUTES})

    def test_report_coverage_knows_every_layout(self):
        self.assertEqual(set(REPORT_LAYOUTS), set(ROUTES))
