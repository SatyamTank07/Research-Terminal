"""The segment and revenue-mix tables embedded verbatim in the report."""

import unittest

from app.agents.specialized.business_strategist.modules.revenue_mix.calculate import extract_segment_disclosures
from app.agents.specialized.business_strategist.modules.revenue_mix.render import render_revenue_mix_markdown, render_segment_table_markdown
from tests.business_strategist.filing_fixtures import filing


class TestRenderers(unittest.TestCase):
    def test_renderers(self):
        ex = extract_segment_disclosures(filing(), 2025)
        seg_md = render_segment_table_markdown(ex.segments)
        self.assertIn("| Cloud | 200 | 20.0% | 33.3% | 70 | 35.0% | 63.6% |", seg_md)
        self.assertIn("Operating income ($M)", seg_md)
        mix_md = render_revenue_mix_markdown(ex.revenue_lines)
        self.assertIn("| Online stores | 500 | 50.0% | 11.1% |", mix_md)
        self.assertIn("No reportable-segment", render_segment_table_markdown([]))
        self.assertIn("No product", render_revenue_mix_markdown([]))


if __name__ == "__main__":
    unittest.main()
