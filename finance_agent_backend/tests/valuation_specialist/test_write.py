"""The Valuation Specialist's write step: commentary trimming and the template fallback."""

import unittest

from app.agents.specialized.valuation_specialist.write import trim_to_complete_sentence


class TestCommentaryTruncation(unittest.TestCase):
    def test_trailing_fragment_is_trimmed(self):
        cut = "Fair value is $83.75 per share. The reverse DCF implies 23.0% growth, which is significantly higher than the modeled growth"
        self.assertEqual(trim_to_complete_sentence(cut), "Fair value is $83.75 per share.")
        whole = 'The base case is "Overvalued." Margin of safety is -68.1%.'
        self.assertEqual(trim_to_complete_sentence(whole), whole)
        self.assertEqual(trim_to_complete_sentence("no sentence end at all"), "no sentence end at all")


if __name__ == "__main__":
    unittest.main()
