"""Prompt templates resolve by agent folder and path, never through a global namespace."""

import unittest

from jinja2 import TemplateNotFound

from app.agents.shared.prompts import render_prompt


class TestRenderPrompt(unittest.TestCase):
    def test_system_prompt_renders_from_its_agent_folder(self):
        self.assertIn("Senior Equity Research Director", render_prompt("supervisor", "prompt_supervisor.j2"))

    def test_another_agents_template_is_not_found(self):
        with self.assertRaises(TemplateNotFound):
            render_prompt("risk_analyst", "prompt_supervisor.j2")

    def test_unknown_agent_folder_raises(self):
        with self.assertRaises(ValueError):
            render_prompt("no_such_agent", "prompt_no_such_agent.j2")

    def test_context_may_use_the_argument_names(self):
        self.assertIsInstance(render_prompt("supervisor", "prompt_supervisor.j2", agent_name="x", template_path="y"), str)


if __name__ == "__main__":
    unittest.main()
