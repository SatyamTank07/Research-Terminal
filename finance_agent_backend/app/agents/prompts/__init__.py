from pathlib import Path
from typing import Any
from jinja2 import Environment, FileSystemLoader

PROMPTS_DIR = Path(__file__).parent

# Centralized Jinja2 environment for all agent prompt templates
_jinja_env = Environment(
    loader=FileSystemLoader(PROMPTS_DIR),
    trim_blocks=True,
    lstrip_blocks=True,
)


def render_prompt(template_name: str, **context: Any) -> str:
    """Renders any .j2 prompt template by name (e.g. 'financial_analyst' or 'financial_analyst.j2').

    Args:
        template_name: Name of the template with or without .j2 extension.
        **context: Keyword arguments to pass into the Jinja2 template.

    Returns:
        Rendered string prompt.
    """
    if not template_name.endswith(".j2"):
        template_name = f"{template_name}.j2"

    template = _jinja_env.get_template(template_name)
    return template.render(**context).strip()


__all__ = ["render_prompt", "PROMPTS_DIR"]
