"""Renders an agent's Jinja2 templates by agent folder and path inside it."""

from functools import lru_cache
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader

AGENTS_DIR = Path(__file__).resolve().parent.parent / "specialized"


@lru_cache(maxsize=None)
def _agent_environment(agent_name: str) -> Environment:
    agent_dir = AGENTS_DIR / agent_name
    if not agent_dir.is_dir():
        raise ValueError(f"No agent folder named {agent_name!r} under {AGENTS_DIR}")
    return Environment(loader=FileSystemLoader(str(agent_dir)), trim_blocks=True, lstrip_blocks=True)


def render_prompt(agent_name: str, template_path: str, /, **context: Any) -> str:
    """Renders `specialized/<agent_name>/<template_path>`; includes and imports resolve inside that folder."""
    return _agent_environment(agent_name).get_template(template_path).render(**context).strip()


__all__ = ["AGENTS_DIR", "render_prompt"]
