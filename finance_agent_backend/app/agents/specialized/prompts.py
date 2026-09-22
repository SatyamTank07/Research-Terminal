from pathlib import Path
from typing import Any
from jinja2 import Environment, FileSystemLoader

SPECIALIZED_DIR = Path(__file__).parent


def _get_template_search_dirs() -> list:
    """Collects all specialized agent directories containing .j2 templates."""
    search_dirs = [str(SPECIALIZED_DIR)]
    if SPECIALIZED_DIR.exists():
        for p in SPECIALIZED_DIR.iterdir():
            if p.is_dir() and not p.name.startswith("__"):
                search_dirs.append(str(p))
    return search_dirs


# Centralized Jinja2 environment searching across specialized/<agent>/ folders
_jinja_env = Environment(
    loader=FileSystemLoader(_get_template_search_dirs()),
    trim_blocks=True,
    lstrip_blocks=True,
)


def render_prompt(template_name: str, **context: Any) -> str:
    """Renders any .j2 prompt template by name (e.g. 'business_strategist' or 'prompt_business_strategist.j2').

    Args:
        template_name: Name of the template with or without .j2 extension.
        **context: Keyword arguments to pass into the Jinja2 template.

    Returns:
        Rendered string prompt.
    """
    clean_name = template_name[:-3] if template_name.endswith(".j2") else template_name

    candidate_filenames = [
        f"{clean_name}.j2",
        f"{clean_name}_prompt.j2",
        f"prompt_{clean_name}.j2",
    ]
    if clean_name.endswith("_prompt"):
        candidate_filenames.append(f"{clean_name[:-7]}.j2")
    if clean_name.startswith("prompt_"):
        candidate_filenames.append(f"{clean_name[7:]}.j2")

    for filename in candidate_filenames:
        try:
            template = _jinja_env.get_template(filename)
            return template.render(**context).strip()
        except Exception:
            continue

    # Final attempt with the literal name to produce the standard TemplateNotFound if missing
    fallback_name = f"{template_name}.j2" if not template_name.endswith(".j2") else template_name
    template = _jinja_env.get_template(fallback_name)
    return template.render(**context).strip()


__all__ = ["render_prompt", "SPECIALIZED_DIR"]
