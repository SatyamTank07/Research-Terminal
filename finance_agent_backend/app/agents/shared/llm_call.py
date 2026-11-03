"""One tool-free LLM completion (temperature 0) used by agent judge, repair and write calls, and the parser for
the JSON object in a model reply."""

import json
import re
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI


def complete_once(
    model_name: str,
    user_prompt: str,
    *,
    system_prompt: Optional[str] = None,
    max_tokens: Optional[int] = None,
    max_retries: int = 5,
    json_mode: bool = False,
    callbacks: Optional[List[Any]] = None,
) -> str:
    """Sends one prompt (optionally after a system prompt) and returns the reply text."""
    load_dotenv(override=True)
    llm_kwargs: Dict[str, Any] = {"model": model_name.replace("openai:", ""), "temperature": 0, "max_retries": max_retries}
    if max_tokens is not None:
        llm_kwargs["max_tokens"] = max_tokens
    if json_mode:
        llm_kwargs["model_kwargs"] = {"response_format": {"type": "json_object"}}
    messages: List[BaseMessage] = [SystemMessage(content=system_prompt)] if system_prompt is not None else []
    messages.append(HumanMessage(content=user_prompt))
    config = {"callbacks": callbacks} if callbacks else {}
    return ChatOpenAI(**llm_kwargs).invoke(messages, config=config).content


def parse_json_content(raw_content: Any) -> Dict[str, Any]:
    """Parses a JSON object from model output (strips code fences, falls back to the outermost {...})."""
    if isinstance(raw_content, list):
        raw_content = "".join(
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in raw_content
        )
    clean_json = (raw_content or "").strip() if isinstance(raw_content, str) else ""
    if clean_json.startswith("```"):
        lines = clean_json.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        clean_json = "\n".join(lines).strip()

    parsed: Any = {}
    if clean_json:
        try:
            parsed = json.loads(clean_json)
        except Exception:
            match = re.search(r"\{.*\}", clean_json, re.DOTALL)
            if match:
                try:
                    parsed = json.loads(match.group(0))
                except Exception:
                    pass
    return parsed if isinstance(parsed, dict) else {}


__all__ = ["complete_once", "parse_json_content"]
