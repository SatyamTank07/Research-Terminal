import json
import logging
import os
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
from langchain.agents import create_agent

from app.agents.base import AgentOutput, BaseAgent
from app.agents.specialized.prompts import render_prompt
from app.agents.registry import AgentRegistry
from app.agents.tools.tavily_search import get_tavily_tool

logger = logging.getLogger("finance_agent.agents.financial_analyst")


@AgentRegistry.register("financial_analyst")
class FinancialAnalystAgent(BaseAgent):
    """General financial assistant and market research analyst."""

    def __init__(
        self,
        model_name: str = "openai:gpt-4o-mini",
        recursion_limit: int = 10,
    ):
        self.model_name = model_name
        self.recursion_limit = recursion_limit
        self._cached_agent = None
        self._last_tavily_key: Optional[str] = None

    def _get_or_create_agent(self):
        load_dotenv(override=True)
        tavily_key = os.getenv("TAVILY_API_KEY")

        if self._cached_agent is not None and self._last_tavily_key == tavily_key:
            return self._cached_agent

        tools = []
        tavily_tool = get_tavily_tool()
        if tavily_tool:
            tools.append(tavily_tool)

        agent = create_agent(
            model=self.model_name,
            tools=tools,
            system_prompt=render_prompt("financial_analyst"),
        )
        self._cached_agent = agent
        self._last_tavily_key = tavily_key
        return agent

    def run(self, messages: List[Dict[str, str]]) -> AgentOutput:
        active_agent = self._get_or_create_agent()
        result = active_agent.invoke(
            {"messages": messages},
            config={"recursion_limit": self.recursion_limit},
        )

        response_content = result["messages"][-1].content
        if isinstance(response_content, list):
            response_content = "\n".join(
                item.get("text", "") if isinstance(item, dict) else str(item)
                for item in response_content
            )

        sources: List[Dict[str, Any]] = []
        for msg in result.get("messages", []):
            msg_name = getattr(msg, "name", "")
            if msg_name and "tavily" in msg_name.lower():
                raw_tool_content = getattr(msg, "content", "")
                if isinstance(raw_tool_content, list):
                    for item in raw_tool_content:
                        if isinstance(item, dict) and "url" in item:
                            sources.append({
                                "title": item.get("title", "Source"),
                                "url": item.get("url", ""),
                                "snippet": (item.get("content") or "")[:200],
                            })
                elif isinstance(raw_tool_content, str):
                    try:
                        parsed = json.loads(raw_tool_content)
                        if isinstance(parsed, list):
                            for item in parsed:
                                if isinstance(item, dict) and "url" in item:
                                    sources.append({
                                        "title": item.get("title", "Source"),
                                        "url": item.get("url", ""),
                                        "snippet": (item.get("content") or "")[:200],
                                    })
                    except Exception:
                        pass

        return AgentOutput(content=str(response_content), sources=sources)
