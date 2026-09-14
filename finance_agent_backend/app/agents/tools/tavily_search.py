import logging
import os
from typing import Optional
from dotenv import load_dotenv
from langchain_tavily import TavilySearch

logger = logging.getLogger("finance_agent.tools.tavily")


def get_tavily_tool() -> Optional[TavilySearch]:
    """Initializes and returns TavilySearch if TAVILY_API_KEY is configured in environment."""
    load_dotenv(override=True)
    tavily_key = os.getenv("TAVILY_API_KEY")

    if not tavily_key:
        logger.warning(
            "TAVILY_API_KEY not found in environment. Agent is running without real-time search capabilities."
        )
        return None

    logger.info("Initializing Tavily search tool (topic='finance').")
    return TavilySearch(
        max_results=5,
        topic="finance",
    )
