import json
import logging
from typing import Any, Dict, Generic, List, Optional, Type, TypeVar, Union
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from app.agents.shared.llm_call import parse_json_content

logger = logging.getLogger("finance_agent.agents.base")


T = TypeVar("T", bound=BaseModel)


class StructuredAgent(Generic[T]):
    """Extensible declarative base class for agents emitting structured Pydantic artifacts."""

    prompt_name: Optional[str] = None
    tools: List[Any] = []
    output_schema: Optional[Type[T]] = None
    default_recursion_limit: int = 50

    def __init__(
        self,
        model_name: str = "openai:gpt-4o-mini",
        recursion_limit: Optional[int] = None,
    ):
        self.model_name = model_name
        self.recursion_limit = (
            recursion_limit if recursion_limit is not None else self.default_recursion_limit
        )
        self._cached_agent = None

    def get_tools(self) -> List[Any]:
        """Returns the tools bound to this agent. Can be overridden if tools are dynamic."""
        return self.tools

    def get_system_prompt(self) -> str:
        """Renders and returns the system prompt."""
        if self.prompt_name:
            from app.agents.shared.prompts import render_prompt
            return render_prompt(self.prompt_name, f"prompt_{self.prompt_name}.j2")
        raise NotImplementedError(
            f"Class {self.__class__.__name__} must define 'prompt_name' (its agent folder) or override 'get_system_prompt()'."
        )


    def _get_or_create_agent(self):
        """Initializes and caches the LangChain agent instance."""
        load_dotenv(override=True)
        if self._cached_agent is not None:
            return self._cached_agent

        tools = self.get_tools()
        system_prompt = self.get_system_prompt()

        model_clean = self.model_name.replace("openai:", "")
        llm = ChatOpenAI(
            model=model_clean,
            temperature=0,
            max_retries=5,
        )

        agent = create_agent(
            model=llm,
            tools=tools,
            system_prompt=system_prompt,
        )

        self._cached_agent = agent
        return agent

    def _extract_citations_from_messages(self, messages: List[Any]) -> List[Dict[str, Any]]:
        """Extracts chunk citations and section breadcrumbs from tool message payloads."""
        citations: List[Dict[str, Any]] = []
        seen_chunk_ids = set()

        for msg in messages:
            raw_content = getattr(msg, "content", "")
            if isinstance(raw_content, str) and "chunk_id" in raw_content:
                try:
                    parsed = json.loads(raw_content)
                    items = parsed if isinstance(parsed, list) else [parsed] if isinstance(parsed, dict) else []
                    for item in items:
                        if isinstance(item, dict) and "chunk_id" in item:
                            cid = item.get("chunk_id")
                            if cid and cid not in seen_chunk_ids:
                                seen_chunk_ids.add(cid)
                                citation: Dict[str, Any] = {"chunk_id": cid}
                                for key in ["ticker", "fiscal_year", "item", "breadcrumb", "sub_section"]:
                                    val = item.get(key)
                                    if val is not None:
                                        citation[key] = val
                                citations.append(citation)
                except Exception:
                    pass

        return citations

    def _pre_validate_data(
        self,
        data: Dict[str, Any],
        result: Dict[str, Any],
        **kwargs,
    ) -> Dict[str, Any]:
        """Lifecycle hook to modify or overlay tool data onto the dictionary prior to Pydantic validation."""
        return data

    def _extract_structured_output(
        self,
        result: Dict[str, Any],
        ticker: Optional[str] = None,
        fiscal_year: Optional[int] = None,
        fallback_defaults: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> T:
        """Extracts, repairs, and validates a Pydantic artifact from agent execution results."""
        if not self.output_schema:
            raise ValueError(f"No output_schema defined on {self.__class__.__name__}")

        messages = result.get("messages", [])
        structured = result.get("structured_response")

        if isinstance(structured, self.output_schema):
            output = structured
        else:
            if isinstance(structured, dict):
                parsed = dict(structured)
            else:
                # Fallback: Parse from last message content if model emitted JSON string
                last_msg = messages[-1] if messages else None
                parsed = parse_json_content(getattr(last_msg, "content", "") if last_msg else "")

            if fallback_defaults:
                for k, v in fallback_defaults.items():
                    parsed.setdefault(k, v)

            if ticker:
                parsed.setdefault("ticker", ticker.upper())
            if fiscal_year:
                parsed.setdefault("fiscal_year", fiscal_year)

            # Pre-validation hook: allows agents to overlay deterministic tool math before Pydantic validates
            parsed = self._pre_validate_data(parsed, result, ticker=ticker, fiscal_year=fiscal_year, **kwargs)

            try:
                output = self.output_schema.model_validate(parsed)
            except Exception as e:
                raw_content = getattr(messages[-1], "content", "") if messages else ""
                logger.error(
                    f"Failed to parse structured output for {self.__class__.__name__}: {e}. "
                    f"Raw content: {str(raw_content)[:500]}"
                )
                raise ValueError(f"Agent did not return a valid {self.output_schema.__name__}: {e}")

        # Post-processing: backfill citations from tool messages if empty
        if hasattr(output, "citations"):
            collected_citations = self._extract_citations_from_messages(messages)
            if not getattr(output, "citations", None) and collected_citations:
                output.citations = collected_citations

        # Ensure ticker and fiscal year match target query
        if ticker and hasattr(output, "ticker"):
            if not getattr(output, "ticker", None) or getattr(output, "ticker") != ticker.upper():
                output.ticker = ticker.upper()
        if fiscal_year and hasattr(output, "fiscal_year"):
            if not getattr(output, "fiscal_year", None):
                output.fiscal_year = fiscal_year

        return output

    def _post_process_output(self, output: T, result: Dict[str, Any], **kwargs) -> T:
        """Lifecycle hook for specialized post-processing (e.g. math overlays, markdown tables)."""
        return output

    def execute_structured(
        self,
        query_or_messages: Union[str, List[Dict[str, str]]],
        ticker: Optional[str] = None,
        fiscal_year: Optional[int] = None,
        fallback_defaults: Optional[Dict[str, Any]] = None,
        callbacks: Optional[List[Any]] = None,
        **kwargs,
    ) -> T:
        """Unified execution pipeline: invokes agent, extracts structured output, and runs post-processing."""
        active_agent = self._get_or_create_agent()
        if isinstance(query_or_messages, str):
            messages = [{"role": "user", "content": query_or_messages}]
        else:
            messages = query_or_messages

        config: Dict[str, Any] = {"recursion_limit": self.recursion_limit}
        if callbacks:
            config["callbacks"] = callbacks

        result = active_agent.invoke(
            {"messages": messages},
            config=config,
        )

        output = self._extract_structured_output(
            result,
            ticker=ticker,
            fiscal_year=fiscal_year,
            fallback_defaults=fallback_defaults,
            **kwargs,
        )

        output = self._post_process_output(output, result, ticker=ticker, fiscal_year=fiscal_year, **kwargs)
        return output
