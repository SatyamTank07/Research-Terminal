import logging
from typing import Callable, Dict, List, Type
from app.agents.base import BaseAgent

logger = logging.getLogger("finance_agent.agents.registry")


class AgentRegistry:
    """Factory and registry for plug-and-play specialized agents (OCP & DIP)."""

    _registry: Dict[str, Type[BaseAgent]] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[Type[BaseAgent]], Type[BaseAgent]]:
        """Decorator to register a new agent class with a unique identifier."""
        def decorator(agent_cls: Type[BaseAgent]) -> Type[BaseAgent]:
            cls._registry[name.lower()] = agent_cls
            logger.info(f"Registered agent '{name}': {agent_cls.__name__}")
            return agent_cls

        return decorator

    @classmethod
    def get(cls, name: str = "financial_analyst") -> BaseAgent:
        """Retrieves and instantiates an agent by name. Falls back to default if not found."""
        normalized_name = (name or "financial_analyst").lower()
        agent_cls = cls._registry.get(normalized_name)
        if not agent_cls:
            logger.warning(
                f"Agent '{name}' not found in registry. Falling back to 'financial_analyst'."
            )
            agent_cls = cls._registry.get("financial_analyst")

        if not agent_cls:
            raise ValueError(
                f"Requested agent '{name}' is not registered and fallback 'financial_analyst' is missing."
            )

        return agent_cls()

    @classmethod
    def list_agents(cls) -> List[str]:
        """Returns the names of all registered agents."""
        return list(cls._registry.keys())
