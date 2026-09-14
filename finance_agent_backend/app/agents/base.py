from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class AgentOutput:
    content: str
    sources: List[Dict[str, Any]] = field(default_factory=list)


class BaseAgent(ABC):
    """Abstract Base Class for all specialized agents (LSP)."""

    @abstractmethod
    def run(self, messages: List[Dict[str, str]]) -> AgentOutput:
        """Executes the agent with conversational history and returns standardized AgentOutput."""
        pass
