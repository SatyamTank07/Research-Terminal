from app.agents.base import AgentOutput, BaseAgent
from app.agents.registry import AgentRegistry
import app.agents.specialized  # Triggers self-registration of agents

__all__ = ["BaseAgent", "AgentOutput", "AgentRegistry"]
