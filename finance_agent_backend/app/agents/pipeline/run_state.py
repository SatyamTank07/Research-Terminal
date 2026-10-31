"""The state one research run carries through the graph; it imports no agent schema."""

import operator
from typing import Annotated, Any, Dict, List, Optional, TypedDict

from app.agents.pipeline.contract import Conversation, RunNote


def merge_outputs(current: Optional[Dict[str, Any]], update: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Parallel agents each add their own key; merging keeps every one of them."""
    return {**(current or {}), **(update or {})}


class ResearchRunState(TypedDict, total=False):
    conversation: Conversation
    run: Any  # the Supervisor's TurnResolution: route, RunContext, session update
    outputs: Annotated[Dict[str, Any], merge_outputs]
    notes: Annotated[List[RunNote], operator.add]
    final: Any  # Final10KResearchReport, or the conversational reply
