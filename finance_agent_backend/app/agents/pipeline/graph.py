"""Builds the LangGraph state machine from the route table; the only module that knows LangGraph."""

from typing import Callable, Dict, List

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.agents.pipeline.node import make_agent_node, supervisor_node
from app.agents.pipeline.routes import ROUTES, Step, all_specs, step_specs
from app.agents.pipeline.run_state import ResearchRunState


def first_step(route: str, routes: Dict[str, List[Step]] = ROUTES) -> List[str]:
    steps = routes.get(route)
    return [spec.name for spec in step_specs(steps[0])] if steps else [END]


def next_step(agent_name: str, route: str, routes: Dict[str, List[Step]] = ROUTES) -> List[str]:
    steps = routes.get(route) or []
    for index, step in enumerate(steps):
        if agent_name in [spec.name for spec in step_specs(step)]:
            return [spec.name for spec in step_specs(steps[index + 1])] if index + 1 < len(steps) else [END]
    return [END]


def _after_supervisor(routes: Dict[str, List[Step]]) -> Callable[[ResearchRunState], List[str]]:
    def after_supervisor(state: ResearchRunState) -> List[str]:
        turn = state["run"]
        return first_step(turn.route, routes) if turn.run_context is not None else [END]

    return after_supervisor


def _after_agent(agent_name: str, routes: Dict[str, List[Step]]) -> Callable[[ResearchRunState], List[str]]:
    def after_agent(state: ResearchRunState) -> List[str]:
        return next_step(agent_name, state["run"].route, routes)

    after_agent.__name__ = f"after_{agent_name}"
    return after_agent


def build_graph(routes: Dict[str, List[Step]] = ROUTES) -> CompiledStateGraph:
    builder = StateGraph(ResearchRunState)
    builder.add_node("supervisor", supervisor_node)
    specs = all_specs(routes)
    for spec in specs:
        builder.add_node(spec.name, make_agent_node(spec))

    builder.add_edge(START, "supervisor")
    starts = {name for route in routes for name in first_step(route, routes)} | {END}
    builder.add_conditional_edges("supervisor", _after_supervisor(routes), sorted(starts))
    for spec in specs:
        successors = {name for route in routes for name in next_step(spec.name, route, routes)}
        builder.add_conditional_edges(spec.name, _after_agent(spec.name, routes), sorted(successors))
    return builder.compile()
