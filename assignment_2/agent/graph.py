"""
Graph construction for the Stock Analysis Agent.

LangGraph uses a directed graph where:
  - Nodes  = Python functions that transform state
  - Edges  = define execution order (normal or conditional)
  - START  = the entry-point sentinel
  - END    = the exit-point sentinel

Graph topology:

    START
      │
      ▼
  fetch_data
      │
      ▼  (conditional — skip to format_report on error)
  calculate_indicators
      │
      ▼
  generate_recommendation
      │
      ▼
  format_report
      │
      ▼
     END
"""

from langgraph.graph import StateGraph, START, END

from agent.state import StockAgentState
from agent.nodes import (
    fetch_data,
    calculate_indicators,
    generate_recommendation,
    format_report,
)


def _route_after_fetch(state: StockAgentState) -> str:
    """
    Conditional edge function.

    If fetch_data set an error we skip straight to format_report
    (which knows how to render an error report).
    Otherwise we proceed normally to calculate_indicators.
    """
    if state.get("error"):
        return "format_report"
    return "calculate_indicators"


def build_graph() -> StateGraph:
    """
    Assemble and compile the LangGraph StateGraph.

    Returns a compiled graph ready to be invoked with:
        graph.invoke({"ticker": "AAPL"})
    """
    graph = StateGraph(StockAgentState)

    # ── Register nodes ────────────────────────────────────────────
    graph.add_node("fetch_data",              fetch_data)
    graph.add_node("calculate_indicators",    calculate_indicators)
    graph.add_node("generate_recommendation", generate_recommendation)
    graph.add_node("format_report",           format_report)

    # ── Wire edges ────────────────────────────────────────────────
    # Entry point
    graph.add_edge(START, "fetch_data")

    # After fetch: conditional branch on error
    graph.add_conditional_edges(
        "fetch_data",
        _route_after_fetch,
        {
            "calculate_indicators": "calculate_indicators",
            "format_report":        "format_report",
        },
    )

    # Happy path continues linearly
    graph.add_edge("calculate_indicators",    "generate_recommendation")
    graph.add_edge("generate_recommendation", "format_report")

    # All paths end here
    graph.add_edge("format_report", END)

    return graph.compile()
