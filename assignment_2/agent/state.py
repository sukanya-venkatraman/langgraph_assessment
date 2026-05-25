"""
State definition for the Stock Analysis Agent.

In LangGraph, the State is the single source of truth that flows through
every node. Each node reads from it and writes back to it.
"""

from typing import TypedDict, Optional
import pandas as pd


class StockAgentState(TypedDict):
    """
    The shared state that travels through the entire agent graph.

    Fields are populated incrementally as each node runs:
      - ticker:          set by the user before the graph starts
      - raw_data:        populated by the 'fetch_data' node
      - indicators:      populated by the 'calculate_indicators' node
      - recommendation:  populated by the 'generate_recommendation' node
      - report:          populated by the 'format_report' node
      - error:           set by any node that encounters a problem
    """
    ticker: str
    raw_data: Optional[pd.DataFrame]
    indicators: Optional[dict]
    recommendation: Optional[dict]
    report: Optional[str]
    error: Optional[str]
