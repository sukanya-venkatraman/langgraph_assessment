"""
Command-line entry point for the Stock Analysis Agent.

Usage:
    python main.py             # Activates interactive input prompt
    python main.py AAPL        # Directly runs specified ticker
"""

import sys
from agent.graph import build_graph

def analyze(ticker: str) -> None:
    """Compiles the LangGraph workflow and runs a live analysis on the ticker."""
    print(f"\nCompiling workflow and analyzing market data for {ticker.upper()}...")
    graph = build_graph()
    result = graph.invoke({"ticker": ticker})
    print(result["report"])

if __name__ == "__main__":
    # Check if any tickers were passed as terminal arguments
    if len(sys.argv) > 1:
        tickers = sys.argv[1:]
        for t in tickers:
            analyze(t)
    else:
        # No arguments passed -> Fallback to interactive input prompt
        print("=" * 55)
        print(" WELCOME TO THE LANGGRAPH STOCK ANALYSIS AGENT ")
        print("=" * 55)
        
        user_ticker = input("Enter a stock ticker symbol (e.g., AAPL, TSLA, NVDA): ")
        
        # Guard against empty/whitespace inputs
        if user_ticker.strip():
            analyze(user_ticker)
        else:
            print("Input cannot be empty. Exiting analysis.")