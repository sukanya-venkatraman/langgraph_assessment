# Stock Market Analysis Agent

A LangGraph agent that analyzes stock tickers using technical indicators.

## Project Structure

```
stock_agent/
├── agent/
│   ├── __init__.py       # package exports
│   ├── state.py          # StockAgentState TypedDict
│   ├── nodes.py          # 4 node functions
│   └── graph.py          # StateGraph assembly + conditional routing
├── main.py               # CLI entry point
├── notebook.ipynb        # Jupyter demonstration
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

**CLI:**
```bash
python main.py AAPL
python main.py AAPL MSFT TSLA   # multiple tickers
```

**Python:**
```python
from agent.graph import build_graph
graph = build_graph()
result = graph.invoke({"ticker": "AAPL"})
print(result["report"])
```

## Agent Graph

```
START -> fetch_data (error?) -> format_report -> END
                  => (ok)
          calculate_indicators
                  =>
        generate_recommendation
                  =>
           format_report → END
```

## Recommendation Logic

| Condition               | Signal |
|-------------------------|--------|
| price > SMA-10 > SMA-20 | BUY    |
| price < SMA-10 < SMA-20 | SELL   |
| SMA-10 > SMA-20         | BUY    |
| SMA-10 < SMA-20         | SELL   |
| RSI < 30 (oversold)     | BUY    |
| RSI > 70 (overbought)   | SELL   |

More BUY signals = **BUY**, more SELL = **SELL**, tie = **HOLD**.
