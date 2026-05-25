"""
Node implementations for the Stock Analysis Agent.

Each node is a plain Python function with this signature:
    def node_name(state: StockAgentState) -> dict

LangGraph merges the returned dict back into the state automatically,
so nodes only need to return the keys they modify.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from agent.state import StockAgentState



# Node 1: Fetch Data

def fetch_data(state: StockAgentState) -> dict:
    """
    Downloads 60 days of OHLCV data from Yahoo Finance.

    Returns raw_data (a DataFrame) or sets error on failure.
    """
    ticker = state["ticker"].upper().strip()

    try:
        end_date = datetime.today()
        start_date = end_date - timedelta(days=90)  # extra buffer for weekends/holidays

        tk = yf.Ticker(ticker)
        df = tk.history(start=start_date, end=end_date)

        if df.empty:
            return {"error": f"No data returned for ticker '{ticker}'. Is it valid?"}

        # Keep only the last 60 trading days
        df = df.tail(60).copy()
        df.index = pd.to_datetime(df.index)

        return {"raw_data": df, "ticker": ticker}

    except Exception as exc:
        return {"error": f"Network or data error: {exc}"}


# Node 2: Calculate Indicators

def calculate_indicators(state: StockAgentState) -> dict:
    """
    Computes three technical indicators from the raw OHLCV data:

      SMA-10  – 10-day Simple Moving Average
      SMA-20  – 20-day Simple Moving Average
      RSI-14  – 14-day Relative Strength Index

    All three are appended as new columns on the DataFrame, and a
    summary dict of the *latest* values is stored in state['indicators'].
    """
    if state.get("error"):
        return {}

    df = state["raw_data"].copy()
    close = df["Close"]

    # Simple Moving Averages
    df["SMA_10"] = close.rolling(window=10).mean()
    df["SMA_20"] = close.rolling(window=20).mean()

    # RSI 
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # First average: simple mean over the initial 14-period window
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI_14"] = 100 - (100 / (1 + rs))

    # Snapshot of the most-recent values
    latest = df.iloc[-1]
    indicators = {
        "current_price": round(float(latest["Close"]), 2),
        "sma_10":        round(float(latest["SMA_10"]), 2) if pd.notna(latest["SMA_10"]) else None,
        "sma_20":        round(float(latest["SMA_20"]), 2) if pd.notna(latest["SMA_20"]) else None,
        "rsi_14":        round(float(latest["RSI_14"]), 2) if pd.notna(latest["RSI_14"]) else None,
        "volume":        int(latest["Volume"]),
        "data_df":       df,          # keep enriched DataFrame for the report
    }

    return {"indicators": indicators}


# Node 3: Generate Recommendation

def generate_recommendation(state: StockAgentState) -> dict:
    """
    Produces a BUY / HOLD / SELL signal from the indicator snapshot.

    Scoring rubric (each criterion adds +1 BUY signal or +1 SELL signal):

      Trend signals (SMA crossover):
        BUY  – price > SMA-10 > SMA-20  (uptrend)
        SELL – price < SMA-10 < SMA-20  (downtrend)
        else – neutral

      Momentum signals (RSI thresholds):
        BUY  – RSI < 30  (oversold)
        SELL – RSI > 70  (overbought)
        else – neutral

    Final decision:
        buy_signals > sell_signals  → BUY
        sell_signals > buy_signals  → SELL
        tie                         → HOLD
    """
    if state.get("error"):
        return {}

    ind = state["indicators"]
    price   = ind["current_price"]
    sma10   = ind["sma_10"]
    sma20   = ind["sma_20"]
    rsi     = ind["rsi_14"]

    buy_signals  = []
    sell_signals = []
    reasons      = []

    # Trend analysis 
    if sma10 and sma20:
        if price > sma10 > sma20:
            buy_signals.append("price_above_smas")
            reasons.append(f"Price (${price}) is above both SMA-10 (${sma10}) and SMA-20 (${sma20}) — uptrend")
        elif price < sma10 < sma20:
            sell_signals.append("price_below_smas")
            reasons.append(f"Price (${price}) is below both SMA-10 (${sma10}) and SMA-20 (${sma20}) — downtrend")
        else:
            reasons.append(f"Mixed SMA signal — price ${price}, SMA-10 ${sma10}, SMA-20 ${sma20}")

        if sma10 > sma20:
            buy_signals.append("golden_cross")
            reasons.append(f"SMA-10 (${sma10}) > SMA-20 (${sma20}) — bullish crossover")
        else:
            sell_signals.append("death_cross")
            reasons.append(f"SMA-10 (${sma10}) < SMA-20 (${sma20}) — bearish crossover")

    # RSI / momentum analysis 
    if rsi is not None:
        if rsi < 30:
            buy_signals.append("oversold")
            reasons.append(f"RSI {rsi} < 30 — oversold territory, potential reversal")
        elif rsi > 70:
            sell_signals.append("overbought")
            reasons.append(f"RSI {rsi} > 70 — overbought territory, potential pullback")
        else:
            reasons.append(f"RSI {rsi} is neutral (30–70 range)")

    # Final verdict 
    nb = len(buy_signals)
    ns = len(sell_signals)

    if nb > ns:
        action     = "BUY"
        confidence = "High" if nb - ns >= 2 else "Moderate"
    elif ns > nb:
        action     = "SELL"
        confidence = "High" if ns - nb >= 2 else "Moderate"
    else:
        action     = "HOLD"
        confidence = "Moderate"

    recommendation = {
        "action":       action,
        "confidence":   confidence,
        "buy_signals":  buy_signals,
        "sell_signals": sell_signals,
        "reasons":      reasons,
    }

    return {"recommendation": recommendation}


# Node 4: Format Report

def format_report(state: StockAgentState) -> dict:
    """
    Renders a plain-text analysis report from all accumulated state.
    This is intentionally the *last* node — it only reads, never writes
    back anything that other nodes depend on.
    """
    if state.get("error"):
        report = f"""

STOCK ANALYSIS — ERROR             

Ticker : {state.get('ticker', 'N/A')}
Error  : {state['error']}

Please check the ticker symbol and your internet connection.
"""
        return {"report": report}

    ticker = state["ticker"]
    ind    = state["indicators"]
    rec    = state["recommendation"]
    df     = ind["data_df"]


    # Price change over the 60-day window
    first_close = float(df["Close"].iloc[0])
    last_close  = ind["current_price"]
    pct_change  = ((last_close - first_close) / first_close) * 100
    direction   = "up" if pct_change >= 0 else "down"

    report = f"""

  STOCK ANALYSIS REPORT — {ticker:<6}          


  Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
  Data range: {df.index[0].date()}  ->  {df.index[-1].date()}  ({len(df)} trading days)


  PRICE SUMMARY                                       

  Current Price : ${last_close:>10.2f}
  60-day Start  : ${first_close:>10.2f}
  60-day Change : {direction} {abs(pct_change):>6.2f}%
  Volume (last) : {ind['volume']:>12,}


  TECHNICAL INDICATORS                                

  SMA-10  : ${ind['sma_10'] or 'N/A':>10}
  SMA-20  : ${ind['sma_20'] or 'N/A':>10}
  RSI-14  : {ind['rsi_14'] or 'N/A':>11}

   SIGNAL ANALYSIS                                     

"""
    for reason in rec["reasons"]:
        report += f"  {reason}\n"

    report += f"""
    
     RECOMMENDATION                                     
     
  Decision   : {rec['action']}
  Confidence : {rec['confidence']}
  Buy signals : {len(rec['buy_signals'])}   |   Sell signals: {len(rec['sell_signals'])}

"""

    return {"report": report}
