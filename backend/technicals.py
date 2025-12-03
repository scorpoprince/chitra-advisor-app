"""
Technical indicator computations.

Functions in this module compute basic indicators such as moving averages,
volatility, RSI and support/resistance levels based on historical price
data retrieved via the prices module.
"""

import math
from .prices import _download_history


def get_basic_technicals(symbol: str) -> dict:
    """
    Compute a handful of simple technical statistics for a ticker.

    Returns a dict containing:
      - last_close
      - ma50
      - ma200
      - vol_annual
      - rsi14
      - one_year_return_pct
      - support
      - resistance
      - pivot
    """
    # Use one year of daily data for indicators.
    df = _download_history(symbol, period="1y", interval="1d")
    closes = df["Close"].astype(float)
    # Annualised volatility based on daily returns.
    daily_returns = closes.pct_change().dropna()
    vol_annual = float(daily_returns.std() * math.sqrt(252)) if not daily_returns.empty else 0.0
    # Moving averages.
    ma50 = float(closes.rolling(50).mean().iloc[-1])
    ma200 = float(closes.rolling(200).mean().iloc[-1])
    # RSI(14)
    delta = closes.diff()
    up = delta.clip(lower=0).rolling(14).mean()
    down = -delta.clip(upper=0).rolling(14).mean()
    rs = up / (down + 1e-9)
    rsi14 = 100 - (100 / (1 + rs))
    rsi_val = float(rsi14.iloc[-1])
    # Return since start of period (approx one year)
    start_price = float(closes.iloc[0])
    last_price = float(closes.iloc[-1])
    one_year_return_pct = ((last_price / start_price) - 1.0) * 100
    # Support/resistance/pivot based on the last 60 days
    recent = closes.tail(60)
    support = float(recent.min())
    resistance = float(recent.max())
    pivot = float(recent.iloc[-1])
    return {
        "symbol": symbol,
        "last_close": round(last_price, 2),
        "ma50": round(ma50, 2),
        "ma200": round(ma200, 2),
        "vol_annual": round(vol_annual, 4),
        "rsi14": round(rsi_val, 2),
        "one_year_return_pct": round(one_year_return_pct, 2),
        "support": round(support, 2),
        "resistance": round(resistance, 2),
        "pivot": round(pivot, 2),
    }


__all__ = ["get_basic_technicals"]