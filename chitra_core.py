# chitra_core.py
import math
import time
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from openai import OpenAI, RateLimitError, APIError

client = OpenAI()


# ---------- Helper: safe OpenAI call with retries ----------

def safe_openai_chat(model: str, messages: List[Dict], max_tokens: int = 900):
    """
    Wraps OpenAI chat call with small retry / backoff.
    Avoids hard crashes on transient rate limits.
    """
    backoff_seconds = [1, 2, 4]  # three attempts
    last_error = None

    for delay in backoff_seconds:
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.6,
            )
            return resp.choices[0].message.content.strip()
        except RateLimitError as e:
            last_error = e
            time.sleep(delay)
        except APIError as e:
            last_error = e
            time.sleep(delay)

    # If we are here, all retries failed
    raise last_error


# ---------- Market data & technicals ----------

def fetch_history(
    ticker: str, period: str = "1y", interval: str = "1d"
) -> pd.DataFrame:
    """
    Fetch OHLCV data from Yahoo Finance.
    Ensures we always return a clean DataFrame, or raise if no data.
    """
    data = yf.download(
        ticker,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False,
        threads=False,
    )

    if data is None or data.empty:
        raise ValueError(f"No price data found for ticker '{ticker}'")

    # Standardise columns
    data = data[["Open", "High", "Low", "Close", "Volume"]].dropna()
    return data


def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def technical_snapshot(
    ticker: str, holding_days: int = 90
) -> Dict:
    """
    Returns a compact technical snapshot used by the AI.
    Everything is converted to Python floats (no numpy arrays / Series).
    """
    df = fetch_history(ticker, period="1y", interval="1d")
    closes = df["Close"]
    volumes = df["Volume"]

    # Basic stats
    latest_close = float(closes.iloc[-1])
    ma50 = float(closes.rolling(50).mean().iloc[-1])
    ma200 = float(closes.rolling(200).mean().iloc[-1])

    daily_returns = closes.pct_change().dropna()
    vol_annual = float(daily_returns.std() * math.sqrt(252))

    # RSI
    rsi14 = compute_rsi(closes, 14)
    latest_rsi = float(rsi14.iloc[-1]) if not math.isnan(rsi14.iloc[-1]) else 50.0

    # Volume trend
    vol50 = float(volumes.rolling(50).mean().iloc[-1])
    latest_vol = float(volumes.iloc[-1])
    volume_ratio = latest_vol / vol50 if vol50 > 0 else 1.0

    # Recent support / resistance
    lookback = min(60, len(closes))
    recent = closes.iloc[-lookback:]

    support = float(recent.min())
    resistance = float(recent.max())
    pivot = float(recent.iloc[-1])

    # Very simple “trend strength” number between 0–1
    trend_score = 0.0
    if ma50 > ma200 and latest_close > ma50:
        trend_score = 1.0
    elif latest_close > ma50:
        trend_score = 0.7
    elif latest_close > ma200:
        trend_score = 0.4

    return {
        "ticker": ticker.upper(),
        "latest_close": round(latest_close, 2),
        "ma50": round(ma50, 2),
        "ma200": round(ma200, 2),
        "vol_annual": round(vol_annual, 4),
        "rsi14": round(latest_rsi, 2),
        "volume_ratio": round(volume_ratio, 2),
        "support": round(support, 2),
        "resistance": round(resistance, 2),
        "pivot": round(pivot, 2),
        "trend_score": round(trend_score, 2),
        "holding_days": holding_days,
    }


# ---------- AI layer: single stock idea ----------

RISK_TEXT = {
    "Low": "very conservative; prefers mega / large caps, 5–10% per stock, hates drawdowns.",
    "Medium": "balanced; comfortable with 10–15% per stock, accepts normal volatility.",
    "High": "aggressive; okay with 15–20% per stock and deeper but temporary drawdowns.",
    "All or Nothing": "extremely aggressive; can concentrate 30–50% into a single idea.",
}


def build_single_stock_messages(
    metrics: Dict,
    capital: float,
    risk_profile: str,
    language: str,
) -> List[Dict]:
    """
    Builds the system + user messages for the AI.
    We keep it to ONE call per stock idea.
    """
    lang = "Hindi" if language.lower().startswith("hi") else "English"

    system_msg = {
        "role": "system",
        "content": (
            "You are ChitraAdvisor, a calm, risk-aware stock guide for an Indian retail investor. "
            "You MUST NOT give direct financial advice, and you MUST remind that this is educational only. "
            "Use simple, clear language. Assume the person is not a trader."
        ),
    }

    user_msg = {
        "role": "user",
        "content": f"""
We are analysing an Indian stock.

Technical snapshot (NSE):
- Ticker: {metrics['ticker']}
- Last close: ₹{metrics['latest_close']}
- 50DMA: ₹{metrics['ma50']}
- 200DMA: ₹{metrics['ma200']}
- Annualised volatility: {metrics['vol_annual']:.4f}
- RSI(14): {metrics['rsi14']}
- Volume vs 50-day avg: {metrics['volume_ratio']}x
- Support: ₹{metrics['support']}
- Resistance: ₹{metrics['resistance']}
- Pivot region: ₹{metrics['pivot']}
- Trend strength (0–1): {metrics['trend_score']}

Investor context:
- Capital considered for this idea: ₹{capital:,.0f}
- Intended holding period: ~{metrics['holding_days']} days
- Risk profile: {risk_profile} ({RISK_TEXT.get(risk_profile, '')})
- Output language: {lang}

TASK:

1. First line MUST be exactly in this format (UPPERCASE rating):
   RATING: BUY
   or
   RATING: HOLD
   or
   RATING: SELL
   or
   RATING: AVOID

   - BUY = reasonable to slowly accumulate with discipline.
   - HOLD = okay to keep if already owning, but fresh buying should be cautious.
   - SELL = suitable to exit or reduce.
   - AVOID = do NOT touch for this risk profile.

2. After that, give a short explanation in bullet points covering:
   - Trend & momentum (mention 50/200DMA and RSI)
   - Risk level / volatility in plain language
   - Suggested entry zone (around which price range), stop-loss region, and rough upside zone
   - How much maximum % of total portfolio this stock could be for THIS risk profile
   - A one-line 'Plan' (e.g. 'staggered buying near support over 3–4 weeks').

3. End with a final line:
   NOTE: This is not investment advice, only an educational model output.

Write the whole answer in {lang}. Keep it compact and friendly.
""",
    }

    return [system_msg, user_msg]


def analyse_single_stock(
    ticker: str,
    capital: float,
    holding_days: int,
    risk_profile: str,
    language: str,
    model_name: str,
) -> Dict:
    """
    Main entry used by Streamlit's cached function.
    Returns dict: { 'metrics': {...}, 'rating': 'BUY', 'explanation': '...' }
    """
    metrics = technical_snapshot(ticker, holding_days=holding_days)
    messages = build_single_stock_messages(
        metrics, capital, risk_profile, language
    )
    text = safe_openai_chat(model_name, messages)

    # Parse first line "RATING: X"
    rating = "UNKNOWN"
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if lines and lines[0].upper().startswith("RATING:"):
        rating = lines[0].split(":", 1)[1].strip().upper()

    explanation = "\n".join(lines[1:]).strip()

    return {
        "metrics": metrics,
        "rating": rating,
        "explanation": explanation,
    }


# ---------- Portfolio suggestion helpers (no OpenAI) ----------

# Basic universes – simplified examples, just to have stable tickers.
UNIVERSES: Dict[str, List[str]] = {
    "NIFTY50": [
        "RELIANCE.NS",
        "TCS.NS",
        "INFY.NS",
        "HDFCBANK.NS",
        "ICICIBANK.NS",
        "KOTAKBANK.NS",
        "AXISBANK.NS",
        "SBIN.NS",
        "ITC.NS",
        "LT.NS",
        "HINDUNILVR.NS",
        "ASIANPAINT.NS",
        "BAJFINANCE.NS",
        "SUNPHARMA.NS",
        "MARUTI.NS",
        "ULTRACEMCO.NS",
        "NESTLEIND.NS",
        "HCLTECH.NS",
        "POWERGRID.NS",
        "TITAN.NS",
    ],
    "NIFTY100": [
        "RELIANCE.NS",
        "TCS.NS",
        "INFY.NS",
        "HDFCBANK.NS",
        "ICICIBANK.NS",
        "KOTAKBANK.NS",
        "AXISBANK.NS",
        "SBIN.NS",
        "ITC.NS",
        "LT.NS",
        "HINDUNILVR.NS",
        "ASIANPAINT.NS",
        "BAJFINANCE.NS",
        "SUNPHARMA.NS",
        "MARUTI.NS",
        "ULTRACEMCO.NS",
        "NESTLEIND.NS",
        "HCLTECH.NS",
        "POWERGRID.NS",
        "TITAN.NS",
        "TECHM.NS",
        "JSWSTEEL.NS",
        "ONGC.NS",
        "COALINDIA.NS",
        "BAJAJFINSV.NS",
        "ADANIPORTS.NS",
        "ADANIENT.NS",
        "CIPLA.NS",
        "DRREDDY.NS",
    ],
    "NIFTYNEXT50": [
        "BEL.NS",
        "DABUR.NS",
        "GAIL.NS",
        "PIDILITIND.NS",
        "ICICIGI.NS",
        "ICICIPRULI.NS",
        "BANKBARODA.NS",
        "INDUSINDBK.NS",
        "SRF.NS",
        "NAUKRI.NS",
        "MUTHOOTFIN.NS",
        "PEL.NS",
        "UBL.NS",
        "HAVELLS.NS",
        "CHOLAFIN.NS",
        "DIVISLAB.NS",
        "LUPIN.NS",
        "AUROPHARMA.NS",
        "BERGEPAINT.NS",
        "COLPAL.NS",
    ],
    "NIFTY200": [],  # we can treat as NIFTY100 universe in app if needed
    "MIDCAP150": [
        "MUTHOOTFIN.NS",
        "SRF.NS",
        "TATAPOWER.NS",
        "BANDHANBNK.NS",
        "DIXON.NS",
        "INDHOTEL.NS",
        "PNB.NS",
        "IDFCFIRSTB.NS",
        "JKCEMENT.NS",
        "MPHASIS.NS",
        "LTI.NS",
        "AUBANK.NS",
    ],
    "SMALLCAP100": [
        "IEX.NS",
        "PERSISTENT.NS",
        "AFFLE.NS",
        "AARTIIND.NS",
        "DEEPAKNTR.NS",
        "ALEMBICLTD.NS",
        "AMBER.NS",
        "GARFIBRES.NS",
        "GUJGASLTD.NS",
        "KEI.NS",
        "NAVINFLUOR.NS",
        "OFSS.NS",
        "PIIND.NS",
    ],
}

CUSTOM_HQ20_KEY = "CUSTOM_HQ20"

UNIVERSES[CUSTOM_HQ20_KEY] = [
    "RELIANCE.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "KOTAKBANK.NS",
    "AXISBANK.NS",
    "SBIN.NS",
    "TCS.NS",
    "INFY.NS",
    "HCLTECH.NS",
    "HINDUNILVR.NS",
    "ASIANPAINT.NS",
    "NESTLEIND.NS",
    "ITC.NS",
    "SUNPHARMA.NS",
    "DRREDDY.NS",
    "ULTRACEMCO.NS",
    "MARUTI.NS",
    "BAJFINANCE.NS",
    "TITAN.NS",
    "POWERGRID.NS",
]


def get_universe_symbols(key: str) -> List[str]:
    if key == "NIFTY200":
        # fallback so it still works:
        return UNIVERSES["NIFTY100"]
    return UNIVERSES.get(key, [])


def simple_equal_weight_portfolio(
    symbols: List[str],
    total_amount: float,
    risk_profile: str,
) -> List[Dict]:
    """
    Very simple allocation: equal-weight by amount, rounded to nearest share.
    Does NOT call OpenAI.
    """
    if not symbols:
        return []

    prices: Dict[str, float] = {}
    for sym in symbols:
        try:
            df = fetch_history(sym, period="5d", interval="1d")
            prices[sym] = float(df["Close"].iloc[-1])
        except Exception:
            # skip if price unavailable
            continue

    usable = list(prices.keys())
    if not usable:
        return []

    n_positions = len(usable)
    per_stock_amount = total_amount / n_positions

    rows = []
    cash_left = total_amount

    for sym in usable:
        price = prices[sym]
        qty = int(per_stock_amount // price)
        alloc_value = qty * price
        alloc_pct = (alloc_value / total_amount * 100) if total_amount > 0 else 0

        cash_left -= alloc_value

        rows.append(
            {
                "symbol": sym,
                "price": round(price, 2),
                "allocation_pct": round(alloc_pct, 2),
                "allocation_value": round(alloc_value, 2),
                "qty": qty,
            }
        )

    # We will return rows + a separate cash_left from Streamlit side.
    # (Streamlit will compute / show the cash text.)
    return rows
