"""
Core logic for ChitraAdvisor – Stock Helper

- Handles symbol normalisation (".NS" auto-handling)
- Fetches prices with caching + retry (FMP primary, yfinance fallback)
- Single–stock idea via OpenAI
- Portfolio suggestion (universe + risk based)
"""

import json
import math
import time
import os
from functools import lru_cache
from typing import List, Dict, Tuple

import requests
import numpy as np
import pandas as pd
import yfinance as yf
from openai import OpenAI

# ============ OpenAI client & helpers ============

client = OpenAI()

def _safe_chat(
    system_prompt: str,
    user_prompt: str,
    model: str = "gpt-4.1-mini",
    max_tokens: int = 900,
    temperature: float = 0.4,
) -> str:
    """
    Wrapper around OpenAI Chat API with automatic handling of
    model-specific token parameters.

    Newer models (gpt-5.x / o-series) DO NOT accept `max_tokens`.
    They require `max_completion_tokens`.
    """
    last_err = None

    # Decide parameter name based on model
    if model.startswith("gpt-5") or model.startswith("o"):
        token_param = {"max_completion_tokens": max_tokens}
    else:
        token_param = {"max_tokens": max_tokens}

    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                **token_param,
            )
            return resp.choices[0].message.content.strip()
        except Exception as exc:
            last_err = exc
            msg = str(exc).lower()

            # retry for transient errors
            if "rate" in msg or "timeout" in msg or "overloaded" in msg:
                time.sleep(2 ** attempt)
                continue

            break  # non-retryable

    raise RuntimeError(f"OpenAI call failed: {last_err}")

# ============ Symbol helpers & price cache ============

# Very small in-memory cache for AI-resolved symbols
_SYMBOL_RESOLUTION_CACHE: Dict[str, str] = {}

# FMP config & simple price cache
FMP_API_KEY = os.getenv("FMP_API_KEY", "").strip()

# symbol -> (timestamp, price)
_PRICE_CACHE: Dict[str, Tuple[float, float]] = {}
_PRICE_TTL_SEC = 300  # 5 minutes


def normalize_symbol(user_input: str) -> str:
    """
    Try to turn whatever user typed into a usable NSE symbol.

    Rules:
    - Trim spaces, uppercase.
    - If it already ends with ".NS" or any ".XYZ", keep as-is.
    - If no dot, assume NSE and append ".NS".
    """
    raw = (user_input or "").strip()
    if not raw:
        raise ValueError("Please enter a company name or symbol.")

    s_up = raw.upper()
    if "." in s_up:
        return s_up
    # Plain symbol like "TCS" -> "TCS.NS"
    return s_up + ".NS"


def _maybe_resolve_company_name_with_ai(user_input: str, model: str) -> str:
    """
    If user typed a long name like 'Maruti Suzuki India Limited',
    ask the model for the closest NSE symbol once and cache it.
    """
    key = user_input.strip().lower()
    if key in _SYMBOL_RESOLUTION_CACHE:
        return _SYMBOL_RESOLUTION_CACHE[key]

    system = (
        "You are an assistant that maps Indian stock names/descriptions to NSE symbols. "
        "Always return ONLY the NSE symbol like 'MARUTI.NS'. "
        "If you are not sure, return the word UNKNOWN."
    )
    user = (
        f"User typed: '{user_input}'.\n"
        "What is the most likely NSE symbol? Reply with only the symbol or UNKNOWN."
    )
    try:
        ans = _safe_chat(system, user, model=model, max_tokens=20)
    except Exception:
        return "UNKNOWN"

    sym = ans.strip().upper()
    if "UNKNOWN" in sym:
        sym = "UNKNOWN"

    _SYMBOL_RESOLUTION_CACHE[key] = sym
    return sym


@lru_cache(maxsize=256)
def _download_history(symbol: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    """
    Download price history with retry & small cache.

    This is used as a fallback (yfinance) when FMP is unavailable,
    and also for computing technicals.
    """
    last_err = None
    for attempt in range(3):
        try:
            df = yf.download(
                symbol,
                period=period,
                interval=interval,
                progress=False,
                auto_adjust=False,
                threads=True,
            )
            if df is None or df.empty:
                raise RuntimeError(f"No price data found for symbol {symbol}")
            return df
        except Exception as exc:
            last_err = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Could not download data for {symbol}: {last_err}")


def _get_price_from_fmp(symbol: str) -> float:
    """
    Fetch latest price from Financial Modeling Prep.
    Raises if API key missing or response invalid.
    """
    if not FMP_API_KEY:
        raise RuntimeError("FMP_API_KEY is not set in the environment.")

    url = f"https://financialmodelingprep.com/api/v3/quote/{symbol}?apikey={FMP_API_KEY}"
    resp = requests.get(url, timeout=5)

    if resp.status_code != 200:
        raise RuntimeError(f"FMP HTTP {resp.status_code}: {resp.text[:200]}")

    data = resp.json()
    if not data or not isinstance(data, list):
        raise RuntimeError(f"FMP returned empty/invalid data for {symbol}: {data}")

    item = data[0]
    price = item.get("price") or item.get("previousClose")
    if price is None:
        raise RuntimeError(f"FMP data missing price for {symbol}: {item}")

    price = float(price)
    if price <= 0:
        raise RuntimeError(f"FMP returned non-positive price for {symbol}: {price}")

    return round(price, 2)


def _get_price_from_yf(symbol: str) -> float:
    """
    Fallback price using yfinance (last close from 5 days).
    """
    df = _download_history(symbol, period="5d", interval="1d")
    latest_close = float(df["Close"].iloc[-1])
    return round(latest_close, 2)


def get_latest_price(symbol: str) -> float:
    """
    Get latest price for a symbol.

    Strategy:
    1. If we have a recent cached price (<= 5 min), return it.
    2. Try FMP first (cheap & reliable, up to 250 calls/day).
    3. If FMP fails (quota/network/anything), fall back to yfinance.
    """
    now = time.time()

    # 1) Cache hit
    if symbol in _PRICE_CACHE:
        ts, cached_price = _PRICE_CACHE[symbol]
        if now - ts <= _PRICE_TTL_SEC:
            return cached_price

    # 2) Try FMP
    try:
        price = _get_price_from_fmp(symbol)
    except Exception as exc:
        print(f"[WARN] FMP price fetch failed for {symbol}: {exc}. Falling back to yfinance.")
        # 3) Fallback to yfinance
        price = _get_price_from_yf(symbol)

    _PRICE_CACHE[symbol] = (now, price)
    return price


def get_basic_technicals(symbol: str) -> Dict:
    """
    Compute a handful of simple technical stats.
    Enough to give the model some structure,
    but still cheap & fast.
    """
    df = _download_history(symbol, period="1y", interval="1d")
    closes = df["Close"].astype(float)

    daily_returns = closes.pct_change().dropna()
    if daily_returns.empty:
        vol_annual = 0.0
    else:
        vol_annual = float(daily_returns.std() * math.sqrt(252))

    ma50 = float(closes.rolling(50).mean().iloc[-1])
    ma200 = float(closes.rolling(200).mean().iloc[-1])

    # simple RSI(14)
    delta = closes.diff()
    up = delta.clip(lower=0).rolling(14).mean()
    down = -delta.clip(upper=0).rolling(14).mean()
    rs = up / (down + 1e-9)
    rsi14 = 100 - (100 / (1 + rs))
    rsi_val = float(rsi14.iloc[-1])

    start_price = float(closes.iloc[0])
    last_price = float(closes.iloc[-1])
    one_year_return_pct = ((last_price / start_price) - 1.0) * 100

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


# ============ Single stock analysis ============

def analyze_single_stock(
    user_typed: str,
    capital: float,
    risk_profile: str,
    holding_period_days: int,
    language: str = "en",
    model: str = "gpt-4.1-mini",
) -> Dict:
    """
    Main entry point for the Single Stock tab.

    Returns dict with:
      - resolved_symbol
      - display_symbol
      - metrics
      - analysis_markdown
    """
    if not user_typed or not user_typed.strip():
        raise ValueError("Please enter a company name or NSE symbol.")

    text = user_typed.strip()
    symbol_guess = normalize_symbol(text)

    # First try the naive guess
    try:
        metrics = get_basic_technicals(symbol_guess)
        resolved_symbol = symbol_guess
    except Exception:
        # If the input looks like a long name, try AI symbol resolution once
        if " " in text.strip():
            ai_symbol = _maybe_resolve_company_name_with_ai(text, model=model)
            if ai_symbol != "UNKNOWN":
                metrics = get_basic_technicals(ai_symbol)
                resolved_symbol = ai_symbol
            else:
                raise RuntimeError(
                    "Could not find price data for what you typed. "
                    "Please try again using the NSE symbol, e.g. 'TCS' or 'TCS.NS'."
                )
        else:
            raise

    # Build prompt for the model
    risk_text = {
        "Low": "very conservative, capital preservation focused",
        "Medium": "balanced risk and return",
        "High": "aggressive, willing to take drawdowns for higher return",
        "All or Nothing": "extremely aggressive, okay with large loss if thesis fails",
    }.get(risk_profile, "balanced risk and return")

    lang_label = "English" if language == "en" else "Hindi"

    system_prompt = (
        "You are ChitraAdvisor, a friendly Indian stock helper for a retail investor. "
        "You DO NOT give guaranteed advice – only model-based, educational views.\n\n"
        "You must always respect risk profile and give clear levels (entry, stop, targets). "
        "Assume Indian equity market, delivery trades (no intraday leverage). "
    )

    user_prompt = f"""
User wants a single stock idea.

Symbol: {resolved_symbol}
Total capital user is considering for THIS idea (not full portfolio): ₹{capital:,.0f}
Risk profile: {risk_profile} ({risk_text})
Intended holding period: ~{holding_period_days} days
Language for explanation: {lang_label}

Recent technical snapshot (approx):
- Last close: ₹{metrics['last_close']}
- 50-day MA: ₹{metrics['ma50']}
- 200-day MA: ₹{metrics['ma200']}
- 1-year return: {metrics['one_year_return_pct']}%
- Annualised volatility: {metrics['vol_annual']}
- RSI(14): {metrics['rsi14']}
- Support: ₹{metrics['support']}
- Resistance: ₹{metrics['resistance']}
- Pivot / reference level: ₹{metrics['pivot']}

TASK:
1. Decide a clear ACTION among:
   BUY, AVOID, HOLD, PARTIAL EXIT, or SELL.
2. Suggest:
   - Ideal entry zone (one price or a narrow range)
   - Stop loss level
   - 1–2 target levels
   - How much of the user's capital is reasonable to allocate (as % of capital)
   - Short comment on expected volatility and risk.
3. Explain the reasoning in simple {lang_label}, in 5–8 bullet points max.
4. Tone: calm, realistic, no hype. Reiterate that this is NOT guaranteed advice.

FORMAT (very important) – answer EXACTLY in this markdown layout:

Action: <one of BUY / AVOID / HOLD / PARTIAL EXIT / SELL>

Entry zone: ₹<x> – ₹<y>  (or just one price)
Stop loss: ₹<z>
Targets: ₹<t1> (and ₹<t2> if you want)
Suggested allocation: <p>% of the capital for this idea

Holding view:
<one short sentence about time horizon & conditions to re-evaluate>

Reasoning:
- point 1
- point 2
- ...
- point N

Risk reminder:
<one short line reminding this is educational only>
"""

    analysis_markdown = _safe_chat(system_prompt, user_prompt, model=model, max_tokens=700)

    return {
        "resolved_symbol": resolved_symbol,
        "display_symbol": resolved_symbol,
        "metrics": metrics,
        "analysis_markdown": analysis_markdown,
    }


# ============ Portfolio suggestion ============

# Very rough universes – enough for experimentation.
UNIVERSES: Dict[str, List[str]] = {
    "NIFTY50": [
        "RELIANCE.NS", "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS",
        "INFY.NS", "TCS.NS", "ITC.NS", "LT.NS", "AXISBANK.NS",
        "KOTAKBANK.NS", "HINDUNILVR.NS", "BAJFINANCE.NS",
        "ASIANPAINT.NS", "MARUTI.NS", "SUNPHARMA.NS",
        "TITAN.NS", "ULTRACEMCO.NS", "HCLTECH.NS", "WIPRO.NS",
        "NESTLEIND.NS",
    ],
    "NIFTY100": [
        "RELIANCE.NS", "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS",
        "INFY.NS", "TCS.NS", "ITC.NS", "LT.NS", "AXISBANK.NS",
        "KOTAKBANK.NS", "HINDUNILVR.NS", "BAJFINANCE.NS",
        "ASIANPAINT.NS", "MARUTI.NS", "SUNPHARMA.NS",
        "TITAN.NS", "ULTRACEMCO.NS", "HCLTECH.NS", "WIPRO.NS",
        "NESTLEIND.NS", "DMART.NS", "PIDILITIND.NS", "DIVISLAB.NS",
    ],
    "NIFTY_NEXT_50": [
        "DMART.NS", "ICICIPRULI.NS", "DIVISLAB.NS", "BERGEPAINT.NS",
        "DABUR.NS", "GODREJCP.NS", "PGHH.NS", "UNITDSPR.NS",
        "BAJAJHLDNG.NS", "BANKBARODA.NS", "HINDPETRO.NS",
        "INDHOTEL.NS", "LUPIN.NS", "MFSL.NS", "SIEMENS.NS",
    ],
    "NIFTY200": [
        "RELIANCE.NS", "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS",
        "INFY.NS", "TCS.NS", "ITC.NS", "LT.NS", "AXISBANK.NS",
        "KOTAKBANK.NS", "BAJFINANCE.NS", "ASIANPAINT.NS",
        "MARUTI.NS", "SUNPHARMA.NS", "DMART.NS", "PIDILITIND.NS",
        "DIVISLAB.NS", "LUPIN.NS", "SIEMENS.NS", "INDHOTEL.NS",
        "BANKBARODA.NS", "HINDPETRO.NS", "DABUR.NS", "GODREJCP.NS",
    ],
    "MIDCAP150": [
        "DMART.NS", "PIDILITIND.NS", "DIVISLAB.NS", "LUPIN.NS",
        "ABBOTINDIA.NS", "PAGEIND.NS", "POLYCAB.NS", "AUBANK.NS",
        "TATACOMM.NS", "TATACONSUM.NS", "MUTHOOTFIN.NS",
        "CHOLAFIN.NS", "INDHOTEL.NS", "TATAPOWER.NS", "DALBHARAT.NS",
        "TRENT.NS", "BERGEPAINT.NS", "MFSL.NS", "NAUKRI.NS",
    ],
    "SMALLCAP100": [
        "ZOMATO.NS", "NYKAA.NS", "MAPMYINDIA.NS", "DEEPAKNTR.NS",
        "TATAELXSI.NS", "INDIAMART.NS", "PVRINOX.NS", "AARTIIND.NS",
        "KEI.NS", "ASTERDM.NS", "SYNGENE.NS", "SRF.NS",
        "JKCEMENT.NS", "NATCOPHARM.NS", "CERA.NS",
    ],
    # Locked custom high-quality list that you defined earlier (broader 20)
    "CUSTOM_HQ20": [
        "RELIANCE.NS", "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS",
        "INFY.NS", "TCS.NS", "ITC.NS", "HINDUNILVR.NS",
        "ASIANPAINT.NS", "BAJFINANCE.NS",
        "DMART.NS", "PIDILITIND.NS", "DIVISLAB.NS",
        "SUNPHARMA.NS", "TITAN.NS", "ULTRACEMCO.NS",
        "MARUTI.NS", "AXISBANK.NS", "LT.NS", "NESTLEIND.NS",
    ],
}

RISK_LEVELS = ["Low", "Medium", "High", "All or Nothing"]

# Number of positions per risk level – this is what changes portfolio concentration
RISK_TO_POSITIONS = {
    "Low": 12,
    "Medium": 8,
    "High": 6,
    "All or Nothing": 3,
}


def _pick_symbols_for_risk(universe_symbols: List[str], risk_profile: str) -> List[str]:
    """Pick a sub-list of symbols based on risk. Very simple heuristic."""
    num_positions = RISK_TO_POSITIONS.get(risk_profile, 8)
    num_positions = min(num_positions, len(universe_symbols))
    # For now just take the first N symbols – later we can rank by volatility, etc.
    return universe_symbols[:num_positions]


def build_portfolio(
    universe_key: str,
    risk_profile: str,
    total_capital: float,
) -> Tuple[pd.DataFrame, float]:
    """
    Build a simple equal-weight portfolio within the chosen universe.

    Returns (df, unallocated_cash)
      df columns: Symbol, Approx Price, Allocation (%), Allocation (₹), Approx Qty
    """
    if total_capital <= 0:
        raise ValueError("Total capital must be positive.")

    if universe_key not in UNIVERSES:
        raise ValueError("Unknown universe selected.")

    all_symbols = UNIVERSES[universe_key]
    selected = _pick_symbols_for_risk(all_symbols, risk_profile)
    n = len(selected)
    if n == 0:
        raise ValueError("No symbols available for the selected universe.")

    equal_weight_pct = round(100.0 / n, 2)

    rows = []
    unallocated_cash = total_capital

    for sym in selected:
        price = get_latest_price(sym)
        allocation_rupees = total_capital * equal_weight_pct / 100.0
        qty = math.floor(allocation_rupees / price) if price > 0 else 0
        used_cash = qty * price
        unallocated_cash -= used_cash

        rows.append(
            {
                "Symbol": sym,
                "Approx Price (₹)": round(price, 2),
                "Allocation (%)": equal_weight_pct,
                "Allocation (₹)": round(used_cash, 2),
                "Approx Qty": int(qty),
            }
        )

    df = pd.DataFrame(rows)
    unallocated_cash = round(unallocated_cash, 2)
    return df, unallocated_cash
