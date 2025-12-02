# chitra_core.py
#
# Core logic for ChitraAdvisor:
# - Price data + indicators
# - Single-stock analysis + AI summary
# - Portfolio suggestion engine

import math
import os
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import yfinance as yf
from openai import OpenAI

client = OpenAI()

# ---------- CONFIG / CONSTANTS ----------

DEFAULT_PERIOD = "1y"       # history for indicators
DEFAULT_INTERVAL = "1d"     # we internally use 1d; timeframe text is just cosmetic

RISK_PROFILES = ["Low", "Medium", "High", "All or Nothing"]

# Universes (simplified, can be updated later from NSE lists)
NIFTY50 = [
    "RELIANCE.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS", "TCS.NS",
    "SBIN.NS", "AXISBANK.NS", "KOTAKBANK.NS", "ITC.NS", "LT.NS",
    "HINDUNILVR.NS", "ASIANPAINT.NS", "BAJFINANCE.NS", "MARUTI.NS",
    "SUNPHARMA.NS", "ULTRACEMCO.NS", "TITAN.NS", "WIPRO.NS", "POWERGRID.NS",
    "NTPC.NS"
]

# For demo we keep 100/200 etc as supersets / variations on this
NIFTY100 = NIFTY50 + [
    "ADANIENT.NS", "ADANIPORTS.NS", "BPCL.NS", "BHARTIARTL.NS",
    "COALINDIA.NS", "HCLTECH.NS", "TECHM.NS", "GRASIM.NS",
    "DIVISLAB.NS", "DRREDDY.NS"
]

NIFTY_NEXT50 = [
    "PIDILITIND.NS", "DABUR.NS", "ICICIPRULI.NS", "ICICIGI.NS",
    "MUTHOOTFIN.NS", "GODREJCP.NS", "INDIGO.NS", "HAVELLS.NS",
    "NAUKRI.NS", "UBL.NS"
]

NIFTY200 = NIFTY100 + NIFTY_NEXT50

MIDCAP150 = [
    "CUMMINSIND.NS", "ABB.NS", "LUPIN.NS", "PERSISTENT.NS", "COROMANDEL.NS",
    "AUROPHARMA.NS", "OFSS.NS", "MINDTREE.NS", "ASTRAL.NS", "POLYCAB.NS"
]

SMALLCAP100 = [
    "APLAPOLLO.NS", "IEX.NS", "BALAMINES.NS", "LAURUSLABS.NS",
    "TATAELXSI.NS", "CROMPTON.NS", "KEI.NS", "DEEPAKNTR.NS",
    "CLEAN.NS", "TANLA.NS"
]

CUSTOM_HIGH_QUALITY_20 = [
    "RELIANCE.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS", "TCS.NS",
    "SBIN.NS", "AXISBANK.NS", "KOTAKBANK.NS", "ITC.NS", "ASIANPAINT.NS",
    "HINDUNILVR.NS", "SUNPHARMA.NS", "ULTRACEMCO.NS", "BAJFINANCE.NS",
    "MARUTI.NS", "TITAN.NS", "NESTLEIND.NS", "DIVISLAB.NS", "DRREDDY.NS",
    "LT.NS",
]

UNIVERSES: Dict[str, List[str]] = {
    "NIFTY50": NIFTY50,
    "NIFTY100": NIFTY100,
    "NIFTY Next 50": NIFTY_NEXT50,
    "NIFTY200": NIFTY200,
    "Midcap 150": MIDCAP150,
    "Smallcap 100": SMALLCAP100,
    "Custom High-Quality 20": CUSTOM_HIGH_QUALITY_20,
}

# ---------- HELPERS ----------

def _ensure_nse_suffix(symbol: str) -> str:
    s = symbol.strip().upper()
    if not s.endswith(".NS"):
        s = s + ".NS"
    return s


def fetch_price_data(symbol: str,
                     period: str = DEFAULT_PERIOD,
                     interval: str = DEFAULT_INTERVAL) -> pd.DataFrame:
    """Fetch OHLCV data from yfinance, ensuring NSE suffix."""
    sym = _ensure_nse_suffix(symbol)
    ticker = yf.Ticker(sym)
    df = ticker.history(period=period, interval=interval, auto_adjust=False)
    if df is None or df.empty:
        raise ValueError(f"No price data found for {sym}")
    df = df.dropna()
    return df


def latest_price(symbol: str) -> float:
    df = fetch_price_data(symbol, period="5d", interval="1d")
    price = float(df["Close"].iloc[-1])
    return price


def compute_indicators(df: pd.DataFrame) -> Dict[str, float]:
    """Compute simple indicators; all outputs plain floats (no numpy round issue)."""
    closes = df["Close"]
    volumes = df["Volume"]

    daily_ret = closes.pct_change().dropna()
    vol_annual = float(daily_ret.std() * math.sqrt(252))

    ma50 = closes.rolling(50).mean()
    ma200 = closes.rolling(200).mean()

    # RSI 14
    delta = closes.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))

    latest_close = float(closes.iloc[-1])
    latest_ma50 = float(ma50.iloc[-1]) if not math.isnan(ma50.iloc[-1]) else latest_close
    latest_ma200 = float(ma200.iloc[-1]) if not math.isnan(ma200.iloc[-1]) else latest_close
    latest_rsi = float(rsi.iloc[-1]) if not math.isnan(rsi.iloc[-1]) else 50.0

    vol50 = float(volumes.rolling(50).mean().iloc[-1])
    latest_vol = float(volumes.iloc[-1])
    volume_ratio = latest_vol / vol50 if vol50 > 0 else 1.0

    recent = closes.tail(60)
    support = float(recent.min())
    resistance = float(recent.max())
    pivot = float(recent.iloc[-1])

    return {
        "latest_close": latest_close,
        "ma50": latest_ma50,
        "ma200": latest_ma200,
        "rsi": latest_rsi,
        "annual_vol": vol_annual,
        "volume_ratio": volume_ratio,
        "support": support,
        "resistance": resistance,
        "pivot": pivot,
    }


def score_technical(ind: Dict[str, float]) -> Dict[str, float]:
    """Simple scoring between 0 and 1. All outputs are plain floats."""
    close = ind["latest_close"]
    ma50 = ind["ma50"]
    ma200 = ind["ma200"]
    rsi = ind["rsi"]
    vol = ind["annual_vol"]
    vol_ratio = ind["volume_ratio"]

    trend_up = 1.0 if (close > ma50 > ma200) else 0.5 if (close > ma200) else 0.0

    if rsi < 30:
        rsi_score = 0.7
    elif rsi < 60:
        rsi_score = 1.0
    elif rsi < 70:
        rsi_score = 0.6
    else:
        rsi_score = 0.3

    vol_norm = max(0.0, min(1.0, (0.6 - vol) / 0.6))  # prefer lower vol up to 60%

    vol_r = 1.0 if 0.8 <= vol_ratio <= 2.0 else 0.6 if 0.5 <= vol_ratio <= 3.0 else 0.3

    tech_score = float(np.clip(0.4 * trend_up + 0.3 * rsi_score + 0.2 * vol_norm + 0.1 * vol_r, 0, 1))
    prob_score = float(np.clip(0.5 * tech_score + 0.5 * trend_up, 0, 1))

    return {
        "technical_score": round(tech_score, 3),
        "probability_score": round(prob_score, 3),
        "trend_strength": round(trend_up, 3),
    }


def risk_adjust_for_profile(risk_profile: str,
                            capital: float,
                            indicators: Dict[str, float],
                            scores: Dict[str, float]) -> Dict[str, float]:
    """Return position sizing & SL/TP based on 1% capital rule."""
    risk_pct_map = {
        "Low": 0.005,
        "Medium": 0.01,
        "High": 0.02,
        "All or Nothing": 0.05,
    }
    risk_pct = risk_pct_map.get(risk_profile, 0.01)

    close = indicators["latest_close"]
    support = indicators["support"]
    resistance = indicators["resistance"]
    pivot = indicators["pivot"]

    risk_amount = capital * risk_pct
    stop_distance = max(close - support, close * 0.04)
    qty = max(1, int(risk_amount / stop_distance)) if stop_distance > 0 else 0

    pos_value = qty * close
    max_loss = qty * stop_distance

    rr_raw = (resistance - close) / stop_distance if stop_distance > 0 else 0.0
    rr = float(max(0.0, rr_raw))

    return {
        "position_value": round(pos_value, 2),
        "quantity": int(qty),
        "max_loss": round(max_loss, 2),
        "max_loss_pct_capital": round((max_loss / capital) * 100, 2) if capital > 0 else 0.0,
        "entry": round(pivot, 2),
        "stop_loss": round(max(support, close * 0.9), 2),
        "take_profit": round(resistance, 2),
        "rr": round(rr, 2),
    }

# ---------- AI LAYER ----------

STOCK_PROMPT_TEMPLATE = """
You are ChitraAdvisor, a calm Indian stock helper for a retail investor.
User profile:
- Risk profile: {risk_profile}
- Capital considered for this stock: ₹{capital}
- Intended holding period: {holding_period_days} days.

Stock data (NSE):
- Symbol: {symbol}
- Latest close: ₹{latest_close}
- 50-day MA: ₹{ma50}
- 200-day MA: ₹{ma200}
- RSI(14): {rsi}
- Annualized volatility: {annual_vol:.2%}
- Volume ratio vs 50-day avg: {volume_ratio:.2f}
- Support: ₹{support}
- Resistance: ₹{resistance}
- Pivot: ₹{pivot}

Quant scores (0–1):
- Technical score: {technical_score}
- Probability score: {probability_score}
- Trend strength: {trend_strength}

Risk engine:
- Suggested entry: ₹{entry}
- Suggested stop-loss: ₹{stop_loss}
- Suggested take-profit: ₹{take_profit}
- Risk–reward estimate: {rr}x
- Approx quantity: {quantity} shares
- Max loss if SL hit: ₹{max_loss} (~{max_loss_pct_capital}% of capital)

TASK:
1. Give a **clear, single-word verdict** for this stock for the given risk profile:
   - One of: BUY, HOLD, SELL, AVOID.
2. In 5–7 short bullet points, explain:
   - Trend & momentum in **simple language**
   - How risky this looks for the given risk profile
   - Why you chose that verdict
   - Simple “when to worry” guidance.
3. End with a one-line summary for Chitra like:
   - "For Chitra: This is a conservative BUY for 3–6 months with strict stop-loss."
Keep it friendly, concise and non-technical. Do NOT give any guarantees or
use words like "sure shot". This is only a model output, not investment advice.
"""


def build_stock_view_text(
    symbol: str,
    risk_profile: str,
    capital: float,
    holding_period_days: int,
    ind: Dict[str, float],
    scores: Dict[str, float],
    risk: Dict[str, float],
    model_name: str,
    language: str = "English",
) -> str:
    prompt = STOCK_PROMPT_TEMPLATE.format(
        symbol=symbol,
        risk_profile=risk_profile,
        capital=int(capital),
        holding_period_days=holding_period_days,
        **ind,
        **scores,
        **risk,
    )

    system_msg = "You are a helpful Indian stock explainer for a non-technical investor."
    if language.lower().startswith("hi"):
        system_msg += " Respond in simple Hindi, using Hinglish words if helpful."

    resp = client.responses.create(
        model=model_name,
        input=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt},
        ],
    )
    text = resp.output[0].content[0].text
    return text

# ---------- SINGLE STOCK ENTRY POINT ----------

@dataclass
class SingleStockResult:
    symbol: str
    indicators: Dict[str, float]
    scores: Dict[str, float]
    risk: Dict[str, float]
    verdict_text: str


def analyze_single_stock(
    symbol: str,
    capital: float,
    risk_profile: str,
    holding_period_days: int,
    model_name: str,
    language: str,
) -> SingleStockResult:
    df = fetch_price_data(symbol)
    indicators = compute_indicators(df)
    scores = score_technical(indicators)
    risk = risk_adjust_for_profile(risk_profile, capital, indicators, scores)
    text = build_stock_view_text(
        _ensure_nse_suffix(symbol),
        risk_profile,
        capital,
        holding_period_days,
        indicators,
        scores,
        risk,
        model_name,
        language,
    )
    return SingleStockResult(
        symbol=_ensure_nse_suffix(symbol),
        indicators=indicators,
        scores=scores,
        risk=risk,
        verdict_text=text,
    )

# ---------- PORTFOLIO ENGINE ----------

def allowed_universes_for_risk(risk_profile: str) -> List[str]:
    """Map risk profile → which universes are allowed."""
    if risk_profile == "Low":
        return ["NIFTY50", "NIFTY100", "Custom High-Quality 20"]
    elif risk_profile == "Medium":
        return ["NIFTY50", "NIFTY100", "NIFTY Next 50", "Custom High-Quality 20"]
    elif risk_profile == "High":
        return ["NIFTY50", "NIFTY100", "NIFTY Next 50", "Midcap 150", "Custom High-Quality 20"]
    else:  # All or Nothing
        return list(UNIVERSES.keys())


def target_positions_for_risk(risk_profile: str) -> int:
    if risk_profile == "Low":
        return 8
    elif risk_profile == "Medium":
        return 10
    elif risk_profile == "High":
        return 6
    else:
        return 3  # very concentrated


def build_portfolio(
    universe_name: str,
    capital: float,
    risk_profile: str,
) -> Tuple[pd.DataFrame, float]:
    """Return portfolio df with price, allocation%, allocation₹, qty, plus leftover cash."""
    symbols = UNIVERSES[universe_name]
    target_pos = min(target_positions_for_risk(risk_profile), len(symbols))

    # naive: take first N; later could be ranked/sorted
    selected = symbols[:target_pos]

    prices = {}
    for sym in selected:
        try:
            prices[sym] = latest_price(sym)
        except Exception:
            prices[sym] = None

    # First pass: equal allocation
    allocations = []
    viable_symbols = []

    equal_pct = 1.0 / len(selected)
    for sym in selected:
        price = prices.get(sym)
        if price is None or price <= 0:
            continue
        alloc_rs = capital * equal_pct
        qty = int(alloc_rs // price)
        if qty < 1:
            continue  # can't even buy 1 share → skip
        real_alloc_rs = qty * price
        pct = real_alloc_rs / capital

        viable_symbols.append(sym)
        allocations.append((sym, price, pct, real_alloc_rs, qty))

    if not allocations:
        raise ValueError("Could not allocate to any symbol with given capital & prices.")

    # Re-normalize percentages
    total_alloc_rs = sum(a[3] for a in allocations)
    leftover = capital - total_alloc_rs

    rows = []
    for sym, price, pct, alloc_rs, qty in allocations:
        pct_norm = alloc_rs / capital
        rows.append({
            "Symbol": sym,
            "Approx Price (₹)": round(price, 2),
            "Allocation (%)": round(pct_norm * 100, 2),
            "Allocation (₹)": round(alloc_rs, 2),
            "Approx Qty": int(qty),
        })

    df = pd.DataFrame(rows)
    return df, round(leftover, 2)
