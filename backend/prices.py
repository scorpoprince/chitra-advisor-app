"""
Price fetching logic.

This module provides functions to retrieve the latest price for a ticker.  It
prefers the Financial Modeling Prep API and falls back to yfinance when
unavailable.  A small in‑memory cache is used to reduce repeated queries.
"""

import time
import requests
import yfinance as yf
from functools import lru_cache

from .utils import logger, FMP_API_KEY

# Cache for latest prices.  Maps symbol -> (timestamp, price).
PRICE_CACHE: dict[str, tuple[float, float]] = {}
# Time‑to‑live for cache entries (seconds).
CACHE_TTL = 300


@lru_cache(maxsize=256)
def _download_history(symbol: str, period: str = "5d", interval: str = "1d"):
    """
    Download historical price data using yfinance.

    This function is cached to avoid repeated network calls.  It is used
    both as a fallback for fetching current prices and by the technicals
    module for computing indicators.
    """
    logger.info("Downloading history (yfinance) for %s", symbol)
    df = yf.download(
        symbol, period=period, interval=interval,
        progress=False, auto_adjust=False
    )
    if df is None or df.empty:
        raise RuntimeError(f"No price data for {symbol}")
    return df


def _get_price_from_yf(symbol: str) -> float:
    """Fallback price retrieval using yfinance."""
    logger.warning("Fallback to yfinance for %s", symbol)
    df = _download_history(symbol)
    return float(df["Close"].iloc[-1])


def _get_price_from_fmp(symbol: str) -> float:
    """
    Retrieve the latest price from Financial Modeling Prep using the stable
    quote endpoint.  If the API key is missing or the request fails,
    an exception is raised.
    """
    if not FMP_API_KEY:
        raise RuntimeError("FMP_API_KEY is not set in Streamlit secrets.")
    url = f"https://financialmodelingprep.com/stable/quote?symbol={symbol}&apikey={FMP_API_KEY}"
    logger.info("Calling FMP for %s", symbol)
    resp = requests.get(url, timeout=5)
    # Check for HTTP errors explicitly.  A 403 may indicate plan limitations.
    if resp.status_code != 200:
        raise RuntimeError(f"FMP HTTP {resp.status_code}: {resp.text[:120]}")
    data = resp.json()
    # Data can be a list or dict depending on the endpoint.
    if isinstance(data, list) and data:
        item = data[0]
    else:
        item = data
    price = item.get("price") or item.get("previousClose") or item.get("close")
    if price is None:
        raise RuntimeError(f"FMP missing price: {item}")
    price = float(price)
    if price <= 0:
        raise RuntimeError(f"FMP returned non‑positive price for {symbol}: {price}")
    return price


def get_latest_price(symbol: str) -> float:
    """
    Return the most recent closing price for a ticker.

    This function first checks an in‑memory cache.  If there is no fresh
    value, it calls FMP and falls back to yfinance upon any error.
    """
    now = time.time()
    if symbol in PRICE_CACHE:
        ts, cached_price = PRICE_CACHE[symbol]
        if now - ts <= CACHE_TTL:
            return cached_price
    # Attempt FMP.  If any exception occurs, fall back to yfinance.
    try:
        price = _get_price_from_fmp(symbol)
    except Exception as e:
        logger.warning("FMP price fetch failed for %s: %s", symbol, e)
        price = _get_price_from_yf(symbol)
    PRICE_CACHE[symbol] = (now, price)
    return price


__all__ = ["get_latest_price", "_download_history"]