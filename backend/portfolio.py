"""
Portfolio construction.

This module takes a selected universe and risk profile and constructs a simple
equal‑weighted portfolio.  Prices are fetched via the prices module and any
symbols whose price cannot be retrieved are skipped.  Remaining weights are
normalised so that allocations sum to 100%.
"""

import math
import pandas as pd

from .models import UNIVERSES, RISK_TO_POSITIONS
from .symbols import normalize_symbol
from .prices import get_latest_price
from .utils import logger


def _pick(universe: list[str], risk: str) -> list[str]:
    """Select a subset of tickers based on risk profile."""
    n = RISK_TO_POSITIONS.get(risk, 8)
    return universe[: min(n, len(universe))]


def build_portfolio(universe_key: str, risk_profile: str, total_capital: float) -> tuple[pd.DataFrame, float]:
    """
    Build an equal‑weight portfolio and return (DataFrame, unallocated_cash).

    The DataFrame has columns:
      Symbol, Approx Price (₹), Allocation (%), Allocation (₹), Approx Qty

    Any tickers whose price fetch fails are skipped and the remaining
    allocations are recomputed accordingly.
    """
    if total_capital <= 0:
        raise ValueError("Total capital must be positive.")
    if universe_key not in UNIVERSES:
        raise ValueError("Unknown universe selected.")

    universe = UNIVERSES[universe_key]
    selected = _pick(universe, risk_profile)
    if not selected:
        raise ValueError("No symbols available for the selected universe.")

    price_map: dict[str, float] = {}
    skipped_symbols: list[str] = []

    for sym in selected:
        try:
            # Normalise in case input universes omit suffixes (rare).
            ticker = normalize_symbol(sym)
            price_map[sym] = get_latest_price(ticker)
        except Exception as exc:
            skipped_symbols.append(sym)
            logger.warning("Skipping %s – price fetch failed: %s", sym, exc)

    if not price_map:
        raise RuntimeError(
            "Could not fetch price data for any stocks in this universe. "
            "Please try again later or choose a different universe."
        )

    n = len(price_map)
    weight = round(100.0 / n, 2)
    rows = []
    unallocated_cash = total_capital

    for sym, price in price_map.items():
        allocation_rupees = total_capital * weight / 100.0
        qty = math.floor(allocation_rupees / price) if price > 0 else 0
        used_cash = qty * price
        unallocated_cash -= used_cash
        rows.append({
            "Symbol": sym,
            "Approx Price (₹)": round(price, 2),
            "Allocation (%)": weight,
            "Allocation (₹)": round(used_cash, 2),
            "Approx Qty": int(qty),
        })

    if skipped_symbols:
        logger.info("Portfolio built, but skipped symbols: %s", ", ".join(skipped_symbols))

    df = pd.DataFrame(rows)
    return df, round(unallocated_cash, 2)


__all__ = ["build_portfolio"]