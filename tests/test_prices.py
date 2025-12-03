import pytest

from backend.prices import get_latest_price, PRICE_CACHE


def test_price_cache(monkeypatch):
    # Force the FMP call to raise and yfinance to return a fixed value.
    monkeypatch.setattr(
        "backend.prices._get_price_from_fmp",
        lambda symbol: (_ for _ in ()).throw(RuntimeError("FMP unavailable"))
    )
    monkeypatch.setattr(
        "backend.prices._get_price_from_yf",
        lambda symbol: 123.45
    )
    # Clear cache before test
    PRICE_CACHE.clear()
    price1 = get_latest_price("TCS.NS")
    price2 = get_latest_price("TCS.NS")
    assert price1 == 123.45
    assert price2 == 123.45  # Should come from cache