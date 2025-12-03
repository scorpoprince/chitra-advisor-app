import pandas as pd
from backend.portfolio import build_portfolio


def test_portfolio_skip(monkeypatch):
    # Patch price retrieval: return price for first symbol, raise for others.
    monkeypatch.setattr(
        "backend.prices.get_latest_price",
        lambda sym: 200.0 if sym == "TCS.NS" else (_ for _ in ()).throw(RuntimeError("no price"))
    )
    # Patch models: small universe with one good and two bad tickers.
    monkeypatch.setattr(
        "backend.models.UNIVERSES",
        {"TEST": ["TCS.NS", "FAIL1.NS", "FAIL2.NS"]}
    )
    df, cash = build_portfolio("TEST", "Medium", 10000)
    assert len(df) == 1
    assert df.iloc[0]["Symbol"] == "TCS.NS"