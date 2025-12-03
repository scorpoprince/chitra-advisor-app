import pandas as pd
from backend.technicals import get_basic_technicals


def test_technicals(monkeypatch):
    # Create a deterministic fake dataframe with monotonically increasing prices.
    data = {"Close": [100 + i for i in range(300)]}
    fake_df = pd.DataFrame(data)
    # Patch the history download function to return fake data.
    monkeypatch.setattr(
        "backend.technicals._download_history",
        lambda symbol, period="1y", interval="1d": fake_df
    )
    metrics = get_basic_technicals("TCS.NS")
    assert metrics["last_close"] > 0
    assert "ma50" in metrics and "ma200" in metrics