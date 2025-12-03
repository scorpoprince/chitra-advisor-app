from backend.single_stock import analyze_single_stock


def test_single_stock(monkeypatch):
    # Patch technicals to avoid network access.
    import pandas as pd
    fake_df = pd.DataFrame({"Close": [100 + i for i in range(300)]})
    monkeypatch.setattr(
        "backend.technicals._download_history",
        lambda symbol, period="1y", interval="1d": fake_df
    )
    # Patch OpenAI client to return a dummy response.
    class DummyResp:
        class Choice:
            class Msg:
                content = (
                    "Action: BUY\n"
                    "Entry zone: ₹100\n"
                    "Stop loss: ₹90\n"
                    "Targets: ₹110\n"
                    "Suggested allocation: 10%\n"
                    "\n"
                    "Holding view:\n"
                    "This is a test.\n"
                    "\n"
                    "Reasoning:\n"
                    "- Good company.\n"
                    "- Strong fundamentals.\n"
                    "\n"
                    "Risk reminder:\n"
                    "This is educational."
                )
            message = Msg()
        choices = [Choice()]

    monkeypatch.setattr(
        "backend.single_stock.client.chat.completions.create",
        lambda **kwargs: DummyResp()
    )
    result = analyze_single_stock("TCS", 50000, "Medium", 30)
    assert "analysis_markdown" in result
    assert "Action:" in result["analysis_markdown"]