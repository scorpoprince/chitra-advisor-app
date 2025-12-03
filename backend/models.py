# Centralised definitions for stock universes and risk configurations.
#
# By keeping these constants in a separate module they can be reused across
# the portfolio builder, the test suite and elsewhere without risking
# accidental modification.

# Stock universes used by the Streamlit UI.  Each entry maps a universe key
# to a list of NSE ticker strings.  These definitions mirror those from the
# original monolithic `chitra_core.py` to ensure feature parity.
UNIVERSES = {
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
    "CUSTOM_HQ20": [
        "RELIANCE.NS", "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS",
        "INFY.NS", "TCS.NS", "ITC.NS", "HINDUNILVR.NS",
        "ASIANPAINT.NS", "BAJFINANCE.NS",
        "DMART.NS", "PIDILITIND.NS", "DIVISLAB.NS",
        "SUNPHARMA.NS", "TITAN.NS", "ULTRACEMCO.NS",
        "MARUTI.NS", "AXISBANK.NS", "LT.NS", "NESTLEIND.NS",
    ],
}

# List of risk levels, maintained in the same order used in the original app.
RISK_LEVELS = ["Low", "Medium", "High", "All or Nothing"]

# Mapping from risk profile to the number of positions.  Lower risk uses more
# positions (greater diversification).
RISK_TO_POSITIONS = {
    "Low": 12,
    "Medium": 8,
    "High": 6,
    "All or Nothing": 3,
}

__all__ = ["UNIVERSES", "RISK_TO_POSITIONS", "RISK_LEVELS"]