import logging
import streamlit as st

# Create a module‑level logger.  Streamlit attaches its own handlers so this
# logger will emit to the Streamlit console.  We keep a single logger for
# the entire backend to make it easy to enable/disable debug output.
logger = logging.getLogger("chitra_backend")
if not logger.handlers:
    logger.setLevel(logging.INFO)

# Read the Financial Modeling Prep (FMP) API key once at module import time.
# We force this to come from Streamlit secrets rather than environment
# variables to avoid relying on the host environment.  If the key is not
# present, the backend will gracefully fall back to yfinance for price data.
try:
    FMP_API_KEY = st.secrets["FMP_API_KEY"].strip()
except Exception:
    FMP_API_KEY = ""

# Emit a debug message about whether the FMP key was found.  This will be
# visible in the Streamlit logs and is useful when diagnosing why FMP calls
# are failing.
logger.info("FMP key present: %s (len=%d)", bool(FMP_API_KEY), len(FMP_API_KEY))

__all__ = ["logger", "FMP_API_KEY"]