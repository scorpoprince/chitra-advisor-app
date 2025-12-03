"""
Streamlit front-end for ChitraAdvisor – Stock Helper (Safety ON)

This file assumes:
- chitra_core.py is in the same folder.
- OPENAI_API_KEY is provided via Streamlit secrets.
"""

import os

import streamlit as st

from chitra_core import (
    analyze_single_stock,
    build_portfolio,
    UNIVERSES,
    RISK_LEVELS,
)

# Bridge Streamlit secrets -> environment for OpenAI client
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

st.set_page_config(
    page_title="ChitraAdvisor – Stock Helper",
    layout="wide",
)

st.title("ChitraAdvisor – Stock Helper")
st.caption(
    "Safety ON · Experimental, educational tool only · Not SEBI-registered advice."
)

tab_single, tab_portfolio = st.tabs(["Single Stock Idea", "Portfolio Suggestion"])

# --------------------------------------------------------------------
# Single Stock Idea tab
# --------------------------------------------------------------------
with tab_single:
    st.subheader("Single Stock Idea")

    col_left, col_right = st.columns([2, 2])

    with col_left:
        user_symbol = st.text_input(
            "Company name or NSE symbol",
            help="Examples: `TCS`, `RELIANCE`, `MARUTI.NS`. "
                 "You can also try a full name like `Maruti Suzuki India Limited`.",
        )

        holding_period_days = st.slider(
            "Intended holding period (days)",
            min_value=30,
            max_value=730,
            value=90,
            step=10,
        )

        language_choice = st.radio(
            "Language for explanation",
            options=["English", "Hindi"],
            index=0,
            horizontal=True,
        )
        language_code = "en" if language_choice == "English" else "hi"

    with col_right:
        capital = st.number_input(
            "Total capital considered for this idea (₹)",
            min_value=1000.0,
            max_value=10_000_000.0,
            value=50_000.0,
            step=5_000.0,
            format="%.2f",
        )

        risk_profile = st.selectbox(
            "Risk profile (for interpretation)",
            options=RISK_LEVELS,
            index=1,
            help=(
                "Approx guide: "
                "Low ≈ 5–10% of portfolio per stock, mostly mega/large caps; "
                "Medium ≈ 10–15%; High ≈ 15–20%; "
                "All or Nothing ≈ concentrated bets 30–50% in a few names."
            ),
        )

        model_choice = st.selectbox(
            "AI model",
            options=["gpt-5.1", "gpt-4.1-mini"],
            index=0,
        )

    if st.button("Generate Stock View", type="primary"):
        try:
            with st.spinner("Asking ChitraAdvisor to think..."):
                result = analyze_single_stock(
                    user_typed=user_symbol,
                    capital=capital,
                    risk_profile=risk_profile,
                    holding_period_days=holding_period_days,
                    language=language_code,
                    model=model_choice,
                )

            st.success(
                f"View generated for **{result['display_symbol']}** "
                f"(approx last close ₹{result['metrics']['last_close']})"
            )

            # Show quick metrics
            with st.expander("Quick technical snapshot", expanded=False):
                m = result["metrics"]
                cols = st.columns(3)
                cols[0].metric("Last close (₹)", m["last_close"])
                cols[1].metric("50D / 200D MA (₹)", f"{m['ma50']} / {m['ma200']}")
                cols[2].metric("1Y return (%)", m["one_year_return_pct"])
                col2 = st.columns(3)
                col2[0].metric("RSI(14)", m["rsi14"])
                col2[1].metric("Annual vol", f"{m['vol_annual']:.2f}")
                col2[2].metric("Support / Resistance (₹)", f"{m['support']} / {m['resistance']}")

            st.markdown("### Chitra’s view")
            st.markdown(result["analysis_markdown"])

        except Exception as exc:
            st.error(f"Something went wrong: {exc}")

    st.markdown(
        "<br><small>This tool is for educational & experimental purposes only. "
        "Do your own research or consult a professional advisor before taking any investment decision.</small>",
        unsafe_allow_html=True,
    )

# --------------------------------------------------------------------
# Portfolio Suggestion tab
# --------------------------------------------------------------------
with tab_portfolio:
    st.subheader("Portfolio Suggestion (Universe-based)")

    col_left, col_right = st.columns([2, 2])

    with col_left:
        universe_type = st.radio(
            "Universe type",
            options=["Index / Custom"],  # Sector mode can come later
            horizontal=True,
        )

        universe_key_ui = st.selectbox(
            "Choose universe",
            options=[
                "NIFTY50",
                "NIFTY100",
                "NIFTY Next 50",
                "NIFTY200",
                "Midcap 150",
                "Smallcap 100",
                "Custom High-Quality 20",
            ],
            index=0,
        )

        # Map UI label to internal key
        universe_map = {
            "NIFTY50": "NIFTY50",
            "NIFTY100": "NIFTY100",
            "NIFTY Next 50": "NIFTY_NEXT_50",
            "NIFTY200": "NIFTY200",
            "Midcap 150": "MIDCAP150",
            "Smallcap 100": "SMALLCAP100",
            "Custom High-Quality 20": "CUSTOM_HQ20",
        }
        universe_key = universe_map[universe_key_ui]

    with col_right:
        total_capital_pf = st.number_input(
            "Total investment amount for portfolio (₹)",
            min_value=10_000.0,
            max_value=50_000_000.0,
            value=200_000.0,
            step=10_000.0,
            format="%.2f",
        )

        risk_profile_pf = st.selectbox(
            "Risk profile for this portfolio",
            options=RISK_LEVELS,
            index=1,
        )

        risk_help_map = {
            "Low": "Approx. 5–10% per stock, diversified across quality names.",
            "Medium": "Approx. 10–15% per stock, diversified across quality names.",
            "High": "Approx. 15–20% per stock, tilt to higher beta ideas.",
            "All or Nothing": "Concentrated 30–50% in a few names. Very high risk.",
        }
        st.caption(risk_help_map[risk_profile_pf])

    if st.button("Generate Portfolio", type="primary"):
        try:
            with st.spinner("Building a simple equal-weight portfolio..."):
                df, unallocated_cash = build_portfolio(
                    universe_key=universe_key,
                    risk_profile=risk_profile_pf,
                    total_capital=total_capital_pf,
                )

            st.markdown(
                f"**Universe:** {universe_key_ui}  "
                f"· **Risk:** {risk_profile_pf}  "
                f"· **Positions:** {len(df)}"
            )

            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
            )

            st.info(
                f"Approx cash left unallocated due to lot sizes: "
                f"₹{unallocated_cash:,.2f}"
            )

        except Exception as exc:
            st.error(f"Something went wrong: {exc}")

    st.markdown(
        "<br><small>This tool does not optimise tax, brokerage or impact cost. "
        "It is only a simple educational allocator based on equal weights.</small>",
        unsafe_allow_html=True,
    )
