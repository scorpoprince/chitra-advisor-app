# chitra_app.py
import os

import streamlit as st

from chitra_core import (
    analyse_single_stock,
    get_universe_symbols,
    simple_equal_weight_portfolio,
    CUSTOM_HQ20_KEY,
)

# --------- Connect Streamlit secrets to OpenAI env ---------
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

st.set_page_config(
    page_title="ChitraAdvisor – Stock Helper",
    layout="wide",
)


# --------- Cached analysis to reduce API calls ---------


@st.cache_data(ttl=900)  # 15 minutes
def cached_single_stock_analysis(
    ticker: str,
    capital: float,
    holding_days: int,
    risk_profile: str,
    language: str,
    model_name: str,
):
    return analyse_single_stock(
        ticker=ticker,
        capital=capital,
        holding_days=holding_days,
        risk_profile=risk_profile,
        language=language,
        model_name=model_name,
    )


@st.cache_data(ttl=900)
def cached_portfolio_allocation(
    universe_key: str,
    amount: float,
    risk_profile: str,
):
    symbols = get_universe_symbols(universe_key)
    rows = simple_equal_weight_portfolio(symbols, amount, risk_profile)
    return rows


# --------- UI Helpers ---------


def risk_help_text(risk_profile: str) -> str:
    if risk_profile == "Low":
        return "Approx. 5–10% of portfolio per stock, focus on mega/large caps."
    if risk_profile == "Medium":
        return "Approx. 10–15% per stock, diversified across quality names."
    if risk_profile == "High":
        return "Approx. 15–20% per stock, tilt to higher beta ideas."
    if risk_profile == "All or Nothing":
        return "Concentrated 30–50% in a few names. Very high risk."
    return ""


# --------- MAIN LAYOUT ---------

st.markdown(
    "## ChitraAdvisor – Stock Helper  \n"
    "**Safety ON · Experimental, educational tool only · Not SEBI-registered advice.**"
)

tab1, tab2 = st.tabs(["Single Stock Idea", "Portfolio Suggestion"])


# =========================================================
# TAB 1 – Single Stock Idea
# =========================================================

with tab1:
    st.subheader("Single Stock Idea")

    col_left, col_right = st.columns([2, 2])

    with col_left:
        ticker_input = st.text_input(
            "Company name or NSE symbol",
            value="RELIANCE.NS",
            help="You can enter 'RELIANCE.NS', 'TCS.NS', etc. NSE symbols preferred.",
        )

        holding_days = st.slider(
            "Intended holding period (days)",
            min_value=30,
            max_value=730,
            value=90,
            step=10,
        )

        language = st.radio(
            "Language for explanation",
            options=["English", "Hindi"],
            horizontal=True,
        )

    with col_right:
        capital = st.number_input(
            "Total capital considered for this idea (₹)",
            min_value=1000.0,
            max_value=5_000_000.0,
            value=50_000.0,
            step=5_000.0,
        )

        risk_profile = st.selectbox(
            "Risk profile (for interpretation)",
            options=["Low", "Medium", "High", "All or Nothing"],
            index=1,
            help="Used only to interpret the idea; the app will not forcibly match index membership.",
        )
        st.caption("Approx guide: " + risk_help_text(risk_profile))

        model_name = st.selectbox(
            "AI model",
            options=["gpt-4.1-mini", "gpt-5.1"],
            index=0,
            help="Mini is cheaper & faster. gpt-5.1 is more powerful but costs more.",
        )

    generate_btn = st.button("Generate Stock View")

    if generate_btn:
        if not ticker_input.strip():
            st.error("Please enter a valid NSE symbol or company name.")
        else:
            with st.spinner("Thinking about this stock..."):
                try:
                    result = cached_single_stock_analysis(
                        ticker=ticker_input.strip(),
                        capital=float(capital),
                        holding_days=int(holding_days),
                        risk_profile=risk_profile,
                        language=language,
                        model_name=model_name,
                    )
                except Exception as exc:
                    st.error(f"Something went wrong: {exc}")
                else:
                    metrics = result["metrics"]
                    rating = result["rating"]
                    explanation = result["explanation"]

                    badge_color = {
                        "BUY": "green",
                        "HOLD": "orange",
                        "SELL": "red",
                        "AVOID": "red",
                    }.get(rating, "gray")

                    st.markdown(
                        f"### Rating: "
                        f"<span style='background-color:{badge_color};"
                        f"color:white;padding:4px 10px;border-radius:8px;'>"
                        f"{rating}</span>",
                        unsafe_allow_html=True,
                    )

                    # Quick metrics panel
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Last Close", f"₹{metrics['latest_close']}")
                    m2.metric("50 / 200 DMA", f"₹{metrics['ma50']} / ₹{metrics['ma200']}")
                    m3.metric("RSI(14)", metrics["rsi14"])
                    m4.metric("Volatility (ann.)", f"{metrics['vol_annual']*100:.1f}%")

                    st.markdown("#### Explanation")
                    st.markdown(explanation)


# =========================================================
# TAB 2 – Portfolio Suggestion
# =========================================================

with tab2:
    st.subheader("Portfolio Suggestion (Universe-based)")

    col_u_left, col_u_right = st.columns([2, 2])

    with col_u_left:
        universe_type = st.radio(
            "Universe type",
            options=["Index / Custom"],
            horizontal=True,
        )

        universe_key = st.selectbox(
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

        # Map UI label -> internal key
        label_to_key = {
            "NIFTY50": "NIFTY50",
            "NIFTY100": "NIFTY100",
            "NIFTY Next 50": "NIFTYNEXT50",
            "NIFTY200": "NIFTY200",
            "Midcap 150": "MIDCAP150",
            "Smallcap 100": "SMALLCAP100",
            "Custom High-Quality 20": CUSTOM_HQ20_KEY,
        }
        internal_universe_key = label_to_key[universe_key]

    with col_u_right:
        portfolio_amount = st.number_input(
            "Total investment amount for portfolio (₹)",
            min_value=10_000.0,
            max_value=10_000_000.0,
            value=200_000.0,
            step=10_000.0,
        )

        portfolio_risk = st.selectbox(
            "Risk profile for this portfolio",
            options=["Low", "Medium", "High", "All or Nothing"],
            index=1,
        )
        st.caption(risk_help_text(portfolio_risk))

    gen_port_btn = st.button("Generate Portfolio")

    if gen_port_btn:
        with st.spinner("Building a rough equal-weight portfolio..."):
            try:
                rows = cached_portfolio_allocation(
                    universe_key=internal_universe_key,
                    amount=float(portfolio_amount),
                    risk_profile=portfolio_risk,
                )
            except Exception as exc:
                st.error(f"Something went wrong: {exc}")
                rows = []

        if not rows:
            st.warning("Could not build portfolio – no usable price data.")
        else:
            import pandas as pd

            df = pd.DataFrame(rows)
            cash_left = round(
                portfolio_amount - float(df["allocation_value"].sum()), 2
            )

            st.markdown("### Suggested Portfolio Allocation")
            st.write(
                f"Universe: **{universe_key}** | "
                f"Risk: **{portfolio_risk}** | "
                f"Positions: **{len(df)}**"
            )

            df_display = df.rename(
                columns={
                    "symbol": "Symbol",
                    "price": "Approx Price (₹)",
                    "allocation_pct": "Allocation (%)",
                    "allocation_value": "Allocation (₹)",
                    "qty": "Approx Qty",
                }
            )
            st.dataframe(df_display, use_container_width=True)

            st.info(
                f"Approx cash left unallocated due to lot sizes: ₹{cash_left:,.2f}"
            )

# Footer
st.caption(
    "This tool is for educational & experimental purposes only. "
    "Do your own research or consult a professional advisor before taking any "
    "investment decision."
)
