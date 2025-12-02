# chitra_app.py
#
# Streamlit UI for ChitraAdvisor – cleaner, mobile-friendly.

import streamlit as st

from chitra_core import (
    RISK_PROFILES,
    allowed_universes_for_risk,
    analyze_single_stock,
    build_portfolio,
)
# Bridge Streamlit secrets → environment variable for OpenAI
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]


st.set_page_config(
    page_title="ChitraAdvisor – Stock Helper",
    layout="wide",
)

# ---------- STYLES ----------

st.markdown(
    """
    <style>
    .small-note {
        font-size: 0.8rem;
        color: #9ca3af;
    }
    .verdict-box {
        padding: 1rem 1.2rem;
        border-radius: 0.5rem;
        background-color: #111827;
        border: 1px solid #374151;
        font-size: 0.95rem;
        line-height: 1.5;
        white-space: pre-wrap;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("ChitraAdvisor – Stock Helper")
st.caption("Safety ON · Experimental, educational tool only · Not SEBI-registered advice.")

tab_single, tab_portfolio = st.tabs(["Single Stock Idea", "Portfolio Suggestion"])

# ---------- SINGLE STOCK TAB ----------

with tab_single:
    st.subheader("Single Stock Idea")

    col_left, col_right = st.columns([1.4, 1])

    with col_left:
        symbol = st.text_input(
            "Company name or NSE symbol",
            value="RELIANCE.NS",
            help="Example: RELIANCE.NS, HDFCBANK.NS, TCS.NS etc.",
        )

        holding_days = st.slider(
            "Intended holding period (days)",
            min_value=30,
            max_value=730,
            value=180,
            step=10,
        )

        lang = st.radio(
            "Language for explanation",
            options=["English", "Hindi"],
            horizontal=True,
        )

    with col_right:
        capital = st.number_input(
            "Total capital considered for this idea (₹)",
            min_value=1000.0,
            value=20000.0,
            step=1000.0,
        )

        risk_profile = st.selectbox(
            "Risk profile (for interpretation)",
            options=RISK_PROFILES,
            index=0,
            help=(
                "Risk here does NOT change the stock price.\n"
                "It only affects how strict/loose the verdict is: "
                "Low risk → more conservative, All or Nothing → aggressive."
            ),
        )

        st.markdown(
            "<p class='small-note'>Approx guide: "
            "<b>Low</b> ≈ 5–10% of portfolio per stock, mega/large caps only; "
            "<b>Medium</b> ≈ 5–15%; "
            "<b>High</b> ≈ 10–20%; "
            "<b>All or Nothing</b> can be very concentrated.</p>",
            unsafe_allow_html=True,
        )

        model_name = st.selectbox(
            "AI model",
            options=["gpt-4.1-mini", "gpt-4.1", "gpt-5.1"],
            index=0,
        )

    st.markdown("---")

    if st.button("Generate Stock View", type="primary"):
        if not symbol.strip():
            st.error("Please enter a company name or symbol.")
        else:
            try:
                with st.spinner("Analyzing stock..."):
                    result = analyze_single_stock(
                        symbol=symbol.strip(),
                        capital=capital,
                        risk_profile=risk_profile,
                        holding_period_days=holding_days,
                        model_name=model_name,
                        language=lang,
                    )

                ind = result.indicators
                scores = result.scores
                risk = result.risk

                st.markdown(f"### Verdict & Explanation for `{result.symbol}`")
                st.markdown("<div class='verdict-box'>" + result.verdict_text + "</div>", unsafe_allow_html=True)

                with st.expander("Technical snapshot & risk numbers"):
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.metric("Last Close (₹)", f"{ind['latest_close']:.2f}")
                        st.metric("50d MA (₹)", f"{ind['ma50']:.2f}")
                        st.metric("200d MA (₹)", f"{ind['ma200']:.2f}")
                    with c2:
                        st.metric("RSI(14)", f"{ind['rsi']:.1f}")
                        st.metric("Volatility (annual)", f"{ind['annual_vol']:.1%}")
                        st.metric("Volume vs 50d", f"{ind['volume_ratio']:.2f}x")
                    with c3:
                        st.metric("Support (₹)", f"{ind['support']:.2f}")
                        st.metric("Resistance (₹)", f"{ind['resistance']:.2f}")
                        st.metric("Pivot (₹)", f"{ind['pivot']:.2f}")

                    st.markdown("**Quant scores (0–1):**")
                    st.write(scores)

                    st.markdown("**Position sizing (based on your capital & risk profile):**")
                    st.write(risk)

            except Exception as exc:
                st.error(f"Something went wrong: {exc}")

    st.markdown(
        "<p class='small-note'>This tool is for educational & experimental purposes only. "
        "Do your own research or consult a professional advisor before investing.</p>",
        unsafe_allow_html=True,
    )

# ---------- PORTFOLIO TAB ----------

with tab_portfolio:
    st.subheader("Portfolio Suggestion (Universe-based)")

    col_left, col_right = st.columns([1.2, 1])

    with col_right:
        port_capital = st.number_input(
            "Total investment amount for portfolio (₹)",
            min_value=10000.0,
            value=200000.0,
            step=5000.0,
        )

        port_risk = st.selectbox(
            "Risk profile for this portfolio",
            options=RISK_PROFILES,
            index=1,
        )

        risk_hint = {
            "Low": "Approx. 8–12 diversified large caps.",
            "Medium": "10–12 stocks, mix of large & quality midcaps.",
            "High": "4–8 focused bets, can include midcaps.",
            "All or Nothing": "2–4 very concentrated, can include smallcaps. Very high risk.",
        }[port_risk]

        st.markdown(f"<p class='small-note'>{risk_hint}</p>", unsafe_allow_html=True)

    with col_left:
        st.markdown("**Universe type**")
        uni_type = st.radio(
            "",
            options=["Index / Custom"],
            horizontal=True,
            label_visibility="collapsed",
        )

        allowed_unis = allowed_universes_for_risk(port_risk)
        universe = st.selectbox(
            "Choose universe",
            options=allowed_unis,
            index=0,
            help="Options filtered automatically based on your risk profile.",
        )

    st.markdown(
        "<p class='small-note'>Note: Small-/micro-cap ideas are only considered for "
        "<b>High</b> and <b>All or Nothing</b> risk profiles. With Safety ON, "
        "conservative profiles stick to larger, more liquid names.</p>",
        unsafe_allow_html=True,
    )

    st.markdown("---")

    if st.button("Generate Portfolio", key="generate_portfolio", type="primary"):
        try:
            with st.spinner("Building portfolio..."):
                df_port, leftover = build_portfolio(
                    universe_name=universe,
                    capital=port_capital,
                    risk_profile=port_risk,
                )

            st.markdown("### Suggested Portfolio Allocation")
            st.markdown(
                f"Universe: **{universe}** &nbsp;&nbsp;|&nbsp;&nbsp; "
                f"Risk: **{port_risk}** &nbsp;&nbsp;|&nbsp;&nbsp; "
                f"Positions: **{len(df_port)}**"
            )

            st.dataframe(
                df_port,
                use_container_width=True,
                hide_index=True,
            )

            st.markdown(
                f"<div class='small-note'>Approx cash left unallocated due to lot sizes: "
                f"₹{leftover:,.2f}</div>",
                unsafe_allow_html=True,
            )

        except Exception as exc:
            st.error(f"Could not generate portfolio: {exc}")

    st.markdown(
        "<p class='small-note'>This portfolio is a mechanical suggestion based on prices and your "
        "risk label. It is NOT personalised financial advice.</p>",
        unsafe_allow_html=True,
    )
