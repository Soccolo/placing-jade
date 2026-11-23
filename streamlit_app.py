import asyncio
import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Placing Jade Terminal", page_icon="💎", layout="wide")


def ensure_encryption_key():
    if os.environ.get("ENCRYPTION_KEY"):
        return
    try:
        if "ENCRYPTION_KEY" in st.secrets:
            os.environ["ENCRYPTION_KEY"] = st.secrets["ENCRYPTION_KEY"]
    except Exception:
        pass
    if not os.environ.get("ENCRYPTION_KEY"):
        st.error(
            "ENCRYPTION_KEY is required. Set it in Streamlit Secrets or your environment."
        )
        st.stop()


ensure_encryption_key()

from app.database import init_db
from app.financial_command_center.charts import (
    build_distribution_chart,
    build_pnl_simulation_chart,
    build_risk_contribution_chart,
    build_stock_analysis_chart,
)
from app.financial_command_center.constants import get_api_keys
from app.financial_command_center.portfolio_inputs import normalize_portfolio_dataframe
from app.financial_command_center.logic import (
    analyze_sentiment,
    calculate_bollinger_bands,
    calculate_macd,
    calculate_rsi,
    common_expirations,
    get_economic_news,
    get_fundamental_data,
    get_insurance_company_data,
    get_insurance_news,
    get_market_data,
    get_market_rates_data,
    get_stock_data_multi_source,
    get_stock_news_multi_source,
    predict_stock_movement,
    simulate_portfolio_pnl,
    options_implied_values,
    summarise_distribution,
)
from app.services.alpaca import fetch_account_data, verify_connection
from app.services.credentials import (
    delete_credentials,
    get_credentials,
    log_audit_event,
    save_credentials,
    update_connection_status,
)
from app.config import TARGET_PORTFOLIO_PATH, WEIGHT_SUM_EPSILON
from app.services.portfolio import clear_portfolio_cache, get_portfolio


def run_async(coro):
    return asyncio.run(coro)


@st.cache_resource
def ensure_db():
    run_async(init_db())
    return True


def show_logo():
    logo_path = Path(__file__).parent / "app" / "static" / "placing_jade_logo.png"
    if logo_path.exists():
        st.sidebar.image(str(logo_path), width=120)


def render_connect():
    st.header("Connect to Alpaca")

    account_mode = st.radio(
        "Account type",
        ["Paper Trading", "Live Trading"],
        horizontal=True,
        help="Paper and live accounts use different API keys. Connect each separately.",
    )
    is_paper = account_mode == "Paper Trading"
    mode_key = "paper" if is_paper else "live"
    mode_label = "paper" if is_paper else "live"

    if not is_paper:
        st.warning("⚠️ You are connecting to a **live** trading account. Real money is at stake.")

    creds = run_async(get_credentials())
    is_connected = bool(creds and creds.is_connected)

    status_col, detail_col = st.columns([1, 3])
    with status_col:
        st.metric("Status", "Connected" if is_connected else "Not Connected")
    with detail_col:
        if creds and creds.last_verified_at:
            st.write(f"Last verified: {creds.last_verified_at} UTC")

    if not is_connected:
        with st.form("connect_form"):
            api_key = st.text_input("API Key", placeholder="PK...", type="password")
            api_secret = st.text_input(
                "API Secret", placeholder="Your API secret", type="password"
            )
            submitted = st.form_submit_button("Connect")

        if submitted:
            if not api_key or not api_secret:
                st.error("API key and secret are required.")
                return
            success, message = verify_connection(api_key, api_secret, paper=is_paper)
            if success:
                run_async(save_credentials(api_key, api_secret))
                run_async(update_connection_status(is_connected=True))
                run_async(log_audit_event("connected", f"Initial {mode_label} connection verified"))
                st.session_state["alpaca_paper"] = is_paper
                st.success(f"Successfully connected to Alpaca {mode_label} trading.")
            else:
                run_async(log_audit_event("connection_failed", message))
                st.error(message)
    else:
        action_col1, action_col2 = st.columns(2)
        with action_col1:
            if st.button("Verify Connection"):
                paper_flag = st.session_state.get("alpaca_paper", True)
                success, message = verify_connection(creds.api_key, creds.api_secret, paper=paper_flag)
                if success:
                    run_async(update_connection_status(is_connected=True))
                    run_async(
                        log_audit_event("connected", "Re-verification successful")
                    )
                    st.success("Connection verified successfully.")
                else:
                    run_async(update_connection_status(is_connected=False))
                    run_async(log_audit_event("connection_failed", message))
                    st.error(message)
        with action_col2:
            if st.button("Disconnect"):
                run_async(delete_credentials())
                run_async(log_audit_event("disconnected", "Credentials deleted by user"))
                st.session_state.pop("alpaca_paper", None)
                st.success("Disconnected and credentials deleted.")

        st.divider()
        st.subheader("Update Credentials")
        with st.form("update_form"):
            api_key = st.text_input("API Key", placeholder="PK...", type="password")
            api_secret = st.text_input(
                "API Secret", placeholder="Your API secret", type="password"
            )
            submitted = st.form_submit_button("Update Credentials")
        if submitted:
            if not api_key or not api_secret:
                st.error("API key and secret are required.")
            else:
                success, message = verify_connection(api_key, api_secret, paper=is_paper)
                if success:
                    run_async(save_credentials(api_key, api_secret))
                    run_async(update_connection_status(is_connected=True))
                    run_async(log_audit_event("connected", f"Credentials updated ({mode_label})"))
                    st.session_state["alpaca_paper"] = is_paper
                    st.success("Credentials updated.")
                else:
                    run_async(log_audit_event("connection_failed", message))
                    st.error(message)


def render_dashboard():
    st.header("Dashboard")
    is_paper = st.session_state.get("alpaca_paper", True)
    mode_label = "paper" if is_paper else "live"
    st.caption(f"Your Alpaca **{mode_label}** trading account overview.")

    creds = run_async(get_credentials())
    if not creds or not creds.is_connected:
        st.warning("Not connected. Please connect your Alpaca account first.")
        return

    if st.button("Refresh account data") or "account_data" not in st.session_state:
        data, message = fetch_account_data(creds.api_key, creds.api_secret, paper=is_paper)
        if data:
            st.session_state["account_data"] = data
            st.session_state["account_message"] = None
            run_async(log_audit_event("refreshed", f"Fetched {len(data.positions)} positions ({mode_label})"))
        else:
            st.session_state["account_data"] = None
            st.session_state["account_message"] = message
            run_async(update_connection_status(is_connected=False))
            run_async(log_audit_event("connection_failed", message))

    data = st.session_state.get("account_data")
    error_message = st.session_state.get("account_message")

    if error_message:
        st.error(error_message)
        return

    if not data:
        st.info("No account data available yet.")
        return

    cols = st.columns(4)
    cols[0].metric("Portfolio Value", f"${data.account.portfolio_value:,.2f}")
    cols[1].metric("Equity", f"${data.account.equity:,.2f}")
    cols[2].metric("Cash", f"${data.account.cash:,.2f}")
    cols[3].metric("Buying Power", f"${data.account.buying_power:,.2f}")
    st.caption(f"Last refreshed: {data.fetched_at} UTC")

    positions_df = pd.DataFrame(
        [
            {
                "Symbol": p.symbol,
                "Qty": p.qty,
                "Market Value": p.market_value,
                "Avg Entry": p.avg_entry_price,
                "Current Price": p.current_price,
                "Unrealized P/L": p.unrealized_pl,
                "P/L %": p.unrealized_pl_pct,
            }
            for p in data.positions
        ]
    )
    st.subheader("Positions")
    if positions_df.empty:
        st.info("No positions in the account.")
    else:
        st.dataframe(positions_df, use_container_width=True)

        # ── Portfolio Risk Simulation ──
        st.divider()
        st.subheader("Portfolio Risk Simulation")

        if st.button("Run Simulation") or "sim_result" in st.session_state:
            tickers = [p.symbol for p in data.positions]
            quantities = [float(p.qty) for p in data.positions]

            if "sim_result" not in st.session_state or st.session_state.get("sim_tickers") != tickers:
                with st.spinner("Running 10,000 Monte Carlo scenarios..."):
                    result = simulate_portfolio_pnl(tickers, quantities)
                    st.session_state["sim_result"] = result
                    st.session_state["sim_tickers"] = tickers

            result = st.session_state.get("sim_result")

            if result is None or "error" in result:
                error_msg = result.get("error", "Unknown error") if result else "Simulation returned no result."
                st.warning(f"⚠️ {error_msg}")
                if result and result.get("diagnostics"):
                    with st.expander("Diagnostics", expanded=False):
                        for d in result["diagnostics"]:
                            st.text(d)
            else:
                fig = build_pnl_simulation_chart(
                    result["sim_pnl"],
                    result["var_95"],
                    result["var_995"],
                    result["tvar_995"],
                )
                st.plotly_chart(fig, use_container_width=True)

                st.caption(
                    "**What mathematics says** — using correlation between the stocks and past experience, "
                    "we simulated 10,000 economic scenarios for your portfolio. "
                    "Above is the distribution of the profits & losses based on these scenarios."
                )

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Portfolio Value", f"${result['total_investment']:,.2f}")
                m2.metric("Daily Volatility", f"${result['port_daily_vol']:,.2f}")
                m3.metric("VaR 99.5% (1-in-200)", f"${result['var_995']:,.2f}")
                m4.metric("TVaR 99.5%", f"${result['tvar_995']:,.2f}")

                # ── Largest contributors to risk ──
                contributions = result.get("risk_contributions") or []
                if contributions:
                    st.divider()
                    st.subheader("Largest Contributors to Risk")
                    st.caption(
                        "**How we attribute risk** — using Euler's theorem on the "
                        "covariance matrix (for daily volatility) and conditional "
                        "expectation on the simulated tail scenarios (for VaR & TVaR), "
                        "we decompose total portfolio risk into the share coming from "
                        "each position. Contributions sum to the headline figure, so "
                        "the largest bars are literally the biggest drivers of your "
                        "downside."
                    )

                    metric_options = {
                        "VaR 99.5%": ("component_var_995", "var_995"),
                        "VaR 99%": ("component_var_99", "var_99"),
                        "VaR 95%": ("component_var_95", "var_95"),
                        "TVaR 99.5%": ("component_tvar_995", "tvar_995"),
                        "Daily Volatility": ("component_vol", "port_daily_vol"),
                    }
                    selected_label = st.selectbox(
                        "Risk measure to decompose",
                        list(metric_options.keys()),
                        index=0,
                    )
                    metric_key, total_key = metric_options[selected_label]

                    chart_fig = build_risk_contribution_chart(
                        contributions,
                        metric=metric_key,
                        title=f"Contribution to {selected_label}",
                    )
                    st.plotly_chart(chart_fig, use_container_width=True)

                    total_value = result.get(total_key, 0.0)
                    top = sorted(
                        contributions, key=lambda r: r[metric_key], reverse=True
                    )[:3]
                    if total_value and top:
                        top_sum = sum(r[metric_key] for r in top)
                        top_pct = top_sum / total_value * 100 if total_value else 0.0
                        top_str = ", ".join(
                            f"{r['ticker']} (${r[metric_key]:,.0f}, "
                            f"{r[metric_key] / total_value * 100:.1f}%)"
                            for r in top
                        )
                        st.info(
                            f"**Top 3 contributors to {selected_label}:** {top_str}. "
                            f"Together they explain {top_pct:.1f}% of total "
                            f"{selected_label.lower()} (${total_value:,.0f})."
                        )

                    contrib_df = pd.DataFrame(contributions)
                    if not contrib_df.empty:
                        display_df = pd.DataFrame(
                            {
                                "Ticker": contrib_df["ticker"],
                                "Position ($)": contrib_df["position_value"].map(
                                    lambda v: f"${v:,.0f}"
                                ),
                                "Weight": contrib_df["weight"].map(
                                    lambda v: f"{v * 100:.1f}%"
                                ),
                                "Component Vol ($)": contrib_df["component_vol"].map(
                                    lambda v: f"${v:,.0f}"
                                ),
                                "Vol Share": contrib_df["component_vol_pct"].map(
                                    lambda v: f"{v * 100:.1f}%"
                                ),
                                "Component VaR 99.5% ($)": contrib_df[
                                    "component_var_995"
                                ].map(lambda v: f"${v:,.0f}"),
                                "VaR 99.5% Share": contrib_df[
                                    "component_var_995_pct"
                                ].map(lambda v: f"{v * 100:.1f}%"),
                                "Component TVaR 99.5% ($)": contrib_df[
                                    "component_tvar_995"
                                ].map(lambda v: f"${v:,.0f}"),
                            }
                        )
                        with st.expander("Full per-position breakdown", expanded=False):
                            st.dataframe(display_df, use_container_width=True, hide_index=True)
                            st.caption(
                                "**Component Vol** comes from Euler decomposition on "
                                "the covariance matrix; columns sum exactly to the "
                                "headline volatility. **Component VaR / TVaR** come "
                                "from conditional expectation on the simulated tail "
                                "scenarios and sum approximately to the headline VaR "
                                "/ TVaR (small Monte Carlo noise on the band estimate)."
                            )

                if result["skipped_tickers"]:
                    st.warning(f"Skipped tickers (no price data): {', '.join(result['skipped_tickers'])}")

                if result.get("diagnostics"):
                    with st.expander("Simulation diagnostics", expanded=False):
                        st.text(f"Tickers used: {', '.join(result['available_tickers'])}")
                        st.text(f"Trading days of history: {result['num_trading_days']}")
                        for d in result["diagnostics"]:
                            st.text(d)


def render_strategy():
    st.header("Target Portfolio")
    st.caption("Portfolio allocation from data/target_portfolio.csv")

    if st.button("Reload from disk"):
        clear_portfolio_cache()
        result = get_portfolio(TARGET_PORTFOLIO_PATH, WEIGHT_SUM_EPSILON, force_reload=True)
    else:
        result = get_portfolio(TARGET_PORTFOLIO_PATH, WEIGHT_SUM_EPSILON)

    if result.is_valid:
        st.success("Portfolio is valid and ready to use.")
    else:
        st.error("Portfolio has validation errors.")

    if result.errors:
        st.error("\n".join(result.errors))
    if result.warnings:
        st.warning("\n".join(result.warnings))

    st.metric("Total Weight", f"{result.total_weight:.6f}")

    if result.entries:
        entries_df = pd.DataFrame(
            [
                {
                    "Symbol": entry.symbol,
                    "Weight": entry.weight,
                    "Percentage": entry.weight * 100,
                }
                for entry in result.entries
            ]
        )
        st.dataframe(entries_df, use_container_width=True)
    else:
        st.info("No portfolio entries found.")


def render_market_ticker():
    st.subheader("Live Market Ticker")
    api_keys = get_api_keys()
    show_debug = st.checkbox("Show API debug info")

    refresh = st.button("Refresh market data")
    state_key = "market_data_debug" if show_debug else "market_data"
    if refresh or state_key not in st.session_state:
        if show_debug:
            data, debug_messages = get_market_data(api_keys, show_debug=True)
            st.session_state["market_data_debug"] = (data, debug_messages)
        else:
            data = get_market_data(api_keys, show_debug=False)
            st.session_state["market_data"] = (data, [])

    data, debug_messages = st.session_state.get(state_key, ({}, []))
    cols = st.columns(6)
    for idx, (name, info) in enumerate(data.items()):
        with cols[idx % 6]:
            if info.get("available"):
                st.metric(
                    name,
                    f"{info['price']:,.2f}",
                    f"{info['change']:+.2f}%",
                )
            elif info.get("cached"):
                st.metric(name, f"{info['price']:,.2f}", "cached")
            else:
                st.metric(name, "--", "unavailable")

    if show_debug and debug_messages:
        with st.expander("API Debug Log", expanded=False):
            for msg in debug_messages:
                st.write(msg)


def render_stock_research():
    st.subheader("Stock Research & Analysis")
    api_keys = get_api_keys()

    with st.form("stock_research_form"):
        symbol = st.text_input("Symbol", value="AAPL").upper().strip()
        period = st.selectbox("Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3)
        submitted = st.form_submit_button("Run Analysis")

    if not submitted:
        return

    history, info, source = get_stock_data_multi_source(symbol, period, api_keys)
    if history is None or history.empty:
        st.warning("No price data available.")
        return

    rsi_series = calculate_rsi(history["Close"])
    macd_series, signal_series, _ = calculate_macd(history["Close"])
    upper_band, middle_band, lower_band = calculate_bollinger_bands(history["Close"])

    fig = build_stock_analysis_chart(
        price_history=history,
        rsi_series=rsi_series,
        macd_series=macd_series,
        signal_series=signal_series,
        upper_band=upper_band,
        middle_band=middle_band,
        lower_band=lower_band,
    )
    st.plotly_chart(fig, use_container_width=True)
    if source:
        st.caption(f"Data source: {source}")

    prediction, score, signals = predict_stock_movement(history)
    st.write(f"Prediction: **{prediction}** ({score:+.1f})")
    if signals:
        st.write("Signals:")
        st.write("\n".join(signals))

    news_items, news_source = get_stock_news_multi_source(symbol, api_keys)
    if news_items:
        st.subheader("News & Sentiment")
        if news_source:
            st.caption(f"News source: {news_source}")
        for item in news_items:
            sentiment_label, sentiment_score = analyze_sentiment(
                f"{item.get('title', '')} {item.get('summary', '')}"
            )
            st.markdown(f"**{item.get('title', 'No title')}**")
            st.caption(f"{item.get('publisher', 'Unknown')} · {sentiment_label}")
            if item.get("summary"):
                st.write(item["summary"])
            st.write(item.get("link", "#"))
            st.divider()

    fundamentals = get_fundamental_data(symbol, api_keys)
    if fundamentals:
        st.subheader("Fundamentals Snapshot")
        metrics = {
            "Market Cap": fundamentals["valuation"].get("market_cap"),
            "P/E (Trailing)": fundamentals["valuation"].get("pe_ratio"),
            "Forward P/E": fundamentals["valuation"].get("forward_pe"),
            "Profit Margin": fundamentals["profitability"].get("profit_margin"),
            "ROE": fundamentals["profitability"].get("return_on_equity"),
            "Debt/Equity": fundamentals["financial_health"].get("debt_to_equity"),
            "Revenue Growth": fundamentals["growth"].get("revenue_growth"),
            "Dividend Yield": fundamentals["dividends"].get("dividend_yield"),
        }
        st.dataframe(pd.DataFrame(metrics.items(), columns=["Metric", "Value"]))


@st.cache_data(ttl=300, show_spinner=False)
def cached_expirations(tickers):
    return common_expirations(list(tickers))


def render_portfolio_calculator():
    st.subheader("Portfolio Calculator")
    st.caption("What the options market implies about your portfolio's value on a future date.")

    with st.expander("Settings"):
        col1, col2, col3 = st.columns(3)
        free_capital = col1.number_input("Free cash ($)", 0.0, 1_000_000_000.0, 100.0, 100.0, help="Uninvested cash, added to the portfolio as is.")
        risk_free_rate = col1.number_input("Risk-free rate (%)", 0.0, 15.0, 4.0, 0.25) / 100
        var_confidence = col2.selectbox("Value-at-risk confidence", [90, 95, 99], index=1, format_func=lambda c: f"{c}%")
        percentiles = col2.multiselect("Percentiles to list", [1, 5, 10, 25, 50, 75, 90, 95, 99], default=[5, 25, 50, 75, 95])
        min_volume = col3.number_input("Minimum option volume", 1, 1000, 20)
        max_spread_ratio = col3.number_input("Max bid/ask spread (share of ask)", 0.01, 1.0, 0.2, 0.01)

    st.markdown("**1 · Portfolio**")
    table_col, import_col = st.columns([3, 2], gap="large")
    columns = ["Stocks", "Value", "Unleveraged Value"]
    base = pd.DataFrame([["AAPL", 2000.0, 1000.0], ["MSFT", 3000.0, 1500.0]], columns=columns)
    with import_col:
        uploaded_file = st.file_uploader("Or import an Excel file", type=["xlsx"], help="Columns: Stocks, Value and optionally Unleveraged Value.")
        if uploaded_file is not None:
            uploaded = pd.read_excel(uploaded_file)
            if {"Stocks", "Value"} <= set(uploaded.columns):
                base = uploaded.reindex(columns=columns)
            else:
                st.error("The file needs 'Stocks' and 'Value' columns.")
        st.caption(
            "**Position** is the full size of the holding, borrowed money included. **Your capital** is the part that is "
            "your own money. With \\$1,000 and 2× leverage: position \\$2,000, your capital \\$1,000. Leave it empty if you use no leverage."
        )
    with table_col:
        edited = st.data_editor(
            base,
            num_rows="dynamic",
            hide_index=True,
            width="stretch",
            column_config={
                "Stocks": st.column_config.TextColumn("Ticker", max_chars=10),
                "Value": st.column_config.NumberColumn("Position ($)", min_value=0, format="dollar"),
                "Unleveraged Value": st.column_config.NumberColumn("Your capital ($)", min_value=0, format="dollar"),
            },
        )
    portfolio = normalize_portfolio_dataframe(edited)
    if portfolio is None:
        st.info("Add at least one ticker with a position above zero.")
        return

    st.markdown("**2 · Expiry date**")
    with st.spinner("Loading option expiry dates…"):
        expirations, no_options = cached_expirations(tuple(portfolio["Stocks"]))
    if no_options:
        st.warning(f"No listed options found for {', '.join(no_options)}. Left out of the analysis.")
        portfolio = portfolio[~portfolio["Stocks"].isin(no_options)]
    if not expirations:
        st.error("These tickers share no option expiry date. Remove the odd one out and try again.")
        return

    today_ny = pd.Timestamp.now(tz="America/New_York").strftime("%Y-%m-%d")
    pick_col, button_col = st.columns([3, 1], vertical_alignment="bottom")
    expiry = pick_col.selectbox(
        "The distribution is calculated for this date",
        expirations,
        label_visibility="collapsed",
        format_func=lambda d: f"{d}  ·  {(pd.Timestamp(d) - pd.Timestamp(today_ny)).days} days away",
    )
    inputs = (portfolio.to_json(), expiry, risk_free_rate, min_volume, max_spread_ratio)
    if button_col.button("Calculate", type="primary", width="stretch"):
        with st.spinner("Reading option chains and simulating…"):
            try:
                result = options_implied_values(portfolio, expiry, risk_free_rate, min_volume, max_spread_ratio)
                st.session_state["calculator_result"] = {**result, "inputs": inputs}
            except Exception as error:
                st.error(f"Could not calculate: {error}")

    # Results live in session state so they survive reruns (downloads, setting changes).
    result = st.session_state.get("calculator_result")
    if not result:
        return

    st.markdown(f"**3 · Portfolio value on {result['expiry']}**")
    if result["inputs"] != inputs:
        st.warning("The portfolio or settings changed after this was calculated. Press Calculate to refresh.")
    if result["skipped"]:
        reasons = "; ".join(f"{ticker} ({why})" for ticker, why in result["skipped"].items())
        st.warning(f"Left out of every figure below: {reasons}.")
    if result["positions"] is None:
        st.error("No position could be analysed. Option quotes are often empty outside US market hours; otherwise loosen the liquidity filters.")
        return

    stats = summarise_distribution(result, free_capital, var_confidence, percentiles)
    today, capital = stats["today"], stats["capital"]
    tiles = st.columns(4)
    tiles[0].metric("Portfolio today", f"${today:,.0f}", border=True)
    tiles[1].metric(
        f"Value at risk ({var_confidence}%)",
        f"${stats['var_loss']:,.0f}",
        delta=f"{-stats['var_loss'] / capital:.1%} of your capital",
        border=True,
        help=f"In {var_confidence} cases out of 100 you lose less than this by the expiry date.",
    )
    tiles[2].metric("Chance of profit", f"{stats['prob_profit']:.0%}", border=True, help="Probability the portfolio is worth more than today.")
    tiles[3].metric(
        "Market-implied mean",
        f"${stats['mean']:,.0f}",
        delta=f"{(stats['mean'] - today) / capital:+.2%} on your capital",
        delta_color="off",
        border=True,
        help="Option prices are risk-neutral: by construction the mean is roughly today's value grown at the "
        "risk-free rate. It is not a forecast of your return. Read this page for the spread of outcomes.",
    )
    leveraged = stats["leverage"] > 1.01
    if leveraged:
        st.caption(f"{stats['leverage']:.2f}× leverage: \\${capital:,.0f} of your own capital controls \\${today:,.0f}.")

    figure = build_distribution_chart(stats["values"], today, stats["var_value"], var_confidence, capital)
    st.plotly_chart(figure, theme=None, width="stretch")

    table = pd.DataFrame(stats["percentiles"], columns=["percentile", "value", "return_on_capital", "return_on_position"])
    table["percentile"] = table["percentile"].map(lambda level: f"{level}th")
    if not leveraged:
        table = table.drop(columns="return_on_position")
    percentile_tab, download_tab, method_tab = st.tabs(["Percentiles", "Download", "Method and limits"])
    with percentile_tab:
        st.dataframe(
            table,
            hide_index=True,
            width="stretch",
            column_config={
                "percentile": "Percentile",
                "value": st.column_config.NumberColumn("Portfolio value", format="dollar"),
                "return_on_capital": st.column_config.NumberColumn("Return on your capital", format="%+.2f%%"),
                "return_on_position": st.column_config.NumberColumn("Return on position", format="%+.2f%%"),
            },
        )
    with download_tab:
        values = stats["values"]
        summary = pd.DataFrame(
            {
                "Metric": ["Portfolio today", "Your capital", "Leverage", "Market-implied mean", f"VaR {var_confidence}% value",
                           f"VaR {var_confidence}% loss", "Chance of profit (%)"] + [f"Percentile {row['percentile']}th" for row in stats["percentiles"]],
                "Value": [today, capital, stats["leverage"], stats["mean"], stats["var_value"], stats["var_loss"],
                          stats["prob_profit"] * 100] + [row["value"] for row in stats["percentiles"]],
            }
        )
        grid = np.linspace(*np.percentile(values, [0.1, 99.9]), 300)
        distribution = pd.DataFrame(
            {
                "Portfolio_Value": grid,
                "CDF": np.searchsorted(np.sort(values), grid) / len(values),
                "Return_On_Capital_Percent": (grid - today) / capital * 100,
                "Return_On_Position_Percent": (grid / today - 1) * 100,
            }
        )
        left, right = st.columns(2)
        left.download_button("Summary (CSV)", summary.to_csv(index=False), f"summary_{result['expiry']}.csv", "text/csv", width="stretch")
        right.download_button("Full distribution (CSV)", distribution.to_csv(index=False), f"distribution_{result['expiry']}.csv", "text/csv", width="stretch")
    with method_tab:
        st.markdown(
            "- **Per stock**: the distribution of each price on the expiry date is read from the prices of its liquid "
            "out-of-the-money calls and puts (Breeden-Litzenberger). Beyond the last liquid strike the tail keeps the "
            "probability the quotes imply and takes a lognormal shape.\n"
            "- **Together**: stocks move together according to the correlation of their daily returns over the past year. "
            "Correlations rise in a crash, so extreme losses are likely somewhat understated.\n"
            "- **Risk-neutral**: these are the probabilities option prices imply, not a forecast. They tell you how wide the "
            "market thinks outcomes are, not whether the portfolio beats cash.\n"
            "- **Not included**: borrowing costs on leverage, margin calls before expiry, and trading costs. Yahoo quotes are delayed by about 15 minutes."
        )


def render_news(title, fetch_fn):
    st.subheader(title)
    refresh = st.button(f"Refresh {title}")
    state_key = f"{title}_news"
    if refresh or state_key not in st.session_state:
        st.session_state[state_key] = fetch_fn()

    news_items = st.session_state.get(state_key, [])
    if not news_items:
        st.info("No news available.")
        return
    for item in news_items:
        st.markdown(f"**{item.get('title', '')}**")
        st.caption(item.get("source", item.get("publisher", "")))
        if item.get("summary"):
            st.write(item["summary"])
        st.write(item.get("link", ""))
        st.divider()


def render_insurance_tracker():
    st.subheader("Insurance Industry Tracker")
    market_rates = get_market_rates_data()
    company_data = get_insurance_company_data()

    st.write(f"Last updated: {market_rates['last_updated']}")
    st.metric("Global Composite", f"{market_rates['global_composite']['rate']:+.1f}%")

    lines_df = pd.DataFrame(
        [
            {
                "Line": line.replace("_", " ").title(),
                "Global": info["global_rate"],
                "UK": info["uk_rate"],
                "US": info["us_rate"],
                "Trend": info["trend"],
                "Market": info["market_condition"],
            }
            for line, info in market_rates["lines_of_business"].items()
        ]
    )
    st.dataframe(lines_df, use_container_width=True)

    st.subheader("UK Insurance Companies")
    st.dataframe(pd.DataFrame(company_data["uk_insurers"]), use_container_width=True)

    st.subheader("Global Insurance Brokers")
    st.dataframe(pd.DataFrame(company_data["global_brokers"]), use_container_width=True)


def render_command_center():
    st.header("Financial Command Center")
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        [
            "Live Market Ticker",
            "Stock Research",
            "Portfolio Calculator",
            "Economic News",
            "Insurance News",
            "Insurance Tracker",
        ]
    )
    with tab1:
        render_market_ticker()
    with tab2:
        render_stock_research()
    with tab3:
        render_portfolio_calculator()
    with tab4:
        render_news("Economic News", get_economic_news)
    with tab5:
        render_news("Insurance News", get_insurance_news)
    with tab6:
        render_insurance_tracker()


def main():
    ensure_db()
    show_logo()
    st.sidebar.title("Placing Jade Terminal")
    page = st.sidebar.radio("Navigation", ["Connect", "Dashboard", "Strategy", "Command Center"])

    if page == "Connect":
        render_connect()
    elif page == "Dashboard":
        render_dashboard()
    elif page == "Strategy":
        render_strategy()
    else:
        render_command_center()


if __name__ == "__main__":
    main()
