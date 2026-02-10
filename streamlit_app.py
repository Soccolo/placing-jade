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
    build_stock_analysis_chart,
)
from app.financial_command_center.constants import get_api_keys
from app.financial_command_center.logic import (
    analyze_sentiment,
    calculate_bollinger_bands,
    calculate_macd,
    calculate_percentile,
    calculate_probability_below,
    calculate_rsi,
    convolve_pdfs,
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
    ticker_prediction,
    intersection_of_lists,
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


def render_portfolio_calculator():
    st.subheader("Daily Profitability Calculator")

    with st.expander("Model Parameters", expanded=True):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            risk_free_rate = st.number_input("Risk-Free Rate", 0.0, 1.0, 0.04, 0.001)
        with col2:
            min_volume = st.number_input("Minimum Volume", 1, 1000, 20, 1)
            max_spread_ratio = st.number_input("Max Spread Ratio", 0.01, 1.0, 0.2, 0.01)
        with col3:
            free_capital = st.number_input("Free Capital ($)", 0.0, 1_000_000.0, 100.0, 100.0)
        with col4:
            var_confidence = st.selectbox("VaR Confidence", [90, 95, 99], index=1)
            show_percentiles = st.multiselect(
                "Show Percentiles",
                options=[1, 5, 10, 25, 50, 75, 90, 95, 99],
                default=[5, 25, 50, 75, 95],
            )

    show_unleveraged = st.checkbox("Show unleveraged metrics", value=True)
    input_method = st.radio("Input method", ["Manual Entry", "Upload Excel"])

    stock_list = None
    if input_method == "Upload Excel":
        uploaded_file = st.file_uploader("Upload Excel", type=["xlsx", "xls"])
        if uploaded_file is not None:
            stock_list = pd.read_excel(uploaded_file)
            if "Unleveraged Value" not in stock_list.columns:
                stock_list["Unleveraged Value"] = stock_list["Value"]
            st.dataframe(stock_list, use_container_width=True)
    else:
        default_df = pd.DataFrame(
            {"Stocks": ["AAPL", "MSFT"], "Value": [2000.0, 3000.0], "Unleveraged Value": [1000.0, 1500.0]}
        )
        edited_df = st.data_editor(default_df, num_rows="dynamic")
        stock_list = edited_df

    if stock_list is None or stock_list.empty:
        st.info("Provide a portfolio to continue.")
        return

    if "Stocks" not in stock_list.columns or "Value" not in stock_list.columns:
        st.error("Portfolio must include Stocks and Value columns.")
        return

    if "Unleveraged Value" not in stock_list.columns:
        stock_list["Unleveraged Value"] = stock_list["Value"]

    tickers = [str(t).strip().upper() for t in stock_list["Stocks"].tolist()]
    expiration_lists = []
    for ticker_symbol in tickers:
        try:
            expiration_lists.append(yf.Ticker(ticker_symbol).options)
        except Exception:
            expiration_lists.append([])

    possible_expirations = sorted(intersection_of_lists(expiration_lists))
    if possible_expirations and (
        pd.to_datetime(possible_expirations[0]) - pd.Timestamp.today()
    ).days <= 0:
        possible_expirations = possible_expirations[1:]

    if not possible_expirations:
        st.error("No valid expiration dates found for tickers.")
        return

    selected_expiration = st.selectbox("Expiration Date", options=possible_expirations)

    if not st.button("Calculate Distribution"):
        return

    investment_grid_list = []
    pdf_smooth_list = []
    failed_tickers = []
    progress = st.progress(0)

    for idx, ticker_symbol in enumerate(tickers):
        try:
            investment_grid, pdf_smooth = ticker_prediction(
                idx,
                stock_list,
                possible_expirations,
                possible_expirations.index(selected_expiration),
                risk_free_rate,
                min_volume,
                max_spread_ratio,
            )
            if investment_grid is not None:
                investment_grid_list.append(investment_grid)
                pdf_smooth_list.append(pdf_smooth)
            else:
                failed_tickers.append(ticker_symbol)
        except Exception:
            failed_tickers.append(ticker_symbol)
        progress.progress((idx + 1) / len(tickers))

    if not investment_grid_list:
        st.error("Could not process any tickers successfully.")
        return

    if len(investment_grid_list) == 1:
        investment_values = np.array(investment_grid_list[0]) + free_capital
        pdf_values = np.array(pdf_smooth_list[0])
    else:
        investment_values, pdf_values = convolve_pdfs(investment_grid_list, pdf_smooth_list)
        investment_values = investment_values + free_capital

    expected_value = float(np.trapz(investment_values * pdf_values, investment_values))
    current_value = float(stock_list["Value"].sum() + free_capital)
    unleveraged_capital = float(stock_list["Unleveraged Value"].sum() + free_capital)
    expected_gain = expected_value - current_value
    expected_return = expected_gain / current_value * 100 if current_value else 0
    expected_return_unleveraged = (
        expected_gain / unleveraged_capital * 100 if unleveraged_capital else 0
    )
    leverage_ratio = current_value / unleveraged_capital if unleveraged_capital else 1
    var_percentile = 100 - var_confidence
    var_value = calculate_percentile(investment_values, pdf_values, var_percentile)
    var_loss = current_value - var_value
    prob_loss = calculate_probability_below(investment_values, pdf_values, current_value)
    prob_profit = 1 - prob_loss

    fig = build_distribution_chart(
        investment_values, pdf_values, current_value, expected_value, var_value, var_confidence
    )
    st.plotly_chart(fig, use_container_width=True)

    st.metric("Expected Value", f"${expected_value:,.2f}")
    st.metric("Expected Return", f"{expected_return:+.2f}%")
    st.metric(f"VaR ({var_confidence}%)", f"${var_loss:,.2f}")
    st.metric("Probability of Profit", f"{prob_profit*100:.1f}%")

    if show_unleveraged:
        st.metric("Unleveraged Return", f"{expected_return_unleveraged:+.2f}%")
        st.metric("Leverage Ratio", f"{leverage_ratio:.2f}x")

    if show_percentiles:
        percentile_values = {
            p: calculate_percentile(investment_values, pdf_values, p)
            for p in show_percentiles
        }
        st.dataframe(
            pd.DataFrame(
                [{"Percentile": p, "Value": v} for p, v in percentile_values.items()]
            )
        )

    if failed_tickers:
        st.warning(f"Skipped tickers: {', '.join(failed_tickers)}")


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
