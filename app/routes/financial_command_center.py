from __future__ import annotations

import json
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.io as plotly_io
import yfinance as yf
from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates

from app.financial_command_center.charts import (
    build_distribution_chart,
    build_stock_analysis_chart,
)
from app.financial_command_center.constants import get_api_keys
from app.financial_command_center.portfolio_inputs import (
    load_portfolio_from_upload,
    normalize_portfolio_dataframe,
    parse_portfolio_text,
)
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
    ticker_prediction,
    intersection_of_lists,
)

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")


def build_fcc_context(request: Request, active_tab: str) -> Dict[str, object]:
    return {
        "request": request,
        "active_tab": active_tab,
    }



@router.get("/command-center")
def command_center_home(request: Request):
    return RedirectResponse(url="/command-center/market-ticker", status_code=302)


@router.get("/command-center/market-ticker")
def market_ticker(request: Request, debug: Optional[bool] = False):
    context = build_fcc_context(request, active_tab="market")
    api_keys = get_api_keys()

    if debug:
        market_data, debug_messages = get_market_data(api_keys, show_debug=True)
    else:
        market_data = get_market_data(api_keys, show_debug=False)
        debug_messages = []

    context.update(
        {
            "market_data": market_data,
            "debug_messages": debug_messages,
            "debug_enabled": bool(debug),
        }
    )
    return templates.TemplateResponse("fcc/market_ticker.html", context)


@router.get("/command-center/stock-research")
def stock_research(
    request: Request, symbol: str = "AAPL", period: str = "1y"
):
    context = build_fcc_context(request, active_tab="research")
    api_keys = get_api_keys()

    query_symbol = symbol.strip().upper() if symbol else "AAPL"
    query_period = period or "1y"

    history, info, source = get_stock_data_multi_source(query_symbol, query_period, api_keys)

    chart_html = None
    prediction = None
    prediction_score = None
    prediction_signals = []
    rsi_latest = None
    macd_latest = None
    signal_latest = None

    if history is not None and not history.empty:
        rsi_series = calculate_rsi(history["Close"])
        macd_series, signal_series, _ = calculate_macd(history["Close"])
        upper_band, middle_band, lower_band = calculate_bollinger_bands(history["Close"])

        rsi_latest = rsi_series.iloc[-1] if not rsi_series.empty else None
        macd_latest = macd_series.iloc[-1] if not macd_series.empty else None
        signal_latest = signal_series.iloc[-1] if not signal_series.empty else None

        figure = build_stock_analysis_chart(
            price_history=history,
            rsi_series=rsi_series,
            macd_series=macd_series,
            signal_series=signal_series,
            upper_band=upper_band,
            middle_band=middle_band,
            lower_band=lower_band,
        )
        chart_html = plotly_io.to_html(
            figure, full_html=False, include_plotlyjs="cdn"
        )

        prediction, prediction_score, prediction_signals = predict_stock_movement(history)

    news_items, news_source = get_stock_news_multi_source(query_symbol, api_keys)
    news_with_sentiment = []
    for item in news_items:
        sentiment_label, sentiment_score = analyze_sentiment(
            f"{item.get('title', '')} {item.get('summary', '')}"
        )
        news_with_sentiment.append(
            {
                **item,
                "sentiment_label": sentiment_label,
                "sentiment_score": sentiment_score,
            }
        )

    fundamentals = get_fundamental_data(query_symbol, api_keys)

    context.update(
        {
            "symbol": query_symbol,
            "period": query_period,
            "chart_html": chart_html,
            "prediction": prediction,
            "prediction_score": prediction_score,
            "prediction_signals": prediction_signals,
            "news_items": news_with_sentiment,
            "news_source": news_source,
            "fundamentals": fundamentals,
            "data_source": source,
            "rsi_latest": rsi_latest,
            "macd_latest": macd_latest,
            "signal_latest": signal_latest,
            "info": info or {},
        }
    )
    return templates.TemplateResponse("fcc/stock_research.html", context)


@router.get("/command-center/portfolio-calculator")
def portfolio_calculator(request: Request):
    context = build_fcc_context(request, active_tab="portfolio")
    context.update(
        {
            "portfolio_text": "",
            "input_method": "manual",
            "risk_free_rate": 0.04,
            "min_volume": 20,
            "max_spread_ratio": 0.2,
            "free_capital": 100.0,
            "var_confidence": 95,
            "show_percentiles": [5, 25, 50, 75, 95],
            "show_unleveraged": True,
            "results": None,
            "errors": [],
            "expirations": [],
            "selected_expiration": "",
        }
    )
    return templates.TemplateResponse("fcc/portfolio_calculator.html", context)


@router.post("/command-center/portfolio-calculator")
async def portfolio_calculator_run(
    request: Request,
    input_method: str = Form("manual"),
    portfolio_text: str = Form(""),
    uploaded_file: UploadFile = File(None),
    risk_free_rate: float = Form(0.04),
    min_volume: int = Form(20),
    max_spread_ratio: float = Form(0.2),
    free_capital: float = Form(100.0),
    var_confidence: int = Form(95),
    show_percentiles: Optional[List[int]] = Form(default=None),
    show_unleveraged: Optional[bool] = Form(False),
    selected_expiration: str = Form(""),
    portfolio_payload: str = Form(""),
):
    context = build_fcc_context(request, active_tab="portfolio")
    errors: List[str] = []

    if show_percentiles is None:
        show_percentiles = []

    percentiles = []
    for percentile in show_percentiles:
        try:
            percentiles.append(int(percentile))
        except Exception:
            continue

    dataframe = None
    has_upload = uploaded_file is not None and bool(uploaded_file.filename)
    if portfolio_payload and not portfolio_text and not has_upload:
        try:
            data = json.loads(portfolio_payload)
            dataframe = pd.DataFrame(data)
        except Exception:
            dataframe = None

    if dataframe is None:
        if input_method == "upload":
            dataframe = load_portfolio_from_upload(uploaded_file)
            if dataframe is None:
                errors.append("Uploaded file must include Stocks and Value columns.")
        else:
            dataframe = parse_portfolio_text(portfolio_text)
            if dataframe is None:
                errors.append("Enter portfolio rows as TICKER,VALUE,UNLEVERAGED.")

    dataframe = normalize_portfolio_dataframe(dataframe)

    expirations: List[str] = []
    if dataframe is not None and errors == []:
        tickers = dataframe["Stocks"].tolist()
        expiration_lists = []
        for ticker_symbol in tickers:
            try:
                ticker_data = yf.Ticker(ticker_symbol)
                expiration_lists.append(ticker_data.options)
            except Exception:
                expiration_lists.append([])
        expirations = sorted(intersection_of_lists(expiration_lists))
        if expirations and (pd.to_datetime(expirations[0]) - pd.Timestamp.today()).days <= 0:
            expirations = expirations[1:]

        if not expirations:
            errors.append("No valid option expiration dates found for tickers.")

    results = None
    distribution_chart = None

    if dataframe is not None and errors == [] and selected_expiration:
        tickers = dataframe["Stocks"].tolist()
        try:
            expiration_index = expirations.index(selected_expiration)
        except ValueError:
            expiration_index = 0

        investment_grid_list = []
        pdf_smooth_list = []
        failed_tickers = []

        for ticker_index, ticker_symbol in enumerate(tickers):
            try:
                investment_grid, pdf_smooth = ticker_prediction(
                    ticker_index,
                    dataframe,
                    expirations,
                    expiration_index,
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

        if not investment_grid_list:
            errors.append("Could not process any tickers successfully.")
        else:
            if len(investment_grid_list) == 1:
                investment_values = np.array(investment_grid_list[0]) + free_capital
                pdf_values = np.array(pdf_smooth_list[0])
            else:
                investment_values, pdf_values = convolve_pdfs(
                    investment_grid_list, pdf_smooth_list
                )
                investment_values = investment_values + free_capital

            expected_value = float(
                np.trapz(investment_values * pdf_values, investment_values)
            )
            current_value_leveraged = float(dataframe["Value"].sum() + free_capital)
            unleveraged_capital = float(
                dataframe["Unleveraged Value"].sum() + free_capital
            )
            expected_gain = expected_value - current_value_leveraged
            expected_return_leveraged = (
                expected_gain / current_value_leveraged * 100
                if current_value_leveraged > 0
                else 0
            )
            expected_return_unleveraged = (
                expected_gain / unleveraged_capital * 100 if unleveraged_capital > 0 else 0
            )
            leverage_ratio = (
                current_value_leveraged / unleveraged_capital
                if unleveraged_capital > 0
                else 1
            )

            var_percentile = 100 - var_confidence
            var_value = calculate_percentile(
                investment_values, pdf_values, var_percentile
            )
            var_loss = current_value_leveraged - var_value
            var_loss_pct_leveraged = (
                var_loss / current_value_leveraged * 100
                if current_value_leveraged > 0
                else 0
            )
            var_loss_pct_unleveraged = (
                var_loss / unleveraged_capital * 100 if unleveraged_capital > 0 else 0
            )

            percentile_values = {
                percentile: calculate_percentile(investment_values, pdf_values, percentile)
                for percentile in percentiles
            }

            prob_loss = calculate_probability_below(
                investment_values, pdf_values, current_value_leveraged
            )
            prob_profit = 1 - prob_loss

            figure = build_distribution_chart(
                investment_values,
                pdf_values,
                current_value_leveraged,
                expected_value,
                var_value,
                var_confidence,
            )
            distribution_chart = plotly_io.to_html(
                figure, full_html=False, include_plotlyjs="cdn"
            )

            results = {
                "expected_value": expected_value,
                "current_value": current_value_leveraged,
                "expected_return_leveraged": expected_return_leveraged,
                "expected_return_unleveraged": expected_return_unleveraged,
                "leverage_ratio": leverage_ratio,
                "var_value": var_value,
                "var_loss": var_loss,
                "var_loss_pct_leveraged": var_loss_pct_leveraged,
                "var_loss_pct_unleveraged": var_loss_pct_unleveraged,
                "prob_profit": prob_profit,
                "prob_loss": prob_loss,
                "percentiles": percentile_values,
                "failed_tickers": failed_tickers,
            }

    portfolio_payload_value = (
        dataframe.to_dict(orient="records") if dataframe is not None else []
    )

    context.update(
        {
            "input_method": input_method,
            "portfolio_text": portfolio_text,
            "risk_free_rate": risk_free_rate,
            "min_volume": min_volume,
            "max_spread_ratio": max_spread_ratio,
            "free_capital": free_capital,
            "var_confidence": var_confidence,
            "show_percentiles": percentiles,
            "show_unleveraged": bool(show_unleveraged),
            "selected_expiration": selected_expiration,
            "expirations": expirations,
            "portfolio_payload": json.dumps(portfolio_payload_value),
            "results": results,
            "distribution_chart": distribution_chart,
            "errors": errors,
        }
    )
    return templates.TemplateResponse("fcc/portfolio_calculator.html", context)


@router.get("/command-center/economic-news")
def economic_news(request: Request):
    context = build_fcc_context(request, active_tab="economic-news")
    news_items = get_economic_news()
    context.update({"news_items": news_items})
    return templates.TemplateResponse("fcc/economic_news.html", context)


@router.get("/command-center/insurance-news")
def insurance_news(request: Request):
    from urllib.parse import urlparse
    
    context = build_fcc_context(request, active_tab="insurance-news")
    news_items_raw = get_insurance_news()
    
    # Validate and sanitize URLs - block dangerous schemes
    dangerous_schemes = {"javascript", "data", "vbscript", "file", "about"}
    news_items = []
    for item in news_items_raw:
        # Create a copy of the item
        validated_item = item.copy()
        # Validate the link - set to '#' if unsafe
        if "link" in item and item["link"]:
            try:
                parsed = urlparse(item["link"])
                # Block dangerous schemes
                if parsed.scheme and parsed.scheme.lower() in dangerous_schemes:
                    validated_item["link"] = "#"
                # Only allow http/https for external links
                elif parsed.scheme and parsed.scheme.lower() not in {"http", "https"}:
                    validated_item["link"] = "#"
            except Exception:
                validated_item["link"] = "#"
        else:
            validated_item["link"] = "#"
        news_items.append(validated_item)
    
    context.update({"news_items": news_items})
    return templates.TemplateResponse("fcc/insurance_news.html", context)


@router.get("/command-center/insurance-tracker")
def insurance_tracker(request: Request):
    context = build_fcc_context(request, active_tab="insurance-tracker")
    market_rates = get_market_rates_data()
    company_data = get_insurance_company_data()
    context.update(
        {
            "market_rates": market_rates,
            "company_data": company_data,
        }
    )
    return templates.TemplateResponse("fcc/insurance_tracker.html", context)
