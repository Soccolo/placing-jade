import logging
import time
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import feedparser
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from scipy import interpolate
from scipy.interpolate import BSpline, splrep
from scipy.optimize import brentq
from scipy.stats import gaussian_kde, norm
from textblob import TextBlob

warnings.filterwarnings("ignore")
logging.getLogger("yfinance").setLevel(logging.CRITICAL)
logging.getLogger("urllib3").setLevel(logging.CRITICAL)
logging.getLogger("requests").setLevel(logging.CRITICAL)


def safe_get_column(dataframe: pd.DataFrame, column_name: str) -> Optional[pd.Series]:
    if dataframe is None or len(dataframe) == 0:
        return None
    try:
        if isinstance(dataframe.columns, pd.MultiIndex):
            if column_name in dataframe.columns.get_level_values(0):
                selected = dataframe[column_name]
                return selected.iloc[:, 0] if isinstance(selected, pd.DataFrame) else selected
            for column_tuple in dataframe.columns:
                if column_tuple[0] == column_name or column_tuple[1] == column_name:
                    return dataframe[column_tuple]
        if column_name in dataframe.columns:
            selected = dataframe[column_name]
            return selected.iloc[:, 0] if isinstance(selected, pd.DataFrame) else selected
    except Exception:
        return None
    return None


def normalize_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    if dataframe is None or len(dataframe) == 0:
        return dataframe
    try:
        if isinstance(dataframe.columns, pd.MultiIndex):
            normalized = pd.DataFrame(index=dataframe.index)
            for column_name in ["Open", "High", "Low", "Close", "Volume"]:
                column_data = safe_get_column(dataframe, column_name)
                if column_data is not None:
                    normalized[column_name] = column_data
            return normalized
        return dataframe
    except Exception:
        return dataframe


def get_finnhub_quote(symbol: str, api_key: str) -> Optional[Dict[str, float]]:
    if not api_key:
        return None
    try:
        symbol_map = {
            "GC=F": "GLD",
            "SI=F": "SLV",
            "HG=F": "CPER",
            "^IXIC": "QQQ",
            "^DJI": "DIA",
        }
        finnhub_symbol = symbol_map.get(symbol, symbol)
        if finnhub_symbol is None:
            return None
        url = f"https://finnhub.io/api/v1/quote?symbol={finnhub_symbol}&token={api_key}"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data and data.get("c") and data.get("c") > 0:
                return {
                    "price": float(data["c"]),
                    "change": float(data.get("dp", 0)),
                    "prev_close": float(data.get("pc", data["c"])),
                }
    except Exception:
        return None
    return None


def get_twelve_data_quote(symbol: str, api_key: str) -> Optional[Dict[str, float]]:
    if not api_key:
        return None
    try:
        symbol_map = {
            "GC=F": "XAU/USD",
            "SI=F": "XAG/USD",
            "HG=F": None,
            "^IXIC": "IXIC",
            "^DJI": "DJI",
        }
        twelve_symbol = symbol_map.get(symbol, symbol)
        if twelve_symbol is None:
            return None
        url = f"https://api.twelvedata.com/quote?symbol={twelve_symbol}&apikey={api_key}"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data and "close" in data and data.get("close"):
                try:
                    current_price = float(data["close"])
                    previous_price = float(data.get("previous_close", current_price))
                    change_pct = float(data.get("percent_change", 0))
                    if change_pct == 0 and previous_price > 0:
                        change_pct = ((current_price - previous_price) / previous_price) * 100
                    return {
                        "price": current_price,
                        "change": change_pct,
                        "prev_close": previous_price,
                    }
                except (ValueError, TypeError):
                    return None
    except Exception:
        return None
    return None


def get_alpha_vantage_quote(symbol: str, api_key: str) -> Optional[Dict[str, float]]:
    if not api_key:
        return None
    try:
        if symbol in ["GC=F", "SI=F", "HG=F"]:
            return None

        clean_symbol = symbol.replace("^", "").replace("=F", "")
        if symbol == "^IXIC":
            clean_symbol = "QQQ"
        elif symbol == "^DJI":
            clean_symbol = "DIA"

        url = (
            "https://www.alphavantage.co/query?function=GLOBAL_QUOTE"
            f"&symbol={clean_symbol}&apikey={api_key}"
        )
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            quote = data.get("Global Quote", {})
            if quote and quote.get("05. price"):
                try:
                    price = float(quote["05. price"])
                    change_str = quote.get("10. change percent", "0%").replace("%", "")
                    change_pct = float(change_str)
                    previous_price = float(quote.get("08. previous close", price))
                    return {
                        "price": price,
                        "change": change_pct,
                        "prev_close": previous_price,
                    }
                except (ValueError, TypeError):
                    return None
    except Exception:
        return None
    return None


def get_fmp_quote(symbol: str, api_key: str) -> Optional[Dict[str, float]]:
    if not api_key:
        return None
    try:
        symbol_map = {
            "GC=F": "GCUSD",
            "SI=F": "SIUSD",
            "HG=F": "HGUSD",
            "^IXIC": "QQQ",
            "^DJI": "DIA",
        }
        fmp_symbol = symbol_map.get(symbol, symbol)
        url = f"https://financialmodelingprep.com/api/v3/quote/{fmp_symbol}?apikey={api_key}"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data and len(data) > 0:
                quote = data[0]
                if quote.get("price"):
                    return {
                        "price": float(quote["price"]),
                        "change": float(quote.get("changesPercentage", 0)),
                        "prev_close": float(quote.get("previousClose", quote["price"])),
                    }
    except Exception:
        return None
    return None


def fetch_from_yahoo_download(symbol: str, period: str = "5d") -> Optional[pd.DataFrame]:
    try:
        import io
        import sys

        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()

        data = yf.download(symbol, period=period, progress=False, auto_adjust=True)

        sys.stdout = old_stdout
        sys.stderr = old_stderr

        if data is not None and len(data) > 0:
            return data
    except Exception:
        return None
    return None


def fetch_from_ticker_history(symbol: str, period: str = "5d") -> Optional[pd.DataFrame]:
    try:
        import io
        import sys

        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()

        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period)

        sys.stdout = old_stdout
        sys.stderr = old_stderr

        if data is not None and len(data) > 0:
            return data
    except Exception:
        return None
    return None


def safe_yf_download(
    symbol: str,
    period: str = "5d",
    max_retries: int = 2,
    delay: float = 1.0,
) -> Optional[pd.DataFrame]:
    for _ in range(max_retries):
        data = fetch_from_yahoo_download(symbol, period)
        if data is not None and len(data) > 0:
            return normalize_dataframe(data)
        time.sleep(delay)

    for _ in range(max_retries):
        data = fetch_from_ticker_history(symbol, period)
        if data is not None and len(data) > 0:
            return normalize_dataframe(data)
        time.sleep(delay)

    return None


def safe_get_info(symbol: str, max_retries: int = 2, delay: float = 1.0) -> Dict:
    import io
    import sys

    for attempt in range(max_retries):
        try:
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            sys.stdout = io.StringIO()
            sys.stderr = io.StringIO()

            ticker = yf.Ticker(symbol)
            info = ticker.info

            sys.stdout = old_stdout
            sys.stderr = old_stderr

            return info
        except Exception:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            if attempt < max_retries - 1:
                time.sleep(delay)
    return {}


def get_multi_source_quote(
    symbol: str,
    api_keys: Dict[str, str],
    debug_info: Optional[Dict[str, List[str]]] = None,
) -> Tuple[Optional[Dict[str, float]], Optional[str]]:
    tried_sources: List[str] = []

    if api_keys.get("finnhub"):
        tried_sources.append("Finnhub")
        quote = get_finnhub_quote(symbol, api_keys["finnhub"])
        if quote and quote.get("price", 0) > 0:
            if debug_info is not None:
                debug_info["tried"] = tried_sources
            return quote, "Finnhub"

    if api_keys.get("twelve_data"):
        tried_sources.append("Twelve Data")
        quote = get_twelve_data_quote(symbol, api_keys["twelve_data"])
        if quote and quote.get("price", 0) > 0:
            if debug_info is not None:
                debug_info["tried"] = tried_sources
            return quote, "Twelve Data"

    if api_keys.get("alpha_vantage"):
        tried_sources.append("Alpha Vantage")
        quote = get_alpha_vantage_quote(symbol, api_keys["alpha_vantage"])
        if quote and quote.get("price", 0) > 0:
            if debug_info is not None:
                debug_info["tried"] = tried_sources
            return quote, "Alpha Vantage"

    if api_keys.get("fmp"):
        tried_sources.append("FMP")
        quote = get_fmp_quote(symbol, api_keys["fmp"])
        if quote and quote.get("price", 0) > 0:
            if debug_info is not None:
                debug_info["tried"] = tried_sources
            return quote, "FMP"

    tried_sources.append("Yahoo")
    history = safe_yf_download(symbol, period="5d", max_retries=1, delay=0.5)
    if history is not None and len(history) >= 1:
        try:
            current_price = float(history["Close"].iloc[-1])
            previous_price = float(history["Close"].iloc[0]) if len(history) > 1 else current_price
            change_pct = ((current_price - previous_price) / previous_price) * 100 if previous_price > 0 else 0
            if debug_info is not None:
                debug_info["tried"] = tried_sources
            return {"price": current_price, "change": change_pct, "prev_close": previous_price}, "Yahoo"
        except Exception:
            return None, None

    if debug_info is not None:
        debug_info["tried"] = tried_sources
    return None, None


def get_cached_market_data() -> Dict[str, Dict[str, object]]:
    return {
        "Gold": {
            "price": 2650.00,
            "change": 0.0,
            "symbol": "GC=F",
            "available": False,
            "cached": True,
            "source": "Cached",
        },
        "Silver": {
            "price": 31.50,
            "change": 0.0,
            "symbol": "SI=F",
            "available": False,
            "cached": True,
            "source": "Cached",
        },
        "Copper": {
            "price": 4.25,
            "change": 0.0,
            "symbol": "HG=F",
            "available": False,
            "cached": True,
            "source": "Cached",
        },
        "SPY": {
            "price": 595.00,
            "change": 0.0,
            "symbol": "SPY",
            "available": False,
            "cached": True,
            "source": "Cached",
        },
        "NASDAQ": {
            "price": 19800.00,
            "change": 0.0,
            "symbol": "^IXIC",
            "available": False,
            "cached": True,
            "source": "Cached",
        },
        "Dow Jones": {
            "price": 42800.00,
            "change": 0.0,
            "symbol": "^DJI",
            "available": False,
            "cached": True,
            "source": "Cached",
        },
    }


def get_market_data(api_keys: Dict[str, str], show_debug: bool = False):
    symbols = {
        "Gold": "GC=F",
        "Silver": "SI=F",
        "Copper": "HG=F",
        "SPY": "SPY",
        "NASDAQ": "^IXIC",
        "Dow Jones": "^DJI",
    }

    market_data: Dict[str, Dict[str, object]] = {}
    debug_messages: List[str] = []

    for name, symbol in symbols.items():
        debug_info: Dict[str, List[str]] = {}
        quote, source = get_multi_source_quote(symbol, api_keys, debug_info)

        if quote and quote.get("price", 0) > 0:
            market_data[name] = {
                "price": quote["price"],
                "change": quote["change"],
                "symbol": symbol,
                "available": True,
                "cached": False,
                "source": source,
            }
            debug_messages.append(f"✅ {name}: Got data from {source}")
        else:
            cached = get_cached_market_data()
            market_data[name] = cached[name]
            tried_sources = debug_info.get("tried", ["None"])
            debug_messages.append(
                f"❌ {name}: Failed (tried: {', '.join(tried_sources)}) - using cached"
            )

        time.sleep(0.2)

    if show_debug:
        return market_data, debug_messages
    return market_data


def get_stock_data_multi_source(
    symbol: str, period: str = "1y", api_keys: Optional[Dict[str, str]] = None
) -> Tuple[Optional[pd.DataFrame], Optional[Dict], Optional[str]]:
    if api_keys is None:
        api_keys = {}

    history = safe_yf_download(symbol, period=period, max_retries=2, delay=1)
    info = safe_get_info(symbol, max_retries=1, delay=1)

    if history is not None and len(history) > 0:
        return history, info, "Yahoo Finance"

    quote, source = get_multi_source_quote(symbol, api_keys)
    if quote:
        today = datetime.now()
        history = pd.DataFrame(
            {
                "Open": [quote["prev_close"]],
                "High": [quote["price"]],
                "Low": [quote["prev_close"]],
                "Close": [quote["price"]],
                "Volume": [0],
            },
            index=[today],
        )
        return history, {}, source

    return None, None, None


def get_stock_news(symbol: str) -> List[Dict[str, str]]:
    import io
    import sys

    try:
        time.sleep(0.3)
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()

        ticker = yf.Ticker(symbol)
        news_items = ticker.news

        sys.stdout = old_stdout
        sys.stderr = old_stderr

        if news_items:
            normalized_news = []
            for item in news_items[:10]:
                if isinstance(item, dict):
                    title = (
                        item.get("title")
                        or item.get("headline")
                        or (
                            item.get("content", {}).get("title")
                            if isinstance(item.get("content"), dict)
                            else None
                        )
                        or "No title"
                    )
                    link = (
                        item.get("link")
                        or item.get("url")
                        or (
                            item.get("content", {})
                            .get("canonicalUrl", {})
                            .get("url")
                            if isinstance(item.get("content"), dict)
                            else None
                        )
                        or "#"
                    )
                    publisher = (
                        item.get("publisher")
                        or item.get("source")
                        or (
                            item.get("content", {})
                            .get("provider", {})
                            .get("displayName")
                            if isinstance(item.get("content"), dict)
                            else None
                        )
                        or "Unknown"
                    )
                    if item.get("content") and isinstance(item.get("content"), dict):
                        content = item["content"]
                        if not title or title == "No title":
                            title = content.get("title", "No title")
                        if link == "#":
                            canonical = content.get("canonicalUrl", {})
                            if isinstance(canonical, dict):
                                link = canonical.get("url", "#")
                            elif isinstance(canonical, str):
                                link = canonical
                        if publisher == "Unknown":
                            provider = content.get("provider", {})
                            if isinstance(provider, dict):
                                publisher = provider.get("displayName", "Unknown")

                    normalized_news.append(
                        {"title": title, "link": link, "publisher": publisher}
                    )
            return normalized_news if normalized_news else []
        return []
    except Exception:
        return []


def get_finnhub_news(symbol: str, api_key: str) -> List[Dict[str, str]]:
    if not api_key:
        return []
    try:
        today = datetime.now().strftime("%Y-%m-%d")
        week_ago = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
        url = (
            "https://finnhub.io/api/v1/company-news"
            f"?symbol={symbol}&from={week_ago}&to={today}&token={api_key}"
        )
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            news_items = response.json()
            if news_items and isinstance(news_items, list):
                return [
                    {
                        "title": item.get("headline", "No title"),
                        "link": item.get("url", "#"),
                        "publisher": item.get("source", "Unknown"),
                        "summary": item.get("summary", ""),
                    }
                    for item in news_items[:10]
                ]
    except Exception:
        return []
    return []


def get_stock_news_multi_source(
    symbol: str, api_keys: Optional[Dict[str, str]] = None
) -> Tuple[List[Dict[str, str]], Optional[str]]:
    if api_keys is None:
        api_keys = {}

    if api_keys.get("finnhub"):
        news_items = get_finnhub_news(symbol, api_keys["finnhub"])
        if news_items and len(news_items) > 0 and news_items[0].get("title") != "No title":
            return news_items, "Finnhub"

    news_items = get_stock_news(symbol)
    if news_items and len(news_items) > 0 and news_items[0].get("title") != "No title":
        return news_items, "Yahoo"

    return [], None


def get_fundamental_data(symbol: str, api_keys: Optional[Dict[str, str]] = None) -> Dict:
    if api_keys is None:
        api_keys = {}

    fundamentals: Dict[str, Dict] = {
        "valuation": {},
        "profitability": {},
        "financial_health": {},
        "growth": {},
        "dividends": {},
        "analyst": {},
        "source": None,
    }

    try:
        import io
        import sys

        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()

        ticker = yf.Ticker(symbol)
        info = ticker.info

        sys.stdout = old_stdout
        sys.stderr = old_stderr

        if info:
            fundamentals["valuation"] = {
                "market_cap": info.get("marketCap"),
                "enterprise_value": info.get("enterpriseValue"),
                "pe_ratio": info.get("trailingPE"),
                "forward_pe": info.get("forwardPE"),
                "peg_ratio": info.get("pegRatio"),
                "price_to_book": info.get("priceToBook"),
                "price_to_sales": info.get("priceToSalesTrailing12Months"),
                "ev_to_revenue": info.get("enterpriseToRevenue"),
                "ev_to_ebitda": info.get("enterpriseToEbitda"),
            }
            fundamentals["profitability"] = {
                "profit_margin": info.get("profitMargins"),
                "operating_margin": info.get("operatingMargins"),
                "gross_margin": info.get("grossMargins"),
                "ebitda_margin": info.get("ebitdaMargins"),
                "return_on_assets": info.get("returnOnAssets"),
                "return_on_equity": info.get("returnOnEquity"),
                "revenue": info.get("totalRevenue"),
                "net_income": info.get("netIncomeToCommon"),
                "eps": info.get("trailingEps"),
                "forward_eps": info.get("forwardEps"),
            }
            fundamentals["financial_health"] = {
                "total_cash": info.get("totalCash"),
                "total_debt": info.get("totalDebt"),
                "debt_to_equity": info.get("debtToEquity"),
                "current_ratio": info.get("currentRatio"),
                "quick_ratio": info.get("quickRatio"),
                "free_cash_flow": info.get("freeCashflow"),
                "operating_cash_flow": info.get("operatingCashflow"),
            }
            fundamentals["growth"] = {
                "revenue_growth": info.get("revenueGrowth"),
                "earnings_growth": info.get("earningsGrowth"),
                "earnings_quarterly_growth": info.get("earningsQuarterlyGrowth"),
                "revenue_per_share": info.get("revenuePerShare"),
                "book_value": info.get("bookValue"),
            }
            fundamentals["dividends"] = {
                "dividend_rate": info.get("dividendRate"),
                "dividend_yield": info.get("dividendYield"),
                "payout_ratio": info.get("payoutRatio"),
                "ex_dividend_date": info.get("exDividendDate"),
                "five_year_avg_dividend_yield": info.get("fiveYearAvgDividendYield"),
            }
            fundamentals["analyst"] = {
                "target_high": info.get("targetHighPrice"),
                "target_low": info.get("targetLowPrice"),
                "target_mean": info.get("targetMeanPrice"),
                "target_median": info.get("targetMedianPrice"),
                "recommendation": info.get("recommendationKey"),
                "recommendation_mean": info.get("recommendationMean"),
                "num_analysts": info.get("numberOfAnalystOpinions"),
            }
            fundamentals["company_info"] = {
                "name": info.get("longName", info.get("shortName", symbol)),
                "sector": info.get("sector"),
                "industry": info.get("industry"),
                "employees": info.get("fullTimeEmployees"),
                "website": info.get("website"),
                "description": info.get("longBusinessSummary", "")[:500]
                if info.get("longBusinessSummary")
                else "",
            }
            fundamentals["source"] = "Yahoo Finance"
    except Exception:
        return fundamentals

    return fundamentals




def analyze_sentiment(text: str) -> Tuple[str, float]:
    try:
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        if polarity > 0.1:
            return "Positive", polarity
        if polarity < -0.1:
            return "Negative", polarity
        return "Neutral", polarity
    except Exception:
        return "Neutral", 0.0


def get_economic_news() -> List[Dict[str, str]]:
    feeds = [
        ("Reuters Business", "https://feeds.reuters.com/reuters/businessNews"),
        ("MarketWatch", "https://feeds.marketwatch.com/marketwatch/topstories/"),
        ("CNBC", "https://www.cnbc.com/id/100003114/device/rss/rss.html"),
    ]

    all_news = []
    for source, url in feeds:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:5]:
                all_news.append(
                    {
                        "source": source,
                        "title": entry.get("title", ""),
                        "link": entry.get("link", ""),
                        "published": entry.get("published", ""),
                        "summary": entry.get("summary", "")[:200] if entry.get("summary") else "",
                    }
                )
        except Exception:
            continue

    return all_news[:15]


def get_insurance_news() -> List[Dict[str, str]]:
    feeds = [
        ("Insurance Journal", "https://www.insurancejournal.com/feed/"),
        ("Insurance News Net", "https://insurancenewsnet.com/feed"),
    ]

    all_news = []
    for source, url in feeds:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:5]:
                all_news.append(
                    {
                        "source": source,
                        "title": entry.get("title", ""),
                        "link": entry.get("link", ""),
                        "published": entry.get("published", ""),
                        "summary": entry.get("summary", "")[:200] if entry.get("summary") else "",
                    }
                )
        except Exception:
            continue

    return all_news[:10]


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    price_delta = prices.diff()
    gain = price_delta.where(price_delta > 0, 0).rolling(window=period).mean()
    loss = (-price_delta.where(price_delta < 0, 0)).rolling(window=period).mean()
    relative_strength = gain / loss
    rsi_values = 100 - (100 / (1 + relative_strength))
    return rsi_values


def calculate_macd(
    prices: pd.Series, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = prices.ewm(span=fast_period, adjust=False).mean()
    ema_slow = prices.ewm(span=slow_period, adjust=False).mean()
    macd_values = ema_fast - ema_slow
    signal_line = macd_values.ewm(span=signal_period, adjust=False).mean()
    histogram = macd_values - signal_line
    return macd_values, signal_line, histogram


def calculate_bollinger_bands(
    prices: pd.Series, period: int = 20, std_dev: int = 2
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    simple_moving_average = prices.rolling(window=period).mean()
    rolling_std = prices.rolling(window=period).std()
    upper_band = simple_moving_average + (rolling_std * std_dev)
    lower_band = simple_moving_average - (rolling_std * std_dev)
    return upper_band, simple_moving_average, lower_band


def predict_stock_movement(hist: pd.DataFrame):
    if hist is None or len(hist) < 30:
        return None, None, None

    close_prices = hist["Close"]

    rsi_value = calculate_rsi(close_prices).iloc[-1]
    macd_series, signal_series, _ = calculate_macd(close_prices)
    macd_current = macd_series.iloc[-1]
    signal_current = signal_series.iloc[-1]

    sma_20 = close_prices.rolling(20).mean().iloc[-1]
    sma_50 = close_prices.rolling(50).mean().iloc[-1] if len(close_prices) >= 50 else sma_20
    sma_200 = close_prices.rolling(200).mean().iloc[-1] if len(close_prices) >= 200 else sma_50
    current_price = close_prices.iloc[-1]

    score = 0.0
    signals: List[str] = []

    if rsi_value < 30:
        score += 2
        signals.append("RSI oversold (bullish)")
    elif rsi_value > 70:
        score -= 2
        signals.append("RSI overbought (bearish)")
    elif rsi_value < 50:
        score += 0.5
        signals.append("RSI below 50 (neutral-bearish)")
    else:
        score -= 0.5
        signals.append("RSI above 50 (neutral-bullish)")

    if macd_current > signal_current:
        score += 1.5
        signals.append("MACD above signal (bullish)")
    else:
        score -= 1.5
        signals.append("MACD below signal (bearish)")

    if current_price > sma_20:
        score += 1
        signals.append("Price above 20-day SMA (bullish)")
    else:
        score -= 1
        signals.append("Price below 20-day SMA (bearish)")

    if current_price > sma_50:
        score += 1
        signals.append("Price above 50-day SMA (bullish)")
    else:
        score -= 1
        signals.append("Price below 50-day SMA (bearish)")

    if sma_20 > sma_50:
        score += 0.5
        signals.append("Short-term trend up (20 > 50 SMA)")
    else:
        score -= 0.5
        signals.append("Short-term trend down (20 < 50 SMA)")

    max_score = 6.5
    normalized_score = (score / max_score) * 100

    if normalized_score > 30:
        prediction = "BULLISH"
    elif normalized_score < -30:
        prediction = "BEARISH"
    else:
        prediction = "NEUTRAL"

    return prediction, normalized_score, signals


def filter_liquid_options(
    options_dataframe: pd.DataFrame, min_volume: int, max_spread_ratio: float
) -> pd.DataFrame:
    return options_dataframe[
        (options_dataframe["volume"] >= min_volume)
        & (options_dataframe["bid"] > 0)
        & ((options_dataframe["ask"] - options_dataframe["bid"]) / options_dataframe["ask"] <= max_spread_ratio)
    ]


def call_bs_price(
    spot_price: float,
    strike_price: float,
    time_to_expiry: float,
    risk_free_rate: float,
    volatility: float,
) -> float:
    if time_to_expiry <= 0:
        return max(spot_price - strike_price, 0)
    d1 = (
        np.log(spot_price / strike_price)
        + (risk_free_rate + 0.5 * volatility**2) * time_to_expiry
    ) / (volatility * np.sqrt(time_to_expiry))
    d2 = d1 - volatility * np.sqrt(time_to_expiry)
    return spot_price * norm.cdf(d1) - strike_price * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(d2)


def implied_vol_call(
    price: float, spot_price: float, strike_price: float, time_to_expiry: float, risk_free_rate: float
) -> float:
    if time_to_expiry <= 0:
        return np.nan

    def objective(volatility):
        return call_bs_price(spot_price, strike_price, time_to_expiry, risk_free_rate, volatility) - price

    try:
        return brentq(objective, 1e-9, 5.0)
    except ValueError:
        return np.nan


def build_pdf(
    strike_grid: np.ndarray,
    iv_spline_tck,
    spot_price: float,
    time_to_expiry: float,
    risk_free_rate: float,
) -> Tuple[np.ndarray, np.ndarray]:
    implied_vols = BSpline(*iv_spline_tck)(strike_grid)
    call_prices = np.array(
        [
            call_bs_price(spot_price, strike, time_to_expiry, risk_free_rate, volatility)
            for strike, volatility in zip(strike_grid, implied_vols)
        ]
    )
    first_derivative = np.gradient(call_prices, strike_grid)
    second_derivative = np.gradient(first_derivative, strike_grid)
    pdf_raw = np.exp(risk_free_rate * time_to_expiry) * second_derivative
    pdf_raw = np.clip(pdf_raw, 0, None)
    return strike_grid, pdf_raw


def smooth_pdf(strike_grid: np.ndarray, pdf_raw: np.ndarray) -> np.ndarray:
    kernel_density = gaussian_kde(strike_grid, weights=pdf_raw)
    pdf_smoothed = kernel_density(strike_grid)
    area = np.trapezoid(pdf_smoothed, strike_grid)
    if area > 0:
        pdf_smoothed /= area
    return pdf_smoothed


def union_of_lists(lists: List[List[str]]) -> List[str]:
    if not lists:
        return []
    intersection_set = set(lists[0])
    for values in lists[1:]:
        intersection_set &= set(values)
    return list(intersection_set)


def convolve_pdfs(x_lists: List[np.ndarray], pdf_lists: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    x_result = np.array(x_lists[0])
    pdf_result = np.array(pdf_lists[0])

    for list_index in range(1, len(x_lists)):
        x_values = np.array(x_lists[list_index])
        pdf_values = np.array(pdf_lists[list_index])
        dx_result = np.mean(np.diff(x_result))
        dx_values = np.mean(np.diff(x_values))
        dx_step = min(dx_result, dx_values)
        x_result_uniform = np.arange(x_result.min(), x_result.max(), dx_step)
        x_values_uniform = np.arange(x_values.min(), x_values.max(), dx_step)
        interpolated_result = interpolate.interp1d(
            x_result, pdf_result, bounds_error=False, fill_value=0
        )
        interpolated_values = interpolate.interp1d(
            x_values, pdf_values, bounds_error=False, fill_value=0
        )
        pdf_result_uniform = interpolated_result(x_result_uniform)
        pdf_values_uniform = interpolated_values(x_values_uniform)
        pdf_convolved = np.convolve(pdf_result_uniform, pdf_values_uniform) * dx_step
        x_min_new = x_result_uniform.min() + x_values_uniform.min()
        x_result = x_min_new + np.arange(len(pdf_convolved)) * dx_step
        pdf_result = pdf_convolved

    pdf_result = pdf_result / np.trapezoid(pdf_result, x_result)
    return x_result, pdf_result


def calculate_percentile(
    x_values: np.ndarray, pdf_values: np.ndarray, percentile: float
) -> float:
    dx_values = np.diff(x_values)
    pdf_midpoints = (pdf_values[:-1] + pdf_values[1:]) / 2
    cdf = np.concatenate([[0], np.cumsum(pdf_midpoints * dx_values)])
    cdf = cdf / cdf[-1]
    target = percentile / 100.0
    index = np.searchsorted(cdf, target)
    if index == 0:
        return x_values[0]
    if index >= len(x_values):
        return x_values[-1]
    x_lower, x_upper = x_values[index - 1], x_values[index]
    cdf_lower, cdf_upper = cdf[index - 1], cdf[index]
    if cdf_upper == cdf_lower:
        return x_lower
    return x_lower + (x_upper - x_lower) * (target - cdf_lower) / (cdf_upper - cdf_lower)


def calculate_probability_below(
    x_values: np.ndarray, pdf_values: np.ndarray, threshold: float
) -> float:
    mask = x_values <= threshold
    if not np.any(mask):
        return 0.0
    x_below = x_values[mask]
    pdf_below = pdf_values[mask]
    return np.trapezoid(pdf_below, x_below)


def ticker_prediction(
    ticker_index: int,
    stock_list: pd.DataFrame,
    possible_expirations: List[str],
    expiration_index: int,
    risk_free_rate: float,
    min_volume: int,
    max_spread_ratio: float,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    import io
    import sys

    investment_value = stock_list["Value"].iloc[ticker_index]
    ticker_symbol = stock_list["Stocks"].iloc[ticker_index]

    time.sleep(0.5)

    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()

    try:
        ticker_data = yf.Ticker(ticker_symbol)
        current_price = ticker_data.history(period="1d")["Close"].iloc[-1]
        number_of_shares = investment_value / current_price

        selected_expiry = possible_expirations[expiration_index]
        option_chain = ticker_data.option_chain(selected_expiry)
        calls_dataframe = option_chain.calls[
            ["strike", "lastPrice", "bid", "ask", "volume"]
        ].copy()

        filtered_calls = filter_liquid_options(
            calls_dataframe, min_volume=min_volume, max_spread_ratio=max_spread_ratio
        )

        time_to_expiry = (
            pd.to_datetime(selected_expiry) - pd.Timestamp.today()
        ).days / 365.0
        spot_price = ticker_data.history().iloc[-1]["Close"]

        filtered_calls["iv"] = filtered_calls.apply(
            lambda row: implied_vol_call(
                row["lastPrice"], spot_price, row["strike"], time_to_expiry, risk_free_rate
            ),
            axis=1,
        )

        filtered_calls.dropna(subset=["iv"], inplace=True)

        if len(filtered_calls) < 4:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            return None, None

        strikes = filtered_calls["strike"].values
        implied_vols = filtered_calls["iv"].values

        iv_spline = splrep(strikes, implied_vols, s=10, k=3)
        strike_grid = np.linspace(strikes.min(), strikes.max(), 300)

        strike_grid_for_pdf, pdf_raw = build_pdf(
            strike_grid, iv_spline, spot_price, time_to_expiry, risk_free_rate
        )
        pdf_smoothed = smooth_pdf(strike_grid_for_pdf, pdf_raw)
        investment_grid = np.array([strike * number_of_shares for strike in strike_grid_for_pdf])
        pdf_smoothed_over_investment = pdf_smoothed / number_of_shares

        sys.stdout = old_stdout
        sys.stderr = old_stderr
        return investment_grid, pdf_smoothed_over_investment

    except Exception as error:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        raise error


def get_insurance_company_data() -> Dict[str, List[Dict[str, object]]]:
    uk_insurers = {
        "AV.L": {"name": "Aviva", "type": "Composite", "market": "FTSE 100"},
        "ADM.L": {"name": "Admiral Group", "type": "Personal Lines", "market": "FTSE 100"},
        "DLG.L": {"name": "Direct Line", "type": "Personal Lines", "market": "FTSE 250"},
        "LGEN.L": {"name": "Legal & General", "type": "Life & Pensions", "market": "FTSE 100"},
        "HSX.L": {"name": "Hiscox", "type": "Specialty/Lloyd's", "market": "FTSE 250"},
        "BEZ.L": {"name": "Beazley", "type": "Specialty/Lloyd's", "market": "FTSE 100"},
        "LRE.L": {"name": "Lancashire Holdings", "type": "Specialty/Lloyd's", "market": "FTSE 250"},
        "PRU.L": {"name": "Prudential", "type": "Life & Pensions", "market": "FTSE 100"},
    }

    cor_data = {
        "AV.L": {"cor": 94.5, "year": "2024", "note": "GI COR"},
        "ADM.L": {"cor": 88.2, "year": "2024", "note": "UK Motor"},
        "DLG.L": {"cor": 98.5, "year": "2024", "note": "Ongoing operations"},
        "LGEN.L": {"cor": None, "year": None, "note": "Life insurer - N/A"},
        "HSX.L": {"cor": 88.7, "year": "2024", "note": "Group COR"},
        "BEZ.L": {"cor": 79.0, "year": "2024", "note": "Group COR"},
        "LRE.L": {"cor": 83.5, "year": "2024", "note": "Group COR"},
        "PRU.L": {"cor": None, "year": None, "note": "Life insurer - N/A"},
    }

    global_brokers = {
        "AON": {"name": "Aon", "type": "Broker", "market": "NYSE"},
        "MMC": {"name": "Marsh McLennan", "type": "Broker", "market": "NYSE"},
        "WTW": {"name": "Willis Towers Watson", "type": "Broker", "market": "NASDAQ"},
        "AJG": {"name": "Arthur J. Gallagher", "type": "Broker", "market": "NYSE"},
    }

    results = {"uk_insurers": [], "global_brokers": []}

    import io
    import sys

    for ticker, info in uk_insurers.items():
        try:
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            sys.stdout = io.StringIO()
            sys.stderr = io.StringIO()

            stock = yf.Ticker(ticker)
            history = stock.history(period="5d")
            stock_info = stock.info

            sys.stdout = old_stdout
            sys.stderr = old_stderr

            if not history.empty:
                current_price = history["Close"].iloc[-1]
                previous_price = history["Close"].iloc[0] if len(history) > 1 else current_price
                change_pct = ((current_price - previous_price) / previous_price) * 100

                pe_ratio = stock_info.get("trailingPE")
                dividend_yield = stock_info.get("dividendYield")
                market_cap = stock_info.get("marketCap")

                cor_info = cor_data.get(ticker, {})

                results["uk_insurers"].append(
                    {
                        "ticker": ticker,
                        "name": info["name"],
                        "type": info["type"],
                        "market": info["market"],
                        "price": current_price,
                        "change_pct": change_pct,
                        "pe_ratio": pe_ratio,
                        "dividend_yield": dividend_yield * 100
                        if dividend_yield and dividend_yield < 1
                        else dividend_yield,
                        "market_cap": market_cap,
                        "currency": "GBP",
                        "cor": cor_info.get("cor"),
                        "cor_year": cor_info.get("year"),
                        "cor_note": cor_info.get("note"),
                    }
                )
        except Exception:
            continue

    for ticker, info in global_brokers.items():
        try:
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            sys.stdout = io.StringIO()
            sys.stderr = io.StringIO()

            stock = yf.Ticker(ticker)
            history = stock.history(period="5d")
            stock_info = stock.info

            sys.stdout = old_stdout
            sys.stderr = old_stderr

            if not history.empty:
                current_price = history["Close"].iloc[-1]
                previous_price = history["Close"].iloc[0] if len(history) > 1 else current_price
                change_pct = ((current_price - previous_price) / previous_price) * 100

                pe_ratio = stock_info.get("trailingPE")
                dividend_yield = stock_info.get("dividendYield")
                market_cap = stock_info.get("marketCap")

                results["global_brokers"].append(
                    {
                        "ticker": ticker,
                        "name": info["name"],
                        "type": info["type"],
                        "market": info["market"],
                        "price": current_price,
                        "change_pct": change_pct,
                        "pe_ratio": pe_ratio,
                        "dividend_yield": dividend_yield * 100
                        if dividend_yield and dividend_yield < 1
                        else dividend_yield,
                        "market_cap": market_cap,
                        "currency": "USD",
                    }
                )
        except Exception:
            continue

    return results


def get_market_rates_data() -> Dict[str, object]:
    market_rates = {
        "last_updated": "Q4 2024 / Q1 2025",
        "sources": [
            "Marsh Global Insurance Market Index",
            "Aon UK Market Insights",
            "WTW FINEX",
            "IUMI Stats Report",
            "Risk Strategies",
        ],
        "global_composite": {
            "rate": -2.0,
            "trend": "decreasing",
            "note": "Second consecutive quarterly decrease after 7 years of increases",
        },
        "lines_of_business": {
            "cyber": {
                "global_rate": -7.0,
                "uk_rate": -7.0,
                "us_rate": -5.0,
                "trend": "decreasing",
                "market_condition": "Soft - Buyer Friendly",
                "capacity": "Abundant",
                "notes": "Stabilized rates after significant increases in 2021-2022. Strong cybersecurity controls can achieve better rates. Claims volume increased in 2024 with ransomware and privacy breaches.",
                "source": "Marsh Q4 2024",
            },
            "directors_officers": {
                "global_rate": -5.0,
                "uk_rate": -10.0,
                "us_rate": -5.0,
                "trend": "decreasing",
                "market_condition": "Soft - Buyer Friendly",
                "capacity": "Abundant",
                "notes": "UK D&O rates down 10-15% in 2024. 81% of UK clients saw premium decreases. Pricing stabilized with single-digit decreases becoming more common.",
                "source": "Aon/WTW Q4 2024",
            },
            "professional_indemnity": {
                "global_rate": -7.5,
                "uk_rate": -10.0,
                "us_rate": -1.0,
                "trend": "decreasing",
                "market_condition": "Soft - Competitive",
                "capacity": "Increased",
                "notes": "London market PI rates decreased 5-10% in 2024, correcting 2019-2022 hard market increases. New capacity entering market. Competitive environment expected to continue in 2025.",
                "source": "WTW Feb 2025",
            },
            "financial_institutions": {
                "global_rate": -6.0,
                "uk_rate": -8.0,
                "us_rate": -2.0,
                "trend": "decreasing",
                "market_condition": "Soft",
                "capacity": "Adequate",
                "notes": "FI rates moderated with decreases ranging 5-10%. Limited capital markets activity restricted new business opportunities.",
                "source": "Marsh Q4 2024",
            },
            "property": {
                "global_rate": -3.0,
                "uk_rate": -4.0,
                "us_rate": -4.0,
                "trend": "decreasing",
                "market_condition": "Moderate - Improving",
                "capacity": "Strong",
                "notes": "Increased insurer capacity and competition. Strong financial performance over past 3 years. Nat cat exposed risks still face scrutiny.",
                "source": "Marsh Q4 2024",
            },
            "casualty_liability": {
                "global_rate": 4.0,
                "uk_rate": -6.0,
                "us_rate": 7.0,
                "trend": "increasing (US), decreasing (UK)",
                "market_condition": "Mixed",
                "capacity": "Constrained for US exposure",
                "notes": "US casualty rates up 7% due to social inflation and litigation trends. UK casualty (ex-motor) down 6%. US exposed risks face continued pressure.",
                "source": "Marsh Q4 2024",
            },
            "property_cat": {
                "global_rate": 6.0,
                "uk_rate": 6.0,
                "us_rate": 8.0,
                "trend": "increasing",
                "market_condition": "Firming",
                "capacity": "Constrained",
                "notes": "Property cat reinsurance pricing increased 6% at Jan 2025 renewals. Higher retentions and tighter terms in cat-prone areas.",
                "source": "Marsh Q1 2025",
            },
            "aviation": {
                "global_rate": 5.0,
                "uk_rate": 5.0,
                "us_rate": 5.0,
                "trend": "increasing",
                "market_condition": "Moderate - Firming",
                "capacity": "Constrained",
                "notes": "Continued claims activity and higher cost of parts/labor. Rate increases continued in 2024.",
                "source": "WTW Q1 2025",
            },
            "energy": {
                "global_rate": 2.0,
                "uk_rate": 2.0,
                "us_rate": 3.0,
                "trend": "increasing",
                "market_condition": "Stable",
                "capacity": "Adequate",
                "notes": "Energy market remained stable with adequate capacity, but nat cat exposure continues to pressure pricing.",
                "source": "Marsh Q4 2024",
            },
            "construction": {
                "global_rate": 6.0,
                "uk_rate": 6.0,
                "us_rate": 8.0,
                "trend": "increasing",
                "market_condition": "Hard",
                "capacity": "Constrained",
                "notes": "High inflation, supply chain issues, and large loss activity driving rate increases.",
                "source": "Aon Q4 2024",
            },
            "motor": {
                "global_rate": 5.0,
                "uk_rate": 7.0,
                "us_rate": 5.0,
                "trend": "increasing",
                "market_condition": "Hard",
                "capacity": "Constrained",
                "notes": "UK motor market improving after worst performance in a decade in 2023. Care inflation reached 25% due to provider shortages.",
                "source": "Marsh Q1 2025",
            },
            "business_travel": {
                "global_rate": 0.0,
                "uk_rate": 0.0,
                "us_rate": 0.0,
                "trend": "stable",
                "market_condition": "Stable - Growth Focus",
                "capacity": "Adequate",
                "notes": "Market valued at ~$5.7B in 2024, growing at 5-8% CAGR. Focus on digital platforms and comprehensive coverage. Post-pandemic demand recovery.",
                "source": "Industry Reports 2024",
            },
            "marine_hull": {
                "global_rate": 4.0,
                "uk_rate": 3.0,
                "us_rate": 5.0,
                "trend": "stabilizing",
                "market_condition": "Moderate - Stabilizing",
                "capacity": "Adequate",
                "notes": "Hull market settled after 5+ years of 10%+ annual increases. Baltimore Key Bridge (MV Dali) incident impacting reinsurance. Inflation driving claims costs up. New capacity entering London market.",
                "source": "WTW/Risk Strategies 2025",
            },
            "marine_cargo": {
                "global_rate": -2.5,
                "uk_rate": -3.0,
                "us_rate": -2.0,
                "trend": "softening",
                "market_condition": "Moderate - Buyer Friendly",
                "capacity": "Increased",
                "notes": "Increased insurer capacity causing modest premium reductions since Q4 2023. Cargo theft up 27% in 2024 (avg $202K per theft). Stock inventory increases softening. War risk for Red Sea/Indian Ocean priced daily.",
                "source": "Risk Strategies/IUMI 2025",
            },
            "marine_liability_pi": {
                "global_rate": 5.0,
                "uk_rate": 5.0,
                "us_rate": 7.0,
                "trend": "increasing",
                "market_condition": "Moderate - Firming",
                "capacity": "Adequate",
                "notes": "P&I Club February 2025 renewals concluded at ~5% increases. Baltimore Key Bridge case creating uncertainty for 2026. Crew injury claims and US litigation driving pressure. Excess marine casualty seeing competition in higher layers.",
                "source": "Risk Strategies/P&I Clubs 2025",
            },
        },
        "regional_summary": {
            "UK": {
                "overall_rate": -5.0,
                "trend": "Softening",
                "outlook": "Buyer-friendly conditions expected to continue",
            },
            "US": {
                "overall_rate": 0.0,
                "trend": "Flat",
                "outlook": "Casualty pressure, property improving",
            },
            "Europe": {
                "overall_rate": -2.0,
                "trend": "Softening",
                "outlook": "Moderate conditions",
            },
            "Asia_Pacific": {
                "overall_rate": -8.0,
                "trend": "Softening",
                "outlook": "Led global declines",
            },
        },
    }

    return market_rates
