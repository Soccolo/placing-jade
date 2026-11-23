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
from textblob import TextBlob

from app.financial_command_center.options_risk import implied_cdf, simulate

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
    import io
    import sys

    old_stdout = sys.stdout
    old_stderr = sys.stderr
    try:
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()

        data = yf.download(symbol, period=period, progress=False, auto_adjust=True)

        if data is not None and len(data) > 0:
            return data
    except Exception:
        return None
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
    return None


def fetch_from_ticker_history(symbol: str, period: str = "5d") -> Optional[pd.DataFrame]:
    import io
    import sys

    old_stdout = sys.stdout
    old_stderr = sys.stderr
    try:
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()

        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period)

        if data is not None and len(data) > 0:
            return data
    except Exception:
        return None
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
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

    old_stdout = sys.stdout
    old_stderr = sys.stderr
    try:
        time.sleep(0.3)
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()

        ticker = yf.Ticker(symbol)
        news_items = ticker.news

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
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr


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

    import io
    import sys

    old_stdout = sys.stdout
    old_stderr = sys.stderr
    try:
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()

        ticker = yf.Ticker(symbol)
        info = ticker.info

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
        pass
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr

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
    
    # Calculate relative strength, handling division by zero
    relative_strength = gain / loss
    rsi_values = 100 - (100 / (1 + relative_strength))
    
    # Handle edge cases to avoid inf/NaN
    # When loss == 0 and gain > 0, RSI = 100
    rsi_values = rsi_values.where(~((loss == 0) & (gain > 0)), 100)
    # When gain == 0 and loss > 0, RSI = 0
    rsi_values = rsi_values.where(~((gain == 0) & (loss > 0)), 0)
    # When both gain and loss are 0, RSI = 50
    rsi_values = rsi_values.where(~((gain == 0) & (loss == 0)), 50)
    
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

    if len(close_prices) >= 200:
        if current_price > sma_200:
            score += 1
            signals.append("Price above 200-day SMA (bullish)")
        else:
            score -= 1
            signals.append("Price below 200-day SMA (bearish)")

    max_score = 7.5 if len(close_prices) >= 200 else 6.5
    normalized_score = (score / max_score) * 100

    if normalized_score > 30:
        prediction = "BULLISH"
    elif normalized_score < -30:
        prediction = "BEARISH"
    else:
        prediction = "NEUTRAL"

    return prediction, normalized_score, signals


NEW_YORK = "America/New_York"


def common_expirations(tickers: List[str]) -> Tuple[List[str], List[str]]:
    """Future option expiry dates that every ticker trades, soonest first.

    Returns (dates, tickers with no listed options). Those tickers are ignored when intersecting.
    """
    expirations, no_options = [], []
    for ticker_symbol in tickers:
        try:
            dates = set(yf.Ticker(ticker_symbol).options)
        except Exception:
            dates = set()
        if dates:
            expirations.append(dates)
        else:
            no_options.append(ticker_symbol)
    if not expirations:
        return [], no_options
    today = pd.Timestamp.now(tz=NEW_YORK).strftime("%Y-%m-%d")
    return sorted(date for date in set.intersection(*expirations) if date > today), no_options


def return_correlation(tickers: List[str]) -> np.ndarray:
    """Correlation of daily returns over the past year, in the order given."""
    if len(tickers) == 1:
        return np.ones((1, 1))
    closes = yf.download(list(tickers), period="1y", progress=False, auto_adjust=True)["Close"]
    correlation = closes.pct_change().corr().reindex(index=tickers, columns=tickers).fillna(0).to_numpy(copy=True)
    np.fill_diagonal(correlation, 1)
    return correlation


def options_implied_values(
    portfolio: pd.DataFrame,
    expiry: str,
    risk_free_rate: float,
    min_volume: int,
    max_spread_ratio: float,
) -> Dict[str, object]:
    """Simulate what the positions are worth on the expiry date, as implied by option prices.

    Returns the simulated values ("positions", None if nothing could be modelled), the
    rows that were modelled ("included") and why each other ticker was left out ("skipped").
    """
    years = (pd.Timestamp(f"{expiry} 16:00", tz=NEW_YORK) - pd.Timestamp.now(tz=NEW_YORK)).total_seconds() / (365 * 86400)
    marginals, kept, skipped = [], [], {}
    for ticker_symbol in portfolio["Stocks"]:
        try:
            chain = yf.Ticker(ticker_symbol).option_chain(expiry)
            marginal = implied_cdf(chain.calls, chain.puts, years, risk_free_rate, min_volume, max_spread_ratio)
        except Exception as error:
            skipped[ticker_symbol] = f"data error: {str(error)[:80]}"
            continue
        if marginal is None:
            skipped[ticker_symbol] = "too few liquid option quotes"
        else:
            marginals.append(marginal)
            kept.append(ticker_symbol)
    # Positions that could not be modelled leave every figure, including today's value.
    included = portfolio[portfolio["Stocks"].isin(kept)]
    positions = simulate(marginals, included["Value"], return_correlation(kept)) if kept else None
    return {"expiry": expiry, "included": included, "skipped": skipped, "positions": positions}


def summarise_distribution(
    result: Dict[str, object], free_capital: float, var_confidence: int, percentiles: List[int]
) -> Dict[str, object]:
    """Headline figures for the simulated values from options_implied_values."""
    included = result["included"]
    values = result["positions"] + free_capital
    today = float(included["Value"].sum() + free_capital)
    capital = float(included["Unleveraged Value"].sum() + free_capital)
    var_value = float(np.percentile(values, 100 - var_confidence))
    levels = sorted(percentiles)
    return {
        "values": values,
        "today": today,
        "capital": capital,
        "leverage": today / capital,
        "var_value": var_value,
        "var_loss": today - var_value,
        "mean": float(values.mean()),
        "prob_profit": float((values > today).mean()),
        "percentiles": [
            {
                "percentile": level,
                "value": float(value),
                "return_on_capital": float((value - today) / capital * 100),
                "return_on_position": float((value / today - 1) * 100),
            }
            for level, value in zip(levels, np.percentile(values, levels) if levels else [])
        ],
    }


def options_outlook(holdings: Dict[str, float], cash: float) -> Optional[Dict[str, object]]:
    """Chance the holdings are worth more on the first option expiry at least two weeks away.

    `holdings` maps ticker to position value. Returns None when no holding has usable option quotes.
    """
    held = {ticker: value for ticker, value in holdings.items() if value > 0}
    expirations, no_options = common_expirations(list(held))
    portfolio = pd.DataFrame(
        [(ticker, value, value) for ticker, value in held.items() if ticker not in no_options],
        columns=["Stocks", "Value", "Unleveraged Value"],
    )
    if not expirations or portfolio.empty:
        return None
    cutoff = (pd.Timestamp.now(tz=NEW_YORK) + pd.Timedelta(days=14)).strftime("%Y-%m-%d")
    expiry = next((date for date in expirations if date >= cutoff), expirations[-1])
    result = options_implied_values(portfolio, expiry, risk_free_rate=0.04, min_volume=20, max_spread_ratio=0.2)
    if result["positions"] is None:
        return None
    stats = summarise_distribution(result, max(cash, 0.0), var_confidence=95, percentiles=[])
    return {
        "prob_profit": stats["prob_profit"] * 100,
        "expiry": expiry,
        "coverage": float(result["included"]["Value"].sum() / sum(held.values()) * 100),
    }


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
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        try:
            sys.stdout = io.StringIO()
            sys.stderr = io.StringIO()

            stock = yf.Ticker(ticker)
            history = stock.history(period="5d")
            stock_info = stock.info

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
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

    for ticker, info in global_brokers.items():
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        try:
            sys.stdout = io.StringIO()
            sys.stderr = io.StringIO()

            stock = yf.Ticker(ticker)
            history = stock.history(period="5d")
            stock_info = stock.info

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
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

    return results


def simulate_portfolio_pnl(
    tickers: List[str],
    quantities: List[float],
    start_date: str = "2022-01-01",
    num_simulations: int = 10_000,
) -> Optional[Dict[str, object]]:
    """
    Run a one-day Monte Carlo simulation on a portfolio using
    Cholesky-correlated GBM draws on the log-return covariance matrix.

    Returns a dict with simulation results — VaR/TVaR figures, the full
    P&L distribution, and a per-asset `risk_contributions` table that
    decomposes both daily volatility (Euler decomposition on the
    covariance matrix) and VaR/TVaR (conditional expectation on the
    simulated scenarios) into contributions from each position.

    Returns a dict with just an 'error' key explaining what went wrong
    if the simulation cannot be run.
    """
    if not tickers or not quantities:
        return {"error": "No tickers or quantities provided."}

    diagnostics = []

    # ── Step 1: Download price data ──
    # Try bulk download first, then fall back to individual downloads
    price_series = {}

    try:
        raw = yf.download(tickers, start=start_date, progress=False)
        if raw is not None and not raw.empty:
            if isinstance(raw.columns, pd.MultiIndex):
                close_df = raw["Close"]
            else:
                close_df = raw[["Close"]]
                close_df.columns = tickers

            if isinstance(close_df, pd.Series):
                close_df = close_df.to_frame(name=tickers[0])

            for t in tickers:
                if t in close_df.columns:
                    col = close_df[t].dropna()
                    if len(col) >= 10:
                        price_series[t] = col
                    else:
                        diagnostics.append(f"{t}: only {len(col)} price points from bulk download")
                else:
                    diagnostics.append(f"{t}: not found in bulk download")
    except Exception as e:
        diagnostics.append(f"Bulk download failed: {str(e)[:100]}")

    # Individual fallback for any tickers we missed
    missing = [t for t in tickers if t not in price_series]
    for t in missing:
        try:
            single = yf.download(t, start=start_date, progress=False)
            if single is not None and not single.empty:
                single = normalize_dataframe(single)
                col = safe_get_column(single, "Close")
                if col is not None:
                    col = col.dropna()
                    if len(col) >= 10:
                        price_series[t] = col
                        diagnostics.append(f"{t}: recovered via individual download ({len(col)} days)")
                    else:
                        diagnostics.append(f"{t}: only {len(col)} price points even individually")
                else:
                    diagnostics.append(f"{t}: no Close column in individual download")
            else:
                diagnostics.append(f"{t}: individual download returned empty")
        except Exception as e:
            diagnostics.append(f"{t}: individual download failed — {str(e)[:80]}")

    if not price_series:
        return {
            "error": "Could not download price data for any ticker. "
                     "This may be a network issue or the tickers may not be available on Yahoo Finance.",
            "diagnostics": diagnostics,
        }

    # ── Step 2: Align into a DataFrame ──
    # Use forward-fill + back-fill to handle minor date gaps rather than dropping all rows
    prices = pd.DataFrame(price_series)
    prices = prices.sort_index().ffill().bfill().dropna()

    available = list(prices.columns)
    skipped = [t for t in tickers if t not in available]

    if len(prices) < 20:
        return {
            "error": f"Only {len(prices)} overlapping trading days across tickers (need at least 20). "
                     f"Tickers with data: {available}. Try a more recent start date.",
            "diagnostics": diagnostics,
        }

    if len(available) == 0:
        return {
            "error": "No tickers had enough price history after alignment.",
            "diagnostics": diagnostics,
        }

    # Map back to quantities for available tickers only
    avail_shares = np.array(
        [quantities[tickers.index(t)] for t in available], dtype=float
    )

    current_prices = prices.iloc[-1].values.astype(float)
    holdings = avail_shares * current_prices
    total_investment = float(holdings.sum())

    # ── Step 3: Log returns & covariance ──
    log_returns = np.log(prices / prices.shift(1)).dropna()

    if len(log_returns) < 15:
        return {
            "error": f"Only {len(log_returns)} return observations — need at least 15 for a meaningful simulation.",
            "diagnostics": diagnostics,
        }

    cov_matrix = log_returns.cov().values
    corr_matrix = log_returns.corr().values
    mu = log_returns.mean().values
    sigma = log_returns.std().values
    n = len(available)

    # ── Step 4: Cholesky decomposition ──
    try:
        L = np.linalg.cholesky(corr_matrix)
    except np.linalg.LinAlgError:
        # Near-PD fix: clamp negative eigenvalues
        eigvals, eigvecs = np.linalg.eigh(corr_matrix)
        eigvals = np.maximum(eigvals, 1e-8)
        corr_fixed = eigvecs @ np.diag(eigvals) @ eigvecs.T
        try:
            L = np.linalg.cholesky(corr_fixed)
        except np.linalg.LinAlgError:
            # Last resort: treat as uncorrelated
            L = np.eye(n)
            diagnostics.append("Correlation matrix not positive-definite; treating stocks as uncorrelated.")

    # ── Step 5: Simulate ──
    rng = np.random.default_rng(42)
    epsilon = rng.standard_normal((n, num_simulations))
    Z = L @ epsilon

    sim_prices = current_prices.reshape(-1, 1) * np.exp(
        mu.reshape(-1, 1) + sigma.reshape(-1, 1) * Z
    )

    sim_values = avail_shares @ sim_prices
    sim_pnl = sim_values - total_investment

    # Per-asset simulated P&L (shape: n × num_simulations).
    # By construction per_asset_pnl.sum(axis=0) == sim_pnl, so any
    # decomposition we build from it is internally consistent.
    per_asset_pnl = avail_shares.reshape(-1, 1) * (
        sim_prices - current_prices.reshape(-1, 1)
    )

    var_95 = float(-np.percentile(sim_pnl, 5))
    var_99 = float(-np.percentile(sim_pnl, 1))
    var_995 = float(-np.percentile(sim_pnl, 0.5))

    tail = sim_pnl[sim_pnl <= -var_995]
    tvar_995 = float(-np.mean(tail)) if len(tail) > 0 else var_995

    port_daily_vol = float(np.sqrt(holdings @ cov_matrix @ holdings))

    # ── Step 6: Risk attribution ──
    # Two decompositions, each using a different (rigorous) definition of
    # "contribution":
    #
    #   (a) Component volatility via Euler decomposition on the covariance
    #       matrix. Sums to port_daily_vol exactly.
    #         CV_i = w_i * (Σ w)_i / σ_p
    #
    #   (b) Component VaR / TVaR via conditional expectation on the
    #       simulated scenarios. For VaR we average per-asset P&L in a
    #       narrow band around the VaR quantile; for TVaR we average
    #       across the full tail beyond VaR_99.5%.
    #         C_VaR_i = -E[ pnl_i | portfolio P&L ≈ -VaR_α ]
    #         C_TVaR_i = -E[ pnl_i | portfolio P&L ≤ -VaR_α ]
    #       By Euler's theorem these sum (approximately) to VaR_α / TVaR_α.

    # (a) Euler decomposition of daily volatility
    cov_holdings = cov_matrix @ holdings  # shape (n,)
    if port_daily_vol > 0:
        component_vol = holdings * cov_holdings / port_daily_vol
        marginal_vol = cov_holdings / port_daily_vol
    else:
        component_vol = np.zeros(n)
        marginal_vol = np.zeros(n)

    # (b) Conditional-expectation component VaR using a narrow band around
    #     the quantile. The band collects scenarios within ±0.5% of the VaR
    #     percentile, falling back to the tail mean if too few samples land
    #     inside the band (rare with 10k sims but possible for tight tails).
    def _component_var_band(confidence: float, band_width: float = 0.005) -> np.ndarray:
        alpha = 1.0 - confidence
        lower_pct = max(0.0, (alpha - band_width) * 100.0)
        upper_pct = min(100.0, (alpha + band_width) * 100.0)
        lo = np.percentile(sim_pnl, lower_pct)
        hi = np.percentile(sim_pnl, upper_pct)
        band_mask = (sim_pnl >= lo) & (sim_pnl <= hi)
        if band_mask.sum() < 5:
            tail_mask = sim_pnl <= np.percentile(sim_pnl, alpha * 100.0)
            if tail_mask.sum() == 0:
                return np.zeros(n)
            return -per_asset_pnl[:, tail_mask].mean(axis=1)
        return -per_asset_pnl[:, band_mask].mean(axis=1)

    comp_var_95 = _component_var_band(0.95)
    comp_var_99 = _component_var_band(0.99)
    comp_var_995 = _component_var_band(0.995)

    # Component TVaR 99.5% = mean per-asset P&L over the full 99.5% tail
    tail_mask_995 = sim_pnl <= -var_995
    if tail_mask_995.sum() > 0:
        comp_tvar_995 = -per_asset_pnl[:, tail_mask_995].mean(axis=1)
    else:
        comp_tvar_995 = comp_var_995.copy()

    # Assemble a per-asset table. Sorted by largest contribution to VaR 99.5%
    # so the worst offenders surface first.
    def _safe_pct(numerator: float, denominator: float) -> float:
        return float(numerator / denominator) if denominator not in (0.0, 0) else 0.0

    risk_contributions: List[Dict[str, object]] = []
    for i, ticker in enumerate(available):
        position_value = float(holdings[i])
        weight = _safe_pct(position_value, total_investment)
        risk_contributions.append(
            {
                "ticker": ticker,
                "position_value": position_value,
                "weight": weight,
                "marginal_vol": float(marginal_vol[i]),
                "component_vol": float(component_vol[i]),
                "component_vol_pct": _safe_pct(float(component_vol[i]), port_daily_vol),
                "component_var_95": float(comp_var_95[i]),
                "component_var_95_pct": _safe_pct(float(comp_var_95[i]), var_95),
                "component_var_99": float(comp_var_99[i]),
                "component_var_99_pct": _safe_pct(float(comp_var_99[i]), var_99),
                "component_var_995": float(comp_var_995[i]),
                "component_var_995_pct": _safe_pct(float(comp_var_995[i]), var_995),
                "component_tvar_995": float(comp_tvar_995[i]),
                "component_tvar_995_pct": _safe_pct(float(comp_tvar_995[i]), tvar_995),
            }
        )

    risk_contributions.sort(key=lambda row: row["component_var_995"], reverse=True)

    return {
        "sim_pnl": sim_pnl,
        "sim_values": sim_values,
        "total_investment": total_investment,
        "var_95": var_95,
        "var_99": var_99,
        "var_995": var_995,
        "tvar_995": tvar_995,
        "port_daily_vol": port_daily_vol,
        "available_tickers": available,
        "skipped_tickers": skipped,
        "num_trading_days": len(log_returns),
        "diagnostics": diagnostics,
        "risk_contributions": risk_contributions,
    }


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
