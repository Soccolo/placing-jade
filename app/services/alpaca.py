"""
Alpaca API Service

Handles communication with Alpaca's paper and live trading APIs.
"""
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Tuple
import logging

from alpaca.trading.client import TradingClient
from alpaca.common.exceptions import APIError

# Configure logging - never log secrets
logger = logging.getLogger(__name__)


@dataclass
class AccountInfo:
    """Account information from Alpaca."""
    cash: float
    buying_power: float
    portfolio_value: float
    equity: float
    currency: str


@dataclass
class Position:
    """A single position in the account."""
    symbol: str
    qty: float
    market_value: float
    avg_entry_price: float
    current_price: float
    unrealized_pl: float
    unrealized_pl_pct: float


@dataclass
class AlpacaAccountData:
    """Complete account data from Alpaca."""
    account: AccountInfo
    positions: List[Position]
    fetched_at: datetime


def create_trading_client(api_key: str, api_secret: str, paper: bool = True) -> TradingClient:
    """
    Create an Alpaca TradingClient.

    Args:
        api_key: Alpaca API key
        api_secret: Alpaca API secret
        paper: True for paper trading, False for live trading

    Returns:
        Configured TradingClient instance
    """
    return TradingClient(
        api_key=api_key,
        secret_key=api_secret,
        paper=paper,
    )


def verify_connection(api_key: str, api_secret: str, paper: bool = True) -> Tuple[bool, str]:
    """
    Verify that the provided credentials can connect to Alpaca.

    Args:
        api_key: Alpaca API key
        api_secret: Alpaca API secret
        paper: True for paper trading, False for live trading

    Returns:
        Tuple of (success: bool, message: str)
    """
    mode = "paper" if paper else "live"
    try:
        client = create_trading_client(api_key, api_secret, paper=paper)
        client.get_account()
        logger.info(f"Successfully verified Alpaca {mode} connection")
        return True, f"Connected to Alpaca {mode} trading account"

    except APIError as e:
        logger.warning(f"Alpaca API error during {mode} verification: {e.status_code}")
        if e.status_code == 401:
            return False, "Invalid API credentials"
        elif e.status_code == 403:
            return False, "API credentials do not have required permissions"
        else:
            return False, f"Alpaca API error: {e.status_code}"

    except Exception as e:
        logger.error(f"Unexpected error during Alpaca {mode} verification: {type(e).__name__}")
        return False, f"Connection failed: {type(e).__name__}"


def fetch_account_data(api_key: str, api_secret: str, paper: bool = True) -> Tuple[Optional[AlpacaAccountData], str]:
    """
    Fetch account and position data from Alpaca.

    Args:
        api_key: Alpaca API key
        api_secret: Alpaca API secret
        paper: True for paper trading, False for live trading

    Returns:
        Tuple of (data: AlpacaAccountData or None, message: str)
    """
    mode = "paper" if paper else "live"
    try:
        client = create_trading_client(api_key, api_secret, paper=paper)

        account = client.get_account()

        account_info = AccountInfo(
            cash=float(account.cash),
            buying_power=float(account.buying_power),
            portfolio_value=float(account.portfolio_value),
            equity=float(account.equity),
            currency=account.currency,
        )

        raw_positions = client.get_all_positions()

        positions = []
        for pos in raw_positions:
            positions.append(Position(
                symbol=pos.symbol,
                qty=float(pos.qty),
                market_value=float(pos.market_value),
                avg_entry_price=float(pos.avg_entry_price),
                current_price=float(pos.current_price),
                unrealized_pl=float(pos.unrealized_pl),
                unrealized_pl_pct=float(pos.unrealized_plpc) * 100,
            ))

        positions.sort(key=lambda p: p.market_value, reverse=True)

        data = AlpacaAccountData(
            account=account_info,
            positions=positions,
            fetched_at=datetime.utcnow(),
        )

        logger.info(f"Successfully fetched {mode} account data with {len(positions)} positions")
        return data, f"Successfully fetched {mode} account data"

    except APIError as e:
        logger.warning(f"Alpaca API error during {mode} fetch: {e.status_code}")
        if e.status_code == 401:
            return None, "Invalid API credentials"
        elif e.status_code == 403:
            return None, "API credentials do not have required permissions"
        else:
            return None, f"Alpaca API error: {e.status_code}"

    except Exception as e:
        logger.error(f"Unexpected error during Alpaca {mode} fetch: {type(e).__name__}")
        return None, f"Failed to fetch data: {type(e).__name__}"
