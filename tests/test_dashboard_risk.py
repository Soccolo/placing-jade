"""The dashboard's overview and risk check, end to end, without the network or a real account."""
import os
from datetime import datetime

import numpy as np
import pandas as pd
from cryptography.fernet import Fernet

os.environ.setdefault("ENCRYPTION_KEY", Fernet.generate_key().decode())

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402
from app.routes import dashboard  # noqa: E402
from app.services.alpaca import AccountInfo, AlpacaAccountData, Position  # noqa: E402


def position(symbol, value):
    return Position(symbol=symbol, qty=value / 100, market_value=value, avg_entry_price=90, current_price=100,
                    unrealized_pl=value / 10, unrealized_pl_pct=11.1)


def fake_download(tickers, **_):
    days = pd.bdate_range("2024-01-01", periods=300)
    returns = np.random.default_rng(0).normal(0.0005, 0.02, (len(days), len(tickers)))
    closes = 100 * np.exp(returns.cumsum(axis=0))
    return pd.DataFrame(closes, index=days, columns=pd.MultiIndex.from_product([["Close"], tickers]))


def test_strategy_alignment_is_the_share_already_at_target():
    targets = {"AAA": 0.5, "BBB": 0.5}
    assert dashboard.strategy_alignment([position("AAA", 500), position("BBB", 500)], 1000, targets) == 100
    # Half in the right stock, half in cash: half the portfolio still has to move.
    assert dashboard.strategy_alignment([position("AAA", 500)], 1000, targets) == 50
    # Nothing in common with the strategy.
    assert dashboard.strategy_alignment([position("ZZZ", 1000)], 1000, targets) == 0
    assert dashboard.strategy_alignment([position("AAA", 500)], 1000, {}) is None


def test_allocation_strip_groups_the_tail_and_shows_cash():
    positions = [position(symbol, value) for symbol, value in zip("ABCDEFG", (300, 200, 150, 100, 80, 40, 30))]
    parts = dashboard.allocation_strip(positions, 1000)
    assert [part["label"] for part in parts] == ["A", "B", "C", "D", "E", "Other", "Cash"]
    assert round(parts[5]["weight"]) == 7 and round(parts[6]["weight"]) == 10
    assert round(sum(part["weight"] for part in parts)) == 100


def test_dashboard_runs_risk_check_only_when_asked_then_remembers_it(monkeypatch):
    class Credentials:
        api_key = api_secret = "x"
        is_connected = True

    async def get_credentials():
        return Credentials()

    async def no_audit(*_):
        return None

    positions = [position("AAA", 1000), position("BBB", 3000)]
    account = AccountInfo(cash=1000, buying_power=2000, portfolio_value=5000, equity=5000, currency="USD")
    data = AlpacaAccountData(account=account, positions=positions, fetched_at=datetime(2026, 1, 1))

    dashboard._risk_cache.clear()
    monkeypatch.setattr(dashboard, "get_credentials", get_credentials)
    monkeypatch.setattr(dashboard, "log_audit_event", no_audit)
    monkeypatch.setattr(dashboard, "fetch_account_data", lambda *_: (data, "ok"))
    monkeypatch.setattr(dashboard, "options_outlook", lambda *_: {"prob_profit": 56.0, "expiry": "2026-12-18", "coverage": 75.0})
    monkeypatch.setattr("app.financial_command_center.logic.yf.download", fake_download)
    client = TestClient(app)

    plain = client.get("/dashboard")
    assert plain.status_code == 200
    assert "In line with strategy" in plain.text and "Where the money is" in plain.text
    assert "Run the risk check to fill this in" in plain.text and "VaR 99.5% (1 day in 200)" not in plain.text

    checked = client.get("/dashboard?simulate=1")
    assert checked.status_code == 200
    assert "VaR 99.5% (1 day in 200)" in checked.text and "Largest contributors to risk" in checked.text
    assert "of a 5% one-day loss limit" in checked.text
    assert "Chance of profit by 2026-12-18" in checked.text and "covering 75% of holdings" in checked.text
    assert checked.text.count('<td class="symbol">BBB</td>') == 2  # positions table and risk table

    # A later plain visit shows the remembered risk check instead of empty dials.
    remembered = client.get("/dashboard")
    assert "Chance of profit by 2026-12-18" in remembered.text
