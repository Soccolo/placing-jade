"""The dashboard's risk simulation, end to end, without the network or a real account."""
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


def fake_download(tickers, **_):
    days = pd.bdate_range("2024-01-01", periods=300)
    returns = np.random.default_rng(0).normal(0.0005, 0.02, (len(days), len(tickers)))
    closes = 100 * np.exp(returns.cumsum(axis=0))
    return pd.DataFrame(closes, index=days, columns=pd.MultiIndex.from_product([["Close"], tickers]))


def test_dashboard_runs_risk_simulation_only_when_asked(monkeypatch):
    class Credentials:
        api_key = api_secret = "x"
        is_connected = True

    async def get_credentials():
        return Credentials()

    async def no_audit(*_):
        return None

    positions = [
        Position(symbol=s, qty=q, market_value=q * 100, avg_entry_price=90, current_price=100,
                 unrealized_pl=q * 10, unrealized_pl_pct=11.1)
        for s, q in (("AAA", 10), ("BBB", 30))
    ]
    account = AccountInfo(cash=1000, buying_power=2000, portfolio_value=5000, equity=5000, currency="USD")
    data = AlpacaAccountData(account=account, positions=positions, fetched_at=datetime(2026, 1, 1))

    monkeypatch.setattr(dashboard, "get_credentials", get_credentials)
    monkeypatch.setattr(dashboard, "log_audit_event", no_audit)
    monkeypatch.setattr(dashboard, "fetch_account_data", lambda *_: (data, "ok"))
    monkeypatch.setattr("app.financial_command_center.logic.yf.download", fake_download)
    client = TestClient(app)

    plain = client.get("/dashboard")
    assert plain.status_code == 200
    assert "Run simulation" in plain.text and "VaR 99.5%" not in plain.text

    simulated = client.get("/dashboard?simulate=1")
    assert simulated.status_code == 200
    assert "VaR 99.5% (1 day in 200)" in simulated.text
    assert "Largest contributors to risk" in simulated.text
    assert simulated.text.count('<td class="symbol">BBB</td>') == 2  # positions table and risk table
