"""
Dashboard Route

Displays account summary and positions from Alpaca.
"""
import asyncio
import time

import plotly.io as plotly_io
from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates

from app.services.credentials import get_credentials, update_connection_status, log_audit_event
from app.services.alpaca import fetch_account_data
from app.financial_command_center.charts import build_pnl_simulation_chart, build_risk_contribution_chart
from app.financial_command_center.logic import options_outlook, simulate_portfolio_pnl
from app.config import RISK_BUDGET_PCT, TARGET_PORTFOLIO_PATH, WEIGHT_SUM_EPSILON
from app.services.portfolio import get_portfolio

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")

RISK_CACHE_SECONDS = 30 * 60
# shortcut: in memory and per process, move to the database once there is more than one user or worker
_risk_cache: dict = {}


def strategy_alignment(positions, portfolio_value: float, targets: dict) -> float | None:
    """Percentage of the portfolio already where the strategy wants it; 100 means identical."""
    if portfolio_value <= 0 or not targets:
        return None
    actual = {p.symbol: p.market_value / portfolio_value for p in positions}
    gap = sum(abs(actual.get(symbol, 0.0) - targets.get(symbol, 0.0)) for symbol in actual.keys() | targets.keys())
    # Cash is whatever is left over on each side.
    gap += abs((1 - sum(actual.values())) - (1 - sum(targets.values())))
    return max(0.0, 100 * (1 - gap / 2))


def allocation_strip(positions, portfolio_value: float, top: int = 5) -> list:
    """The largest holdings, the rest and cash as percentages of the portfolio."""
    if portfolio_value <= 0:
        return []
    held = sorted(
        ((p.symbol, p.market_value / portfolio_value * 100) for p in positions if p.market_value > 0),
        key=lambda part: part[1],
        reverse=True,
    )
    parts = held[:top]
    if len(held) > top:
        parts.append(("Other", sum(weight for _, weight in held[top:])))
    cash = 100 - sum(weight for _, weight in held)
    if cash > 0.5:
        parts.append(("Cash", cash))
    return [{"label": label, "weight": weight} for label, weight in parts]


def build_overview(account, positions) -> dict:
    target = get_portfolio(TARGET_PORTFOLIO_PATH, WEIGHT_SUM_EPSILON)
    targets = {entry.symbol: entry.weight for entry in target.entries} if target.is_valid else {}
    return {
        "alignment": strategy_alignment(positions, account.portfolio_value, targets),
        "allocation": allocation_strip(positions, account.portfolio_value),
    }


def build_risk_view(positions, account) -> dict:
    """Run the risk check on the account's positions: Monte Carlo simulation plus the options outlook."""
    result = simulate_portfolio_pnl([p.symbol for p in positions], [float(p.qty) for p in positions])
    if result is None or "error" in result:
        return {
            "error": (result or {}).get("error", "Simulation returned no result."),
            "diagnostics": (result or {}).get("diagnostics", []),
        }
    try:
        outlook = options_outlook({p.symbol: p.market_value for p in positions}, account.cash)
    except Exception:
        outlook = None  # the simulation still stands without it; the dial says option data is missing
    pnl_chart = build_pnl_simulation_chart(result["sim_pnl"], result["var_95"], result["var_995"], result["tvar_995"])
    contribution_chart = build_risk_contribution_chart(result["risk_contributions"])
    budget = RISK_BUDGET_PCT / 100 * account.portfolio_value
    return {
        "pnl_chart": plotly_io.to_html(pnl_chart, full_html=False, include_plotlyjs="cdn"),
        "contribution_chart": plotly_io.to_html(contribution_chart, full_html=False, include_plotlyjs=False),
        "risk_used": result["var_995"] / budget * 100 if budget > 0 else 0.0,
        "risk_budget_pct": RISK_BUDGET_PCT,
        "outlook": outlook,
        # The raw samples stay out of the template context.
        **{key: value for key, value in result.items() if key not in ("sim_pnl", "sim_values")},
    }


async def risk_view(positions, account, run: bool) -> dict | None:
    """The risk check for these exact positions: fresh if asked for, otherwise a recent one if there is one."""
    key = tuple((p.symbol, p.qty) for p in positions)
    if run and positions:
        risk = await asyncio.to_thread(build_risk_view, positions, account)
        _risk_cache.clear()
        if "error" not in risk:
            _risk_cache[key] = (time.monotonic(), risk)
        return risk
    cached = _risk_cache.get(key)
    return cached[1] if cached and time.monotonic() - cached[0] < RISK_CACHE_SECONDS else None


def render_dashboard(
    request: Request,
    connected: bool,
    account=None,
    positions=None,
    fetched_at=None,
    error: str | None = None,
    message: str | None = None,
    risk: dict | None = None,
):
    return templates.TemplateResponse(request, "dashboard.html",
        {
            "request": request,
            "connected": connected,
            "account": account,
            "positions": positions or [],
            "fetched_at": fetched_at,
            "error": error,
            "message": message,
            "risk": risk,
            "overview": build_overview(account, positions or []) if account else None,
        },
    )


@router.get("")
async def dashboard_page(request: Request, simulate: bool = False):
    """Display the dashboard with account info and positions, and the risk simulation on request."""
    creds = await get_credentials()
    
    if not creds:
        return render_dashboard(
            request,
            connected=False,
            error="Not connected. Please connect your Alpaca account first.",
        )
    
    if not creds.is_connected:
        return render_dashboard(
            request,
            connected=False,
            error="Connection not verified. Please verify your connection.",
        )
    
    # Fetch account data
    data, message = await asyncio.to_thread(fetch_account_data, creds.api_key, creds.api_secret)
    
    if data:
        await log_audit_event("refreshed", f"Fetched {len(data.positions)} positions")
        risk = await risk_view(data.positions, data.account, run=simulate)

        return render_dashboard(
            request,
            connected=True,
            account=data.account,
            positions=data.positions,
            fetched_at=data.fetched_at,
            risk=risk,
        )
    else:
        await update_connection_status(is_connected=False)
        await log_audit_event("connection_failed", message)
        
        return render_dashboard(
            request,
            connected=False,
            error=message,
        )


@router.post("/refresh")
async def refresh_dashboard(request: Request):
    """Manually refresh account data."""
    creds = await get_credentials()
    
    if not creds or not creds.is_connected:
        return render_dashboard(
            request,
            connected=False,
            error="Not connected. Please connect your Alpaca account first.",
        )
    
    # Fetch fresh data
    data, message = await asyncio.to_thread(fetch_account_data, creds.api_key, creds.api_secret)
    
    if data:
        await update_connection_status(is_connected=True)
        await log_audit_event("refreshed", f"Manual refresh - {len(data.positions)} positions")
        
        return render_dashboard(
            request,
            connected=True,
            account=data.account,
            positions=data.positions,
            fetched_at=data.fetched_at,
            message="Data refreshed successfully",
            risk=await risk_view(data.positions, data.account, run=False),
        )
    else:
        await update_connection_status(is_connected=False)
        await log_audit_event("connection_failed", message)
        
        return render_dashboard(
            request,
            connected=False,
            error=message,
        )
