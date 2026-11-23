"""
Dashboard Route

Displays account summary and positions from Alpaca.
"""
import asyncio

import plotly.io as plotly_io
from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates

from app.services.credentials import get_credentials, update_connection_status, log_audit_event
from app.services.alpaca import fetch_account_data
from app.financial_command_center.charts import build_pnl_simulation_chart, build_risk_contribution_chart
from app.financial_command_center.logic import simulate_portfolio_pnl

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")

def build_risk_view(positions) -> dict:
    """Run the Monte Carlo risk simulation on the account's positions and render its charts."""
    result = simulate_portfolio_pnl([p.symbol for p in positions], [float(p.qty) for p in positions])
    if result is None or "error" in result:
        return {
            "error": (result or {}).get("error", "Simulation returned no result."),
            "diagnostics": (result or {}).get("diagnostics", []),
        }
    pnl_chart = build_pnl_simulation_chart(result["sim_pnl"], result["var_95"], result["var_995"], result["tvar_995"])
    contribution_chart = build_risk_contribution_chart(result["risk_contributions"])
    return {
        "pnl_chart": plotly_io.to_html(pnl_chart, full_html=False, include_plotlyjs="cdn"),
        "contribution_chart": plotly_io.to_html(contribution_chart, full_html=False, include_plotlyjs=False),
        # The raw samples stay out of the template context.
        **{key: value for key, value in result.items() if key not in ("sim_pnl", "sim_values")},
    }


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
        risk = None
        if simulate and data.positions:
            risk = await asyncio.to_thread(build_risk_view, data.positions)

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
        )
    else:
        await update_connection_status(is_connected=False)
        await log_audit_event("connection_failed", message)
        
        return render_dashboard(
            request,
            connected=False,
            error=message,
        )
