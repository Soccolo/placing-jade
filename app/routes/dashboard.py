"""
Dashboard Route

Displays account summary and positions from Alpaca.
"""
import asyncio
from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates

from app.services.credentials import get_credentials, update_connection_status, log_audit_event
from app.services.alpaca import fetch_account_data

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")

def render_dashboard(
    request: Request,
    connected: bool,
    account=None,
    positions=None,
    fetched_at=None,
    error: str | None = None,
    message: str | None = None,
):
    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "connected": connected,
            "account": account,
            "positions": positions or [],
            "fetched_at": fetched_at,
            "error": error,
            "message": message,
        },
    )


@router.get("")
async def dashboard_page(request: Request):
    """Display the dashboard with account info and positions."""
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
        
        return render_dashboard(
            request,
            connected=True,
            account=data.account,
            positions=data.positions,
            fetched_at=data.fetched_at,
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
