"""
Dashboard Route

Displays account summary and positions from Alpaca.
"""
from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates

from app.services.credentials import get_credentials, update_connection_status, log_audit_event
from app.services.alpaca import fetch_account_data

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")


@router.get("")
async def dashboard_page(request: Request):
    """Display the dashboard with account info and positions."""
    creds = await get_credentials()
    
    if not creds:
        return templates.TemplateResponse("dashboard.html", {
            "request": request,
            "connected": False,
            "account": None,
            "positions": [],
            "fetched_at": None,
            "error": "Not connected. Please connect your Alpaca account first.",
            "message": None
        })
    
    if not creds.is_connected:
        return templates.TemplateResponse("dashboard.html", {
            "request": request,
            "connected": False,
            "account": None,
            "positions": [],
            "fetched_at": None,
            "error": "Connection not verified. Please verify your connection.",
            "message": None
        })
    
    # Fetch account data
    data, message = fetch_account_data(creds.api_key, creds.api_secret)
    
    if data:
        await log_audit_event("refreshed", f"Fetched {len(data.positions)} positions")
        
        return templates.TemplateResponse("dashboard.html", {
            "request": request,
            "connected": True,
            "account": data.account,
            "positions": data.positions,
            "fetched_at": data.fetched_at,
            "error": None,
            "message": None
        })
    else:
        await update_connection_status(is_connected=False)
        await log_audit_event("connection_failed", message)
        
        return templates.TemplateResponse("dashboard.html", {
            "request": request,
            "connected": False,
            "account": None,
            "positions": [],
            "fetched_at": None,
            "error": message,
            "message": None
        })


@router.post("/refresh")
async def refresh_dashboard(request: Request):
    """Manually refresh account data."""
    creds = await get_credentials()
    
    if not creds or not creds.is_connected:
        return templates.TemplateResponse("dashboard.html", {
            "request": request,
            "connected": False,
            "account": None,
            "positions": [],
            "fetched_at": None,
            "error": "Not connected. Please connect your Alpaca account first.",
            "message": None
        })
    
    # Fetch fresh data
    data, message = fetch_account_data(creds.api_key, creds.api_secret)
    
    if data:
        await update_connection_status(is_connected=True)
        await log_audit_event("refreshed", f"Manual refresh - {len(data.positions)} positions")
        
        return templates.TemplateResponse("dashboard.html", {
            "request": request,
            "connected": True,
            "account": data.account,
            "positions": data.positions,
            "fetched_at": data.fetched_at,
            "error": None,
            "message": "Data refreshed successfully"
        })
    else:
        await update_connection_status(is_connected=False)
        await log_audit_event("connection_failed", message)
        
        return templates.TemplateResponse("dashboard.html", {
            "request": request,
            "connected": False,
            "account": None,
            "positions": [],
            "fetched_at": None,
            "error": message,
            "message": None
        })
