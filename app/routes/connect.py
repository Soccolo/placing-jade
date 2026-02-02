"""
Connect Route

Handles Alpaca credential management and connection verification.
"""
from datetime import datetime
from fastapi import APIRouter, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import RedirectResponse
from starlette.concurrency import run_in_threadpool

from app.services.credentials import (
    save_credentials, 
    get_credentials, 
    update_connection_status,
    delete_credentials,
    log_audit_event
)
from app.services.alpaca import verify_connection

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")


@router.get("")
async def connect_page(request: Request):
    """Display the connection page."""
    creds = await get_credentials()
    
    return templates.TemplateResponse("connect.html", {
        "request": request,
        "is_connected": creds.is_connected if creds else False,
        "last_verified": creds.last_verified_at if creds else None,
        "message": None,
        "error": None
    })


@router.post("/save")
async def save_and_verify(
    request: Request,
    api_key: str = Form(...),
    api_secret: str = Form(...)
):
    """Save credentials and verify connection."""
    # Basic validation
    api_key = api_key.strip()
    api_secret = api_secret.strip()
    
    if not api_key or not api_secret:
        return templates.TemplateResponse("connect.html", {
            "request": request,
            "is_connected": False,
            "last_verified": None,
            "message": None,
            "error": "API key and secret are required"
        })
    
    # Verify connection before saving
    success, message = await run_in_threadpool(verify_connection, api_key, api_secret)
    
    if success:
        # Save encrypted credentials
        await save_credentials(api_key, api_secret)
        await update_connection_status(is_connected=True)
        await log_audit_event("connected", "Initial connection verified")
        
        creds = await get_credentials()
        
        return templates.TemplateResponse("connect.html", {
            "request": request,
            "is_connected": True,
            "last_verified": creds.last_verified_at if creds else datetime.utcnow(),
            "message": "Successfully connected to Alpaca paper trading",
            "error": None
        })
    else:
        await log_audit_event("connection_failed", message)
        
        return templates.TemplateResponse("connect.html", {
            "request": request,
            "is_connected": False,
            "last_verified": None,
            "message": None,
            "error": message
        })


@router.post("/verify")
async def verify_existing(request: Request):
    """Re-verify existing credentials."""
    creds = await get_credentials()
    
    if not creds:
        return templates.TemplateResponse("connect.html", {
            "request": request,
            "is_connected": False,
            "last_verified": None,
            "message": None,
            "error": "No credentials stored. Please enter your API key and secret."
        })
    
    success, message = await run_in_threadpool(verify_connection, creds.api_key, creds.api_secret)
    
    if success:
        await update_connection_status(is_connected=True)
        await log_audit_event("connected", "Re-verification successful")
        
        creds = await get_credentials()
        
        return templates.TemplateResponse("connect.html", {
            "request": request,
            "is_connected": True,
            "last_verified": creds.last_verified_at,
            "message": "Connection verified successfully",
            "error": None
        })
    else:
        await update_connection_status(is_connected=False)
        await log_audit_event("connection_failed", message)
        
        return templates.TemplateResponse("connect.html", {
            "request": request,
            "is_connected": False,
            "last_verified": None,
            "message": None,
            "error": message
        })


@router.post("/disconnect")
async def disconnect(request: Request):
    """Delete stored credentials."""
    await delete_credentials()
    await log_audit_event("disconnected", "Credentials deleted by user")
    
    return templates.TemplateResponse("connect.html", {
        "request": request,
        "is_connected": False,
        "last_verified": None,
        "message": "Disconnected and credentials deleted",
        "error": None
    })
