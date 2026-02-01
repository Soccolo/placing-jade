"""
Notrix - Main Application Entry Point
"""
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from contextlib import asynccontextmanager

from app.database import init_db
from app.routes import connect, dashboard, strategy

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler - runs on startup and shutdown."""
    # Startup: Initialize database
    await init_db()
    yield
    # Shutdown: nothing needed for SQLite

# Create FastAPI application
app = FastAPI(
    title="Notrix",
    description="A bridge between portfolio recipes and brokerage accounts",
    version="1.0.0",
    lifespan=lifespan
)

# Mount static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Setup templates
templates = Jinja2Templates(directory="app/templates")

# Include routers
app.include_router(connect.router, prefix="/connect", tags=["connect"])
app.include_router(dashboard.router, prefix="/dashboard", tags=["dashboard"])
app.include_router(strategy.router, prefix="/strategy", tags=["strategy"])

@app.get("/")
async def root(request: Request):
    """Redirect to dashboard or connect page based on connection status."""
    from app.services.credentials import get_credentials
    
    creds = await get_credentials()
    if creds and creds.is_connected:
        return templates.TemplateResponse(
            "redirect.html", 
            {"request": request, "url": "/dashboard"}
        )
    return templates.TemplateResponse(
        "redirect.html", 
        {"request": request, "url": "/connect"}
    )

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "app": "notrix"}
