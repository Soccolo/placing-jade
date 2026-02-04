"""
Strategy Route

Displays the target portfolio loaded from CSV.
"""
from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates

from app.config import TARGET_PORTFOLIO_PATH, WEIGHT_SUM_EPSILON
from app.services.portfolio import get_portfolio, clear_portfolio_cache

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")

def render_strategy(request: Request, result, message: str | None = None):
    return templates.TemplateResponse(
        "strategy.html",
        {
            "request": request,
            "portfolio": result.entries,
            "is_valid": result.is_valid,
            "errors": result.errors,
            "warnings": result.warnings,
            "total_weight": result.total_weight,
            "file_path": TARGET_PORTFOLIO_PATH,
            "message": message,
        },
    )


@router.get("")
async def strategy_page(request: Request):
    """Display the target portfolio from CSV."""
    result = get_portfolio(TARGET_PORTFOLIO_PATH, WEIGHT_SUM_EPSILON)
    
    return render_strategy(request, result)


@router.post("/reload")
async def reload_portfolio(request: Request):
    """Force reload the portfolio from disk."""
    clear_portfolio_cache()
    result = get_portfolio(TARGET_PORTFOLIO_PATH, WEIGHT_SUM_EPSILON, force_reload=True)
    
    return render_strategy(request, result, message="Portfolio reloaded from disk")
