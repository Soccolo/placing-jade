"""
Dashboard overview figures shared by the FastAPI site and the Streamlit app.
"""
from app.config import RISK_BUDGET_PCT, TARGET_PORTFOLIO_PATH, WEIGHT_SUM_EPSILON
from app.services.portfolio import get_portfolio


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


def risk_budget_used(var_995: float, portfolio_value: float) -> float:
    """One-day VaR 99.5% as a percentage of the loss limit (RISK_BUDGET_PCT of portfolio value)."""
    budget = RISK_BUDGET_PCT / 100 * portfolio_value
    return var_995 / budget * 100 if budget > 0 else 0.0
