from typing import Optional

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import gaussian_kde
from plotly.subplots import make_subplots


def build_stock_analysis_chart(
    price_history: pd.DataFrame,
    rsi_series: Optional[pd.Series],
    macd_series: Optional[pd.Series],
    signal_series: Optional[pd.Series],
    upper_band: Optional[pd.Series],
    middle_band: Optional[pd.Series],
    lower_band: Optional[pd.Series],
) -> go.Figure:
    figure = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.6, 0.2, 0.2],
    )

    if {"Open", "High", "Low", "Close"}.issubset(price_history.columns):
        figure.add_trace(
            go.Candlestick(
                x=price_history.index,
                open=price_history["Open"],
                high=price_history["High"],
                low=price_history["Low"],
                close=price_history["Close"],
                name="Price",
            ),
            row=1,
            col=1,
        )
    else:
        # Fallback to line chart with safe column selection
        if "Close" in price_history.columns:
            y_data = price_history["Close"]
        else:
            # Use the first numeric column if Close is missing
            numeric_cols = price_history.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                y_data = price_history[numeric_cols[0]]
            else:
                raise ValueError("No numeric columns found in price_history DataFrame")
        
        figure.add_trace(
            go.Scatter(
                x=price_history.index,
                y=y_data,
                mode="lines",
                name="Price",
            ),
            row=1,
            col=1,
        )

    if upper_band is not None and lower_band is not None and middle_band is not None:
        figure.add_trace(
            go.Scatter(
                x=price_history.index,
                y=upper_band,
                mode="lines",
                name="Bollinger Upper",
                line=dict(color="#94a3b8", width=1),
            ),
            row=1,
            col=1,
        )
        figure.add_trace(
            go.Scatter(
                x=price_history.index,
                y=middle_band,
                mode="lines",
                name="Bollinger Mid",
                line=dict(color="#cbd5f5", width=1),
            ),
            row=1,
            col=1,
        )
        figure.add_trace(
            go.Scatter(
                x=price_history.index,
                y=lower_band,
                mode="lines",
                name="Bollinger Lower",
                line=dict(color="#94a3b8", width=1),
            ),
            row=1,
            col=1,
        )

    if rsi_series is not None:
        figure.add_trace(
            go.Scatter(
                x=price_history.index,
                y=rsi_series,
                mode="lines",
                name="RSI",
            ),
            row=2,
            col=1,
        )
        figure.add_hline(y=70, row=2, col=1, line_dash="dash", line_color="#ef4444")
        figure.add_hline(y=30, row=2, col=1, line_dash="dash", line_color="#22c55e")

    if macd_series is not None and signal_series is not None:
        figure.add_trace(
            go.Scatter(
                x=price_history.index,
                y=macd_series,
                mode="lines",
                name="MACD",
            ),
            row=3,
            col=1,
        )
        figure.add_trace(
            go.Scatter(
                x=price_history.index,
                y=signal_series,
                mode="lines",
                name="Signal",
            ),
            row=3,
            col=1,
        )

    figure.update_layout(
        height=700,
        margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(orientation="h"),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return figure


# Colours for the dark theme both frontends use.
PROFIT, LOSS, INK, MUTED, GRID = "#3987e5", "#e66767", "#f0efec", "#898781", "#2c2c2a"


def build_distribution_chart(values, today, var_value, var_confidence, capital) -> go.Figure:
    """Density of the simulated portfolio values, split into loss and profit around today's value."""
    low, high = np.percentile(values, [0.1, 99.9])
    density = gaussian_kde(values[:20_000])  # samples are independent, so a slice is a fair subsample
    prob_loss = (values < today).mean()
    figure = go.Figure()
    for name, share, colour, fill, x in (
        ("Loss", prob_loss, LOSS, "rgba(230,103,103,0.35)", np.linspace(low, today, 150)),
        ("Profit", 1 - prob_loss, PROFIT, "rgba(57,135,229,0.35)", np.linspace(today, high, 150)),
    ):
        figure.add_trace(
            go.Scatter(
                x=x,
                y=density(x),
                customdata=(x - today) / capital,
                name=f"{name} ({share:.0%} chance)",
                mode="lines",
                line=dict(color=colour, width=2),
                fill="tozeroy",
                fillcolor=fill,
                hovertemplate="$%{x:,.0f} · %{customdata:+.1%} on your capital<extra></extra>",
            )
        )
    figure.add_vline(
        x=today,
        line=dict(color=INK, width=1),
        annotation=dict(text=f"Today ${today:,.0f}", font_color=INK),
        annotation_position="top right",
    )
    figure.add_vline(
        x=var_value,
        line=dict(color=MUTED, width=1, dash="dash"),
        annotation=dict(text=f"VaR {var_confidence}% ${var_value:,.0f}", font_color=INK),
        annotation_position="top left",
    )
    figure.update_layout(
        height=420,
        margin=dict(l=0, r=0, t=40, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=MUTED),
        hovermode="x",
        legend=dict(orientation="h", y=1.12, x=0, font_color=INK),
        xaxis=dict(tickprefix="$", tickformat=",.0f", gridcolor=GRID, zeroline=False),
        yaxis=dict(visible=False, rangemode="tozero"),
    )
    return figure


def build_pnl_simulation_chart(
    sim_pnl,
    var_95: float,
    var_995: float,
    tvar_995: float,
) -> go.Figure:
    figure = go.Figure()

    figure.add_trace(
        go.Histogram(
            x=sim_pnl,
            nbinsx=100,
            marker_color="#0ea5e9",
            opacity=0.75,
            name="Simulated P&L",
        )
    )

    figure.add_vline(
        x=0,
        line_width=1,
        line_color="black",
        opacity=0.4,
    )
    figure.add_vline(
        x=-var_95,
        line_dash="dash",
        line_color="#f59e0b",
        line_width=2,
        annotation_text=f"VaR 95%: -${var_95:,.0f}",
        annotation_position="top left",
    )
    figure.add_vline(
        x=-var_995,
        line_dash="dash",
        line_color="#ef4444",
        line_width=2,
        annotation_text=f"VaR 99.5%: -${var_995:,.0f}",
        annotation_position="top left",
        annotation_yshift=16,
    )
    figure.add_vline(
        x=-tvar_995,
        line_dash="dot",
        line_color="#991b1b",
        line_width=2,
        annotation_text=f"TVaR 99.5%: -${tvar_995:,.0f}",
        annotation_position="top left",
        annotation_yshift=32,
    )
    figure.add_vline(
        x=float(sim_pnl.mean()),
        line_dash="dash",
        line_color="#22c55e",
        line_width=2,
        annotation_text=f"Mean: ${float(sim_pnl.mean()):+,.0f}",
        annotation_position="top right",
    )

    figure.update_layout(
        xaxis_title="P&L ($)",
        yaxis_title="Frequency",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=420,
        margin=dict(l=20, r=20, t=60, b=20),
        showlegend=False,
    )

    return figure


def build_risk_contribution_chart(
    risk_contributions: list,
    metric: str = "component_var_995",
    title: str = "Contribution to VaR 99.5%",
) -> go.Figure:
    """
    Horizontal bar chart showing each position's contribution to a risk
    metric (component volatility, component VaR, or component TVaR).

    `risk_contributions` is the list of per-asset dicts returned by
    `simulate_portfolio_pnl`. `metric` picks which column to plot — e.g.
    `component_vol`, `component_var_95`, `component_var_99`,
    `component_var_995`, `component_tvar_995`.
    """
    if not risk_contributions:
        return go.Figure()

    sorted_rows = sorted(
        risk_contributions, key=lambda row: row.get(metric, 0.0), reverse=False
    )  # ascending so the largest bar lands at the top of the chart

    tickers = [row["ticker"] for row in sorted_rows]
    values = [row.get(metric, 0.0) for row in sorted_rows]
    pct_key = f"{metric}_pct"
    pcts = [row.get(pct_key, 0.0) for row in sorted_rows]
    weights = [row.get("weight", 0.0) for row in sorted_rows]

    # Color bars by sign so we can see hedging / diversification benefits
    # (negative contributions reduce total portfolio risk).
    colors = ["#ef4444" if v >= 0 else "#22c55e" for v in values]

    hover_text = [
        f"<b>{tk}</b><br>"
        f"Contribution: ${v:,.0f} ({p * 100:.1f}% of total)<br>"
        f"Position weight: {w * 100:.1f}%"
        for tk, v, p, w in zip(tickers, values, pcts, weights)
    ]

    figure = go.Figure(
        go.Bar(
            x=values,
            y=tickers,
            orientation="h",
            marker_color=colors,
            text=[f"${v:,.0f}" for v in values],
            textposition="auto",
            hovertext=hover_text,
            hoverinfo="text",
        )
    )

    figure.update_layout(
        title=title,
        xaxis_title="Contribution ($)",
        yaxis_title="",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=max(280, 40 * len(tickers) + 120),
        margin=dict(l=20, r=20, t=50, b=20),
        showlegend=False,
    )

    return figure
