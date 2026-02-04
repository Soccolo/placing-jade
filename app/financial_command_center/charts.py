from typing import Optional

import pandas as pd
import plotly.graph_objects as go
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
        template="plotly_white",
    )
    return figure


def build_distribution_chart(
    x_values,
    pdf_values,
    current_value,
    expected_value,
    var_value,
    var_confidence,
) -> go.Figure:
    figure = go.Figure()

    figure.add_trace(
        go.Scatter(
            x=x_values,
            y=pdf_values,
            mode="lines",
            name="Implied PDF",
            line=dict(color="#0ea5e9", width=3),
            fill="tozeroy",
            fillcolor="rgba(14, 165, 233, 0.15)",
        )
    )

    figure.add_vline(
        x=current_value,
        line_dash="dash",
        line_color="#ef4444",
        line_width=2,
        annotation_text=f"Current: ${current_value:,.2f}",
    )
    figure.add_vline(
        x=expected_value,
        line_dash="dash",
        line_color="#22c55e",
        line_width=2,
        annotation_text=f"Expected: ${expected_value:,.2f}",
    )
    figure.add_vline(
        x=var_value,
        line_dash="dot",
        line_color="#f59e0b",
        line_width=2,
        annotation_text=f"VaR {var_confidence}%: ${var_value:,.2f}",
    )

    figure.update_layout(
        title="Implied Probability Density",
        xaxis_title="Portfolio Value ($)",
        yaxis_title="Probability Density",
        template="plotly_white",
        height=500,
    )

    return figure
