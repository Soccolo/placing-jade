from typing import Optional

import pandas as pd
from fastapi import UploadFile


def parse_portfolio_text(portfolio_text: str) -> Optional[pd.DataFrame]:
    if not portfolio_text:
        return None
    lines = [line.strip() for line in portfolio_text.splitlines() if line.strip()]
    if not lines:
        return None

    header = lines[0].lower()
    has_header = "stocks" in header and "value" in header
    if has_header:
        lines = lines[1:]

    rows = []
    for line in lines:
        parts = [part.strip() for part in line.split(",") if part.strip()]
        if len(parts) < 2:
            continue
        ticker_symbol = parts[0].upper()
        try:
            value = float(parts[1])
        except ValueError:
            continue
        unleveraged_value = value
        if len(parts) >= 3:
            try:
                unleveraged_value = float(parts[2])
            except ValueError:
                unleveraged_value = value
        rows.append(
            {
                "Stocks": ticker_symbol,
                "Value": value,
                "Unleveraged Value": unleveraged_value,
            }
        )
    if not rows:
        return None
    return pd.DataFrame(rows)


def load_portfolio_from_upload(uploaded_file: UploadFile) -> Optional[pd.DataFrame]:
    if uploaded_file is None:
        return None
    try:
        dataframe = pd.read_excel(uploaded_file.file)
    except Exception:
        return None

    required_columns = {"Stocks", "Value"}
    if not required_columns.issubset(set(dataframe.columns)):
        return None

    if "Unleveraged Value" not in dataframe.columns:
        dataframe["Unleveraged Value"] = dataframe["Value"]

    dataframe = dataframe.dropna(subset=["Stocks", "Value"])
    dataframe["Stocks"] = dataframe["Stocks"].astype(str).str.upper().str.strip()
    dataframe = dataframe[dataframe["Stocks"] != ""]
    dataframe = dataframe[dataframe["Value"] > 0]
    return dataframe.reset_index(drop=True)


def normalize_portfolio_dataframe(dataframe: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if dataframe is None or dataframe.empty:
        return None
    if "Unleveraged Value" not in dataframe.columns:
        dataframe["Unleveraged Value"] = dataframe["Value"]
    dataframe["Unleveraged Value"] = dataframe["Unleveraged Value"].fillna(
        dataframe["Value"]
    )
    return dataframe.reset_index(drop=True)
