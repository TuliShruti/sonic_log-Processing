from __future__ import annotations

from typing import Any

import pandas as pd


def check_schema(dataframe: pd.DataFrame, required_columns: list[str]) -> dict[str, Any]:
    missing_columns = [column for column in required_columns if column not in dataframe.columns]
    return {
        "valid": len(missing_columns) == 0,
        "missing_columns": missing_columns,
        "available_columns": list(dataframe.columns),
    }


def check_nans(dataframe: pd.DataFrame, columns: list[str] | None = None) -> dict[str, int]:
    target_columns = columns or list(dataframe.columns)
    valid_columns = [column for column in target_columns if column in dataframe.columns]
    return dataframe[valid_columns].isna().sum().to_dict()


def ensure_depth_monotonicity(dataframe: pd.DataFrame, depth_column: str = "DEPTH") -> dict[str, Any]:
    if depth_column not in dataframe.columns:
        return {
            "valid": False,
            "reason": f"Depth column '{depth_column}' not found.",
        }

    depth_series = pd.to_numeric(dataframe[depth_column], errors="coerce")

    if depth_series.isna().any():
        return {
            "valid": False,
            "reason": f"Depth column '{depth_column}' contains non-numeric or missing values.",
        }

    is_monotonic = depth_series.is_monotonic_increasing or depth_series.is_monotonic_decreasing
    return {
        "valid": is_monotonic,
        "reason": None if is_monotonic else f"Depth column '{depth_column}' is not monotonic.",
    }


def validate_dataframe(
    dataframe: pd.DataFrame,
    required_columns: list[str],
    depth_column: str = "DEPTH",
    nan_columns: list[str] | None = None,
) -> dict[str, Any]:
    schema_result = check_schema(dataframe, required_columns)
    nan_result = check_nans(dataframe, nan_columns)
    depth_result = ensure_depth_monotonicity(dataframe, depth_column)

    return {
        "valid": schema_result["valid"] and depth_result["valid"],
        "schema": schema_result,
        "nans": nan_result,
        "depth": depth_result,
    }
