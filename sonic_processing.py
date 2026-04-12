from __future__ import annotations

import numpy as np
import pandas as pd


def _slowness_to_velocity(slowness: pd.Series, unit: str) -> pd.Series:
    numeric_slowness = pd.to_numeric(slowness, errors="coerce")
    safe_slowness = numeric_slowness.where(numeric_slowness > 0)

    if unit == "us/ft":
        return 0.3048e6 / safe_slowness

    if unit == "us/m":
        return 1.0e6 / safe_slowness

    raise ValueError("Unsupported slowness unit. Use 'us/ft' or 'us/m'.")


def compute_velocity(
    df: pd.DataFrame,
    dt_col: str,
    unit: str,
    shear_dt_col: str | None = None,
    shear_unit: str | None = None,
    vp_output_col: str = "Vp",
    vs_output_col: str = "Vs",
) -> pd.DataFrame:
    if dt_col not in df.columns:
        raise ValueError(f"{dt_col} not found in DataFrame")

    if shear_dt_col is not None and shear_dt_col not in df.columns:
        raise ValueError(f"{shear_dt_col} not found in DataFrame")

    if shear_dt_col is not None and shear_unit is None:
        raise ValueError("shear_unit must be provided when shear_dt_col is used")

    result = df.copy()
    result[vp_output_col] = _slowness_to_velocity(result[dt_col], unit)

    if shear_dt_col is not None:
        result[vs_output_col] = _slowness_to_velocity(result[shear_dt_col], shear_unit)

    return result


def clean_sonic(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    min_value: float | None = None,
    max_value: float | None = None,
) -> pd.DataFrame:
    if columns is None:
        raise ValueError("Specify columns to clean explicitly")

    result = df.copy()
    target_columns = columns

    for column in target_columns:
        if column not in result.columns:
            continue

        numeric_values = pd.to_numeric(result[column], errors="coerce")

        if min_value is not None:
            numeric_values = numeric_values.where(numeric_values >= min_value)

        if max_value is not None:
            numeric_values = numeric_values.where(numeric_values <= max_value)

        result[column] = numeric_values

    return result
