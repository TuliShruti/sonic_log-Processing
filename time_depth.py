from __future__ import annotations

import pandas as pd


def time_to_depth(
    df: pd.DataFrame,
    velocity_col: str,
    dt: float,
    depth_output_col: str = "Depth",
) -> pd.DataFrame:
    """
    Convert velocity to cumulative depth.

    Parameters
    ----------
    dt : float
        Sampling interval in seconds.
    """
    if velocity_col not in df.columns:
        raise ValueError(f"{velocity_col} not found in DataFrame")

    if dt <= 0:
        raise ValueError("dt must be positive and in seconds")

    result = df.copy()
    velocity = pd.to_numeric(result[velocity_col], errors="coerce")
    depth_increment = velocity.fillna(0) * dt
    result[depth_output_col] = depth_increment.cumsum()
    return result
