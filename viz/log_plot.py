from __future__ import annotations

import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go


def plot_logs(df: pd.DataFrame, depth_col: str, log_columns: list[str]) -> go.Figure:
    if depth_col not in df.columns:
        raise ValueError(f"{depth_col} not found in DataFrame")

    if not log_columns:
        raise ValueError("log_columns must contain at least one column")

    missing_columns = [column for column in log_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing log columns: {missing_columns}")

    figure = make_subplots(
        rows=1,
        cols=len(log_columns),
        shared_yaxes=True,
        horizontal_spacing=0.03,
        subplot_titles=log_columns,
    )

    depth_values = df[depth_col]

    for index, column in enumerate(log_columns, start=1):
        figure.add_trace(
            go.Scatter(
                x=df[column],
                y=depth_values,
                mode="lines",
                name=column,
                showlegend=False,
            ),
            row=1,
            col=index,
        )
        figure.update_xaxes(title_text=column, row=1, col=index)

    figure.update_yaxes(title_text=depth_col, autorange="reversed", row=1, col=1)
    figure.update_layout(
        title="Log Tracks",
        template="plotly_white",
        height=800,
    )

    return figure
