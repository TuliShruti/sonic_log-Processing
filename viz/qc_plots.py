from __future__ import annotations

import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go


def plot_qc(df: pd.DataFrame, columns: list[str]) -> go.Figure:
    if not columns:
        raise ValueError("columns must contain at least one column")

    missing_columns = [column for column in columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")

    figure = make_subplots(
        rows=1,
        cols=len(columns),
        horizontal_spacing=0.06,
        subplot_titles=columns,
    )

    for index, column in enumerate(columns, start=1):
        clean_series = pd.to_numeric(df[column], errors="coerce").dropna()
        figure.add_trace(
            go.Histogram(
                x=clean_series,
                nbinsx=30,
                name=column,
                showlegend=False,
            ),
            row=1,
            col=index,
        )
        figure.update_xaxes(title_text=column, row=1, col=index)
        figure.update_yaxes(title_text="Count", row=1, col=index)

    figure.update_layout(
        title="QC Plots",
        template="plotly_white",
        bargap=0.1,
    )

    return figure
