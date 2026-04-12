from __future__ import annotations

import numpy as np
import plotly.graph_objects as go


def plot_wiggle(data: np.ndarray, time: np.ndarray, scale: float = 1.0) -> go.Figure:
    traces = np.asarray(data, dtype=float)
    time_axis = np.asarray(time, dtype=float)

    if traces.ndim != 2:
        raise ValueError("data must have shape (n_traces, n_samples)")

    if time_axis.ndim != 1:
        raise ValueError("time must be a 1D array")

    if traces.shape[1] != time_axis.size:
        raise ValueError("time length must match the number of samples in data")

    if scale <= 0:
        raise ValueError("scale must be positive")

    figure = go.Figure()
    max_amplitude = np.nanmax(np.abs(traces))
    normalization = max_amplitude if max_amplitude > 0 else 1.0

    for index, trace in enumerate(traces):
        normalized_trace = (trace / normalization) * scale
        x_values = index + normalized_trace
        figure.add_trace(
            go.Scatter(
                x=x_values,
                y=time_axis,
                mode="lines",
                line={"width": 1},
                name=f"Trace {index + 1}",
                showlegend=False,
            )
        )

    figure.update_layout(
        title="Wiggle Plot",
        xaxis_title="Trace Offset",
        yaxis_title="Time",
        template="plotly_white",
    )
    figure.update_yaxes(autorange="reversed")

    return figure
