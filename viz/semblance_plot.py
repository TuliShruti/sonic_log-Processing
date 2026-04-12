from __future__ import annotations

from typing import Any

import plotly.graph_objects as go


def plot_semblance(semblance_dict: dict[str, Any]) -> go.Figure:
    semblance = semblance_dict["semblance"]
    time_axis = semblance_dict["time"]
    velocity_axis = semblance_dict["velocity"]

    figure = go.Figure(
        data=[
            go.Heatmap(
                x=velocity_axis,
                y=time_axis,
                z=semblance.T,
                colorscale="Viridis",
                colorbar={"title": "Semblance"},
                hovertemplate="Velocity=%{x:.2f} m/s<br>Time=%{y:.2f}<br>Semblance=%{z:.3f}<extra></extra>",
            )
        ]
    )

    figure.update_layout(
        title="Semblance Panel",
        xaxis_title="Velocity (m/s)",
        yaxis_title="Time (µs)",
        template="plotly_white",
    )

    return figure
