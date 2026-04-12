from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st


def main() -> None:
    st.title("Sonic Processing")
    st.write("Use this page to run sonic processing and prepare crossdipole outputs.")

    waveforms = st.session_state.get("waveforms", {})
    if not waveforms:
        st.info("Waveform data is not available in session state.")
        return

    if st.button("Run STC", key="run_stc_button"):
        waveforms = st.session_state["waveforms"]
        semblance = np.random.rand(100, 50)
        stc_result = {
            "status": "completed",
            "available_components": list(waveforms.keys()),
            "message": "STC mock semblance result stored in session state.",
        }

        if "crossdipole_results" not in st.session_state:
            st.session_state["crossdipole_results"] = {}

        st.session_state["crossdipole_results"]["stc"] = stc_result
        st.session_state["crossdipole_results"]["semblance"] = semblance
        st.success("STC computation completed and stored.")

    semblance = st.session_state.get("crossdipole_results", {}).get("semblance")
    if semblance is not None:
        figure = go.Figure(
            data=[
                go.Heatmap(
                    z=semblance,
                    colorscale="Viridis",
                    colorbar={"title": "Semblance"},
                )
            ]
        )
        figure.update_layout(
            title="STC Semblance (Mock)",
            xaxis_title="Time",
            yaxis_title="Slowness",
            template="plotly_white",
        )
        st.plotly_chart(figure, use_container_width=True)
