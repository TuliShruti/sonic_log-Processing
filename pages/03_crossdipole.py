from __future__ import annotations

from typing import Any

import numpy as np
import streamlit as st

from crossdipole.semblance import compute_semblance
from viz.semblance_plot import plot_semblance


@st.cache_data(show_spinner=False)
def _cached_compute_semblance(
    data: np.ndarray,
    dt: float,
    params_items: tuple[tuple[str, Any], ...],
) -> dict[str, np.ndarray]:
    params = dict(params_items)
    return compute_semblance(data, dt, params)


def _get_default_params() -> dict[str, Any]:
    return {
        "rec_spacing": 0.1524,
        "s_min": 300.0,
        "s_max": 2500.0,
        "s_step": 4.0,
        "win": 40,
        "run_semblance": False,
    }


def main() -> None:
    st.title("Crossdipole Processing")
    st.subheader("STC")

    waveforms = st.session_state.get("waveforms", {})
    required = ["XX", "XY", "YX", "YY"]
    if not all(key in waveforms for key in required):
        st.error("Crossdipole waveform data is not available.")
        return

    results = dict(st.session_state.get("crossdipole_results", {}))
    if "stc" in results:
        st.success("STC results available.")
        st.write(results["stc"])
    else:
        st.info("Run STC from the Sonic page to populate crossdipole results.")

    params = dict(_get_default_params())
    params.update(st.session_state.get("crossdipole_params", {}))
    params["run_semblance"] = st.sidebar.checkbox("Enable Semblance Plot", value=False)
    st.session_state["crossdipole_params"] = params

    if params["run_semblance"]:
        st.subheader("Semblance")
        dt = float(st.session_state.get("crossdipole_dt", 1.0))
        data = np.asarray(waveforms["XX"], dtype=np.float64)
        if data.ndim == 3:
            data = data[0]
        params_items = tuple(sorted(params.items()))
        semblance_output = _cached_compute_semblance(data, dt, params_items)

        updated_results = dict(results)
        updated_results["semblance"] = semblance_output
        st.session_state["crossdipole_results"] = updated_results
        st.session_state["semblance_output"] = semblance_output

        st.plotly_chart(plot_semblance(semblance_output), use_container_width=True)
