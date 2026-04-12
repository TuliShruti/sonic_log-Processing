from __future__ import annotations

from typing import Any

import numpy as np
import streamlit as st

from crossdipole.semblance import compute_semblance, merge_semblance_output
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


def _render_missing_data_message() -> None:
    st.info(
        "Crossdipole waveform data is not available in session state. "
        "Expected keys: 'crossdipole_data', 'crossdipole_dt', and optionally "
        "'crossdipole_results' and 'crossdipole_params'."
    )


def main() -> None:
    st.title("Crossdipole Processing")
    st.subheader("STC")
    st.write("Existing STC panel output should appear here.")

    data = st.session_state.get("crossdipole_data")
    dt = st.session_state.get("crossdipole_dt")

    if data is None or dt is None:
        _render_missing_data_message()
        return

    params = dict(_get_default_params())
    params.update(st.session_state.get("crossdipole_params", {}))
    params["run_semblance"] = st.sidebar.checkbox("Enable Semblance Plot", value=False)

    results = dict(st.session_state.get("crossdipole_results", {}))
    updated_results = dict(results)
    dt_microseconds = float(dt)
    params_items = tuple(sorted(params.items()))

    if params["run_semblance"]:
        st.subheader("Semblance")
        semblance_output = _cached_compute_semblance(np.asarray(data), dt_microseconds, params_items)
        updated_results["semblance"] = semblance_output
        st.session_state["crossdipole_results"] = updated_results
        st.plotly_chart(plot_semblance(semblance_output), use_container_width=True)
    else:
        st.session_state["crossdipole_results"] = updated_results

    st.session_state["crossdipole_params"] = params


main()
