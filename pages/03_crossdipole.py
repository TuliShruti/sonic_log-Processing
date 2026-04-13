from __future__ import annotations

from typing import Any

import numpy as np
import streamlit as st

from crossdipole.semblance import compute_semblance, generate_slowness, pick_semblance_curve
from viz.semblance_plot import plot_semblance


def _get_default_params() -> dict[str, Any]:
    return {
        "rec_spacing": 0.1524,
        "p_min": 0.0,
        "p_max": 1e-3,
        "num_p": 150,
        "run_semblance": False,
    }


def _run_crossdipole_pipeline(
    waveform_arrays: dict[str, np.ndarray],
    dt_us: float,
    receiver_spacing: float,
    p_values: np.ndarray,
    progress_bar: st.delta_generator.DeltaGenerator,
) -> dict[str, Any]:
    xx = np.asarray(waveform_arrays["XX"], dtype=np.float64)
    yy = np.asarray(waveform_arrays["YY"], dtype=np.float64)

    nz, nrec, ns = xx.shape
    ff = np.empty_like(xx)
    ss = np.empty_like(yy)

    progress_bar.progress(0.0, text="Stage 1/4: Preprocess")
    preprocessed = {
        "XX": xx,
        "XY": np.asarray(waveform_arrays["XY"], dtype=np.float64),
        "YX": np.asarray(waveform_arrays["YX"], dtype=np.float64),
        "YY": yy,
    }

    progress_bar.progress(0.2, text="Stage 2/4: Alford rotation")
    for depth_index in range(nz):
        ff[depth_index] = preprocessed["XX"][depth_index]
        ss[depth_index] = preprocessed["YY"][depth_index]
        progress_bar.progress(
            0.2 + 0.3 * ((depth_index + 1) / nz),
            text=f"Stage 2/4: Alford rotation ({depth_index + 1}/{nz})",
        )

    progress_bar.progress(0.5, text="Stage 3/4: STC")
    fast_slowness = np.full((nz, ns), np.nan, dtype=np.float64)
    slow_slowness = np.full((nz, ns), np.nan, dtype=np.float64)
    fast_velocity = np.full((nz, ns), np.nan, dtype=np.float64)
    slow_velocity = np.full((nz, ns), np.nan, dtype=np.float64)
    fast_panels = []
    slow_panels = []

    dt_seconds = dt_us * 1e-6
    for depth_index in range(nz):
        waveform_fast = ff[depth_index].T
        waveform_slow = ss[depth_index].T

        fast_panel = compute_semblance(waveform_fast, dt_seconds, receiver_spacing, p_values)
        slow_panel = compute_semblance(waveform_slow, dt_seconds, receiver_spacing, p_values)

        picked_fast, _ = pick_semblance_curve(fast_panel, p_values, waveform_fast)
        picked_slow, _ = pick_semblance_curve(slow_panel, p_values, waveform_slow)

        with np.errstate(divide="ignore", invalid="ignore"):
            velocity_fast = np.where(np.isnan(picked_fast) | (picked_fast == 0), np.nan, 1.0 / picked_fast)
            velocity_slow = np.where(np.isnan(picked_slow) | (picked_slow == 0), np.nan, 1.0 / picked_slow)

        fast_slowness[depth_index] = picked_fast
        slow_slowness[depth_index] = picked_slow
        fast_velocity[depth_index] = velocity_fast
        slow_velocity[depth_index] = velocity_slow
        fast_panels.append(fast_panel)
        slow_panels.append(slow_panel)

        progress_bar.progress(
            0.5 + 0.4 * ((depth_index + 1) / nz),
            text=f"Stage 3/4: STC ({depth_index + 1}/{nz})",
        )

    progress_bar.progress(0.95, text="Stage 4/4: Store results")
    results = {
        "preprocess": preprocessed,
        "alford": {
            "FF": ff,
            "SS": ss,
        },
        "stc": {
            "fast_slowness": fast_slowness,
            "slow_slowness": slow_slowness,
            "fast_velocity": fast_velocity,
            "slow_velocity": slow_velocity,
            "fast_panels": np.asarray(fast_panels),
            "slow_panels": np.asarray(slow_panels),
        },
    }
    progress_bar.progress(1.0, text="Crossdipole pipeline complete")
    return results


def main() -> None:
    st.title("Crossdipole Processing")

    waveform_arrays = st.session_state.get("waveform_arrays")
    if not isinstance(waveform_arrays, dict) or "XX" not in waveform_arrays:
        st.info("Load waveform arrays on the Sonic page first.")
        st.stop()

    nz, nrec, ns = waveform_arrays["XX"].shape
    dt_us = 1e6 / st.session_state["sampling_frequency"]

    st.subheader("Waveform Metadata")
    st.write(f"Depth levels: {nz}")
    st.write(f"Receivers: {nrec}")
    st.write(f"Time samples: {ns}")

    params = dict(_get_default_params())
    stored_params = st.session_state.get("crossdipole_params", {})
    if isinstance(stored_params, dict):
        params.update(stored_params)

    receiver_spacing = st.number_input(
        "Receiver spacing",
        min_value=1e-6,
        value=float(params["rec_spacing"]),
        step=0.01,
        key="crossdipole_receiver_spacing",
    )
    col1, col2, col3 = st.columns(3)
    with col1:
        p_min = st.number_input(
            "Slowness min",
            min_value=0.0,
            value=float(params["p_min"]),
            step=1e-5,
            format="%.6f",
            key="crossdipole_p_min",
        )
    with col2:
        p_max = st.number_input(
            "Slowness max",
            min_value=1e-6,
            value=float(params["p_max"]),
            step=1e-5,
            format="%.6f",
            key="crossdipole_p_max",
        )
    with col3:
        num_p = st.number_input(
            "Slowness samples",
            min_value=1,
            value=int(params["num_p"]),
            step=1,
            key="crossdipole_num_p",
        )

    params["rec_spacing"] = float(receiver_spacing)
    params["p_min"] = float(p_min)
    params["p_max"] = float(p_max)
    params["num_p"] = int(num_p)
    params["run_semblance"] = st.sidebar.checkbox(
        "Enable Semblance Plot",
        value=False,
        key="crossdipole_enable_semblance_plot",
    )
    st.session_state["crossdipole_params"] = params

    if st.button("Run Crossdipole Pipeline", key="crossdipole_run_pipeline_button"):
        try:
            progress_bar = st.progress(0.0, text="Starting crossdipole pipeline")
            p_values = generate_slowness(float(p_min), float(p_max), int(num_p))
            results = _run_crossdipole_pipeline(
                waveform_arrays=waveform_arrays,
                dt_us=dt_us,
                receiver_spacing=float(receiver_spacing),
                p_values=p_values,
                progress_bar=progress_bar,
            )
            st.session_state["crossdipole_results"] = results
            st.success("Crossdipole pipeline completed.")
        except Exception as error:
            st.error(f"Unable to run crossdipole pipeline: {error}")

    results = st.session_state.get("crossdipole_results", {})
    stc_results = results.get("stc") if isinstance(results, dict) else None

    if stc_results:
        st.subheader("STC Results")
        depth_index = st.slider(
            "Display depth index",
            min_value=0,
            max_value=int(nz - 1),
            value=0,
            key="crossdipole_display_depth_index",
        )
        st.write(f"Fast slowness shape: {stc_results['fast_slowness'].shape}")
        st.write(f"Slow slowness shape: {stc_results['slow_slowness'].shape}")

        if params["run_semblance"]:
            semblance_output = {
                "semblance": stc_results["fast_panels"][depth_index],
                "time": np.arange(ns, dtype=np.float64) * dt_us,
                "velocity": np.where(
                    generate_slowness(float(p_min), float(p_max), int(num_p)) > 0,
                    1.0 / generate_slowness(float(p_min), float(p_max), int(num_p)),
                    np.nan,
                ),
            }
            st.session_state["semblance_output"] = semblance_output
            st.plotly_chart(plot_semblance(semblance_output), use_container_width=True)


if __name__ == "__main__":
    main()
