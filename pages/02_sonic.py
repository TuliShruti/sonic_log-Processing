from __future__ import annotations

from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from scipy.signal import butter, filtfilt

from binary_loader import load_ldeo_binary
from crossdipole.semblance import compute_semblance, find_first_arrival, generate_slowness, pick_semblance_curve


def _init_state() -> None:
    if "waveforms" not in st.session_state:
        st.session_state["waveforms"] = {}
    if "waveform_arrays" not in st.session_state:
        st.session_state["waveform_arrays"] = {}
    if "filtered_waveforms" not in st.session_state:
        st.session_state["filtered_waveforms"] = {}
    if "sonic_data" not in st.session_state:
        st.session_state["sonic_data"] = None
    if "depth_levels" not in st.session_state:
        st.session_state["depth_levels"] = None
    if "time_samples" not in st.session_state:
        st.session_state["time_samples"] = None
    if "receivers" not in st.session_state:
        st.session_state["receivers"] = None
    if "semblance" not in st.session_state:
        st.session_state["semblance"] = None
    if "p_values" not in st.session_state:
        st.session_state["p_values"] = None
    if "picked_p" not in st.session_state:
        st.session_state["picked_p"] = None
    if "velocity" not in st.session_state:
        st.session_state["velocity"] = None
    if "first_arrival_idx" not in st.session_state:
        st.session_state["first_arrival_idx"] = None


def _extract_file_bytes(entry: object) -> bytes:
    if isinstance(entry, dict) and "bytes" in entry:
        return entry["bytes"]
    if isinstance(entry, bytes):
        return entry
    raise ValueError("Waveform entry does not contain raw bytes.")


def bandpass(data: np.ndarray, lowcut: float, highcut: float, fs: float) -> np.ndarray:
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist

    if not 0 < low < high < 1:
        raise ValueError("Bandpass frequencies must satisfy 0 < lowcut < highcut < fs/2.")

    b, a = butter(4, [low, high], btype="band")
    return filtfilt(b, a, data, axis=-1)


def _plot_filter_comparison(raw_trace: np.ndarray, filtered_trace: np.ndarray) -> go.Figure:
    sample_axis = np.arange(raw_trace.shape[0])
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=sample_axis, y=raw_trace, mode="lines", name="Raw"))
    figure.add_trace(go.Scatter(x=sample_axis, y=filtered_trace, mode="lines", name="Filtered"))
    figure.update_layout(
        title="XX Trace Before and After Filtering",
        xaxis_title="Sample",
        yaxis_title="Amplitude",
        template="plotly_white",
    )
    return figure


def _plot_semblance_panel(semblance: np.ndarray, p_values: np.ndarray, dt: float, picked_p: np.ndarray) -> plt.Figure:
    ns = semblance.shape[1]
    time_axis = np.arange(ns, dtype=np.float64) * dt

    figure, axis = plt.subplots(figsize=(10, 5))
    image = axis.imshow(
        semblance,
        aspect="auto",
        origin="lower",
        extent=[time_axis[0], time_axis[-1], float(p_values[0]), float(p_values[-1])],
        cmap="viridis",
    )

    valid_mask = ~np.isnan(picked_p)
    axis.plot(time_axis[valid_mask], picked_p[valid_mask], "w-", linewidth=1.5, label="Picked Slowness")

    axis.set_title("Semblance Panel")
    axis.set_xlabel("Time (s)")
    axis.set_ylabel("Slowness (s/m)")
    axis.legend(loc="upper right")
    figure.colorbar(image, ax=axis, label="Semblance")
    figure.tight_layout()
    return figure


def _plot_velocity_curve(velocity: np.ndarray, dt: float, first_arrival_idx: int) -> plt.Figure:
    time_axis = np.arange(velocity.shape[0], dtype=np.float64) * dt
    valid_mask = ~np.isnan(velocity)

    figure, axis = plt.subplots(figsize=(10, 4))
    axis.plot(time_axis[valid_mask], velocity[valid_mask], "b-", linewidth=1.2)
    axis.set_title("Velocity vs Time")
    axis.set_xlabel("Time (s)")
    axis.set_ylabel("Velocity (m/s)")
    axis.set_xlim(left=first_arrival_idx * dt)
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    return figure


def _load_component_volume(file_bytes: bytes) -> tuple[np.ndarray, dict[str, int | float | str]]:
    data, metadata = load_ldeo_binary(BytesIO(file_bytes))
    data = np.asarray(data, dtype=np.float32)

    if data.ndim != 3:
        raise ValueError("Loaded waveform data must have shape (depth_levels, receivers, time_samples).")

    depth_levels, receivers, time_samples = data.shape
    assert data.shape == (depth_levels, receivers, time_samples)

    if data.size != depth_levels * receivers * time_samples:
        raise ValueError("Binary data size mismatch with header metadata")

    return data, metadata


def main() -> None:
    _init_state()
    st.title("Sonic Processing")
    st.write("Load waveform volumes and apply preprocessing on a selected depth slice.")

    waveforms = st.session_state.get("waveforms", {})
    required = ["XX", "XY", "YX", "YY"]
    if not all(key in waveforms and waveforms[key] is not None for key in required):
        st.error("Crossdipole waveform data is not available.")
        return

    st.caption("Waveforms loaded from session state. Ready for binary parsing.")

    sampling_frequency = st.number_input(
        "Sampling frequency (Hz)",
        min_value=1.0,
        value=25000.0,
        step=100.0,
        key="sonic_sampling_frequency",
    )

    if st.button("Load Waveform Arrays", key="load_waveform_arrays_button"):
        try:
            arrays: dict[str, np.ndarray] = {}
            metadata_by_component: dict[str, dict[str, int | float | str]] = {}

            for component in required:
                data, metadata = _load_component_volume(_extract_file_bytes(waveforms[component]))
                arrays[component] = data
                metadata_by_component[component] = metadata

            if len(set(array.shape for array in arrays.values())) != 1:
                raise ValueError(
                    f"Waveform components do not share the same shape: "
                    f"{ {component: array.shape for component, array in arrays.items()} }"
                )

            if len(
                {
                    (
                        meta["depth_levels"],
                        meta["receivers"],
                        meta["time_samples"],
                    )
                    for meta in metadata_by_component.values()
                }
            ) != 1:
                raise ValueError("Waveform metadata mismatch across components.")

            first_component = required[0]
            st.session_state["waveform_arrays"] = arrays
            st.session_state["sonic_data"] = arrays[first_component]
            st.session_state["depth_levels"] = int(metadata_by_component[first_component]["depth_levels"])
            st.session_state["receivers"] = int(metadata_by_component[first_component]["receivers"])
            st.session_state["time_samples"] = int(metadata_by_component[first_component]["time_samples"])
            st.session_state["crossdipole_dt"] = float(metadata_by_component[first_component]["dt"])
            st.session_state["sampling_frequency"] = float(sampling_frequency)
            st.success("Waveform arrays loaded and validated.")
        except Exception as error:
            st.error(f"Unable to load waveform arrays: {error}")

    waveform_arrays = st.session_state.get("waveform_arrays", {})
    if waveform_arrays:
        st.subheader("Waveform Array Shapes")
        for component in required:
            if component in waveform_arrays:
                st.write(f"{component}: {waveform_arrays[component].shape}")

        st.subheader("Binary Metadata")
        st.write(f"Depth levels: {st.session_state['depth_levels']}")
        st.write(f"Receivers: {st.session_state['receivers']}")
        st.write(f"Time samples: {st.session_state['time_samples']}")

        depth_levels = int(st.session_state["depth_levels"])
        depth_idx = st.slider("Select Depth Index", 0, depth_levels - 1, 0, key="sonic_depth_idx")

        col1, col2, col3 = st.columns(3)
        with col1:
            dx = st.number_input(
                "Receiver spacing",
                min_value=1e-6,
                value=0.15,
                step=0.01,
                key="sonic_receiver_spacing",
            )
        with col2:
            p_min = st.number_input(
                "Slowness min",
                min_value=0.0,
                value=0.0,
                format="%.6f",
                key="sonic_p_min",
            )
            p_max = st.number_input(
                "Slowness max",
                min_value=1e-6,
                value=0.001,
                format="%.6f",
                key="sonic_p_max",
            )
        with col3:
            num_p = st.number_input(
                "Slowness samples",
                min_value=1,
                value=150,
                step=10,
                key="sonic_num_p",
            )

        lowcut = st.number_input(
            "Low cut (Hz)",
            min_value=1.0,
            value=500.0,
            key="sonic_lowcut",
        )
        highcut = st.number_input(
            "High cut (Hz)",
            min_value=1.0,
            value=5000.0,
            key="sonic_highcut",
        )

        if st.button("Apply Bandpass Filter", key="apply_bandpass_filter_button"):
            try:
                filtered = {
                    component: bandpass(
                        waveform_arrays[component],
                        float(lowcut),
                        float(highcut),
                        float(sampling_frequency),
                    )
                    for component in required
                }
                st.session_state["filtered_waveforms"] = filtered
                st.success("Bandpass filter applied to all waveform components.")
            except Exception as error:
                st.error(f"Unable to filter waveforms: {error}")

        data = st.session_state["sonic_data"]
        depth_levels, receivers, time_samples = data.shape
        assert data.shape == (depth_levels, receivers, time_samples)

        waveform = data[depth_idx]
        waveform = waveform.T
        st.write(f"Selected waveform shape for processing: {waveform.shape}")

        if st.button("Compute Semblance", key="compute_semblance_button"):
            dt = 1.0 / float(sampling_frequency)
            p_values = generate_slowness(float(p_min), float(p_max), int(num_p))
            semblance = compute_semblance(waveform, dt, float(dx), p_values)
            picked_p, first_arrival_idx = pick_semblance_curve(semblance, p_values, waveform)

            with np.errstate(divide="ignore", invalid="ignore"):
                velocity = np.where(np.isnan(picked_p) | (picked_p == 0), np.nan, 1.0 / picked_p)

            st.session_state["semblance"] = semblance
            st.session_state["picked_p"] = picked_p
            st.session_state["velocity"] = velocity
            st.session_state["first_arrival_idx"] = first_arrival_idx
            st.session_state["p_values"] = p_values
            st.session_state["stc_done"] = True
            st.session_state["stc_slowness"] = picked_p
            st.session_state["stc_velocity"] = velocity
            st.session_state["stc_dt"] = dt
            st.session_state["waveform_arrays"] = waveform_arrays

    filtered_waveforms = st.session_state.get("filtered_waveforms", {})
    if waveform_arrays and filtered_waveforms:
        st.subheader("Filter Comparison")
        depth_idx = int(st.session_state.get("sonic_depth_idx", 0))

        raw_waveform = waveform_arrays["XX"][depth_idx].T
        filtered_waveform = filtered_waveforms["XX"][depth_idx].T

        raw = raw_waveform[:, 0]
        filtered = filtered_waveform[:, 0]

        st.plotly_chart(
            _plot_filter_comparison(raw, filtered),
            use_container_width=True,
        )

    semblance = st.session_state.get("semblance")
    p_values = st.session_state.get("p_values")
    picked_p = st.session_state.get("picked_p")
    velocity = st.session_state.get("velocity")
    first_arrival_idx = st.session_state.get("first_arrival_idx")
    if semblance is not None and p_values is not None and picked_p is not None:
        st.subheader("Semblance Panel")
        dt = 1.0 / float(sampling_frequency)
        figure = _plot_semblance_panel(np.asarray(semblance), np.asarray(p_values), dt, np.asarray(picked_p))
        st.pyplot(figure)

    if velocity is not None and first_arrival_idx is not None:
        st.subheader("Velocity vs Time")
        dt = 1.0 / float(sampling_frequency)
        velocity_figure = _plot_velocity_curve(np.asarray(velocity), dt, int(first_arrival_idx))
        st.pyplot(velocity_figure)


if __name__ == "__main__":
    main()
