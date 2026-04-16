from __future__ import annotations

import os

import numpy as np
import streamlit as st

from crossdipole.semblance import compute_semblance, generate_slowness
from stoneley.inference import predict_stoneley
from stoneley.model import load_model


def preprocess_waveform(wave: object) -> np.ndarray:
    if isinstance(wave, dict) and "data" in wave:
        data = np.asarray(wave["data"], dtype=np.float64)
    else:
        data = np.asarray(wave, dtype=np.float64)

    if data.ndim == 3:
        waveform = data[0].T
    elif data.ndim == 2:
        waveform = data.T
    else:
        raise ValueError("Stoneley waveform must be 2D or 3D.")

    scale = np.nanmax(np.abs(waveform))
    if scale > 0:
        waveform = waveform / scale

    return waveform

def main() -> None:
    st.title("Stoneley")

    wave = st.session_state.get("stoneley_waveform")
    if wave is None:
        st.warning("Upload monopole waveform first")
        st.stop()

    sampling_frequency = float(st.session_state.get("sampling_frequency", 25000.0))
    dt = 1.0 / sampling_frequency

    receiver_spacing = st.number_input(
        "Receiver spacing",
        min_value=1e-6,
        value=0.1524,
        step=0.01,
        key="stoneley_receiver_spacing",
    )
    col1, col2, col3 = st.columns(3)
    with col1:
        p_min = st.number_input(
            "Slowness min",
            min_value=0.0,
            value=0.0,
            format="%.6f",
            key="stoneley_p_min",
        )
    with col2:
        p_max = st.number_input(
            "Slowness max",
            min_value=1e-6,
            value=0.001,
            format="%.6f",
            key="stoneley_p_max",
        )
    with col3:
        num_p = st.number_input(
            "Slowness samples",
            min_value=1,
            value=150,
            step=10,
            key="stoneley_num_p",
        )

    try:
        waveform = preprocess_waveform(wave)

        p_values = generate_slowness(float(p_min), float(p_max), int(num_p))
        panel = compute_semblance(waveform, dt, float(receiver_spacing), p_values)
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(BASE_DIR, "..", "stoneley", "stoneley_model.joblib")
        model_path = os.path.abspath(model_path)
        if not os.path.exists(model_path):
            st.warning("Stoneley model file not found. Skipping prediction.")
            return

        model = load_model(model_path)
        stoneley_velocity = predict_stoneley(panel, model, p_values)
        depth = np.arange(len(stoneley_velocity))

        st.session_state["stoneley_results"] = {
            "v_st": stoneley_velocity,
            "depth": depth,
            "semblance": panel,
        }

        st.success("Stoneley processing completed.")
        st.write(f"Waveform shape: {waveform.shape}")
        st.write(f"Semblance panel shape: {panel.shape}")
        st.write(f"Stoneley velocity shape: {stoneley_velocity.shape}")
    except Exception as error:
        st.error(f"Unable to process Stoneley waveform: {error}")


if __name__ == "__main__":
    main()
