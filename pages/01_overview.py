from __future__ import annotations

from io import BytesIO

import pandas as pd
import streamlit as st

from binary_loader import load_ldeo_binary
from data_loader import load_csv, load_las
from validator import validate_dataframe


def _load_tabular_data(file_bytes: bytes, file_type: str) -> pd.DataFrame:
    buffer = BytesIO(file_bytes)

    if file_type == ".las":
        return load_las(buffer)

    if file_type == ".csv":
        return load_csv(buffer)

    raise ValueError(f"Unsupported tabular file type: {file_type}")


def _render_binary_controls() -> dict[str, object]:
    st.subheader("Binary Parameters")
    col1, col2 = st.columns(2)
    with col1:
        samples_per_trace = st.number_input("Samples per trace", min_value=1, value=480, step=1)
        channels = st.number_input("Channels", min_value=1, value=8, step=1)
        dtype = st.selectbox("Binary dtype", ["float32", "float64", "int16", "int32"], index=0)
    with col2:
        byte_order = st.selectbox("Byte order", ["<", ">"], index=0)
        depth_start = st.number_input("Depth start", value=0.0)
        depth_step = st.number_input("Depth step", value=0.1524)

    return {
        "samples_per_trace": int(samples_per_trace),
        "channels": int(channels),
        "dtype": dtype,
        "byte_order": byte_order,
        "depth_start": float(depth_start),
        "depth_step": float(depth_step),
    }


def _load_binary_data(file_bytes: bytes, params: dict[str, object]) -> tuple[object, dict[str, object]]:
    return load_ldeo_binary(
        BytesIO(file_bytes),
        samples_per_trace=int(params["samples_per_trace"]),
        channels=int(params["channels"]),
        dtype=str(params["dtype"]),
        byte_order=str(params["byte_order"]),
        depth_start=float(params["depth_start"]),
        depth_step=float(params["depth_step"]),
    )


def main() -> None:
    st.title("Overview")

    file_bytes = st.session_state.get("file_bytes")
    file_name = st.session_state.get("file_name")
    file_type = st.session_state.get("file_type")

    if not file_bytes or not file_name or not file_type:
        st.info("Upload a LAS, CSV, or binary file from the sidebar to begin.")
        return

    st.write(f"Loaded file: `{file_name}`")

    try:
        if file_type in {".las", ".csv"}:
            if st.session_state.get("raw_df") is None:
                st.session_state["raw_df"] = _load_tabular_data(file_bytes, file_type)
                st.session_state["raw_data"] = st.session_state["raw_df"]
                st.session_state["raw_metadata"] = {"file_type": file_type, "file_name": file_name}

            df = st.session_state["raw_df"]
            st.subheader("Data Preview")
            st.dataframe(df.head(), use_container_width=True)

            depth_default = "DEPTH" if "DEPTH" in df.columns else str(df.columns[0])
            depth_col = st.selectbox(
                "Depth column",
                options=list(df.columns),
                index=list(df.columns).index(depth_default) if depth_default in df.columns else 0,
            )
            required_columns = st.multiselect("Required columns", options=list(df.columns), default=[depth_col])

            validation_results = validate_dataframe(
                df,
                required_columns=required_columns,
                depth_column=depth_col,
                nan_columns=required_columns or None,
            )
            st.session_state["validation_results"] = validation_results

            st.subheader("Validation Results")
            st.json(validation_results)

        elif file_type == ".bin":
            binary_params = _render_binary_controls()

            if st.button("Load Binary Data"):
                waveform, metadata = _load_binary_data(file_bytes, binary_params)
                st.session_state["raw_data"] = waveform
                st.session_state["raw_metadata"] = metadata
                st.session_state["crossdipole_data"] = waveform
                st.session_state["crossdipole_dt"] = float(metadata["byte_order"] and metadata["depth_step"] * 0 + metadata["depth_step"] * 0 + metadata["samples_per_trace"] * 0 + metadata["channels"] * 0)  # placeholder to keep state shape stable
                st.session_state["crossdipole_dt"] = float(metadata.get("dt", 0.0))

            if st.session_state.get("raw_data") is not None:
                st.subheader("Binary Summary")
                st.write(f"Waveform shape: {st.session_state['raw_data'].shape}")
                st.json(st.session_state.get("raw_metadata", {}))
            else:
                st.info("Set binary parameters and click 'Load Binary Data'.")

        else:
            st.error(f"Unsupported file type: {file_type}")

    except Exception as error:
        st.error(f"Unable to load or validate the file: {error}")
