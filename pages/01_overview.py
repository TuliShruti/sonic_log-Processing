from __future__ import annotations

from io import BytesIO

import streamlit as st

from data_loader import load_csv, load_las
from validator import validate_dataframe


def _init_state() -> None:
    if "waveforms" not in st.session_state:
        st.session_state["waveforms"] = {}


def _render_waveform_summary() -> None:
    waveforms = st.session_state.get("waveforms", {})
    if not waveforms:
        return

    st.subheader("Waveform Uploads")
    for component in ("XX", "XY", "YX", "YY"):
        entry = waveforms.get(component)
        if isinstance(entry, dict) and entry.get("name"):
            st.write(f"{component}: {entry['name']}")
        else:
            st.write(f"{component}: not uploaded")


def main() -> None:
    _init_state()
    st.title("Overview")

    file_bytes = st.session_state.get("file_bytes")
    file_name = st.session_state.get("file_name")
    file_type = st.session_state.get("file_type")

    uploaded_waveforms = st.session_state.get("waveforms", {})
    st.session_state["waveforms"] = uploaded_waveforms
    _render_waveform_summary()

    if not file_bytes or not file_name or not file_type:
        st.info("Upload a LAS or CSV file from the sidebar to begin.")
        return

    try:
        if file_type == ".las":
            df = load_las(BytesIO(file_bytes))
        elif file_type == ".csv":
            df = load_csv(BytesIO(file_bytes))
        else:
            st.warning("Overview supports LAS and CSV preview only.")
            return

        st.session_state["raw_df"] = df
        st.subheader("Data Preview")
        st.dataframe(df.head(), use_container_width=True)

        depth_col = "DEPTH" if "DEPTH" in df.columns else df.columns[0]
        validation = validate_dataframe(
            df,
            required_columns=[depth_col],
            depth_column=depth_col,
            nan_columns=[depth_col],
        )
        st.session_state["validation_results"] = validation

        st.subheader("Validation Results")
        st.json(validation)
    except Exception as error:
        st.error(f"Unable to load file: {error}")


if __name__ == "__main__":
    main()
