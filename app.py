from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any

import streamlit as st

from binary_loader import load_binary


APP_DIR = Path(__file__).resolve().parent

PAGE_CONFIG = {
    "Overview": APP_DIR / "pages" / "01_overview.py",
    "Sonic": APP_DIR / "pages" / "02_sonic.py",
    "Crossdipole": APP_DIR / "pages" / "03_crossdipole.py",
    "Rock Physics": APP_DIR / "pages" / "04_rockphysics.py",
    "Export": APP_DIR / "pages" / "05_export.py",
    "Stoneley": APP_DIR / "pages" / "06_stoneley.py",
}

SESSION_DEFAULTS: dict[str, Any] = {
    "file_name": None,
    "file_bytes": None,
    "file_type": None,
    "waveforms": {"XX": None, "XY": None, "YX": None, "YY": None},
    "waveform_upload_complete": False,
    "stoneley_waveform": None,
    "raw_df": None,
    "raw_data": None,
    "raw_metadata": None,
    "validation_results": None,
    "processed_df": None,
    "crossdipole_data": None,
    "crossdipole_dt": None,
    "crossdipole_params": {},
    "crossdipole_results": {},
    "semblance_output": None,
    "stoneley_results": None,
}


def _initialize_session_state() -> None:
    for key, value in SESSION_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _reset_loaded_state() -> None:
    st.session_state["raw_df"] = None
    st.session_state["raw_data"] = None
    st.session_state["raw_metadata"] = None
    st.session_state["validation_results"] = None
    st.session_state["processed_df"] = None
    st.session_state["crossdipole_data"] = None
    st.session_state["crossdipole_dt"] = None
    st.session_state["crossdipole_params"] = {}
    st.session_state["crossdipole_results"] = {}
    st.session_state["semblance_output"] = None


def _detect_waveform_component(file_name: str) -> str | None:
    upper_name = file_name.upper()
    for component in ("XX", "XY", "YX", "YY"):
        if component in upper_name:
            return component
    return None


def _render_waveform_checklist() -> None:
    waveforms = st.session_state.get("waveforms", {})

    st.sidebar.markdown("**Waveform Upload Status**")
    complete = True

    for component in ("XX", "XY", "YX", "YY"):
        uploaded = waveforms.get(component) is not None
        complete = complete and uploaded
        status = "OK" if uploaded else "Missing"
        st.sidebar.write(f"{component}: {status}")

    st.session_state["waveform_upload_complete"] = complete


def _load_page(page_path: Path) -> None:
    if not page_path.exists():
        st.warning(f"Page not available yet: {page_path.name}")
        return

    spec = spec_from_file_location(page_path.stem, page_path)
    if spec is None or spec.loader is None:
        st.error(f"Unable to load page: {page_path.name}")
        return

    module = module_from_spec(spec)
    spec.loader.exec_module(module)

    if hasattr(module, "main"):
        module.main()
    else:
        st.error(f"{page_path.name} does not define a main() function.")


def main() -> None:
    st.set_page_config(page_title="Sonic Dashboard", layout="wide")
    _initialize_session_state()

    st.sidebar.title("Navigation")
    selected_page = st.sidebar.radio("Page", list(PAGE_CONFIG.keys()))

    uploaded_file = st.sidebar.file_uploader(
        "Upload file",
        type=["las", "csv"],
    )

    uploaded_waveforms = st.sidebar.file_uploader(
        "Upload binary waveforms",
        type=["bin"],
        accept_multiple_files=True,
    )
    st.sidebar.subheader("Upload monopole waveform")
    monopole_file = st.sidebar.file_uploader(
        "Upload Stoneley (.bin)",
        type=["bin"],
        key="stoneley_file",
    )

    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        file_name = uploaded_file.name
        file_type = Path(file_name).suffix.lower()

        if (
            st.session_state["file_name"] != file_name
            or st.session_state["file_bytes"] != file_bytes
        ):
            st.session_state["file_name"] = file_name
            st.session_state["file_bytes"] = file_bytes
            st.session_state["file_type"] = file_type
            _reset_loaded_state()

    if uploaded_waveforms:
        waveforms = dict(st.session_state.get("waveforms", {}))

        for waveform_file in uploaded_waveforms:
            component = _detect_waveform_component(waveform_file.name)
            if component is None:
                continue

            waveforms[component] = {
                "name": waveform_file.name,
                "bytes": waveform_file.getvalue(),
            }

        st.session_state["waveforms"] = waveforms

    if monopole_file is not None:
        st.session_state["stoneley_waveform"] = load_binary(monopole_file)

    _render_waveform_checklist()
    _load_page(PAGE_CONFIG[selected_page])


if __name__ == "__main__":
    main()
