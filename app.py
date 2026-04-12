from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any

import streamlit as st


APP_DIR = Path(__file__).resolve().parent

PAGE_CONFIG = {
    "Overview": APP_DIR / "pages" / "01_overview.py",
    "Sonic": APP_DIR / "pages" / "02_sonic.py",
    "Crossdipole": APP_DIR / "pages" / "03_crossdipole.py",
    "Rock Physics": APP_DIR / "pages" / "04_rockphysics.py",
    "Export": APP_DIR / "pages" / "05_export.py",
}

SESSION_DEFAULTS: dict[str, Any] = {
    "file_name": None,
    "file_bytes": None,
    "file_type": None,
    "raw_df": None,
    "raw_data": None,
    "raw_metadata": None,
    "validation_results": None,
    "processed_df": None,
    "crossdipole_results": {},
    "semblance_output": None,
    "crossdipole_data": None,
    "crossdipole_dt": None,
    "crossdipole_params": {},
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
    st.session_state["crossdipole_results"] = {}
    st.session_state["semblance_output"] = None
    st.session_state["crossdipole_data"] = None
    st.session_state["crossdipole_dt"] = None
    st.session_state["crossdipole_params"] = {}


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
        type=["las", "csv", "bin"],
    )

    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        if (
            st.session_state["file_name"] != uploaded_file.name
            or st.session_state["file_bytes"] != file_bytes
        ):
            st.session_state["file_name"] = uploaded_file.name
            st.session_state["file_bytes"] = file_bytes
            st.session_state["file_type"] = Path(uploaded_file.name).suffix.lower()
            _reset_loaded_state()

    _load_page(PAGE_CONFIG[selected_page])


if __name__ == "__main__":
    main()
