from __future__ import annotations

from io import BytesIO

import numpy as np
import pandas as pd
import streamlit as st

from rock_physics import compute_elastic_properties
from viz.log_plot import plot_logs


def main() -> None:
    st.title("Rock Physics")

    cross = st.session_state.get("crossdipole_results")
    if not cross or "logs" not in cross:
        st.warning("Run Crossdipole STC first.")
        st.stop()

    logs = cross["logs"]
    depth = cross.get("depth")
    common_depth = np.asarray(depth, dtype=float)

    df = pd.DataFrame(
        {
            "DEPTH": common_depth,
            "Vs": logs["vs_slow"],
        }
    )

    if "sonic_df" in st.session_state:
        sonic = st.session_state["sonic_df"]
        if "RHOB" in sonic:
            df["RHO"] = np.interp(
                common_depth,
                pd.to_numeric(sonic["DEPTH"], errors="coerce"),
                pd.to_numeric(sonic["RHOB"], errors="coerce"),
            )
        else:
            df["RHO"] = 2500
        if "DTCO" in sonic:
            df["Vp"] = 1e6 / pd.to_numeric(sonic["DTCO"], errors="coerce")
    else:
        df["RHO"] = 2500

    if "Vp" not in df.columns:
        df["Vp"] = pd.Series(logs["vp_fast"])

    stoneley = st.session_state.get("stoneley_results")
    if stoneley and "v_st" in stoneley:
        df["V_st"] = np.interp(
            common_depth,
            np.asarray(stoneley["depth"], dtype=float),
            np.asarray(stoneley["v_st"], dtype=float),
        )

    df["Vs_fast"] = df["Vp"]
    df["Vs_slow"] = df["Vs"]

    rho = pd.to_numeric(df["RHO"], errors="coerce")
    vp = pd.to_numeric(df["Vp"], errors="coerce")
    vs_fast = pd.to_numeric(df["Vs_fast"], errors="coerce")
    vs_slow = pd.to_numeric(df["Vs_slow"], errors="coerce")

    df["C33"] = rho * vp**2
    df["C44"] = rho * vs_fast**2
    df["C55"] = rho * vs_slow**2
    if "V_st" in df.columns:
        df["C66"] = rho * pd.to_numeric(df["V_st"], errors="coerce") ** 2
        df["C13"] = df["C33"] - 2 * df["C44"]
        df["C11"] = df["C13"] + 2 * df["C66"]

    for column in ["C33", "C44", "C55", "C66", "C13", "C11"]:
        if column in df.columns:
            df[column] = df[column] / 1e9

    st.subheader("Crossdipole Velocity Logs")
    st.dataframe(df.head(), use_container_width=True)

    working_df = df.copy()

    if st.button("Compute Rock Physics", key="rockphysics_compute_button"):
        try:
            result_df = compute_elastic_properties(
                working_df,
                vp_col="Vp",
                vs_col="Vs",
                rho_col="RHO",
            )
            st.session_state["rockphysics_df"] = result_df
        except Exception as error:
            st.error(f"Unable to compute rock physics: {error}")

    result_df = st.session_state.get("rockphysics_df")
    if result_df is not None:
        st.subheader("Elastic Properties")
        st.dataframe(result_df.head(), use_container_width=True)

        log_columns = [column for column in ["Vp", "Vs", "Youngs_modulus", "Poissons_ratio"] if column in result_df.columns]
        if log_columns:
            st.plotly_chart(
                plot_logs(result_df, depth_col="DEPTH", log_columns=log_columns),
                use_container_width=True,
            )

        output_buffer = BytesIO()
        with pd.ExcelWriter(output_buffer, engine="openpyxl") as writer:
            result_df.to_excel(writer, index=False)
        output_buffer.seek(0)

        st.download_button(
            "Download stiffness_output_v2.xlsx",
            data=output_buffer.getvalue(),
            file_name="stiffness_output_v2.xlsx",
        )


if __name__ == "__main__":
    main()
