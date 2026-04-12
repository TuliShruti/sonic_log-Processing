from __future__ import annotations

import streamlit as st

from rock_physics import compute_elastic_properties
from viz.log_plot import plot_logs


def main() -> None:
    st.title("Rock Physics Analysis")

    # Check prerequisites using safe .get() method
    df = st.session_state.get("sonic_df")

    if df is None:
        st.warning("Run Sonic Analysis first to compute velocities.")
        return

    if hasattr(df, "empty") and df.empty:
        st.error("Dataframe is empty. No data to process.")
        return

    # Select depth column
    depth_default = "DEPTH" if "DEPTH" in df.columns else df.columns[0]
    depth_col = st.selectbox(
        "Depth column",
        options=list(df.columns),
        index=list(df.columns).index(depth_default),
    )

    st.subheader("Select Log Columns")

    col1, col2, col3 = st.columns(3)

    with col1:
        vp_default = "Vp" if "Vp" in df.columns else df.columns[0]
        vp_col = st.selectbox(
            "Vp column",
            options=list(df.columns),
            index=list(df.columns).index(vp_default),
        )

    with col2:
        vs_default = "Vs" if "Vs" in df.columns else df.columns[-1]
        vs_col = st.selectbox(
            "Vs column",
            options=list(df.columns),
            index=list(df.columns).index(vs_default) if vs_default in df.columns else 0,
        )

    with col3:
        rho_default = "RHO" if "RHO" in df.columns else df.columns[-1]
        rho_col = st.selectbox(
            "Density column",
            options=list(df.columns),
            index=list(df.columns).index(rho_default) if rho_default in df.columns else 0,
        )

    # Compute Rock Physics button
    if st.button("Compute Rock Physics", type="primary"):
        try:
            # Call compute_elastic_properties (only external function)
            result_df = compute_elastic_properties(
                df=df,
                vp_col=vp_col,
                vs_col=vs_col,
                rho_col=rho_col,
            )

            # Store in session_state
            st.session_state["rockphysics_df"] = result_df

            st.success("Rock physics properties computed!")

        except Exception as e:
            st.error(f"Error during computation: {str(e)}")

    # Display persisted results
    if "rockphysics_df" in st.session_state:
        result_df = st.session_state["rockphysics_df"]

        st.subheader("Elastic Properties")

        log_columns = [
            "Youngs_modulus",
            "Poissons_ratio",
            "Bulk_modulus",
            "Shear_modulus",
        ]

        # Check which columns exist
        available_columns = [col for col in log_columns if col in result_df.columns]

        if available_columns:
            fig = plot_logs(
                df=result_df,
                depth_col=depth_col,
                log_columns=available_columns,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No elastic property columns found in result.")

        st.subheader("Statistics")
        st.dataframe(
            result_df[[depth_col] + available_columns].describe(),
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
