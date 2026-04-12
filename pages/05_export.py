from __future__ import annotations

import streamlit as st


def _export_csv(df) -> str:
    """Convert DataFrame to CSV string."""
    return df.to_csv(index=False)


def main() -> None:
    st.title("Export Data")

    # Check available datasets
    has_sonic_df = "sonic_df" in st.session_state
    has_rockphysics_df = "rockphysics_df" in st.session_state
    has_crossdipole_results = "crossdipole_results" in st.session_state

    if not (has_sonic_df or has_rockphysics_df or has_crossdipole_results):
        st.warning("No processed data available for export. Run analyses first.")
        return

    # Dataset selection
    st.subheader("Select Dataset")

    dataset_options = []
    if has_sonic_df:
        dataset_options.append("Sonic Analysis")
    if has_rockphysics_df:
        dataset_options.append("Rock Physics")
    if has_crossdipole_results:
        dataset_options.append("Crossdipole Results (metadata only)")

    selected_dataset = st.selectbox("Dataset to export", options=dataset_options)

    # Format selection
    st.subheader("Export Format")
    export_format = st.radio("Format", options=["CSV"], horizontal=True)

    # Export button
    if st.button("Export Data", type="primary"):
        try:
            if selected_dataset == "Sonic Analysis":
                df = st.session_state["sonic_df"]
                filename = "sonic_analysis.csv"

            elif selected_dataset == "Rock Physics":
                df = st.session_state["rockphysics_df"]
                filename = "rock_physics.csv"

            elif selected_dataset == "Crossdipole Results (metadata only)":
                # Export metadata as simple info
                results = st.session_state["crossdipole_results"]
                st.info("Crossdipole results contain array data. Download the Sonic or Rock Physics datasets for tabular export.")
                return

            else:
                st.error("Invalid dataset selection.")
                return

            # Convert to CSV and store in session_state
            csv_data = _export_csv(df)
            st.session_state["export_csv_data"] = csv_data
            st.session_state["export_filename"] = filename

            st.success(f"Data prepared for download: {filename}")

        except Exception as e:
            st.error(f"Error during export: {str(e)}")

    # Display download button outside of button block
    if "export_csv_data" in st.session_state:
        st.download_button(
            label=f"Download {st.session_state['export_filename']}",
            data=st.session_state["export_csv_data"],
            file_name=st.session_state["export_filename"],
            mime="text/csv",
        )

    # Display dataset info
    st.subheader("Dataset Info")

    if selected_dataset == "Sonic Analysis" and has_sonic_df:
        df = st.session_state["sonic_df"]
        st.write(f"Rows: {len(df)}")
        st.write(f"Columns: {len(df.columns)}")
        st.write("Preview:")
        st.dataframe(df.head(), use_container_width=True)

    elif selected_dataset == "Rock Physics" and has_rockphysics_df:
        df = st.session_state["rockphysics_df"]
        st.write(f"Rows: {len(df)}")
        st.write(f"Columns: {len(df.columns)}")
        st.write("Preview:")
        st.dataframe(df.head(), use_container_width=True)

    elif selected_dataset == "Crossdipole Results (metadata only)" and has_crossdipole_results:
        results = st.session_state["crossdipole_results"]
        st.write("Available results:")
        for key, value in results.items():
            if hasattr(value, "shape"):
                st.write(f"  - {key}: array shape {value.shape}")
            else:
                st.write(f"  - {key}: {type(value).__name__}")


if __name__ == "__main__":
    main()
