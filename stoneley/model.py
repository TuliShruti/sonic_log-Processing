from __future__ import annotations

import joblib
import numpy as np


def load_model(path: str = "stoneley_model.joblib"):
    return joblib.load(path)


def extract_features(panel: np.ndarray) -> np.ndarray:
    panel_array = np.asarray(panel, dtype=np.float64)

    if panel_array.ndim != 2:
        raise ValueError("panel must be a 2D semblance matrix")

    peak_by_time = np.nanmax(panel_array, axis=0)
    picked_index_by_time = np.nanargmax(panel_array, axis=0).astype(np.float64)
    mean_by_time = np.nanmean(panel_array, axis=0)
    std_by_time = np.nanstd(panel_array, axis=0)

    features = np.column_stack(
        [
            peak_by_time,
            picked_index_by_time,
            mean_by_time,
            std_by_time,
        ]
    )

    return features
