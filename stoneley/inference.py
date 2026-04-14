from __future__ import annotations

import numpy as np

from stoneley.model import extract_features


def predict_stoneley(panel: np.ndarray, model, p_values: np.ndarray) -> np.ndarray:
    features = extract_features(panel)
    pred_idx = model.predict(features).astype(int)
    pred_idx = np.clip(pred_idx, 0, len(p_values) - 1)
    slowness = p_values[pred_idx]
    velocity = np.where(slowness > 0, 1.0 / slowness, np.nan)
    return velocity
