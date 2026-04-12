from __future__ import annotations

from typing import Any

import numpy as np


def _build_slowness_grid(params: dict[str, Any]) -> np.ndarray:
    s_min = float(params.get("s_min", 300.0))
    s_max = float(params.get("s_max", 2500.0))
    s_step = float(params.get("s_step", 4.0))

    if s_min <= 0 or s_max <= 0 or s_step <= 0:
        raise ValueError("s_min, s_max, and s_step must be positive.")

    if s_min > s_max:
        raise ValueError("s_min must be less than or equal to s_max.")

    return np.arange(s_min, s_max + s_step, s_step, dtype=np.float64)


def _compute_offsets(
    nrec: int,
    rec_spacing: float,
    offset_mode: str = "zero_based",
) -> np.ndarray:
    if rec_spacing <= 0:
        raise ValueError("rec_spacing must be positive.")
    if nrec <= 0:
        raise ValueError("nrec must be positive.")

    indices = np.arange(nrec, dtype=np.float64)
    if offset_mode == "centered":
        return (indices - (nrec // 2)) * rec_spacing

    return indices * rec_spacing


def _remove_moveout(
    traces: np.ndarray,
    slowness: float,
    dt: float,
    offsets: np.ndarray,
) -> np.ndarray:
    nrec, ns = traces.shape
    corrected = np.zeros((nrec, ns), dtype=np.float64)
    sample_axis = np.arange(ns, dtype=np.float64)
    delay_samples = slowness * offsets / dt

    for receiver in range(nrec):
        delay = float(delay_samples[receiver])
        shifted_axis = sample_axis + delay
        valid = (shifted_axis >= 0.0) & (shifted_axis <= ns - 1)

        if np.any(valid):
            corrected[receiver, valid] = np.interp(
                shifted_axis[valid],
                sample_axis,
                traces[receiver],
            )

    return corrected


def _windowed_semblance(corrected: np.ndarray, win: int) -> np.ndarray:
    nrec, ns = corrected.shape
    panel_row = np.zeros(ns, dtype=np.float64)

    for sample in range(ns):
        lo = max(0, sample - win)
        hi = min(ns - 1, sample + win)
        window = corrected[:, lo : hi + 1]
        stack = np.sum(window, axis=0)
        numerator = np.sum(stack * stack)
        denominator = nrec * np.sum(window * window) + 1e-10
        panel_row[sample] = numerator / denominator

    return panel_row


def compute_semblance(data: np.ndarray, dt: float, params: dict[str, Any]) -> dict[str, np.ndarray]:
    """
    Compute a time-velocity semblance panel from waveform data.

    Parameters
    ----------
    data
        Waveform array with shape (n_receivers, n_samples).
    dt
        Sample interval in microseconds.
    params
        Parameter dictionary. Supported keys:
        - rec_spacing: receiver spacing in meters
        - s_min: minimum trial slowness in microseconds per meter
        - s_max: maximum trial slowness in microseconds per meter
        - s_step: slowness step in microseconds per meter
        - win: semblance half-window in samples

    Returns
    -------
    dict
        {
            "semblance": 2D ndarray with shape (n_velocities, n_times),
            "time": 1D ndarray in microseconds,
            "velocity": 1D ndarray in meters/second
        }
    """
    traces = np.asarray(data, dtype=np.float64)

    if traces.ndim != 2:
        raise ValueError("data must have shape (n_receivers, n_samples).")

    if dt <= 0:
        raise ValueError("dt must be positive.")

    win = int(params.get("win", 40))
    if win <= 0:
        raise ValueError("win must be a positive integer.")

    rec_spacing = float(params.get("rec_spacing", 0.1524))
    offset_mode = str(params.get("offset_mode", "zero_based"))
    if offset_mode not in {"zero_based", "centered"}:
        raise ValueError("offset_mode must be 'zero_based' or 'centered'.")

    slownesses = _build_slowness_grid(params)
    offsets = _compute_offsets(traces.shape[0], rec_spacing, offset_mode)

    panel = np.zeros((slownesses.size, traces.shape[1]), dtype=np.float64)

    for index, slowness in enumerate(slownesses):
        corrected = _remove_moveout(traces, slowness, dt, offsets)
        panel[index] = _windowed_semblance(corrected, win)

    time_axis = np.arange(traces.shape[1], dtype=np.float64) * dt
    velocity_axis = 1e6 / slownesses

    return {
        "semblance": panel,
        "time": time_axis,
        "velocity": velocity_axis,
    }


def merge_semblance_output(
    results: dict[str, Any],
    data: np.ndarray,
    dt: float,
    params: dict[str, Any],
) -> dict[str, Any]:
    """
    Return a crossdipole results dictionary with optional semblance output added.
    """
    merged = dict(results)

    if params.get("run_semblance", False):
        merged["semblance"] = compute_semblance(data, dt, params)

    return merged
