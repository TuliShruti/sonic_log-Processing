from __future__ import annotations

from typing import Any

import numpy as np
from scipy.signal import medfilt


def generate_slowness(p_min: float, p_max: float, num_p: int) -> np.ndarray:
    if num_p <= 0:
        raise ValueError("num_p must be positive")
    if p_min < 0 or p_max <= p_min:
        raise ValueError("Require 0 <= p_min < p_max")
    return np.linspace(p_min, p_max, num_p, dtype=np.float64)


def find_first_arrival(waveform: np.ndarray, threshold_ratio: float = 0.05) -> int:
    """
    Returns the sample index of the first arrival onset.
    waveform shape: (ns, nrec)
    """
    energy = np.sum(waveform ** 2, axis=1)
    threshold = threshold_ratio * np.max(energy)
    indices = np.where(energy > threshold)[0]
    return int(indices[0]) if len(indices) > 0 else 0


def _shift_trace(trace: np.ndarray, delay_samples: float) -> np.ndarray:
    sample_axis = np.arange(trace.shape[0], dtype=np.float64)
    shifted_axis = sample_axis + delay_samples
    shifted = np.zeros_like(trace, dtype=np.float64)

    valid = (shifted_axis >= 0.0) & (shifted_axis <= sample_axis[-1])
    if np.any(valid):
        shifted[valid] = np.interp(shifted_axis[valid], sample_axis, trace)

    return shifted


def compute_semblance(
    waveform: np.ndarray,
    dt: float,
    dx: float,
    p_values: np.ndarray,
) -> np.ndarray:
    """
    waveform: (ns, nrec)
    dt: sampling interval (seconds)
    dx: receiver spacing
    p_values: array of slowness values
    Returns:
        semblance_panel: (len(p_values), ns)
    """
    traces = np.asarray(waveform, dtype=np.float64)
    slowness_values = np.asarray(p_values, dtype=np.float64)

    if traces.ndim != 2:
        raise ValueError("waveform must have shape (time_samples, receivers)")
    if dt <= 0:
        raise ValueError("dt must be positive")
    if dx <= 0:
        raise ValueError("dx must be positive")
    if slowness_values.ndim != 1 or slowness_values.size == 0:
        raise ValueError("p_values must be a non-empty 1D array")

    time_samples, receivers = traces.shape
    offsets = np.arange(receivers, dtype=np.float64) * dx
    panel = np.zeros((slowness_values.size, time_samples), dtype=np.float64)

    for slowness_index, slowness in enumerate(slowness_values):
        delays = (slowness * offsets) / dt
        corrected = np.column_stack(
            [_shift_trace(traces[:, receiver], delays[receiver]) for receiver in range(receivers)]
        )

        stack = np.sum(corrected, axis=1)
        stack_energy = stack * stack
        trace_energy = np.sum(corrected * corrected, axis=1)

        numerator = np.cumsum(stack_energy)
        denominator = receivers * np.cumsum(trace_energy) + 1e-10
        panel[slowness_index] = numerator / denominator

    return panel


def pick_semblance_curve(semblance: np.ndarray, p_values: np.ndarray, waveform: np.ndarray) -> tuple[np.ndarray, int]:
    first_arrival_idx = find_first_arrival(waveform, threshold_ratio=0.15)

    picked_p = np.full(semblance.shape[1], np.nan, dtype=np.float64)
    for sample_index in range(first_arrival_idx, semblance.shape[1]):
        picked_p[sample_index] = p_values[np.argmax(semblance[:, sample_index])]

    valid_mask = ~np.isnan(picked_p)
    if np.sum(valid_mask) > 11:
        picked_p[valid_mask] = medfilt(picked_p[valid_mask], kernel_size=11)

    return picked_p, first_arrival_idx


def compute_semblance_from_params(
    waveform: np.ndarray,
    dt_microseconds: float,
    params: dict[str, Any],
) -> dict[str, np.ndarray]:
    slowness_min = float(params.get("p_min", 0.0))
    slowness_max = float(params.get("p_max", 1e-3))
    num_slowness = int(params.get("num_p", 150))
    receiver_spacing = float(params.get("rec_spacing", 0.1524))

    slowness_s_per_m = generate_slowness(
        p_min=slowness_min,
        p_max=slowness_max,
        num_p=num_slowness,
    )

    panel = compute_semblance(
        waveform=np.asarray(waveform, dtype=np.float64),
        dt=dt_microseconds * 1e-6,
        dx=receiver_spacing,
        p_values=slowness_s_per_m,
    )

    time_axis_microseconds = np.arange(waveform.shape[0], dtype=np.float64) * dt_microseconds
    velocity_axis = np.where(slowness_s_per_m > 0, 1.0 / slowness_s_per_m, np.nan)

    return {
        "semblance": panel,
        "time": time_axis_microseconds,
        "velocity": velocity_axis,
    }


def merge_semblance_output(
    results: dict[str, Any],
    waveform: np.ndarray,
    dt_microseconds: float,
    params: dict[str, Any],
) -> dict[str, Any]:
    merged = dict(results)

    if params.get("run_semblance", False):
        merged["semblance"] = compute_semblance_from_params(waveform, dt_microseconds, params)

    return merged
