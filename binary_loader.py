from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def _read_binary(file: Any) -> bytes:
    if isinstance(file, (str, Path)):
        return Path(file).read_bytes()

    if hasattr(file, "getvalue"):
        return file.getvalue()

    if hasattr(file, "read"):
        data = file.read()
        if hasattr(file, "seek"):
            file.seek(0)
        return data

    raise TypeError("Unsupported binary input. Expected path, bytes buffer, or file-like object.")


def load_ldeo_binary(
    file: Any,
    samples_per_trace: int,
    channels: int = 1,
    dtype: str = "float32",
    byte_order: str = "<",
    depth_start: float | None = None,
    depth_step: float | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    raw_bytes = _read_binary(file)
    np_dtype = np.dtype(f"{byte_order}{np.dtype(dtype).str[1:]}")
    flat_data = np.frombuffer(raw_bytes, dtype=np_dtype)

    trace_width = samples_per_trace * channels
    if trace_width <= 0:
        raise ValueError("samples_per_trace and channels must be positive integers.")

    if flat_data.size % trace_width != 0:
        raise ValueError("Binary size is not compatible with the provided trace dimensions.")

    trace_count = flat_data.size // trace_width
    waveform = flat_data.reshape(trace_count, channels, samples_per_trace)

    metadata = {
        "format": "LDEO",
        "dtype": str(np_dtype),
        "byte_order": byte_order,
        "samples_per_trace": samples_per_trace,
        "channels": channels,
        "trace_count": trace_count,
        "depth_start": depth_start,
        "depth_step": depth_step,
    }

    if depth_start is not None and depth_step is not None:
        metadata["depth"] = depth_start + np.arange(trace_count) * depth_step

    return waveform, metadata
