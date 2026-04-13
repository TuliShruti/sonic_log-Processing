from __future__ import annotations

from io import BytesIO
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
    samples_per_trace: int | None = None,
    channels: int | None = None,
    dtype: str = "float32",
    byte_order: str = "<",
    depth_start: float | None = None,
    depth_step: float | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    raw_bytes = _read_binary(file)
    buffer = BytesIO(raw_bytes)

    depth_levels = int(np.frombuffer(buffer.read(4), dtype=">i4")[0])
    time_samples = int(np.frombuffer(buffer.read(4), dtype=">i4")[0])
    receivers = int(np.frombuffer(buffer.read(4), dtype=">i4")[0])
    _ntool = int(np.frombuffer(buffer.read(4), dtype=">i4")[0])
    _mode = int(np.frombuffer(buffer.read(4), dtype=">i4")[0])
    dz = float(np.frombuffer(buffer.read(4), dtype=">f4")[0])
    _scale = float(np.frombuffer(buffer.read(4), dtype=">f4")[0])
    dt = float(np.frombuffer(buffer.read(4), dtype=">f4")[0])

    if depth_levels <= 0 or time_samples <= 0 or receivers <= 0:
        raise ValueError("Invalid binary header metadata.")

    payload = np.frombuffer(buffer.read(), dtype=">f4")
    record_length = 1 + receivers * time_samples
    expected_values = depth_levels * record_length

    if payload.size < expected_values:
        raise ValueError("Binary data size mismatch with header metadata")

    records = payload[:expected_values].reshape(depth_levels, record_length)
    data = records[:, 1:].reshape(depth_levels, receivers, time_samples).astype(np.float32, copy=False)

    if data.size != depth_levels * receivers * time_samples:
        raise ValueError("Binary data size mismatch with header metadata")

    metadata = {
        "format": "LDEO",
        "depth_levels": depth_levels,
        "time_samples": time_samples,
        "receivers": receivers,
        "dz": dz,
        "dt": dt,
        "receiver_spacing_m": 0.1524,
        "dtype": "float32",
        "byte_order": ">",
        "depth_start": depth_start if depth_start is not None else 0.0,
        "depth_step": depth_step if depth_step is not None else dz,
    }

    return data, metadata
