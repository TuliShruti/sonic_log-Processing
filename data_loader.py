from __future__ import annotations

from io import BytesIO, StringIO
from pathlib import Path
from typing import Any

import lasio
import pandas as pd


def _read_bytes(file: Any) -> bytes:
    if isinstance(file, (str, Path)):
        return Path(file).read_bytes()

    if hasattr(file, "getvalue"):
        return file.getvalue()

    if hasattr(file, "read"):
        data = file.read()
        if hasattr(file, "seek"):
            file.seek(0)
        return data

    raise TypeError("Unsupported file input. Expected path, bytes buffer, or file-like object.")


def load_las(file: Any) -> pd.DataFrame:
    file_bytes = _read_bytes(file)
    text = file_bytes.decode("utf-8", errors="ignore")
    las = lasio.read(StringIO(text))
    dataframe = las.df().reset_index()

    if dataframe.columns.size > 0:
        dataframe = dataframe.rename(columns={dataframe.columns[0]: "DEPTH"})

    return dataframe


def load_csv(file: Any) -> pd.DataFrame:
    raw_bytes = _read_bytes(file)
    return pd.read_csv(BytesIO(raw_bytes))
