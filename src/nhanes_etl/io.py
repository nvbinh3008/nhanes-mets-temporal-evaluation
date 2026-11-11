from __future__ import annotations

from pathlib import Path

import pandas as pd

from .exceptions import PipelineError
from .utils import ensure_dir


def output_path(base: Path, stem: str, output_format: str) -> Path:
    extension = ".parquet" if output_format == "parquet" else ".csv"
    return base / f"{stem}{extension}"


def write_frame(frame: pd.DataFrame, path: Path, output_format: str) -> None:
    ensure_dir(path.parent)
    tmp = path.with_name(f".{path.name}.tmp")
    try:
        if output_format == "parquet":
            frame.to_parquet(tmp, index=False)
        elif output_format == "csv":
            frame.to_csv(tmp, index=False)
        else:
            raise PipelineError(f"Unsupported output format: {output_format}")
        tmp.replace(path)
    finally:
        if tmp.exists():
            tmp.unlink()


def read_frame(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, low_memory=False)
    raise PipelineError(f"Unsupported staged file format: {path}")


def write_report(frame: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    frame.to_csv(path, index=False)
