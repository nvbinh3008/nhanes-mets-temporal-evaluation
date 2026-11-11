from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def cycle_slug(cycle: str) -> str:
    return str(cycle).strip().replace("-", "_")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_bytes(path: Path, content: bytes) -> None:
    ensure_dir(path.parent)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def atomic_write_text(path: Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def write_json(path: Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, ensure_ascii=False, default=str))


def normalize_column_name(value: Any) -> str:
    return str(value).strip()


def clean_dataframe_strings(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for column in out.columns:
        if pd.api.types.is_object_dtype(out[column]) or pd.api.types.is_string_dtype(out[column]):
            out[column] = out[column].map(
                lambda x: x.decode("utf-8", errors="replace").strip()
                if isinstance(x, (bytes, bytearray))
                else (x.strip() if isinstance(x, str) else x)
            )
    return out


def normalize_drug_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).lower()
    text = text.replace("®", " ").replace("™", " ")
    text = re.sub(r"[^a-z0-9+/\- ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def setup_logging(log_path: Path, verbose: bool = False) -> logging.Logger:
    ensure_dir(log_path.parent)
    logger = logging.getLogger("nhanes_etl")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def parse_csv_list(values: str | None) -> list[str]:
    if not values:
        return []
    return [item.strip() for item in values.split(",") if item.strip()]


def ordered_union(items: Iterable[Iterable[str]]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for group in items:
        for item in group:
            if item not in seen:
                out.append(item)
                seen.add(item)
    return out
