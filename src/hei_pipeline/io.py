from __future__ import annotations

from pathlib import Path
import pandas as pd


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path, low_memory=False)
    if suffix == ".xpt":
        return pd.read_sas(path, format="xport", encoding="utf-8")
    if suffix in {".sas7bdat", ".sas7bcat"}:
        return pd.read_sas(path, format="sas7bdat", encoding="utf-8")
    if suffix in {".xls", ".xlsx"}:
        return pd.read_excel(path, sheet_name=0)
    raise ValueError(f"Unsupported table format: {path}")


def locate_table(directory: Path, stems: list[str]) -> Path | None:
    if not directory.exists():
        return None
    lower = {p.name.lower(): p for p in directory.iterdir() if p.is_file()}
    for stem in stems:
        for ext in (".parquet", ".csv", ".xpt", ".sas7bdat", ".xls", ".xlsx"):
            candidate = lower.get((stem + ext).lower())
            if candidate is not None:
                return candidate
    return None


def write_parquet_csv(frame: pd.DataFrame, base: Path) -> tuple[Path, Path]:
    base.parent.mkdir(parents=True, exist_ok=True)
    parquet = base.with_suffix(".parquet")
    csv = base.with_suffix(".csv")
    frame.to_parquet(parquet, index=False)
    frame.to_csv(csv, index=False)
    return parquet, csv
