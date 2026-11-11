from __future__ import annotations

from pathlib import Path
import hashlib
import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_fped(fped_root: Path, cycles: dict) -> pd.DataFrame:
    records = []
    for cycle, meta in cycles.items():
        folder = fped_root / meta["folder"]
        if not folder.exists():
            records.append({
                "cycle": cycle, "folder": str(folder), "kind": "missing",
                "path": "", "bytes": 0, "sha256": "", "usable": False,
            })
            continue
        for path in sorted(folder.rglob("*")):
            if not path.is_file():
                continue
            name = path.name.lower()
            if name.startswith("fped_dr1iff"):
                kind = "participant_line_day1_archive"
            elif name.startswith("fped_dr2iff"):
                kind = "participant_line_day2_archive"
            elif name.startswith("fped_dr1tot"):
                kind = "participant_total_day1_archive"
            elif name.startswith("fped_dr2tot"):
                kind = "participant_total_day2_archive"
            elif name.startswith("fped_") and path.suffix.lower() in {".xls", ".xlsx", ".csv", ".sas7bdat", ".xpt"}:
                kind = "food_code_reference"
            elif name.startswith("fpid_"):
                kind = "ingredient_reference"
            else:
                kind = "support"
            records.append({
                "cycle": cycle,
                "folder": str(folder),
                "kind": kind,
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "usable": path.suffix.lower() in {".xls", ".xlsx", ".csv", ".sas7bdat", ".xpt"},
            })
    return pd.DataFrame(records)
