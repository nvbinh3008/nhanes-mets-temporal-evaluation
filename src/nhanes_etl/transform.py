from __future__ import annotations

from pathlib import Path

import pandas as pd

from .audit import read_xpt
from .config import DictionaryConfig, Settings
from .exceptions import SchemaAuditError
from .io import output_path, write_frame
from .utils import cycle_slug
from .validation import clean_known_missing


def canonicalize_raw_file(
    row: dict[str, object],
    dictionary: DictionaryConfig,
    settings: Settings,
) -> tuple[pd.DataFrame, Path]:
    cycle = str(row["cycle"])
    cycle_role = str(row["cycle_role"])
    suffix = str(row["suffix"])
    stem = str(row["file_stem"]).upper()
    file_name = str(row["xpt_file"]).upper()
    raw_path = settings.raw_dir / cycle_slug(cycle) / file_name
    if not raw_path.exists():
        raise SchemaAuditError(f"Cannot stage missing raw file: {raw_path}")

    frame = read_xpt(raw_path)
    expected = dictionary.expected_variables(cycle, file_name)
    missing = sorted(set(expected["original_variable"]) - set(frame.columns))
    if missing:
        raise SchemaAuditError(f"{cycle}/{file_name} is missing locked variables: {missing}")

    # SEQN is a structural join key in every NHANES component file even when
    # the dictionary's administrative mapping is recorded only once per cycle.
    keep = ["SEQN"] if "SEQN" in frame.columns else []
    keep.extend(expected["original_variable"].tolist())
    keep = list(dict.fromkeys(keep))
    staged = frame[keep].copy()

    rename_map = dict(
        zip(expected["original_variable"], expected["normalized_variable"])
    )
    staged = staged.rename(columns=rename_map)
    if "SEQN" in staged.columns:
        staged = staged.rename(columns={"SEQN": "seqn"})

    # Apply only prespecified questionnaire missing codes. Continuous values
    # outside expected ranges remain visible to the audit and are not silently
    # winsorized or set to missing.
    for canonical in expected["normalized_variable"]:
        if canonical in staged.columns:
            staged[canonical] = clean_known_missing(staged[canonical], canonical)

    staged.insert(0, "source_file", file_name)
    staged.insert(0, "suffix", suffix)
    staged.insert(0, "cycle_role", cycle_role)
    staged.insert(0, "cycle_id", cycle)

    cycle_dir = settings.staged_dir / cycle_slug(cycle)
    target = output_path(cycle_dir, stem.lower(), settings.output_format)
    write_frame(staged, target, settings.output_format)
    return staged, target


def stage_plan(
    plan: pd.DataFrame,
    dictionary: DictionaryConfig,
    settings: Settings,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for row in plan.to_dict(orient="records"):
        staged, target = canonicalize_raw_file(row, dictionary, settings)
        rows.append({
            "cycle": row["cycle"],
            "file_stem": row["file_stem"],
            "xpt_file": row["xpt_file"],
            "rows": len(staged),
            "columns": len(staged.columns),
            "output": str(target),
        })
    return pd.DataFrame(rows)
