from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import DictionaryConfig, Settings
from .exceptions import SchemaAuditError
from .utils import cycle_slug, clean_dataframe_strings
from .validation import clean_known_missing, invalid_mask, summarize_series


def read_xpt(path: Path) -> pd.DataFrame:
    try:
        frame = pd.read_sas(path, format="xport", encoding="utf-8")
    except UnicodeDecodeError:
        frame = pd.read_sas(path, format="xport")
    frame.columns = [str(col).upper() for col in frame.columns]
    return clean_dataframe_strings(frame)


def audit_raw_file(
    row: dict[str, object],
    dictionary: DictionaryConfig,
    settings: Settings,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cycle = str(row["cycle"])
    file_name = str(row["xpt_file"]).upper()
    stem = str(row["file_stem"]).upper()
    path = settings.raw_dir / cycle_slug(cycle) / file_name
    file_rows: list[dict[str, object]] = []
    variable_rows: list[dict[str, object]] = []

    if not path.exists():
        file_rows.append({
            "cycle": cycle, "file": file_name, "stem": stem, "path": str(path),
            "readable": False, "rows": 0, "columns": 0, "seqn_present": False,
            "seqn_unique": False, "structure_expected": "one_to_many" if stem in settings.one_to_many_stems else "person_level",
            "hard_fail": True, "detail": "raw file missing",
        })
        return pd.DataFrame(file_rows), pd.DataFrame(variable_rows)

    try:
        frame = read_xpt(path)
    except Exception as exc:
        file_rows.append({
            "cycle": cycle, "file": file_name, "stem": stem, "path": str(path),
            "readable": False, "rows": 0, "columns": 0, "seqn_present": False,
            "seqn_unique": False, "structure_expected": "one_to_many" if stem in settings.one_to_many_stems else "person_level",
            "hard_fail": True, "detail": f"could not read XPT: {exc}",
        })
        return pd.DataFrame(file_rows), pd.DataFrame(variable_rows)

    seqn_present = "SEQN" in frame.columns
    seqn_unique = bool(frame["SEQN"].is_unique) if seqn_present else False
    is_one_to_many = stem in settings.one_to_many_stems
    uniqueness_fail = (not is_one_to_many) and (not seqn_unique)
    file_rows.append({
        "cycle": cycle, "file": file_name, "stem": stem, "path": str(path),
        "readable": True, "rows": len(frame), "columns": len(frame.columns),
        "seqn_present": seqn_present, "seqn_unique": seqn_unique,
        "structure_expected": "one_to_many" if is_one_to_many else "person_level",
        "hard_fail": (not seqn_present) or uniqueness_fail,
        "detail": "" if seqn_present and not uniqueness_fail else "SEQN missing or unexpected duplicates",
    })

    expected = dictionary.expected_variables(cycle, file_name)
    for mapping in expected.to_dict(orient="records"):
        raw_name = str(mapping["original_variable"]).upper()
        canonical = str(mapping["normalized_variable"])
        present = raw_name in frame.columns
        unit_or_coding = str(mapping.get("unit_or_coding", ""))
        expected_type = (
            "text"
            if "string" in unit_or_coding.lower()
            or canonical in {"rx_generic_name", "rx_generic_drug_id"}
            else "numeric"
        )
        record = {
            "cycle": cycle,
            "file": file_name,
            "stem": stem,
            "original_variable": raw_name,
            "normalized_variable": canonical,
            "present": present,
            "expected_unit_or_coding": unit_or_coding,
            "valid_condition": mapping.get("valid_condition", ""),
            "analysis_role": mapping.get("analysis_role", ""),
            "blacklist_status": mapping.get("blacklist_status", ""),
            "expected_type_class": expected_type,
            "dtype": "",
            "type_valid": False,
            "rows": len(frame),
            "nonmissing": 0,
            "missing": len(frame),
            "distinct_nonmissing": 0,
            "min": "",
            "max": "",
            "special_missing_count": "",
            "invalid_count": "",
            "hard_fail": not present,
        }
        if present:
            raw_series = frame[raw_name]
            summary = summarize_series(raw_series)
            record.update(summary)
            if expected_type == "numeric":
                type_valid = bool(
                    pd.api.types.is_numeric_dtype(raw_series)
                    or pd.to_numeric(raw_series.dropna().head(1000), errors="coerce").notna().all()
                )
            else:
                type_valid = bool(
                    pd.api.types.is_object_dtype(raw_series)
                    or pd.api.types.is_string_dtype(raw_series)
                )
            cleaned = clean_known_missing(raw_series, canonical)
            special_missing_count = int((raw_series.notna() & cleaned.isna()).sum())
            invalid_count = int(
                invalid_mask(cleaned, canonical, str(mapping.get("valid_condition", ""))).sum()
            )
            record["type_valid"] = type_valid
            record["special_missing_count"] = special_missing_count
            record["invalid_count"] = invalid_count
            record["hard_fail"] = (not type_valid) or invalid_count > 0
        variable_rows.append(record)

    return pd.DataFrame(file_rows), pd.DataFrame(variable_rows)


def audit_plan(
    plan: pd.DataFrame,
    dictionary: DictionaryConfig,
    settings: Settings,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    file_reports = []
    variable_reports = []
    for row in plan.to_dict(orient="records"):
        fr, vr = audit_raw_file(row, dictionary, settings)
        file_reports.append(fr)
        variable_reports.append(vr)
    files = pd.concat(file_reports, ignore_index=True) if file_reports else pd.DataFrame()
    variables = pd.concat(variable_reports, ignore_index=True) if variable_reports else pd.DataFrame()
    return files, variables


def schema_has_hard_fail(files: pd.DataFrame, variables: pd.DataFrame) -> bool:
    return bool(
        (not files.empty and files["hard_fail"].astype(bool).any())
        or (not variables.empty and variables["hard_fail"].astype(bool).any())
    )
