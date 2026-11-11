from __future__ import annotations

from pathlib import Path
import re
import pandas as pd

from .fped import CANONICAL, REQUIRED, _base_name


SUPPORTED = {".parquet", ".csv", ".xpt", ".sas7bdat", ".xls", ".xlsx"}


def find_participant_line_table(
    cycle_folder: Path,
    extracted_root: Path,
    tag: str,
    day: int,
) -> Path | None:
    stems = [
        f"fped_dr{day}iff_{tag}",
        f"dr{day}iff_fped_{tag}",
        f"fped_dr{day}iff",
    ]
    candidates: list[Path] = []

    for root in (cycle_folder, extracted_root):
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in SUPPORTED:
                continue
            if any(path.stem.lower().startswith(stem) for stem in stems):
                candidates.append(path)

    if not candidates:
        return None

    priority = {
        ".parquet": 0,
        ".csv": 1,
        ".sas7bdat": 2,
        ".xpt": 3,
        ".xlsx": 4,
        ".xls": 5,
    }
    return sorted(
        candidates,
        key=lambda path: (
            priority.get(path.suffix.lower(), 99),
            len(path.parts),
            str(path).lower(),
        ),
    )[0]


def _participant_component_name(column: object, day: int) -> str:
    """Remove the official DR1I_/DR2I_ prefix from FPED component names."""
    base = _base_name(column)
    return re.sub(rf"^DR{day}I_?", "", base, count=1)


def normalize_participant_line(
    frame: pd.DataFrame,
    cycle: str,
    day: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rename: dict[object, str] = {}
    lower = {str(column).strip().lower(): column for column in frame.columns}

    aliases = {
        "seqn": ["seqn"],
        "food_line": [f"dr{day}iline", f"day{day}_food_line", "food_line"],
        "food_code": [
            f"dr{day}ifdcd",
            f"day{day}_usda_food_code",
            "food_code",
        ],
        "modifier_code": [
            f"dr{day}mc",
            f"day{day}_modifier_code",
            "modifier_code",
            "modcode",
        ],
        "food_grams": [
            f"dr{day}igrms",
            f"day{day}_food_grams",
            "food_grams",
        ],
        "description": ["description"],
    }

    for target, candidates in aliases.items():
        for candidate in candidates:
            original = lower.get(candidate.lower())
            if original is not None:
                rename[original] = target
                break

    for column in frame.columns:
        component = _participant_component_name(column, day)
        if component in CANONICAL:
            rename[column] = CANONICAL[component]

    output = frame.rename(columns=rename).copy()
    output["cycle_id"] = cycle
    output["recall_day"] = day

    required = {"seqn", "food_line"} | (REQUIRED - {"food_code"})
    missing = sorted(required - set(output.columns))

    audit = {
        "cycle": cycle,
        "day": day,
        "source_type": "participant_line",
        "rows": int(len(output)),
        "columns": int(len(output.columns)),
        "missing_required": "|".join(missing),
        "missing_key_rows": 0,
        "duplicate_keys": 0,
        "hard_fail": bool(missing),
    }

    if missing:
        return output, pd.DataFrame([audit])

    output["seqn"] = pd.to_numeric(
        output["seqn"], errors="coerce"
    ).astype("Int64")
    output["food_line"] = pd.to_numeric(
        output["food_line"], errors="coerce"
    ).astype("Int64")

    for optional_integer in ("food_code", "modifier_code"):
        if optional_integer in output.columns:
            output[optional_integer] = pd.to_numeric(
                output[optional_integer], errors="coerce"
            ).astype("Int64")

    if "food_grams" in output.columns:
        output["food_grams"] = pd.to_numeric(
            output["food_grams"], errors="coerce"
        )

    measure_columns = [
        column
        for column in output.columns
        if column.endswith(("_cup", "_oz", "_g", "_tsp"))
    ]
    for column in measure_columns:
        output[column] = pd.to_numeric(output[column], errors="coerce")
        if output[column].dropna().lt(0).any():
            audit["hard_fail"] = True
            problem = f"negative:{column}"
            audit["missing_required"] = "|".join(
                item
                for item in (audit["missing_required"], problem)
                if item
            )

    missing_key = output[["seqn", "food_line"]].isna().any(axis=1)
    duplicate_key = output.duplicated(
        ["seqn", "food_line"], keep=False
    )

    audit["missing_key_rows"] = int(missing_key.sum())
    audit["duplicate_keys"] = int(duplicate_key.sum())
    if missing_key.any() or duplicate_key.any():
        audit["hard_fail"] = True

    return output, pd.DataFrame([audit])
