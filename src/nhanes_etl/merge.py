from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import Settings
from .exceptions import MergeError
from .io import output_path, read_frame, write_frame
from .utils import cycle_slug


def staged_path(settings: Settings, cycle: str, stem: str) -> Path:
    return output_path(
        settings.staged_dir / cycle_slug(cycle),
        stem.lower(),
        settings.output_format,
    )


def load_staged_if_exists(settings: Settings, cycle: str, stem: str) -> pd.DataFrame | None:
    path = staged_path(settings, cycle, stem)
    if not path.exists():
        return None
    return read_frame(path)


def _assert_person_level(frame: pd.DataFrame, cycle: str, stem: str) -> None:
    if "seqn" not in frame:
        raise MergeError(f"{cycle}/{stem} has no seqn after staging")
    if frame["seqn"].isna().any():
        raise MergeError(f"{cycle}/{stem} has missing seqn")
    duplicates = int(frame["seqn"].duplicated().sum())
    if duplicates:
        raise MergeError(f"{cycle}/{stem} unexpectedly repeats seqn ({duplicates} rows)")


def aggregate_food_qc(frame: pd.DataFrame, day: int) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["seqn"])
    grams = f"day{day}_food_grams"
    code = f"day{day}_usda_food_code"
    line = f"day{day}_food_line"
    grouped = frame.groupby("seqn", dropna=False)
    out = grouped.size().rename(f"day{day}_food_line_count").to_frame()
    if grams in frame:
        out[f"day{day}_food_grams_sum_qc"] = grouped[grams].sum(min_count=1)
    if code in frame:
        out[f"day{day}_unique_food_codes_qc"] = grouped[code].nunique(dropna=True)
    if line in frame:
        out[f"day{day}_duplicate_line_number_qc"] = grouped[line].apply(
            lambda x: int(x.duplicated().any())
        )
    return out.reset_index()


def merge_person_level_cycle(
    settings: Settings,
    cycle: str,
    medication_flags: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    demo = load_staged_if_exists(settings, cycle, "DEMO")
    if demo is None:
        raise MergeError(f"{cycle}: DEMO staged file is required as the merge base")
    _assert_person_level(demo, cycle, "DEMO")

    administrative = {"cycle_id", "cycle_role", "suffix", "source_file"}
    base = demo.copy()
    merge_audit: list[dict[str, object]] = [{
        "cycle": cycle,
        "source": "DEMO",
        "rows_before": len(base),
        "rows_source": len(demo),
        "matched": len(base),
        "unmatched_base": 0,
        "rows_after": len(base),
        "status": "base",
    }]

    for stem in settings.person_level_stems:
        stem = stem.upper()
        if stem == "DEMO":
            continue
        frame = load_staged_if_exists(settings, cycle, stem)
        if frame is None:
            continue
        _assert_person_level(frame, cycle, stem)
        source = frame.drop(columns=[c for c in administrative if c in frame], errors="ignore")
        duplicate_columns = sorted((set(base.columns) & set(source.columns)) - {"seqn"})
        if duplicate_columns:
            raise MergeError(
                f"{cycle}/{stem} would overwrite canonical columns: {duplicate_columns}"
            )
        before = len(base)
        source_seqn = set(source["seqn"].dropna())
        matched = int(base["seqn"].isin(source_seqn).sum())
        base = base.merge(source, on="seqn", how="left", validate="one_to_one")
        if len(base) != before:
            raise MergeError(f"{cycle}/{stem} changed the DEMO row count")
        merge_audit.append({
            "cycle": cycle,
            "source": stem,
            "rows_before": before,
            "rows_source": len(source),
            "matched": matched,
            "unmatched_base": before - matched,
            "rows_after": len(base),
            "status": "left_join_one_to_one",
        })

    for stem, day in (("DR1IFF", 1), ("DR2IFF", 2)):
        frame = load_staged_if_exists(settings, cycle, stem)
        if frame is None:
            continue
        if "seqn" not in frame:
            raise MergeError(f"{cycle}/{stem} has no seqn")
        qc = aggregate_food_qc(frame, day)
        before = len(base)
        matched = int(base["seqn"].isin(set(qc["seqn"])).sum())
        base = base.merge(qc, on="seqn", how="left", validate="one_to_one")
        merge_audit.append({
            "cycle": cycle,
            "source": f"{stem}_QC",
            "rows_before": before,
            "rows_source": len(qc),
            "matched": matched,
            "unmatched_base": before - matched,
            "rows_after": len(base),
            "status": "one_to_many_aggregated_before_join",
        })

    if medication_flags is not None and not medication_flags.empty:
        if medication_flags["seqn"].duplicated().any():
            raise MergeError(f"{cycle}/medication_flags repeats seqn")
        before = len(base)
        matched = int(base["seqn"].isin(set(medication_flags["seqn"])).sum())
        base = base.merge(medication_flags, on="seqn", how="left", validate="one_to_one")
        # Preserve missing medication ascertainment for DEMO participants absent
        # from RXQ_RX; zeros are assigned only inside map_medications for
        # participants with an RXQ_RX record.
        merge_audit.append({
            "cycle": cycle,
            "source": "RXQ_RX_flags",
            "rows_before": before,
            "rows_source": len(medication_flags),
            "matched": matched,
            "unmatched_base": before - matched,
            "rows_after": len(base),
            "status": "one_to_many_mapped_and_aggregated_before_join",
        })

    if base["seqn"].duplicated().any():
        raise MergeError(f"{cycle}: final person-level table repeats seqn")

    audit = pd.DataFrame(merge_audit)
    target = output_path(
        settings.processed_dir / cycle_slug(cycle),
        "person_level_harmonized",
        settings.output_format,
    )
    write_frame(base, target, settings.output_format)
    return base, audit
