from __future__ import annotations

import numpy as np
import pandas as pd


def canonicalize_iff(frame: pd.DataFrame, day: int) -> pd.DataFrame:
    prefix = f"day{day}_"
    aliases = {
        "seqn": ["seqn", "SEQN"],
        "food_line": [f"{prefix}food_line", f"DR{day}ILINE"],
        "food_code": [f"{prefix}usda_food_code", f"DR{day}IFDCD"],
        "food_grams": [f"{prefix}food_grams", f"DR{day}IGRMS"],
        "modifier_code": [f"{prefix}modifier_code", f"DR{day}MC", "MODCODE"],
    }
    rename = {}
    lower = {str(c).lower(): c for c in frame.columns}
    for target, names in aliases.items():
        for name in names:
            found = lower.get(name.lower())
            if found is not None:
                rename[found] = target
                break
    out = frame.rename(columns=rename).copy()
    needed = {"seqn", "food_line", "food_code", "food_grams"}
    missing = needed - set(out.columns)
    if missing:
        raise ValueError(f"Day {day} IFF is missing columns: {sorted(missing)}")
    out["seqn"] = pd.to_numeric(out["seqn"], errors="coerce").astype("Int64")
    out["food_code"] = pd.to_numeric(out["food_code"], errors="coerce").astype("Int64")
    out["food_grams"] = pd.to_numeric(out["food_grams"], errors="coerce")
    if "modifier_code" in out.columns:
        out["modifier_code"] = pd.to_numeric(out["modifier_code"], errors="coerce").fillna(0).astype("Int64")
    return out


def link_food_lines(
    iff: pd.DataFrame,
    reference: pd.DataFrame,
    cycle: str,
    day: int,
    use_modcode: bool,
    values_per_100g: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    iff = canonicalize_iff(iff, day)
    ref = reference.copy()

    if use_modcode and "modifier_code" in iff.columns and "modifier_code" in ref.columns:
        keys = ["food_code", "modifier_code"]
        duplicate_keys = ref.duplicated(keys, keep=False)
        if duplicate_keys.any():
            raise ValueError(f"{cycle}: duplicate FPED keys for {keys}")
    else:
        keys = ["food_code"]
        if "modifier_code" in ref.columns:
            ref = ref.loc[ref["modifier_code"].fillna(0).eq(0)].copy()
        duplicate_keys = ref.duplicated(keys, keep=False)
        if duplicate_keys.any():
            dup = ref.loc[duplicate_keys, keys].drop_duplicates().head(20)
            raise ValueError(
                f"{cycle}: food-code fallback is not unique after modifier=0 filter: "
                f"{dup.to_dict('records')}"
            )

    ref_cols = [c for c in ref.columns if c not in {"cycle_id", "DESCRIPTION", "description"}]
    linked = iff.merge(ref[ref_cols], on=keys, how="left", validate="many_to_one", indicator=True)
    linked["fped_matched"] = linked["_merge"].eq("both")
    linked.drop(columns="_merge", inplace=True)

    pattern_cols = [c for c in ref_cols if c.endswith(("_cup", "_oz", "_g", "_tsp"))]
    scale = linked["food_grams"] / 100.0 if values_per_100g else 1.0
    for col in pattern_cols:
        linked[col] = pd.to_numeric(linked[col], errors="coerce") * scale

    valid_lines = linked["food_code"].notna() & linked["food_grams"].notna()
    line_cov = linked.loc[valid_lines, "fped_matched"].mean() if valid_lines.any() else np.nan
    participant_match = linked.groupby("seqn", dropna=True)["fped_matched"].all()
    participant_cov = participant_match.mean() if len(participant_match) else np.nan

    fallback_without_modifier = (
        use_modcode
        and "modifier_code" not in iff.columns
        and "modifier_code" in reference.columns
    )
    coverage = pd.DataFrame([{
        "cycle": cycle,
        "day": day,
        "linkage_mode": (
            "foodcode_modifier0_fallback"
            if fallback_without_modifier
            else "foodcode_reference"
        ),
        "provisional": bool(fallback_without_modifier),
        "join_keys": "+".join(keys),
        "food_lines": int(len(linked)),
        "valid_food_lines": int(valid_lines.sum()),
        "matched_food_lines": int((linked["fped_matched"] & valid_lines).sum()),
        "line_coverage": line_cov,
        "participants": int(linked["seqn"].nunique()),
        "participants_all_lines_matched": int(participant_match.sum()),
        "participant_coverage": participant_cov,
    }])

    unmatched = (
        linked.loc[valid_lines & ~linked["fped_matched"], keys + ["food_grams"]]
        .groupby(keys, dropna=False)
        .agg(unmatched_lines=("food_grams", "size"), unmatched_grams=("food_grams", "sum"))
        .reset_index()
        .sort_values(["unmatched_grams", "unmatched_lines"], ascending=False)
    )
    return linked, coverage, unmatched


def participant_day_totals(linked: pd.DataFrame, cycle: str, day: int) -> pd.DataFrame:
    pattern_cols = [c for c in linked.columns if c.endswith(("_cup", "_oz", "_g", "_tsp"))]
    # Exclude food grams from pattern totals.
    pattern_cols = [c for c in pattern_cols if c != "food_grams"]
    totals = linked.groupby("seqn", as_index=False)[pattern_cols].sum(min_count=1)
    totals["cycle_id"] = cycle
    totals["recall_day"] = day
    totals["food_line_count"] = linked.groupby("seqn").size().reindex(totals["seqn"]).to_numpy()
    totals["all_food_lines_matched"] = (
        linked.groupby("seqn")["fped_matched"].all().reindex(totals["seqn"]).to_numpy()
    )
    return totals


def link_participant_line_table(
    iff: pd.DataFrame,
    participant_fped: pd.DataFrame,
    cycle: str,
    day: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Link official FPED participant-line data by SEQN and recall line."""
    base = canonicalize_iff(iff, day)
    official = participant_fped.copy()

    keys = ["seqn", "food_line"]
    duplicate = official.duplicated(keys, keep=False)
    if duplicate.any():
        raise ValueError(
            f"{cycle} day {day}: duplicate official participant-line keys"
        )

    pattern_cols = [
        col for col in official.columns
        if col.endswith(("_cup", "_oz", "_g", "_tsp"))
    ]
    keep = keys + pattern_cols
    linked = base.merge(
        official[keep],
        on=keys,
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    linked["fped_matched"] = linked["_merge"].eq("both")
    linked.drop(columns="_merge", inplace=True)

    valid_lines = (
        linked["food_code"].notna()
        & linked["food_grams"].notna()
    )
    line_coverage = (
        linked.loc[valid_lines, "fped_matched"].mean()
        if valid_lines.any() else np.nan
    )
    participant_match = linked.groupby("seqn", dropna=True)["fped_matched"].all()
    participant_coverage = (
        participant_match.mean() if len(participant_match) else np.nan
    )

    coverage = pd.DataFrame([{
        "cycle": cycle,
        "day": day,
        "linkage_mode": "official_participant_line",
        "join_keys": "seqn+food_line",
        "provisional": False,
        "food_lines": int(len(linked)),
        "valid_food_lines": int(valid_lines.sum()),
        "matched_food_lines": int(
            (linked["fped_matched"] & valid_lines).sum()
        ),
        "line_coverage": line_coverage,
        "participants": int(linked["seqn"].nunique()),
        "participants_all_lines_matched": int(participant_match.sum()),
        "participant_coverage": participant_coverage,
    }])

    unmatched = (
        linked.loc[
            valid_lines & ~linked["fped_matched"],
            ["seqn", "food_line", "food_code", "food_grams"],
        ]
        .copy()
        .sort_values("food_grams", ascending=False)
    )
    return linked, coverage, unmatched
