from __future__ import annotations

from pathlib import Path
import json
import pandas as pd

from .archive import extract_archives
from .config import load_config, resolve
from .discovery import discover_fped
from .fped import choose_reference_file, load_food_reference
from .io import locate_table, read_table, write_parquet_csv
from .linkage import (
    link_food_lines,
    link_participant_line_table,
    participant_day_totals,
)
from .participant_line import (
    find_participant_line_table,
    normalize_participant_line,
)
from .representations import build_representations
from .scoring import score_hei2020, validate_scores


RAW_AMOUNT_COLUMNS = [
    "fruit_total_cup",
    "fruit_juice_cup",
    "vegetable_total_cup",
    "dark_green_vegetable_cup",
    "legumes_vegetable_cup",
    "whole_grain_oz",
    "refined_grain_oz",
    "protein_food_total_oz",
    "seafood_high_omega3_oz",
    "seafood_low_omega3_oz",
    "soy_oz",
    "nuts_seeds_oz",
    "legumes_protein_oz",
    "dairy_total_cup",
    "added_sugars_tsp",
    "energy_kcal",
    "protein_g",
    "carbohydrate_g",
    "total_sugars_g",
    "fiber_g",
    "total_fat_g",
    "saturated_fat_g",
    "monounsaturated_fat_g",
    "polyunsaturated_fat_g",
    "sodium_mg",
    "alcohol_g",
]


def _write_accumulated(frames: list[pd.DataFrame], path: Path) -> None:
    if frames:
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.concat(frames, ignore_index=True).to_csv(path, index=False)


def _nutrient_totals(staged_dir: Path, day: int) -> pd.DataFrame:
    path = locate_table(staged_dir, [f"dr{day}tot"])
    if path is None:
        raise FileNotFoundError(f"Missing staged dr{day}tot in {staged_dir}")

    frame = read_table(path).copy()
    suffix = f"_day{day}"
    aliases = {
        f"energy_kcal{suffix}": "energy_kcal",
        f"protein_g{suffix}": "protein_g",
        f"carbohydrate_g{suffix}": "carbohydrate_g",
        f"total_sugars_g{suffix}": "total_sugars_g",
        f"fiber_g{suffix}": "fiber_g",
        f"total_fat_g{suffix}": "total_fat_g",
        f"saturated_fat_g{suffix}": "saturated_fat_g",
        f"monounsaturated_fat_g{suffix}": "monounsaturated_fat_g",
        f"polyunsaturated_fat_g{suffix}": "polyunsaturated_fat_g",
        f"sodium_mg{suffix}": "sodium_mg",
        f"alcohol_g{suffix}": "alcohol_g",
    }

    lower = {str(column).lower(): column for column in frame.columns}
    rename = {}
    if "seqn" in lower:
        rename[lower["seqn"]] = "seqn"
    for source, target in aliases.items():
        if source.lower() in lower:
            rename[lower[source.lower()]] = target
    frame.rename(columns=rename, inplace=True)

    required = {
        "seqn",
        "energy_kcal",
        "saturated_fat_g",
        "monounsaturated_fat_g",
        "polyunsaturated_fat_g",
        "sodium_mg",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            f"Staged DR{day}TOT is missing normalized variables: {sorted(missing)}"
        )

    keep = ["seqn"] + [
        target for target in aliases.values() if target in frame.columns
    ]
    return frame[keep].drop_duplicates("seqn")


def _processed_cycle(processed_root: Path, folder: str) -> pd.DataFrame | None:
    cycle_dir = processed_root / folder
    path = locate_table(cycle_dir, ["analytic_cohort"])
    if path is None:
        return None
    frame = read_table(path).copy()
    frame.columns = [str(column).lower() for column in frame.columns]
    return frame


def _average_two_days(day1: pd.DataFrame, day2: pd.DataFrame) -> pd.DataFrame:
    columns = [
        column for column in RAW_AMOUNT_COLUMNS
        if column in day1.columns and column in day2.columns
    ]
    left = day1[["seqn", "cycle_id"] + columns].copy()
    right = day2[["seqn", "cycle_id"] + columns].copy()
    merged = left.merge(
        right,
        on=["seqn", "cycle_id"],
        how="inner",
        suffixes=("_d1", "_d2"),
        validate="one_to_one",
    )
    averaged = merged[["seqn", "cycle_id"]].copy()
    for column in columns:
        averaged[column] = merged[[f"{column}_d1", f"{column}_d2"]].mean(
            axis=1
        )
    averaged["recall_day"] = 0
    return averaged


def run(
    config_path: str | Path,
    cycles_requested: list[str] | None = None,
    extract_first: bool = False,
    allow_self_extracting_exe: bool | None = None,
) -> dict:
    cfg = load_config(config_path)
    project_root = Path(cfg["project_root"]).resolve()
    fped_root = resolve(project_root, cfg["fped_root"])
    extracted_root = resolve(project_root, cfg["fped_extracted_root"])
    staged_root = resolve(project_root, cfg["staged_root"])
    processed_root = resolve(project_root, cfg["processed_root"])
    output_root = resolve(project_root, cfg["output_root"])
    report_root = resolve(project_root, cfg["report_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    report_root.mkdir(parents=True, exist_ok=True)

    cycles = cfg["cycles"]
    if cycles_requested:
        cycles = {cycle: cycles[cycle] for cycle in cycles_requested}

    discovery = discover_fped(fped_root, cycles)
    discovery.to_csv(report_root / "fped_archive_discovery.csv", index=False)

    if extract_first:
        allow_exe = (
            cfg["archives"]["allow_self_extracting_exe"]
            if allow_self_extracting_exe is None
            else allow_self_extracting_exe
        )
        extraction = extract_archives(
            fped_root,
            extracted_root,
            allow_self_extracting_exe=bool(allow_exe),
        )
        extraction.to_csv(report_root / "fped_extraction_report.csv", index=False)

    schema_audits: list[pd.DataFrame] = []
    participant_line_audits: list[pd.DataFrame] = []
    coverage_reports: list[pd.DataFrame] = []
    unmatched_reports: list[pd.DataFrame] = []
    score_audits: list[pd.DataFrame] = []
    manifests: list[pd.DataFrame] = []
    all_primary: list[pd.DataFrame] = []
    all_twoday: list[pd.DataFrame] = []

    for cycle, meta in cycles.items():
        folder = meta["folder"]
        staged_dir = staged_root / folder
        cycle_fped_dir = fped_root / folder

        reference_path = choose_reference_file(
            cycle_fped_dir,
            meta["fped_tag"],
        )
        reference, schema = load_food_reference(reference_path, cycle)
        schema["source_path"] = str(reference_path)
        schema_audits.append(schema)
        _write_accumulated(
            schema_audits,
            report_root / "fped_schema_audit.csv",
        )
        if schema["hard_fail"].any():
            raise RuntimeError(f"{cycle}: FPED schema hard failure")

        day_scores: dict[int, pd.DataFrame] = {}
        for day in (1, 2):
            iff_path = locate_table(staged_dir, [f"dr{day}iff"])
            if iff_path is None:
                if day == 2:
                    continue
                raise FileNotFoundError(
                    f"{cycle}: missing staged DR{day}IFF"
                )
            iff = read_table(iff_path)

            participant_path = None
            if cfg["archives"]["prefer_official_participant_line_tables"]:
                participant_path = find_participant_line_table(
                    cycle_fped_dir,
                    extracted_root / folder,
                    meta["fped_tag"],
                    day,
                )

            if participant_path is not None:
                participant_fped, participant_audit = (
                    normalize_participant_line(
                        read_table(participant_path),
                        cycle,
                        day,
                    )
                )
                participant_audit["source_path"] = str(participant_path)
                participant_line_audits.append(participant_audit)
                _write_accumulated(
                    participant_line_audits,
                    report_root / "participant_line_schema_audit.csv",
                )
                if participant_audit["hard_fail"].any():
                    raise RuntimeError(
                        f"{cycle} day {day}: participant-line schema failure"
                    )
                linked, coverage, unmatched = link_participant_line_table(
                    iff,
                    participant_fped,
                    cycle,
                    day,
                )
            else:
                linked, coverage, unmatched = link_food_lines(
                    iff,
                    reference,
                    cycle,
                    day,
                    use_modcode=bool(meta["use_modcode"]),
                    values_per_100g=bool(
                        cfg["linkage"]["fped_values_are_per_100g"]
                    ),
                )
                if (
                    bool(coverage.loc[0, "provisional"])
                    and not cfg["linkage"][
                        "allow_foodcode_modifier0_fallback"
                    ]
                ):
                    raise RuntimeError(
                        f"{cycle} day {day}: modifier-code fallback is disabled"
                    )

            coverage_reports.append(coverage)
            _write_accumulated(
                coverage_reports,
                report_root / "linkage_coverage.csv",
            )

            if not unmatched.empty:
                unmatched["cycle"] = cycle
                unmatched["day"] = day
                unmatched_reports.append(unmatched)
                _write_accumulated(
                    unmatched_reports,
                    report_root / "unmatched_food_codes.csv",
                )

            line_min = float(cfg["linkage"]["minimum_line_coverage"])
            participant_min = float(
                cfg["linkage"]["minimum_participant_coverage"]
            )
            if cfg["linkage"]["fail_below_threshold"]:
                if (
                    coverage.loc[0, "line_coverage"] < line_min
                    or coverage.loc[0, "participant_coverage"]
                    < participant_min
                ):
                    raise RuntimeError(
                        f"{cycle} day {day}: linkage coverage below "
                        "prespecified threshold; inspect "
                        "reports/hei2020/linkage_coverage.csv"
                    )

            totals = participant_day_totals(linked, cycle, day)
            nutrients = _nutrient_totals(staged_dir, day)
            day_frame = totals.merge(
                nutrients,
                on="seqn",
                how="left",
                validate="one_to_one",
            )
            scores = score_hei2020(
                day_frame,
                added_sugar_kcal_per_tsp=float(
                    cfg["scoring"]["added_sugar_kcal_per_tsp"]
                ),
            )
            audit = validate_scores(scores)
            audit["cycle"] = cycle
            audit["day"] = day
            score_audits.append(audit)
            _write_accumulated(
                score_audits,
                report_root / "hei_score_range_audit.csv",
            )
            if audit["hard_fail"].any():
                raise RuntimeError(
                    f"{cycle} day {day}: HEI score range failure"
                )

            write_parquet_csv(
                scores,
                output_root
                / "participant_day"
                / folder
                / f"hei2020_day{day}",
            )
            day_scores[day] = scores

        primary = day_scores[1].copy()
        primary["analysis_view"] = "day1_primary"
        processed = _processed_cycle(processed_root, folder)
        if processed is not None:
            join_keys = ["seqn"]
            if "cycle_id" in processed.columns:
                join_keys.append("cycle_id")
            primary = processed.merge(
                primary,
                on=join_keys,
                how="inner",
                validate="one_to_one",
                suffixes=("", "_hei"),
            )
        write_parquet_csv(
            primary,
            output_root
            / "participant_cycle"
            / folder
            / "hei2020_day1_primary",
        )
        all_primary.append(primary)

        if 2 in day_scores:
            averaged = _average_two_days(day_scores[1], day_scores[2])
            averaged = score_hei2020(
                averaged,
                float(cfg["scoring"]["added_sugar_kcal_per_tsp"]),
            )
            averaged["analysis_view"] = (
                "two_day_average_sensitivity"
            )
            if processed is not None:
                join_keys = ["seqn"]
                if "cycle_id" in processed.columns:
                    join_keys.append("cycle_id")
                averaged = processed.merge(
                    averaged,
                    on=join_keys,
                    how="inner",
                    validate="one_to_one",
                    suffixes=("", "_hei"),
                )
            write_parquet_csv(
                averaged,
                output_root
                / "participant_cycle"
                / folder
                / "hei2020_two_day_average",
            )
            all_twoday.append(averaged)

    primary_all = pd.concat(all_primary, ignore_index=True)
    write_parquet_csv(
        primary_all,
        output_root / "pooled" / "hei2020_day1_primary",
    )
    manifests.append(
        build_representations(
            primary_all,
            cfg,
            output_root / "representations",
            "day1_primary",
        )
    )

    if all_twoday:
        twoday_all = pd.concat(all_twoday, ignore_index=True)
        write_parquet_csv(
            twoday_all,
            output_root / "pooled" / "hei2020_two_day_average",
        )
        manifests.append(
            build_representations(
                twoday_all,
                cfg,
                output_root / "representations",
                "two_day_average",
            )
        )

    manifest_all = pd.concat(manifests, ignore_index=True)
    manifest_all.to_csv(
        report_root / "representation_manifest.csv",
        index=False,
    )

    coverage_all = pd.concat(coverage_reports, ignore_index=True)
    score_all = pd.concat(score_audits, ignore_index=True)
    provisional = coverage_all["provisional"].fillna(False).astype(bool)

    result = {
        "cycles": list(cycles),
        "primary_rows": int(len(primary_all)),
        "linkage_min_line": float(
            coverage_all["line_coverage"].min()
        ),
        "linkage_min_participant": float(
            coverage_all["participant_coverage"].min()
        ),
        "provisional_linkage_rows": int(provisional.sum()),
        "score_hard_failures": int(score_all["hard_fail"].sum()),
        "output_root": str(output_root),
        "report_root": str(report_root),
        "methodological_status": (
            "provisional_modifier0_fallback"
            if provisional.any()
            else "final_official_participant_line_or_unique_foodcode"
        ),
    }
    with (report_root / "run_summary.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(result, handle, indent=2)
    return result
