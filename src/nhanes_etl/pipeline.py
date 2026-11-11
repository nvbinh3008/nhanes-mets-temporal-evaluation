from __future__ import annotations

import shutil
from pathlib import Path
from typing import Iterable

import pandas as pd

from .audit import audit_plan, schema_has_hard_fail
from .config import DictionaryConfig, Settings
from .derive import apply_primary_cohort, derive_cycle_variables
from .downloader import download_plan, download_has_failures
from .features import primary_allowed_columns
from .io import output_path, read_frame, write_frame, write_report
from .medications import (
    map_medications,
    medication_mapping_summary,
    unmapped_medication_summary,
    write_medication_outputs,
)
from .merge import load_staged_if_exists, merge_person_level_cycle
from .summaries import build_weighted_component_summary, build_weighted_cycle_summary
from .transform import stage_plan
from .utils import cycle_slug, ensure_dir, sha256_file, utc_now_iso, write_json




def missing_raw_files(plan: pd.DataFrame, settings: Settings) -> list[Path]:
    missing: list[Path] = []
    for row in plan.to_dict(orient="records"):
        path = settings.raw_dir / cycle_slug(str(row["cycle"])) / str(row["xpt_file"]).upper()
        if not path.exists():
            missing.append(path)
    return missing


def save_config_snapshot(
    dictionary_path: Path,
    settings_path: Path,
    settings: Settings,
) -> None:
    target = ensure_dir(settings.reports_dir / "config_snapshot")
    shutil.copy2(dictionary_path, target / dictionary_path.name)
    shutil.copy2(settings_path, target / settings_path.name)
    write_json(
        target / "snapshot_manifest.json",
        {
            "created_utc": utc_now_iso(),
            "dictionary": str(dictionary_path),
            "dictionary_sha256": sha256_file(dictionary_path),
            "settings": str(settings_path),
            "settings_sha256": sha256_file(settings_path),
        },
    )


def etl_cycle(
    cycle: str,
    cycle_plan: pd.DataFrame,
    dictionary: DictionaryConfig,
    settings: Settings,
) -> dict[str, object]:
    if cycle_plan.empty:
        raise ValueError(f"No selected files for cycle {cycle}")

    stage_manifest = stage_plan(cycle_plan, dictionary, settings)
    write_report(
        stage_manifest,
        settings.reports_dir / "staging" / f"{cycle_slug(cycle)}_stage_manifest.csv",
    )

    rx = load_staged_if_exists(settings, cycle, "RXQ_RX")
    medication_flags = None
    medication_matches = pd.DataFrame()
    if rx is not None:
        medication_flags, medication_matches = map_medications(rx, dictionary, settings, cycle)
        write_medication_outputs(medication_flags, medication_matches, settings, cycle)
        write_report(
            medication_matches,
            settings.reports_dir / "medications" / f"{cycle_slug(cycle)}_medication_matches.csv",
        )
        unmapped = medication_matches[medication_matches["flag"].eq("unmapped")]
        write_report(
            unmapped,
            settings.reports_dir / "medications" / f"{cycle_slug(cycle)}_unmapped_medications.csv",
        )
        write_report(
            medication_mapping_summary(medication_matches),
            settings.reports_dir
            / "medications"
            / f"{cycle_slug(cycle)}_medication_mapping_summary.csv",
        )
        write_report(
            unmapped_medication_summary(medication_matches),
            settings.reports_dir
            / "medications"
            / f"{cycle_slug(cycle)}_unmapped_medication_summary.csv",
        )

    person, merge_audit = merge_person_level_cycle(settings, cycle, medication_flags)
    write_report(
        merge_audit,
        settings.reports_dir / "merge" / f"{cycle_slug(cycle)}_merge_audit.csv",
    )

    derived = derive_cycle_variables(person)
    derived_path = output_path(
        settings.processed_dir / cycle_slug(cycle),
        "person_level_derived",
        settings.output_format,
    )
    write_frame(derived, derived_path, settings.output_format)

    cohort, cohort_flow = apply_primary_cohort(derived)
    cohort_path = output_path(
        settings.processed_dir / cycle_slug(cycle),
        "analytic_cohort",
        settings.output_format,
    )
    write_frame(cohort, cohort_path, settings.output_format)
    write_report(
        cohort_flow,
        settings.reports_dir / "cohort_flow" / f"{cycle_slug(cycle)}_cohort_flow.csv",
    )

    allowed = primary_allowed_columns(cohort, dictionary)
    feature_manifest = pd.DataFrame(
        {
            "column": cohort.columns,
            "in_primary_allowed_set": [column in allowed for column in cohort.columns],
        }
    )
    write_report(
        feature_manifest,
        settings.reports_dir / "features" / f"{cycle_slug(cycle)}_feature_manifest.csv",
    )

    return {
        "cycle": cycle,
        "cycle_role": dictionary.cycle_role(cycle),
        "staged_files": len(stage_manifest),
        "harmonized_rows": len(person),
        "analytic_rows": len(cohort),
        "medication_rows": len(medication_matches),
        "derived_output": str(derived_path),
        "cohort_output": str(cohort_path),
    }


def combine_analytic_cycles(
    cycles: Iterable[str],
    dictionary: DictionaryConfig,
    settings: Settings,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    for cycle in cycles:
        path = output_path(
            settings.processed_dir / cycle_slug(cycle),
            "analytic_cohort",
            settings.output_format,
        )
        if not path.exists():
            continue
        frames.append(read_frame(path))
    if not frames:
        return pd.DataFrame(), pd.DataFrame()

    all_columns = sorted(set().union(*(set(frame.columns) for frame in frames)))
    aligned = [frame.reindex(columns=all_columns) for frame in frames]
    pooled = pd.concat(aligned, ignore_index=True, sort=False)

    duplicate_key = pooled.duplicated(["cycle_id", "seqn"], keep=False)
    if duplicate_key.any():
        raise ValueError("Pooled analytic cohort has duplicate (cycle_id, seqn) keys")

    development = pooled[pooled["cycle_role"].eq("development")].copy()
    temporal_test = pooled[pooled["cycle_role"].eq("temporal_test")].copy()

    write_frame(
        pooled,
        output_path(settings.processed_dir, "analytic_cohort_2007_2018", settings.output_format),
        settings.output_format,
    )
    write_frame(
        development,
        output_path(settings.processed_dir, "development_2007_2014", settings.output_format),
        settings.output_format,
    )
    write_frame(
        temporal_test,
        output_path(settings.processed_dir, "temporal_test_2015_2018", settings.output_format),
        settings.output_format,
    )

    summary = (
        pooled.groupby(["cycle_id", "cycle_role"], dropna=False)
        .agg(
            n=("seqn", "size"),
            mets_cases=("mets_prevalent", "sum"),
            mets_prevalence_unweighted=("mets_prevalent", "mean"),
        )
        .reset_index()
    )
    write_report(summary, settings.reports_dir / "pooled_cycle_summary.csv")
    write_report(
        build_weighted_cycle_summary(pooled),
        settings.reports_dir / "weighted_cycle_summary.csv",
    )
    write_report(
        build_weighted_component_summary(pooled),
        settings.reports_dir / "weighted_component_summary.csv",
    )
    return pooled, summary


def run_pipeline(
    dictionary_path: Path,
    settings_path: Path,
    roles: Iterable[str],
    statuses: Iterable[str],
    cycles: Iterable[str] | None,
    do_download: bool,
    do_audit: bool,
    overwrite: bool,
    workers: int,
    verbose: bool = False,
) -> pd.DataFrame:
    settings = Settings.from_yaml(settings_path, dictionary_path)
    dictionary = DictionaryConfig.load(dictionary_path)
    config_report = dictionary.assert_valid()
    ensure_dir(settings.reports_dir)
    write_report(config_report, settings.reports_dir / "config_audit.csv")
    save_config_snapshot(dictionary_path, settings_path, settings)

    plan = dictionary.file_plan(roles, statuses, cycles)
    write_report(plan, settings.reports_dir / "selected_file_plan.csv")
    if plan.empty:
        raise ValueError("The selected Cycle_File_Matrix plan is empty")

    if do_download:
        manifest = download_plan(plan, settings, overwrite=overwrite, workers=workers)
        write_report(manifest, settings.reports_dir / "download_manifest.csv")
        if settings.strict and download_has_failures(manifest):
            raise RuntimeError("Required downloads failed; inspect reports/download_manifest.csv")

    missing = missing_raw_files(plan, settings)
    if missing:
        examples = "; ".join(str(path) for path in missing[:3])
        raise RuntimeError(
            f"{len(missing)} required raw XPT file(s) are missing. "
            f"Examples: {examples}. Run 'nhanes-etl doctor' and then "
            "'nhanes-etl download --workers 1' before schema audit/ETL."
        )

    if do_audit:
        file_audit, variable_audit = audit_plan(plan, dictionary, settings)
        write_report(file_audit, settings.reports_dir / "schema_file_audit.csv")
        write_report(variable_audit, settings.reports_dir / "schema_variable_audit.csv")
        if settings.strict and schema_has_hard_fail(file_audit, variable_audit):
            raise RuntimeError("Schema audit failed; inspect schema_file_audit.csv and schema_variable_audit.csv")

    cycle_results = []
    for cycle in sorted(plan["cycle"].unique()):
        cycle_results.append(
            etl_cycle(
                cycle,
                plan[plan["cycle"].eq(cycle)].copy(),
                dictionary,
                settings,
            )
        )
    results = pd.DataFrame(cycle_results)
    write_report(results, settings.reports_dir / "etl_cycle_summary.csv")
    combine_analytic_cycles(results["cycle"].tolist(), dictionary, settings)
    return results
