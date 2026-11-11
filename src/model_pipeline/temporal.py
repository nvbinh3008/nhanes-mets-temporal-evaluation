from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .config import resolve_path
from .data import load_representation
from .nested import summarize_paired_oof
from .metrics import (
    classification_at_threshold,
    stratified_psu_bootstrap,
    weighted_calibration_curve,
)
from .utils import (
    config_fingerprint,
    json_dump,
    json_load,
    sha256_file,
    sha256_tree,
    utc_now_iso,
)


def evaluate_temporal_once(
    config: dict,
    *,
    view: str,
    resume_interrupted: bool = False,
) -> dict:
    frozen_root = resolve_path(config, "report_root") / "frozen" / view
    frozen_spec_path = frozen_root / "MODEL_SPEC_FROZEN.json"
    if not frozen_spec_path.exists():
        raise RuntimeError(
            "Frozen model specification not found. Run fit-final before temporal evaluation."
        )

    temporal_root = resolve_path(config, "report_root") / "temporal" / view
    data_root = resolve_path(config, "data_output_root") / "temporal" / view
    temporal_root.mkdir(parents=True, exist_ok=True)
    data_root.mkdir(parents=True, exist_ok=True)

    started_path = temporal_root / config["temporal_evaluation"]["started_marker"]
    completed_path = temporal_root / config["temporal_evaluation"]["completed_lock"]

    if completed_path.exists() and config["temporal_evaluation"]["one_time_only"]:
        completed = json_load(completed_path)
        raise RuntimeError(
            "Temporal evaluation is already locked as complete at "
            f"{completed.get('completed_at')}. The pipeline refuses a second look."
        )

    if started_path.exists() and not resume_interrupted:
        started = json_load(started_path)
        raise RuntimeError(
            "A temporal evaluation was previously started at "
            f"{started.get('started_at')}. Use --resume-interrupted only after "
            "confirming that the prior run did not complete."
        )

    frozen = json_load(frozen_spec_path)
    if frozen.get("smoke_mode"):
        raise RuntimeError("Temporal evaluation is forbidden for smoke-mode frozen models.")

    current_config_hash = config_fingerprint(config)
    if frozen["effective_config_sha256"] != current_config_hash:
        raise RuntimeError(
            "Modeling config changed after the model specification was frozen."
        )

    current_source_hash = sha256_tree(
        Path(config["_project_root"]) / "model_pipeline"
    )
    if frozen["model_pipeline_source_sha256"] != current_source_hash:
        raise RuntimeError(
            "model_pipeline source changed after the model specification was frozen."
        )

    json_dump(
        {
            "started_at": utc_now_iso(),
            "frozen_spec": str(frozen_spec_path),
            "frozen_spec_sha256": sha256_file(frozen_spec_path),
            "resume_interrupted": bool(resume_interrupted),
        },
        started_path,
    )

    all_predictions = []
    all_metrics = []
    all_curves = []
    temporal_files = {}

    for record in frozen["registry"]:
        representation = record["representation"]
        algorithm = record["algorithm"]
        print(
            f"[temporal] evaluating {representation}/{algorithm}/{view}",
            flush=True,
        )
        data = load_representation(
            config,
            "temporal_test",
            representation,
            view,
        )
        temporal_files[representation] = {
            "path": str(data.path),
            "sha256": data.source_sha256,
            "rows": int(len(data.frame)),
        }

        bundle_path = Path(record["bundle_path"])
        if not bundle_path.exists():
            raise FileNotFoundError(f"Frozen model bundle not found: {bundle_path}")
        if sha256_file(bundle_path) != record["bundle_sha256"]:
            raise RuntimeError(f"Frozen model bundle changed: {bundle_path}")

        model = joblib.load(bundle_path)
        if list(model.features) != list(data.features):
            raise RuntimeError(
                f"{representation}/{algorithm}: temporal feature contract differs "
                "from frozen development contract"
            )

        raw = model.predict_raw(data.X)
        calibrated = model.calibrator.predict(raw)
        predicted_class = (calibrated >= model.threshold).astype(int)

        prediction = data.frame[
            [
                "cycle_id",
                "seqn",
                data.target_column,
                data.weight_column,
                data.stratum_column,
                data.psu_column,
            ]
        ].copy()
        prediction.rename(
            columns={
                data.target_column: "outcome",
                data.weight_column: "analysis_weight",
                data.stratum_column: "stratum",
                data.psu_column: "psu",
            },
            inplace=True,
        )
        prediction["representation"] = representation
        prediction["view"] = view
        prediction["algorithm"] = algorithm
        prediction["raw_probability"] = raw
        prediction["calibrated_probability"] = calibrated
        prediction["frozen_threshold"] = float(model.threshold)
        prediction["predicted_class"] = predicted_class
        all_predictions.append(prediction)

        ci = stratified_psu_bootstrap(
            prediction["outcome"].to_numpy(),
            prediction["calibrated_probability"].to_numpy(),
            prediction["analysis_weight"].to_numpy(),
            prediction["stratum"].to_numpy(),
            prediction["psu"].to_numpy(),
            replicates=int(config["uncertainty"]["replicates"]),
            confidence_level=float(config["uncertainty"]["confidence_level"]),
            seed=int(config["seed"]) + len(all_metrics) + 10000,
            n_bins=int(config["calibration"]["curve_bins"]),
        )
        sensitivity, specificity = classification_at_threshold(
            prediction["outcome"].to_numpy(),
            prediction["calibrated_probability"].to_numpy(),
            prediction["analysis_weight"].to_numpy(),
            float(model.threshold),
        )
        ci["representation"] = representation
        ci["algorithm"] = algorithm
        ci["view"] = view
        ci["calibration_method"] = model.calibrator.method
        ci["frozen_threshold"] = float(model.threshold)
        ci["sensitivity_at_frozen_threshold"] = sensitivity
        ci["specificity_at_frozen_threshold"] = specificity
        all_metrics.append(ci)

        curve = weighted_calibration_curve(
            prediction["outcome"].to_numpy(),
            prediction["calibrated_probability"].to_numpy(),
            prediction["analysis_weight"].to_numpy(),
            int(config["calibration"]["curve_bins"]),
        )
        curve["representation"] = representation
        curve["algorithm"] = algorithm
        curve["view"] = view
        all_curves.append(curve)

    predictions = pd.concat(all_predictions, ignore_index=True)
    metrics = pd.concat(all_metrics, ignore_index=True)
    curves = pd.concat(all_curves, ignore_index=True)
    paired = summarize_paired_oof(predictions, config)

    predictions_path = data_root / "temporal_predictions.parquet"
    metrics_path = temporal_root / "temporal_metrics.csv"
    curves_path = temporal_root / "temporal_calibration_curves.csv"
    predictions.to_parquet(predictions_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    curves.to_csv(curves_path, index=False)
    paired_path = temporal_root / "temporal_paired_model_differences.csv"
    if not paired.empty:
        paired.to_csv(paired_path, index=False)

    completed = {
        "completed_at": utc_now_iso(),
        "view": view,
        "frozen_spec_sha256": sha256_file(frozen_spec_path),
        "temporal_files": temporal_files,
        "prediction_rows": int(len(predictions)),
        "prediction_file": str(predictions_path),
        "prediction_sha256": sha256_file(predictions_path),
        "metrics_file": str(metrics_path),
        "metrics_sha256": sha256_file(metrics_path),
        "calibration_curve_file": str(curves_path),
        "calibration_curve_sha256": sha256_file(curves_path),
        "paired_comparison_file": (
            str(paired_path) if paired_path.exists() else None
        ),
        "paired_comparison_sha256": (
            sha256_file(paired_path) if paired_path.exists() else None
        ),
        "refitting_performed": False,
        "recalibration_performed": False,
        "threshold_reselection_performed": False,
        "one_time_evaluation_complete": True,
    }
    json_dump(completed, completed_path)
    return completed
