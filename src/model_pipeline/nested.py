from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .calibration import (
    ProbabilityCalibrator,
    crossfit_calibration,
    fit_calibrator,
)
from .config import resolve_path
from .data import RepresentationData, load_representation
from .metrics import (
    metric_vector,
    paired_stratified_psu_bootstrap,
    select_weighted_youden_threshold,
    stratified_psu_bootstrap,
    weighted_calibration_curve,
)
from .models import TrainedProbabilityModel, fit_base_model
from .splits import assert_group_isolation, make_grouped_folds
from .utils import config_fingerprint, json_dump, sha256_file, sha256_tree, utc_now_iso


@dataclass
class SearchResult:
    algorithm: str
    selected_params: dict[str, Any]
    calibration_method: str
    raw_oof: np.ndarray
    calibrated_oof: np.ndarray
    calibrator: ProbabilityCalibrator
    search_table: pd.DataFrame


def stable_seed(base_seed: int, *parts: object) -> int:
    text = "::".join(str(part) for part in parts)
    value = sum((index + 1) * ord(char) for index, char in enumerate(text))
    return int((int(base_seed) + value) % (2**31 - 1))


def candidate_list(config: dict, algorithm: str) -> list[dict[str, Any]]:
    if algorithm == "logistic":
        return [dict(item) for item in config["models"]["logistic"]["candidates"]]
    if algorithm in {"boost_unconstrained", "boost_monotonic"}:
        return [dict(item) for item in config["models"]["boost"]["candidates"]]
    raise ValueError(f"Unknown algorithm: {algorithm}")


def _fit_model_for_indices(
    data: RepresentationData,
    X: pd.DataFrame,
    y: np.ndarray,
    weights: np.ndarray,
    train_idx: np.ndarray,
    *,
    algorithm: str,
    params: dict[str, Any],
    config: dict,
    seed: int,
) -> TrainedProbabilityModel:
    directions = config["monotonic_constraints"].get(data.representation, {})
    return fit_base_model(
        X.iloc[train_idx],
        y[train_idx],
        weights[train_idx],
        representation=data.representation,
        view=data.view,
        algorithm=algorithm,
        params=params,
        categorical_features=data.categorical_features,
        configured_directions=directions,
        seed=seed,
        logistic_max_iter=config["models"]["logistic"]["max_iter"],
        boost_early_stopping=config["models"]["boost"]["early_stopping"],
    )


def search_inner(
    data: RepresentationData,
    X: pd.DataFrame,
    y: np.ndarray,
    weights: np.ndarray,
    groups: np.ndarray,
    stratification_labels: np.ndarray,
    *,
    algorithm: str,
    config: dict,
    seed: int,
) -> SearchResult:
    cv = config["cross_validation"]
    folds = make_grouped_folds(
        y=y,
        groups=groups,
        stratification_labels=stratification_labels,
        n_splits=int(cv["inner_splits"]),
        seed=stable_seed(seed, "inner_split"),
    )
    assert_group_isolation(folds, groups)

    epsilon = float(config["calibration"]["probability_epsilon"])
    calibration_methods = list(cv["calibration_methods"])
    rows: list[dict[str, Any]] = []
    candidate_outputs: dict[int, np.ndarray] = {}

    for candidate_index, params in enumerate(candidate_list(config, algorithm)):
        raw_oof = np.full(len(y), np.nan, dtype=float)

        for inner_fold, (train_idx, valid_idx) in enumerate(folds):
            fold_seed = stable_seed(
                seed,
                algorithm,
                candidate_index,
                inner_fold,
            )
            model = _fit_model_for_indices(
                data,
                X,
                y,
                weights,
                train_idx,
                algorithm=algorithm,
                params=params,
                config=config,
                seed=fold_seed,
            )
            raw_oof[valid_idx] = model.predict_raw(X.iloc[valid_idx])

        if np.isnan(raw_oof).any():
            raise RuntimeError("Inner CV left missing raw predictions")
        candidate_outputs[candidate_index] = raw_oof

        for method in calibration_methods:
            calibrated_oof = crossfit_calibration(
                method=method,
                raw_probabilities=raw_oof,
                y=y,
                weights=weights,
                groups=groups,
                stratification_labels=stratification_labels,
                n_splits=int(cv["calibration_splits"]),
                seed=stable_seed(seed, "calibration", candidate_index, method),
                epsilon=epsilon,
            )
            metrics = metric_vector(
                y,
                calibrated_oof,
                weights,
                int(config["calibration"]["curve_bins"]),
            )
            rows.append(
                {
                    "candidate_index": candidate_index,
                    "algorithm": algorithm,
                    "params_json": json.dumps(params, sort_keys=True),
                    "calibration_method": method,
                    **metrics,
                }
            )

    search_table = pd.DataFrame(rows)
    primary = cv["selection_metric"]
    secondary = cv["secondary_metric"]
    ascending = [True, True, False]
    ranked = search_table.sort_values(
        [primary, secondary, "roc_auc"],
        ascending=ascending,
        kind="mergesort",
    )
    winner = ranked.iloc[0]
    candidate_index = int(winner["candidate_index"])
    selected_params = candidate_list(config, algorithm)[candidate_index]
    method = str(winner["calibration_method"])
    raw_oof = candidate_outputs[candidate_index]

    calibrated_oof = crossfit_calibration(
        method=method,
        raw_probabilities=raw_oof,
        y=y,
        weights=weights,
        groups=groups,
        stratification_labels=stratification_labels,
        n_splits=int(cv["calibration_splits"]),
        seed=stable_seed(seed, "selected_calibration", candidate_index, method),
        epsilon=epsilon,
    )
    calibrator = fit_calibrator(
        method,
        raw_oof,
        y,
        weights,
        epsilon,
    )

    return SearchResult(
        algorithm=algorithm,
        selected_params=selected_params,
        calibration_method=method,
        raw_oof=raw_oof,
        calibrated_oof=calibrated_oof,
        calibrator=calibrator,
        search_table=search_table,
    )


def nested_validate_one(
    data: RepresentationData,
    algorithm: str,
    config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    X = data.X
    y = data.y
    weights = data.weights
    groups = data.groups
    labels = data.stratification_labels
    cv = config["cross_validation"]
    split_seed = stable_seed(
        config["seed"], data.representation, data.view, "nested_split"
    )

    outer_folds = make_grouped_folds(
        y=y,
        groups=groups,
        stratification_labels=labels,
        n_splits=int(cv["outer_splits"]),
        seed=split_seed,
    )
    assert_group_isolation(outer_folds, groups)

    prediction_rows = []
    search_rows = []

    for outer_fold, (train_idx, valid_idx) in enumerate(outer_folds):
        print(
            f"[nested] {data.representation}/{algorithm} "
            f"outer fold {outer_fold + 1}/{len(outer_folds)}",
            flush=True,
        )
        inner = search_inner(
            data,
            X.iloc[train_idx].reset_index(drop=True),
            y[train_idx],
            weights[train_idx],
            groups[train_idx],
            labels[train_idx],
            algorithm=algorithm,
            config=config,
            seed=stable_seed(split_seed, "outer", outer_fold),
        )

        final_outer_model = _fit_model_for_indices(
            data,
            X,
            y,
            weights,
            train_idx,
            algorithm=algorithm,
            params=inner.selected_params,
            config=config,
            seed=stable_seed(
                split_seed, "outer_refit", outer_fold, algorithm
            ),
        )
        final_outer_model.calibrator = inner.calibrator

        raw = final_outer_model.predict_raw(X.iloc[valid_idx])
        calibrated = final_outer_model.calibrator.predict(raw)

        fold_frame = data.frame.iloc[valid_idx][
            [
                "cycle_id",
                "seqn",
                data.target_column,
                data.weight_column,
                data.stratum_column,
                data.psu_column,
            ]
        ].copy()
        fold_frame.rename(
            columns={
                data.target_column: "outcome",
                data.weight_column: "analysis_weight",
                data.stratum_column: "stratum",
                data.psu_column: "psu",
            },
            inplace=True,
        )
        fold_frame["representation"] = data.representation
        fold_frame["view"] = data.view
        fold_frame["algorithm"] = algorithm
        fold_frame["outer_fold"] = outer_fold
        fold_frame["raw_probability"] = raw
        fold_frame["calibrated_probability"] = calibrated
        fold_frame["calibration_method"] = inner.calibration_method
        prediction_rows.append(fold_frame)

        selected = inner.search_table.copy()
        selected["representation"] = data.representation
        selected["view"] = data.view
        selected["outer_fold"] = outer_fold
        selected["selected"] = (
            selected["params_json"].eq(
                json.dumps(inner.selected_params, sort_keys=True)
            )
            & selected["calibration_method"].eq(inner.calibration_method)
        )
        search_rows.append(selected)

    predictions = pd.concat(prediction_rows, ignore_index=True)
    if predictions.duplicated(["cycle_id", "seqn"]).any():
        raise RuntimeError("Nested OOF predictions contain duplicate participant keys")
    if len(predictions) != len(data.frame):
        raise RuntimeError("Nested OOF predictions do not cover the development set")

    return predictions, pd.concat(search_rows, ignore_index=True)


def summarize_oof(
    predictions: pd.DataFrame,
    config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    curves = []
    uncertainty = config["uncertainty"]
    n_bins = int(config["calibration"]["curve_bins"])

    for (representation, algorithm), group in predictions.groupby(
        ["representation", "algorithm"],
        sort=True,
    ):
        ci = stratified_psu_bootstrap(
            group["outcome"].to_numpy(),
            group["calibrated_probability"].to_numpy(),
            group["analysis_weight"].to_numpy(),
            group["stratum"].to_numpy(),
            group["psu"].to_numpy(),
            replicates=int(uncertainty["replicates"]),
            confidence_level=float(uncertainty["confidence_level"]),
            seed=stable_seed(config["seed"], "bootstrap", representation, algorithm),
            n_bins=n_bins,
        )
        ci["representation"] = representation
        ci["algorithm"] = algorithm
        rows.append(ci)

        curve = weighted_calibration_curve(
            group["outcome"].to_numpy(),
            group["calibrated_probability"].to_numpy(),
            group["analysis_weight"].to_numpy(),
            n_bins,
        )
        curve["representation"] = representation
        curve["algorithm"] = algorithm
        curves.append(curve)

    return pd.concat(rows, ignore_index=True), pd.concat(curves, ignore_index=True)



def summarize_paired_oof(
    predictions: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    comparisons = [
        ("boost_unconstrained", "logistic"),
        ("boost_monotonic", "logistic"),
        ("boost_monotonic", "boost_unconstrained"),
    ]
    rows = []
    uncertainty = config["uncertainty"]

    for representation, rep_frame in predictions.groupby("representation", sort=True):
        for algorithm_a, algorithm_b in comparisons:
            frame_a = rep_frame.loc[
                rep_frame["algorithm"].eq(algorithm_a)
            ].copy()
            frame_b = rep_frame.loc[
                rep_frame["algorithm"].eq(algorithm_b)
            ].copy()
            if frame_a.empty or frame_b.empty:
                continue

            keys = ["cycle_id", "seqn"]
            merged = frame_a[
                keys
                + [
                    "outcome",
                    "analysis_weight",
                    "stratum",
                    "psu",
                    "calibrated_probability",
                ]
            ].merge(
                frame_b[keys + ["calibrated_probability"]],
                on=keys,
                how="inner",
                validate="one_to_one",
                suffixes=("_a", "_b"),
            )
            if len(merged) != len(frame_a) or len(merged) != len(frame_b):
                raise RuntimeError(
                    f"Paired comparison key mismatch for {representation}: "
                    f"{algorithm_a} vs {algorithm_b}"
                )

            comparison = paired_stratified_psu_bootstrap(
                merged["outcome"].to_numpy(),
                merged["calibrated_probability_a"].to_numpy(),
                merged["calibrated_probability_b"].to_numpy(),
                merged["analysis_weight"].to_numpy(),
                merged["stratum"].to_numpy(),
                merged["psu"].to_numpy(),
                replicates=int(uncertainty["replicates"]),
                confidence_level=float(uncertainty["confidence_level"]),
                seed=stable_seed(
                    config["seed"],
                    "paired_bootstrap",
                    representation,
                    algorithm_a,
                    algorithm_b,
                ),
            )
            comparison["representation"] = representation
            comparison["algorithm_a"] = algorithm_a
            comparison["algorithm_b"] = algorithm_b
            rows.append(comparison)

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def run_nested_validation(
    config: dict,
    *,
    view: str,
    representations: list[str],
    algorithms: list[str],
) -> dict:
    report_root = resolve_path(config, "report_root") / "development" / view
    data_root = resolve_path(config, "data_output_root") / "development" / view
    report_root.mkdir(parents=True, exist_ok=True)
    data_root.mkdir(parents=True, exist_ok=True)

    all_predictions = []
    all_search = []

    for representation in representations:
        data = load_representation(
            config,
            "development",
            representation,
            view,
        )
        for algorithm in algorithms:
            print(
                f"[nested] starting {representation}/{algorithm}/{view}",
                flush=True,
            )
            predictions, search = nested_validate_one(data, algorithm, config)
            print(
                f"[nested] completed {representation}/{algorithm}/{view}",
                flush=True,
            )
            all_predictions.append(predictions)
            all_search.append(search)

            task_stem = f"{representation.lower()}__{algorithm}"
            predictions.to_parquet(
                data_root / f"nested_oof__{task_stem}.parquet",
                index=False,
            )
            search.to_csv(
                report_root / f"inner_search__{task_stem}.csv",
                index=False,
            )

    oof = pd.concat(all_predictions, ignore_index=True)
    search_table = pd.concat(all_search, ignore_index=True)
    metrics, curves = summarize_oof(oof, config)
    paired = summarize_paired_oof(oof, config)

    oof.to_parquet(data_root / "nested_oof_predictions.parquet", index=False)
    search_table.to_csv(report_root / "nested_search_details.csv", index=False)
    metrics.to_csv(report_root / "nested_metrics.csv", index=False)
    curves.to_csv(report_root / "nested_calibration_curves.csv", index=False)
    if not paired.empty:
        paired.to_csv(
            report_root / "nested_paired_model_differences.csv",
            index=False,
        )

    metric_wide = (
        metrics.pivot_table(
            index=["representation", "algorithm"],
            columns="metric",
            values="estimate",
            aggfunc="first",
        )
        .reset_index()
    )
    metric_wide["rank_log_loss"] = metric_wide["log_loss"].rank(
        method="min", ascending=True
    )
    metric_wide["rank_brier"] = metric_wide["brier"].rank(
        method="min", ascending=True
    )
    metric_wide["rank_roc_auc"] = metric_wide["roc_auc"].rank(
        method="min", ascending=False
    )
    metric_wide.sort_values(
        ["log_loss", "brier", "roc_auc"],
        ascending=[True, True, False],
        inplace=True,
    )
    metric_wide.to_csv(report_root / "nested_model_ranking.csv", index=False)

    summary = {
        "generated": utc_now_iso(),
        "view": view,
        "representations": representations,
        "algorithms": algorithms,
        "development_rows_per_task": {
            representation: int(
                load_representation(
                    config,
                    "development",
                    representation,
                    view,
                ).frame.shape[0]
            )
            for representation in representations
        },
        "oof_rows_total": int(len(oof)),
        "metrics_rows": int(len(metrics)),
        "method": {
            "preprocessing": "fit within each training fold only",
            "outer_cv": int(config["cross_validation"]["outer_splits"]),
            "inner_cv": int(config["cross_validation"]["inner_splits"]),
            "calibration_cv": int(config["cross_validation"]["calibration_splits"]),
            "grouping": "pooled PSU",
            "weights": "development survey weights in fitting and scoring",
            "uncertainty": config["uncertainty"]["method"],
        },
    }
    json_dump(summary, report_root / "nested_run_summary.json")
    return summary


def fit_final_one(
    data: RepresentationData,
    algorithm: str,
    config: dict,
) -> tuple[TrainedProbabilityModel, pd.DataFrame, pd.DataFrame]:
    X = data.X
    y = data.y
    weights = data.weights
    groups = data.groups
    labels = data.stratification_labels
    split_seed = stable_seed(
        config["seed"], "final", data.representation, data.view
    )

    search = search_inner(
        data,
        X,
        y,
        weights,
        groups,
        labels,
        algorithm=algorithm,
        config=config,
        seed=split_seed,
    )

    all_idx = np.arange(len(y))
    model = _fit_model_for_indices(
        data,
        X,
        y,
        weights,
        all_idx,
        algorithm=algorithm,
        params=search.selected_params,
        config=config,
        seed=stable_seed(split_seed, "refit", algorithm),
    )
    model.calibrator = search.calibrator
    model.threshold = select_weighted_youden_threshold(
        y,
        search.calibrated_oof,
        weights,
        float(config["threshold"]["fallback"]),
    )
    model.metadata.update(
        {
            "trained_at": utc_now_iso(),
            "training_split": "development",
            "training_rows": int(len(y)),
            "development_source": str(data.path),
            "development_sha256": data.source_sha256,
            "calibration_method": search.calibration_method,
            "threshold_strategy": config["threshold"]["strategy"],
            "inner_oof_log_loss": metric_vector(
                y,
                search.calibrated_oof,
                weights,
                int(config["calibration"]["curve_bins"]),
            )["log_loss"],
        }
    )

    internal_oof = data.frame[
        [
            "cycle_id",
            "seqn",
            data.target_column,
            data.weight_column,
            data.stratum_column,
            data.psu_column,
        ]
    ].copy()
    internal_oof.rename(
        columns={
            data.target_column: "outcome",
            data.weight_column: "analysis_weight",
            data.stratum_column: "stratum",
            data.psu_column: "psu",
        },
        inplace=True,
    )
    internal_oof["representation"] = data.representation
    internal_oof["view"] = data.view
    internal_oof["algorithm"] = algorithm
    internal_oof["raw_probability"] = search.raw_oof
    internal_oof["calibrated_probability"] = search.calibrated_oof
    internal_oof["calibration_method"] = search.calibration_method

    return model, search.search_table, internal_oof


def fit_and_freeze_models(
    config: dict,
    *,
    view: str,
    representations: list[str],
    algorithms: list[str],
) -> dict:
    model_root = resolve_path(config, "model_root") / view
    report_root = resolve_path(config, "report_root") / "frozen" / view
    data_root = resolve_path(config, "data_output_root") / "frozen" / view
    model_root.mkdir(parents=True, exist_ok=True)
    report_root.mkdir(parents=True, exist_ok=True)
    data_root.mkdir(parents=True, exist_ok=True)

    registry = []
    feature_contract = []
    all_search = []
    all_internal_oof = []

    for representation in representations:
        data = load_representation(config, "development", representation, view)
        for algorithm in algorithms:
            print(
                f"[freeze] fitting {representation}/{algorithm}/{view}",
                flush=True,
            )
            model, search_table, internal_oof = fit_final_one(
                data,
                algorithm,
                config,
            )
            task_dir = model_root / representation.lower() / algorithm
            task_dir.mkdir(parents=True, exist_ok=True)
            bundle_path = task_dir / "model.joblib"
            joblib.dump(model, bundle_path)

            registry.append(
                {
                    "representation": representation,
                    "view": view,
                    "algorithm": algorithm,
                    "bundle_path": str(bundle_path),
                    "bundle_sha256": sha256_file(bundle_path),
                    "development_source": str(data.path),
                    "development_sha256": data.source_sha256,
                    "features": len(model.features),
                    "categorical_features": len(model.categorical_features),
                    "constraint_count": model.metadata["constraint_count"],
                    "calibration_method": model.calibrator.method,
                    "threshold": model.threshold,
                    "selected_params_json": json.dumps(
                        model.selected_params,
                        sort_keys=True,
                    ),
                }
            )

            for feature in model.features:
                feature_contract.append(
                    {
                        "representation": representation,
                        "view": view,
                        "algorithm": algorithm,
                        "feature": feature,
                        "categorical": feature in model.categorical_features,
                        "prespecified_direction": int(
                            model.feature_directions.get(feature, 0)
                        ),
                        "active_monotonic_constraint": (
                            algorithm == "boost_monotonic"
                            and int(model.feature_directions.get(feature, 0)) != 0
                        ),
                    }
                )

            search_table = search_table.copy()
            search_table["representation"] = representation
            search_table["view"] = view
            search_table["algorithm"] = algorithm
            all_search.append(search_table)
            all_internal_oof.append(internal_oof)

    registry_frame = pd.DataFrame(registry)
    registry_frame.to_csv(report_root / "frozen_model_registry.csv", index=False)
    pd.DataFrame(feature_contract).to_csv(
        report_root / "frozen_feature_contract.csv",
        index=False,
    )
    pd.concat(all_search, ignore_index=True).to_csv(
        report_root / "final_inner_search.csv",
        index=False,
    )
    pd.concat(all_internal_oof, ignore_index=True).to_parquet(
        data_root / "final_training_oof_for_calibration.parquet",
        index=False,
    )

    project_root = Path(config["_project_root"])
    spec = {
        "frozen_at": utc_now_iso(),
        "pipeline_version": config["version"],
        "view": view,
        "representations": representations,
        "algorithms": algorithms,
        "config_path": config["_config_path"],
        "config_file_sha256": sha256_file(Path(config["_config_path"])),
        "effective_config_sha256": config_fingerprint(config),
        "smoke_mode": bool(config.get("_smoke_mode", False)),
        "model_pipeline_source_sha256": sha256_tree(project_root / "model_pipeline"),
        "registry": registry,
        "temporal_data_accessed": False,
    }
    json_dump(spec, report_root / "MODEL_SPEC_FROZEN.json")
    return spec
