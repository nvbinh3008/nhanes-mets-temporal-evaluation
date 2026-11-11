from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import pandas as pd
import numpy as np

from .config import resolve_path
from .utils import sha256_file


@dataclass(frozen=True)
class RepresentationData:
    split: str
    representation: str
    view: str
    path: Path
    frame: pd.DataFrame
    features: list[str]
    categorical_features: list[str]
    target_column: str
    weight_column: str
    stratum_column: str
    psu_column: str
    source_sha256: str

    @property
    def X(self) -> pd.DataFrame:
        return self.frame[self.features].copy()

    @property
    def y(self) -> np.ndarray:
        return self.frame[self.target_column].astype(int).to_numpy()

    @property
    def weights(self) -> np.ndarray:
        return self.frame[self.weight_column].astype(float).to_numpy()

    @property
    def strata(self) -> np.ndarray:
        return self.frame[self.stratum_column].astype(str).to_numpy()

    @property
    def psu(self) -> np.ndarray:
        return self.frame[self.psu_column].astype(str).to_numpy()

    @property
    def groups(self) -> np.ndarray:
        return self.psu

    @property
    def stratification_labels(self) -> np.ndarray:
        return (
            self.frame["cycle_id"].astype(str)
            + "::"
            + self.frame[self.target_column].astype(int).astype(str)
        ).to_numpy()


def representation_path(
    config: dict,
    split: str,
    representation: str,
    view: str,
) -> Path:
    root = resolve_path(config, "representation_root")
    return root / split / f"{representation.lower()}_{view}.parquet"


def _metadata_columns(config: dict) -> set[str]:
    survey = config["data"]["survey_columns"]
    return set(config["data"]["id_columns"]) | {
        config["data"]["target"],
        survey["development_weight"],
        survey["temporal_weight"],
        survey["full_weight"],
        survey["stratum"],
        survey["psu"],
    }


def _check_leakage(features: list[str], patterns: list[str]) -> list[str]:
    compiled = [re.compile(pattern, flags=re.IGNORECASE) for pattern in patterns]
    return [
        feature
        for feature in features
        if any(pattern.search(feature) for pattern in compiled)
    ]


def load_representation(
    config: dict,
    split: str,
    representation: str,
    view: str,
) -> RepresentationData:
    representation = representation.upper()
    allowed = {item.upper() for item in config["data"]["representations"]}
    if representation not in allowed:
        raise ValueError(f"Unknown representation {representation}; allowed={sorted(allowed)}")
    if split not in {"development", "temporal_test"}:
        raise ValueError("split must be development or temporal_test")

    path = representation_path(config, split, representation, view)
    if not path.exists():
        raise FileNotFoundError(f"Representation file not found: {path}")

    frame = pd.read_parquet(path)
    if frame.empty:
        raise ValueError(f"Representation is empty: {path}")

    target = config["data"]["target"]
    survey = config["data"]["survey_columns"]
    weight = survey["development_weight"] if split == "development" else survey["temporal_weight"]
    required = {
        "cycle_id",
        "cycle_role",
        "seqn",
        target,
        weight,
        survey["stratum"],
        survey["psu"],
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    expected_role = config["data"]["expected_roles"][split]
    observed_roles = set(frame["cycle_role"].dropna().astype(str).unique())
    if observed_roles != {expected_role}:
        raise ValueError(
            f"{path}: expected cycle_role={expected_role}, observed={sorted(observed_roles)}"
        )

    if frame["seqn"].isna().any():
        raise ValueError(f"{path}: SEQN contains missing values")
    if frame.duplicated(["cycle_id", "seqn"]).any():
        raise ValueError(f"{path}: duplicate (cycle_id, SEQN) keys")

    y = pd.to_numeric(frame[target], errors="coerce")
    if y.isna().any() or not set(y.astype(int).unique()).issubset({0, 1}):
        raise ValueError(f"{path}: target must be complete and binary")

    w = pd.to_numeric(frame[weight], errors="coerce")
    if w.isna().any() or (w <= 0).any():
        raise ValueError(f"{path}: analysis weight must be complete and positive")

    metadata = _metadata_columns(config)
    features = [column for column in frame.columns if column not in metadata]
    if not features:
        raise ValueError(f"{path}: no predictor columns")

    leakage = _check_leakage(features, config["data"]["leakage_patterns"])
    if leakage:
        raise ValueError(f"{path}: prohibited leakage predictors detected: {leakage}")

    all_missing = [column for column in features if frame[column].isna().all()]
    if all_missing:
        raise ValueError(f"{path}: all-missing predictors detected: {all_missing}")

    categorical = [
        column
        for column in config["data"]["categorical_features"].get(representation, [])
        if column in features
    ]

    return RepresentationData(
        split=split,
        representation=representation,
        view=view,
        path=path,
        frame=frame,
        features=features,
        categorical_features=categorical,
        target_column=target,
        weight_column=weight,
        stratum_column=survey["stratum"],
        psu_column=survey["psu"],
        source_sha256=sha256_file(path),
    )


def preflight_development(
    config: dict,
    view: str,
    representations: list[str] | None = None,
) -> pd.DataFrame:
    representations = representations or config["data"]["representations"]
    rows = []
    key_reference = None
    for representation in representations:
        data = load_representation(config, "development", representation, view)
        keys = data.frame[["cycle_id", "seqn"]].sort_values(
            ["cycle_id", "seqn"]
        ).reset_index(drop=True)
        if key_reference is None:
            key_reference = keys
            same_keys = True
        else:
            same_keys = keys.equals(key_reference)
        rows.append(
            {
                "representation": data.representation,
                "view": view,
                "rows": len(data.frame),
                "features": len(data.features),
                "categorical_features": len(data.categorical_features),
                "target_prevalence_unweighted": float(data.y.mean()),
                "target_prevalence_weighted": float(
                    np.average(data.y, weights=data.weights)
                ),
                "weight_sum": float(data.weights.sum()),
                "strata": int(pd.Series(data.strata).nunique()),
                "psus": int(pd.Series(data.psu).nunique()),
                "same_development_keys_as_first": bool(same_keys),
                "source_sha256": data.source_sha256,
                "hard_fail": not same_keys,
            }
        )
    return pd.DataFrame(rows)
