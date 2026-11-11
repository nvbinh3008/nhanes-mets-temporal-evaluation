from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from .calibration import ProbabilityCalibrator, clip_probabilities
from .preprocess import (
    FeatureSchema,
    build_preprocessor,
    infer_schema,
    transformed_monotonic_constraints,
)
from .utils import normalize_weights


def expand_feature_directions(
    features: list[str],
    configured: dict[str, int],
) -> dict[str, int]:
    directions: dict[str, int] = {}
    prefix_rules = {
        key[len("__prefix__:") :]: int(value)
        for key, value in configured.items()
        if key.startswith("__prefix__:")
    }
    exact_rules = {
        key: int(value)
        for key, value in configured.items()
        if not key.startswith("__prefix__:")
    }

    for feature in features:
        if feature in exact_rules:
            directions[feature] = exact_rules[feature]
            continue
        matches = [
            direction
            for prefix, direction in prefix_rules.items()
            if feature.startswith(prefix)
        ]
        directions[feature] = matches[0] if matches else 0

    invalid = {
        feature: direction
        for feature, direction in directions.items()
        if direction not in {-1, 0, 1}
    }
    if invalid:
        raise ValueError(f"Invalid monotonic directions: {invalid}")
    return directions


@dataclass
class TrainedProbabilityModel:
    representation: str
    view: str
    algorithm: str
    features: list[str]
    categorical_features: list[str]
    preprocessor: Any
    estimator: Any
    calibrator: ProbabilityCalibrator
    threshold: float
    selected_params: dict[str, Any]
    feature_directions: dict[str, int]
    transformed_feature_names: list[str]
    transformed_constraints: list[int]
    metadata: dict[str, Any]

    def predict_raw(self, X: pd.DataFrame) -> np.ndarray:
        missing = sorted(set(self.features) - set(X.columns))
        if missing:
            raise ValueError(f"Prediction data are missing features: {missing}")
        transformed = self.preprocessor.transform(X[self.features])
        return clip_probabilities(self.estimator.predict_proba(transformed)[:, 1])

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.calibrator.predict(self.predict_raw(X))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X) >= self.threshold).astype(int)


def fit_base_model(
    X: pd.DataFrame,
    y,
    weights,
    *,
    representation: str,
    view: str,
    algorithm: str,
    params: dict[str, Any],
    categorical_features: list[str],
    configured_directions: dict[str, int],
    seed: int,
    logistic_max_iter: int,
    boost_early_stopping: bool,
) -> TrainedProbabilityModel:
    y = np.asarray(y, dtype=int)
    weights = normalize_weights(np.asarray(weights, dtype=float))
    schema: FeatureSchema = infer_schema(X, categorical_features)

    is_logistic = algorithm == "logistic"
    preprocessor = build_preprocessor(
        schema,
        scale_numeric=is_logistic,
    )
    transformed = preprocessor.fit_transform(X)
    feature_directions = expand_feature_directions(
        schema.features,
        configured_directions,
    )
    transformed_names, monotonic_vector = transformed_monotonic_constraints(
        preprocessor,
        feature_directions,
    )

    if is_logistic:
        estimator = LogisticRegression(
            C=float(params["C"]),
            solver="lbfgs",
            max_iter=int(logistic_max_iter),
            random_state=seed,
        )
        active_constraints = [0] * len(transformed_names)

    elif algorithm in {"boost_unconstrained", "boost_monotonic"}:
        active_constraints = (
            monotonic_vector
            if algorithm == "boost_monotonic"
            else [0] * len(monotonic_vector)
        )
        estimator = HistGradientBoostingClassifier(
            learning_rate=float(params["learning_rate"]),
            max_iter=int(params["max_iter"]),
            max_leaf_nodes=int(params["max_leaf_nodes"]),
            min_samples_leaf=int(params["min_samples_leaf"]),
            l2_regularization=float(params["l2_regularization"]),
            early_stopping=bool(boost_early_stopping),
            monotonic_cst=active_constraints,
            random_state=seed,
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    estimator.fit(transformed, y, sample_weight=weights)

    return TrainedProbabilityModel(
        representation=representation,
        view=view,
        algorithm=algorithm,
        features=list(schema.features),
        categorical_features=list(schema.categorical_features),
        preprocessor=preprocessor,
        estimator=estimator,
        calibrator=ProbabilityCalibrator(method="none"),
        threshold=0.5,
        selected_params=dict(params),
        feature_directions=feature_directions,
        transformed_feature_names=transformed_names,
        transformed_constraints=active_constraints,
        metadata={
            "constraint_count": int(np.count_nonzero(active_constraints)),
            "seed": int(seed),
        },
    )
