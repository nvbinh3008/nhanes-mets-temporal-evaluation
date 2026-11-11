from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from .splits import make_grouped_folds
from .utils import normalize_weights


def clip_probabilities(probabilities, epsilon: float = 1e-6) -> np.ndarray:
    values = np.asarray(probabilities, dtype=float)
    return np.clip(values, epsilon, 1.0 - epsilon)


def logit(probabilities, epsilon: float = 1e-6) -> np.ndarray:
    p = clip_probabilities(probabilities, epsilon)
    return np.log(p / (1.0 - p))


@dataclass
class ProbabilityCalibrator:
    method: str
    model: object | None = None
    epsilon: float = 1e-6

    def predict(self, raw_probabilities) -> np.ndarray:
        raw = clip_probabilities(raw_probabilities, self.epsilon)
        if self.method == "none":
            return raw
        if self.method == "sigmoid":
            calibrated = self.model.predict_proba(
                logit(raw, self.epsilon).reshape(-1, 1)
            )[:, 1]
            return clip_probabilities(calibrated, self.epsilon)
        if self.method == "isotonic":
            calibrated = self.model.predict(raw)
            return clip_probabilities(calibrated, self.epsilon)
        raise ValueError(f"Unknown calibration method: {self.method}")


def fit_calibrator(
    method: str,
    raw_probabilities,
    y,
    weights,
    epsilon: float = 1e-6,
) -> ProbabilityCalibrator:
    raw = clip_probabilities(raw_probabilities, epsilon)
    y = np.asarray(y, dtype=int)
    weights = normalize_weights(np.asarray(weights, dtype=float))

    if method == "none":
        return ProbabilityCalibrator(method="none", epsilon=epsilon)

    if method == "sigmoid":
        model = LogisticRegression(
            C=1e6,
            solver="lbfgs",
            max_iter=2000,
            random_state=0,
        )
        model.fit(logit(raw, epsilon).reshape(-1, 1), y, sample_weight=weights)
        return ProbabilityCalibrator(
            method="sigmoid",
            model=model,
            epsilon=epsilon,
        )

    if method == "isotonic":
        model = IsotonicRegression(
            y_min=epsilon,
            y_max=1.0 - epsilon,
            increasing=True,
            out_of_bounds="clip",
        )
        model.fit(raw, y, sample_weight=weights)
        return ProbabilityCalibrator(
            method="isotonic",
            model=model,
            epsilon=epsilon,
        )

    raise ValueError(f"Unknown calibration method: {method}")


def crossfit_calibration(
    method: str,
    raw_probabilities,
    y,
    weights,
    groups,
    stratification_labels,
    n_splits: int,
    seed: int,
    epsilon: float,
) -> np.ndarray:
    raw = np.asarray(raw_probabilities, dtype=float)
    y = np.asarray(y, dtype=int)
    weights = np.asarray(weights, dtype=float)

    if method == "none":
        return clip_probabilities(raw, epsilon)

    folds = make_grouped_folds(
        y=y,
        groups=groups,
        stratification_labels=stratification_labels,
        n_splits=n_splits,
        seed=seed,
    )
    calibrated = np.full(len(y), np.nan, dtype=float)

    for train_idx, valid_idx in folds:
        calibrator = fit_calibrator(
            method,
            raw[train_idx],
            y[train_idx],
            weights[train_idx],
            epsilon,
        )
        calibrated[valid_idx] = calibrator.predict(raw[valid_idx])

    if np.isnan(calibrated).any():
        raise RuntimeError("Cross-fitted calibration left missing predictions")

    return calibrated
