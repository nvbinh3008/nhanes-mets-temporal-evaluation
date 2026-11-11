from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
    roc_curve,
)

from .calibration import clip_probabilities, logit
from .utils import normalize_weights


METRIC_NAMES = (
    "roc_auc",
    "average_precision",
    "log_loss",
    "brier",
    "calibration_intercept",
    "calibration_slope",
    "ece",
    "sensitivity_at_0_5",
    "specificity_at_0_5",
)


def weighted_mean(values, weights) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not valid.any():
        return float("nan")
    return float(np.average(values[valid], weights=weights[valid]))


def weighted_quantile(values, quantiles, weights):
    values = np.asarray(values, dtype=float)
    quantiles = np.asarray(quantiles, dtype=float)
    weights = np.asarray(weights, dtype=float)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not valid.any():
        return np.full_like(quantiles, np.nan, dtype=float)
    values = values[valid]
    weights = weights[valid]
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cumulative = np.cumsum(weights) - 0.5 * weights
    cumulative /= weights.sum()
    return np.interp(quantiles, cumulative, values)


def calibration_intercept_slope(y, probabilities, weights) -> tuple[float, float]:
    y = np.asarray(y, dtype=int)
    p = clip_probabilities(probabilities)
    w = np.asarray(weights, dtype=float)
    valid = np.isfinite(w) & (w > 0) & np.isfinite(p)
    if valid.sum() < 3 or np.unique(y[valid]).size < 2:
        return float("nan"), float("nan")
    y = y[valid]
    p = p[valid]
    w = normalize_weights(w[valid])
    x = logit(p).reshape(-1, 1)

    model = LogisticRegression(
        C=1e6,
        solver="lbfgs",
        max_iter=2000,
        random_state=0,
    )
    try:
        model.fit(x, y, sample_weight=w)
        return float(model.intercept_[0]), float(model.coef_[0, 0])
    except Exception:
        return float("nan"), float("nan")


def weighted_calibration_curve(
    y,
    probabilities,
    weights,
    n_bins: int = 10,
) -> pd.DataFrame:
    y = np.asarray(y, dtype=int)
    p = clip_probabilities(probabilities)
    w = np.asarray(weights, dtype=float)

    edges = weighted_quantile(
        p,
        np.linspace(0.0, 1.0, n_bins + 1),
        w,
    )
    edges[0] = 0.0
    edges[-1] = 1.0
    edges = np.maximum.accumulate(edges)

    bins = np.searchsorted(edges[1:-1], p, side="right")
    rows = []
    for bin_id in range(n_bins):
        mask = bins == bin_id
        if not mask.any():
            continue
        rows.append(
            {
                "bin": bin_id + 1,
                "n": int(mask.sum()),
                "weight_sum": float(w[mask].sum()),
                "mean_predicted": weighted_mean(p[mask], w[mask]),
                "observed_rate": weighted_mean(y[mask], w[mask]),
                "lower_probability": float(edges[bin_id]),
                "upper_probability": float(edges[bin_id + 1]),
            }
        )
    return pd.DataFrame(rows)


def expected_calibration_error(y, probabilities, weights, n_bins: int = 10) -> float:
    curve = weighted_calibration_curve(y, probabilities, weights, n_bins)
    if curve.empty:
        return float("nan")
    total_weight = curve["weight_sum"].sum()
    return float(
        (
            curve["weight_sum"]
            * (curve["observed_rate"] - curve["mean_predicted"]).abs()
        ).sum()
        / total_weight
    )


def classification_at_threshold(y, probabilities, weights, threshold: float) -> tuple[float, float]:
    y = np.asarray(y, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    w = np.asarray(weights, dtype=float)
    pred = p >= threshold

    positive = y == 1
    negative = y == 0
    sensitivity = weighted_mean(pred[positive].astype(float), w[positive])
    specificity = weighted_mean((~pred[negative]).astype(float), w[negative])
    return sensitivity, specificity


def metric_vector(y, probabilities, weights, n_bins: int = 10) -> dict[str, float]:
    y = np.asarray(y, dtype=int)
    p = clip_probabilities(probabilities)
    w = np.asarray(weights, dtype=float)

    try:
        auc = float(roc_auc_score(y, p, sample_weight=w))
    except ValueError:
        auc = float("nan")
    try:
        ap = float(average_precision_score(y, p, sample_weight=w))
    except ValueError:
        ap = float("nan")

    intercept, slope = calibration_intercept_slope(y, p, w)
    sensitivity, specificity = classification_at_threshold(y, p, w, 0.5)

    return {
        "roc_auc": auc,
        "average_precision": ap,
        "log_loss": float(log_loss(y, p, sample_weight=w, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p, sample_weight=w)),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "ece": expected_calibration_error(y, p, w, n_bins),
        "sensitivity_at_0_5": sensitivity,
        "specificity_at_0_5": specificity,
    }


def select_weighted_youden_threshold(y, probabilities, weights, fallback: float = 0.5) -> float:
    y = np.asarray(y, dtype=int)
    p = clip_probabilities(probabilities)
    w = np.asarray(weights, dtype=float)
    try:
        fpr, tpr, thresholds = roc_curve(y, p, sample_weight=w)
    except ValueError:
        return float(fallback)
    finite = np.isfinite(thresholds)
    if not finite.any():
        return float(fallback)
    score = tpr[finite] - fpr[finite]
    best = np.flatnonzero(score == np.nanmax(score))
    if len(best) == 0:
        return float(fallback)
    candidate_thresholds = thresholds[finite][best]
    # Deterministic tie break: choose the threshold closest to 0.5.
    return float(candidate_thresholds[np.argmin(np.abs(candidate_thresholds - 0.5))])


def stratified_psu_bootstrap(
    y,
    probabilities,
    weights,
    strata,
    psu,
    *,
    replicates: int,
    confidence_level: float,
    seed: int,
    n_bins: int = 10,
) -> pd.DataFrame:
    y = np.asarray(y, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    w = np.asarray(weights, dtype=float)
    strata = np.asarray(strata).astype(str)
    psu = np.asarray(psu).astype(str)

    point = metric_vector(y, p, w, n_bins)
    rng = np.random.default_rng(seed)
    replicate_values = {name: [] for name in point}

    unique_strata = np.unique(strata)
    psus_by_stratum = {
        stratum: np.unique(psu[strata == stratum])
        for stratum in unique_strata
    }

    for _ in range(int(replicates)):
        multiplier = np.zeros(len(y), dtype=float)
        for stratum, cluster_ids in psus_by_stratum.items():
            sampled = rng.choice(cluster_ids, size=len(cluster_ids), replace=True)
            counts = pd.Series(sampled).value_counts()
            stratum_mask = strata == stratum
            for cluster_id, count in counts.items():
                multiplier[stratum_mask & (psu == cluster_id)] = float(count)

        boot_weights = w * multiplier
        if boot_weights.sum() <= 0:
            continue
        values = metric_vector(y, p, boot_weights, n_bins)
        for name, value in values.items():
            if np.isfinite(value):
                replicate_values[name].append(value)

    alpha = 1.0 - float(confidence_level)
    rows = []
    for name, estimate in point.items():
        values = np.asarray(replicate_values[name], dtype=float)
        if values.size:
            lower, upper = np.quantile(values, [alpha / 2.0, 1.0 - alpha / 2.0])
            standard_error = float(values.std(ddof=1)) if values.size > 1 else float("nan")
        else:
            lower = upper = standard_error = float("nan")
        rows.append(
            {
                "metric": name,
                "estimate": estimate,
                "standard_error": standard_error,
                "ci_lower": float(lower),
                "ci_upper": float(upper),
                "bootstrap_replicates_valid": int(values.size),
            }
        )
    return pd.DataFrame(rows)


def comparison_metric_vector(y, probabilities, weights) -> dict[str, float]:
    y = np.asarray(y, dtype=int)
    p = clip_probabilities(probabilities)
    w = np.asarray(weights, dtype=float)
    try:
        auc = float(roc_auc_score(y, p, sample_weight=w))
    except ValueError:
        auc = float("nan")
    try:
        ap = float(average_precision_score(y, p, sample_weight=w))
    except ValueError:
        ap = float("nan")
    return {
        "roc_auc": auc,
        "average_precision": ap,
        "log_loss": float(log_loss(y, p, sample_weight=w, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p, sample_weight=w)),
    }


def paired_stratified_psu_bootstrap(
    y,
    probabilities_a,
    probabilities_b,
    weights,
    strata,
    psu,
    *,
    replicates: int,
    confidence_level: float,
    seed: int,
) -> pd.DataFrame:
    y = np.asarray(y, dtype=int)
    p_a = np.asarray(probabilities_a, dtype=float)
    p_b = np.asarray(probabilities_b, dtype=float)
    w = np.asarray(weights, dtype=float)
    strata = np.asarray(strata).astype(str)
    psu = np.asarray(psu).astype(str)

    metric_a = comparison_metric_vector(y, p_a, w)
    metric_b = comparison_metric_vector(y, p_b, w)
    point = {name: metric_a[name] - metric_b[name] for name in metric_a}

    rng = np.random.default_rng(seed)
    replicate_values = {name: [] for name in point}
    unique_strata = np.unique(strata)
    psus_by_stratum = {
        stratum: np.unique(psu[strata == stratum])
        for stratum in unique_strata
    }

    for _ in range(int(replicates)):
        multiplier = np.zeros(len(y), dtype=float)
        for stratum, cluster_ids in psus_by_stratum.items():
            sampled = rng.choice(cluster_ids, size=len(cluster_ids), replace=True)
            counts = pd.Series(sampled).value_counts()
            stratum_mask = strata == stratum
            for cluster_id, count in counts.items():
                multiplier[stratum_mask & (psu == cluster_id)] = float(count)
        boot_weights = w * multiplier
        if boot_weights.sum() <= 0:
            continue
        values_a = comparison_metric_vector(y, p_a, boot_weights)
        values_b = comparison_metric_vector(y, p_b, boot_weights)
        for name in point:
            difference = values_a[name] - values_b[name]
            if np.isfinite(difference):
                replicate_values[name].append(difference)

    alpha = 1.0 - float(confidence_level)
    rows = []
    for name, estimate in point.items():
        values = np.asarray(replicate_values[name], dtype=float)
        if values.size:
            lower, upper = np.quantile(
                values, [alpha / 2.0, 1.0 - alpha / 2.0]
            )
            standard_error = (
                float(values.std(ddof=1)) if values.size > 1 else float("nan")
            )
        else:
            lower = upper = standard_error = float("nan")
        rows.append(
            {
                "metric": name,
                "estimate_a_minus_b": estimate,
                "standard_error": standard_error,
                "ci_lower": float(lower),
                "ci_upper": float(upper),
                "bootstrap_replicates_valid": int(values.size),
                "direction_note": (
                    "positive favors A"
                    if name in {"roc_auc", "average_precision"}
                    else "negative favors A"
                ),
            }
        )
    return pd.DataFrame(rows)
