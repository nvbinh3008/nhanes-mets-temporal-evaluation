from __future__ import annotations

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


def make_grouped_folds(
    y,
    groups,
    stratification_labels,
    n_splits: int,
    seed: int,
):
    y = np.asarray(y)
    groups = np.asarray(groups)
    labels = np.asarray(stratification_labels)
    if np.unique(groups).size < n_splits:
        raise ValueError("Not enough PSU groups for requested n_splits")
    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )
    return list(splitter.split(np.zeros(len(y)), labels, groups=groups))


def assert_group_isolation(folds, groups) -> None:
    groups = np.asarray(groups)
    for fold, (train_idx, valid_idx) in enumerate(folds):
        overlap = set(groups[train_idx]).intersection(set(groups[valid_idx]))
        if overlap:
            raise AssertionError(f"Fold {fold} has PSU leakage: {sorted(overlap)[:5]}")
