from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass(frozen=True)
class FeatureSchema:
    features: list[str]
    categorical_features: list[str]
    numeric_features: list[str]


def infer_schema(
    X: pd.DataFrame,
    categorical_features: Iterable[str],
) -> FeatureSchema:
    categorical = [column for column in categorical_features if column in X.columns]
    numeric = [column for column in X.columns if column not in categorical]
    return FeatureSchema(
        features=list(X.columns),
        categorical_features=categorical,
        numeric_features=numeric,
    )


def build_preprocessor(
    schema: FeatureSchema,
    *,
    scale_numeric: bool,
) -> ColumnTransformer:
    transformers = []

    if schema.numeric_features:
        numeric_steps = [
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    add_indicator=True,
                    keep_empty_features=True,
                ),
            )
        ]
        if scale_numeric:
            numeric_steps.append(("scaler", StandardScaler()))
        transformers.append(
            (
                "num",
                Pipeline(numeric_steps),
                schema.numeric_features,
            )
        )

    if schema.categorical_features:
        categorical_pipeline = Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(
                        strategy="most_frequent",
                        keep_empty_features=True,
                    ),
                ),
                (
                    "onehot",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False,
                        dtype=np.float64,
                    ),
                ),
            ]
        )
        transformers.append(
            ("cat", categorical_pipeline, schema.categorical_features)
        )

    if not transformers:
        raise ValueError("No features available for preprocessing")

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0.0,
        verbose_feature_names_out=True,
    )


def transformed_monotonic_constraints(
    preprocessor: ColumnTransformer,
    feature_directions: dict[str, int],
) -> tuple[list[str], list[int]]:
    names = list(preprocessor.get_feature_names_out())
    constraints = []

    for transformed_name in names:
        direction = 0

        if transformed_name.startswith("num__"):
            base = transformed_name[len("num__") :]
            if base.startswith("missingindicator_"):
                direction = 0
            else:
                direction = int(feature_directions.get(base, 0))

        # One-hot encoded categories are intentionally unconstrained.
        constraints.append(direction)

    return names, constraints
