from __future__ import annotations

from pathlib import Path
import pandas as pd


EVALUATION_METADATA = [
    "cycle_id",
    "cycle_role",
    "seqn",
    "recall_day",
    "mets_prevalent",
    "weight_full_2007_2018",
    "weight_development_2007_2014",
    "weight_test_2015_2018",
    "stratum_pool",
    "psu_pool",
]

SAFE_COVARIATES = [
    "age_years",
    "sex",
    "race_ethnicity_legacy",
    "education_adult",
    "family_poverty_income_ratio",
    "smoking_status",
    "ltpa_mvpa_equivalent_min_week",
    "alcohol_g_day1",
]


def available(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    return [column for column in columns if column in frame.columns]


def _feature_columns(frame: pd.DataFrame, cfg: dict) -> dict[str, list[str]]:
    reps = cfg["representations"]
    metadata = available(frame, EVALUATION_METADATA)

    r0_features = available(frame, reps["r0_raw_dietary"])
    r1_features = ["hei2020_total"] if "hei2020_total" in frame.columns else []
    r2_features = [
        column for column in frame.columns
        if column.startswith("hei_")
        and column not in {
            "hei_total_vegetable_cup",
            "hei_greens_and_beans_cup",
            "hei_total_protein_oz",
            "hei_seafood_plant_protein_oz",
        }
    ]
    # R3 follows the manuscript contract: 13 binary full-adherence flags.
    r3_features = [
        column for column in frame.columns
        if column.startswith("rule_full_adherence_")
    ]

    # R4 follows the manuscript contract: ternary rule states plus aggregate
    # burden/count summaries. Individual continuous severities remain available
    # in participant-level HEI outputs for explanation, but are not predictors
    # in R4 because they are one-to-one transformations of component scores.
    r4_features = [
        column for column in frame.columns
        if column.startswith("rule_state_")
    ] + available(frame, reps["r4_rule_features"])
    r5_features = list(
        dict.fromkeys(r4_features + available(frame, SAFE_COVARIATES))
    )

    return {
        "R0": metadata + r0_features,
        "R1": metadata + r1_features,
        "R2": metadata + r2_features,
        "R3": metadata + r3_features,
        "R4": metadata + r4_features,
        "R5": metadata + r5_features,
    }


def _split_name(frame: pd.DataFrame) -> pd.Series:
    if "cycle_role" in frame.columns:
        return frame["cycle_role"].map({
            "development": "development",
            "temporal_test": "temporal_test",
        }).fillna("other")
    return pd.Series("pooled", index=frame.index)


def build_representations(
    analytic: pd.DataFrame,
    cfg: dict,
    output_dir: Path,
    day_label: str,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = analytic.copy()
    frame["_split"] = _split_name(frame)

    manifests = []
    for split, split_frame in frame.groupby("_split", dropna=False):
        split_frame = split_frame.drop(columns="_split")
        columns_by_representation = _feature_columns(split_frame, cfg)

        for representation, columns in columns_by_representation.items():
            data = split_frame[columns].copy()
            target_dir = output_dir / str(split)
            target_dir.mkdir(parents=True, exist_ok=True)
            target = target_dir / f"{representation.lower()}_{day_label}.parquet"
            data.to_parquet(target, index=False)

            metadata = available(data, EVALUATION_METADATA)
            features = [column for column in data.columns if column not in metadata]
            manifests.append({
                "split": split,
                "representation": representation,
                "day_label": day_label,
                "rows": len(data),
                "columns": len(data.columns),
                "feature_count": len(features),
                "path": str(target),
                "features": "|".join(features),
                "evaluation_metadata": "|".join(metadata),
            })

    return pd.DataFrame(manifests)
