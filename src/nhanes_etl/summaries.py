from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


METS_COMPONENTS = [
    "mets_abdominal_obesity",
    "mets_elevated_triglycerides",
    "mets_low_hdl",
    "mets_elevated_blood_pressure",
    "mets_elevated_fasting_glucose",
]


def _weighted_binary_summary(
    frame: pd.DataFrame,
    outcome: str,
    weight: str,
) -> dict[str, float | int]:
    y = pd.to_numeric(frame.get(outcome), errors="coerce")
    w = pd.to_numeric(frame.get(weight), errors="coerce")
    valid = y.notna() & w.notna() & w.gt(0)
    yv = y[valid].astype(float)
    wv = w[valid].astype(float)
    if not valid.any():
        return {
            "n_weighted_complete": 0,
            "sum_weight": np.nan,
            "weighted_cases": np.nan,
            "prevalence_weighted": np.nan,
            "kish_effective_n": np.nan,
        }
    sum_weight = float(wv.sum())
    weighted_cases = float((wv * yv).sum())
    kish = float(sum_weight**2 / (wv.pow(2).sum())) if wv.pow(2).sum() > 0 else np.nan
    return {
        "n_weighted_complete": int(valid.sum()),
        "sum_weight": sum_weight,
        "weighted_cases": weighted_cases,
        "prevalence_weighted": weighted_cases / sum_weight,
        "kish_effective_n": kish,
    }


def _summary_row(
    frame: pd.DataFrame,
    scope: str,
    cycle_id: str,
    cycle_role: str,
    weight_column: str,
) -> dict[str, object]:
    outcome = pd.to_numeric(frame.get("mets_prevalent"), errors="coerce")
    row: dict[str, object] = {
        "scope": scope,
        "cycle_id": cycle_id,
        "cycle_role": cycle_role,
        "weight_column": weight_column,
        "n": int(len(frame)),
        "mets_cases_unweighted": int(outcome.sum(skipna=True)),
        "mets_prevalence_unweighted": float(outcome.mean())
        if outcome.notna().any()
        else np.nan,
        "ltpa_complete_n": int(
            pd.to_numeric(
                frame.get("ltpa_mvpa_equivalent_min_week"), errors="coerce"
            ).notna().sum()
        ),
        "ltpa_missing_n": int(
            pd.to_numeric(
                frame.get("ltpa_mvpa_equivalent_min_week"), errors="coerce"
            ).isna().sum()
        ),
    }
    row.update(_weighted_binary_summary(frame, "mets_prevalent", weight_column))
    row["variance_note"] = (
        "Descriptive weighted mean only; design-based standard errors require "
        "strata/PSU-aware survey estimation."
    )
    return row


def build_weighted_cycle_summary(pooled: pd.DataFrame) -> pd.DataFrame:
    """Create cycle and pooled descriptive summaries with the locked weights."""
    columns = [
        "scope",
        "cycle_id",
        "cycle_role",
        "weight_column",
        "n",
        "mets_cases_unweighted",
        "mets_prevalence_unweighted",
        "n_weighted_complete",
        "sum_weight",
        "weighted_cases",
        "prevalence_weighted",
        "kish_effective_n",
        "ltpa_complete_n",
        "ltpa_missing_n",
        "variance_note",
    ]
    if pooled.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, object]] = []
    for (cycle_id, cycle_role), group in pooled.groupby(
        ["cycle_id", "cycle_role"], dropna=False, sort=True
    ):
        rows.append(
            _summary_row(
                group,
                scope="cycle",
                cycle_id=str(cycle_id),
                cycle_role=str(cycle_role),
                weight_column="fasting_weight_2yr",
            )
        )

    role_specs = [
        ("development_2007_2014", "development", "weight_development_2007_2014"),
        ("temporal_test_2015_2018", "temporal_test", "weight_test_2015_2018"),
    ]
    for scope, role, weight in role_specs:
        group = pooled[pooled["cycle_role"].eq(role)]
        if not group.empty:
            rows.append(
                _summary_row(
                    group,
                    scope=scope,
                    cycle_id="pooled",
                    cycle_role=role,
                    weight_column=weight,
                )
            )

    rows.append(
        _summary_row(
            pooled,
            scope="full_2007_2018",
            cycle_id="pooled",
            cycle_role="all",
            weight_column="weight_full_2007_2018",
        )
    )
    return pd.DataFrame(rows).reindex(columns=columns)


def build_weighted_component_summary(pooled: pd.DataFrame) -> pd.DataFrame:
    """Long-format descriptive prevalence for the five MetS components."""
    columns = [
        "scope",
        "cycle_id",
        "cycle_role",
        "component",
        "weight_column",
        "n_complete",
        "cases_unweighted",
        "prevalence_unweighted",
        "sum_weight",
        "weighted_cases",
        "prevalence_weighted",
        "kish_effective_n",
        "variance_note",
    ]
    if pooled.empty:
        return pd.DataFrame(columns=columns)

    groups: list[tuple[str, str, str, pd.DataFrame, str]] = []
    for (cycle_id, role), group in pooled.groupby(
        ["cycle_id", "cycle_role"], dropna=False, sort=True
    ):
        groups.append(("cycle", str(cycle_id), str(role), group, "fasting_weight_2yr"))
    development = pooled[pooled["cycle_role"].eq("development")]
    temporal = pooled[pooled["cycle_role"].eq("temporal_test")]
    if not development.empty:
        groups.append(
            (
                "development_2007_2014",
                "pooled",
                "development",
                development,
                "weight_development_2007_2014",
            )
        )
    if not temporal.empty:
        groups.append(
            (
                "temporal_test_2015_2018",
                "pooled",
                "temporal_test",
                temporal,
                "weight_test_2015_2018",
            )
        )
    groups.append(
        (
            "full_2007_2018",
            "pooled",
            "all",
            pooled,
            "weight_full_2007_2018",
        )
    )

    rows: list[dict[str, object]] = []
    for scope, cycle_id, role, group, weight in groups:
        for component in METS_COMPONENTS:
            y = pd.to_numeric(group.get(component), errors="coerce")
            weighted = _weighted_binary_summary(group, component, weight)
            rows.append(
                {
                    "scope": scope,
                    "cycle_id": cycle_id,
                    "cycle_role": role,
                    "component": component,
                    "weight_column": weight,
                    "n_complete": int(y.notna().sum()),
                    "cases_unweighted": int(y.sum(skipna=True)),
                    "prevalence_unweighted": float(y.mean())
                    if y.notna().any()
                    else np.nan,
                    "sum_weight": weighted["sum_weight"],
                    "weighted_cases": weighted["weighted_cases"],
                    "prevalence_weighted": weighted["prevalence_weighted"],
                    "kish_effective_n": weighted["kish_effective_n"],
                    "variance_note": (
                        "Descriptive weighted mean only; design-based standard "
                        "errors require strata/PSU-aware survey estimation."
                    ),
                }
            )
    return pd.DataFrame(rows).reindex(columns=columns)
