from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd


EXACT_ALLOWED: dict[str, set[float]] = {
    "sex": {1, 2},
    "mec_exam_status": {1, 2},
    "pregnancy_status": {1, 2, 3},
    "day1_recall_status": {1, 2, 3, 4, 5},
    "day2_recall_status": {1, 2, 3, 4, 5},
    "current_antihypertensive_questionnaire": {1, 2},
    "self_reported_hypertension": {1, 2, 7, 9},
    "current_insulin_use": {1, 2, 7, 9},
    "current_diabetes_pills": {1, 2, 7, 9},
    "self_reported_diabetes": {1, 2, 3, 7, 9},
    "ever_smoked_100_cigarettes": {1, 2, 7, 9},
    "current_smoking_frequency": {1, 2, 3, 7, 9},
    "vigorous_recreation_yes": {1, 2, 7, 9},
    "moderate_recreation_yes": {1, 2, 7, 9},
    "rx_use_past_month": {1, 2, 7, 9},
}

RANGES: dict[str, tuple[float | None, float | None, bool, bool]] = {
    "age_years": (0, 80, True, True),
    "family_poverty_income_ratio": (0, 5, True, True),
    "fasting_hours_reported": (0, 99, True, True),
    "fasting_minutes_reported": (0, 59, True, True),
    "waist_cm": (0, None, False, True),
    "body_weight_kg": (0, None, False, True),
    "bmi_kg_m2": (0, None, False, True),
    "fasting_weight_2yr": (0, None, False, True),
    "interview_weight_2yr": (0, None, False, True),
    "mec_weight_2yr": (0, None, False, True),
    "diet_day1_weight": (0, None, False, True),
    "diet_two_day_weight": (0, None, False, True),
    "day1_food_grams": (0, None, True, True),
    "day2_food_grams": (0, None, True, True),
    "triglycerides_mg_dl": (0, None, False, True),
    "hdl_mg_dl": (0, None, False, True),
    "fasting_glucose_mg_dl": (0, None, False, True),
}

SPECIAL_MISSING: dict[str, set[float]] = {
    "sex": {7, 9},
    "mec_exam_status": {7, 9},
    "pregnancy_status": {7, 9},
    "education_adult": {7, 9, 77, 99},
    "race_ethnicity_legacy": {7, 9},
    "race_ethnicity_expanded": {7, 9},
    "self_reported_hypertension": {7, 9},
    "current_antihypertensive_questionnaire": {7, 9},
    "self_reported_diabetes": {7, 9},
    "current_insulin_use": {7, 9},
    "current_diabetes_pills": {7, 9},
    "ever_smoked_100_cigarettes": {7, 9},
    "current_smoking_frequency": {7, 9},
    "vigorous_recreation_yes": {7, 9},
    "moderate_recreation_yes": {7, 9},
    "vigorous_recreation_days_week": {77, 99},
    "moderate_recreation_days_week": {77, 99},
    "vigorous_recreation_minutes_day": {7777, 9999},
    "moderate_recreation_minutes_day": {7777, 9999},
}


def clean_known_missing(series: pd.Series, canonical: str) -> pd.Series:
    if canonical not in SPECIAL_MISSING:
        return series
    numeric = pd.to_numeric(series, errors="coerce")
    return series.mask(numeric.isin(SPECIAL_MISSING[canonical]))


def invalid_mask(series: pd.Series, canonical: str, valid_condition: str = "") -> pd.Series:
    """Return True only for observed values that violate a machine-actionable rule."""
    observed = series.notna()
    if not observed.any():
        return pd.Series(False, index=series.index)

    if canonical in EXACT_ALLOWED:
        numeric = pd.to_numeric(series, errors="coerce")
        return observed & ~numeric.isin(EXACT_ALLOWED[canonical])

    if canonical in RANGES:
        lower, upper, lower_inclusive, upper_inclusive = RANGES[canonical]
        numeric = pd.to_numeric(series, errors="coerce")
        invalid = observed & numeric.isna()
        if lower is not None:
            invalid |= numeric.lt(lower) if lower_inclusive else numeric.le(lower)
        if upper is not None:
            invalid |= numeric.gt(upper) if upper_inclusive else numeric.ge(upper)
        return invalid

    condition = str(valid_condition)
    numeric = pd.to_numeric(series, errors="coerce")
    invalid = observed & numeric.isna() if "finite" in condition.lower() else pd.Series(False, index=series.index)

    match = re.search(
        r"(-?\d+(?:\.\d+)?)\s*<=\s*value\s*<=\s*(-?\d+(?:\.\d+)?)",
        condition,
        flags=re.I,
    )
    if match:
        low, high = float(match.group(1)), float(match.group(2))
        invalid |= observed & (numeric.lt(low) | numeric.gt(high))

    if re.search(r"\bstrictly\s*>\s*0\b|\bpositive\b", condition, flags=re.I):
        invalid |= observed & numeric.le(0)
    elif re.search(r"\bnon-negative\b", condition, flags=re.I):
        invalid |= observed & numeric.lt(0)

    code_match = re.search(r"code\s+in\s+\{([^}]+)\}", condition, flags=re.I)
    if code_match:
        allowed = {
            float(token.strip())
            for token in code_match.group(1).split(",")
            if token.strip()
        }
        invalid |= observed & ~numeric.isin(allowed)

    return invalid


def summarize_series(series: pd.Series) -> dict[str, Any]:
    numeric = pd.to_numeric(series, errors="coerce")
    nonmissing = int(series.notna().sum())
    return {
        "dtype": str(series.dtype),
        "rows": int(len(series)),
        "nonmissing": nonmissing,
        "missing": int(len(series) - nonmissing),
        "distinct_nonmissing": int(series.nunique(dropna=True)),
        "min": float(numeric.min()) if numeric.notna().any() else "",
        "max": float(numeric.max()) if numeric.notna().any() else "",
    }
