from __future__ import annotations

import re
from collections import defaultdict

import pandas as pd

from .config import DictionaryConfig
from .exceptions import PipelineError


DERIVED_ABSOLUTE_BLACKLIST = {
    "mean_sbp",
    "mean_dbp",
    "sbp_reading_count",
    "dbp_reading_count",
    "dbp_zero_qc",
    "bp_implausible_order_qc",
    "antihypertensive_treatment_primary",
    "glucose_lowering_treatment",
    "triglyceride_specific_treatment",
    "mets_abdominal_obesity",
    "mets_elevated_triglycerides",
    "mets_low_hdl",
    "mets_elevated_blood_pressure",
    "mets_elevated_fasting_glucose",
    "mets_elevated_fasting_glucose_110_sensitivity",
    "mets_component_count",
    "mets_prevalent",
}

ADMINISTRATIVE_COLUMNS = {
    "seqn",
    "cycle_id",
    "cycle_role",
    "suffix",
    "source_file",
    "survey_stratum_raw",
    "survey_psu_raw",
    "stratum_pool",
    "psu_pool",
    "fasting_weight_2yr",
    "interview_weight_2yr",
    "mec_weight_2yr",
    "diet_day1_weight",
    "diet_two_day_weight",
    "weight_full_2007_2018",
    "weight_development_2007_2014",
    "weight_test_2015_2018",
    "weight_sensitivity_2005_2018",
}


def declared_blacklist(dictionary: DictionaryConfig) -> set[str]:
    frame = dictionary.variables_long
    mask = frame["blacklist_status"].astype(str).str.contains(
        "absolute|administrative_not_predictor", case=False, regex=True
    )
    values = set(frame.loc[mask, "normalized_variable"].astype(str))
    values.discard("")
    if not dictionary.predictor_blacklist.empty:
        raw = dictionary.predictor_blacklist.get("raw_or_derived", pd.Series(dtype=str)).astype(str)
        # Raw names are included for defense in depth; canonical matching occurs
        # through Variable_Long wherever a mapping exists.
        values.update(raw.str.strip().tolist())
    return values | DERIVED_ABSOLUTE_BLACKLIST | ADMINISTRATIVE_COLUMNS


def assert_no_blacklisted_features(
    columns: list[str] | pd.Index,
    dictionary: DictionaryConfig,
) -> None:
    blocked = declared_blacklist(dictionary)
    violations = sorted(set(map(str, columns)) & blocked)
    if violations:
        raise PipelineError(
            "Feature matrix contains absolute-blacklist or administrative columns: "
            + ", ".join(violations)
        )


def primary_allowed_columns(
    frame: pd.DataFrame,
    dictionary: DictionaryConfig,
) -> list[str]:
    variable_rows = dictionary.variables_long
    allowed = set(
        variable_rows.loc[
            variable_rows["blacklist_status"].astype(str).str.fullmatch("allowed", case=False),
            "normalized_variable",
        ].astype(str)
    )
    allowed.update({
        "smoking_status",
        "ltpa_mvpa_equivalent_min_week",
        "physical_activity_missing_flag",
    })
    columns = [column for column in frame.columns if column in allowed]
    assert_no_blacklisted_features(columns, dictionary)
    return columns
