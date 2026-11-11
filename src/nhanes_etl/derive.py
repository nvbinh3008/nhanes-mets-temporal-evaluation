from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def _num(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _yes_no(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return pd.Series(
        np.select([numeric.eq(1), numeric.eq(2)], [1.0, 0.0], default=np.nan),
        index=series.index,
    )


def _questionnaire_treatment(
    current: pd.Series,
    diagnosis: pd.Series | None = None,
    no_diagnosis_codes: set[int] | None = None,
) -> pd.Series:
    out = _yes_no(current)
    if diagnosis is not None:
        diag = pd.to_numeric(diagnosis, errors="coerce")
        codes = no_diagnosis_codes or {2}
        out = out.mask(out.isna() & diag.isin(codes), 0.0)
    return out


def _combine_any_treatment(*series: pd.Series) -> pd.Series:
    aligned = pd.concat(series, axis=1)
    positive = aligned.eq(1).any(axis=1)
    all_observed_no = aligned.notna().any(axis=1) & aligned.fillna(0).eq(0).all(axis=1)
    return pd.Series(
        np.select([positive, all_observed_no], [1.0, 0.0], default=np.nan),
        index=aligned.index,
    )


def _activity_minutes_component(
    yes_no: pd.Series,
    days: pd.Series,
    minutes: pd.Series,
) -> pd.Series:
    """Derive weekly minutes for one activity intensity using skip logic.

    A reported ``no`` contributes zero even when follow-up days/minutes are
    structurally missing. A reported ``yes`` requires both follow-up fields.
    If the yes/no item is missing but both follow-up fields are observed, the
    observed product is retained and flagged later as questionnaire-inconsistent.
    """
    response = pd.to_numeric(yes_no, errors="coerce")
    day = pd.to_numeric(days, errors="coerce")
    minute = pd.to_numeric(minutes, errors="coerce")
    complete_detail = day.notna() & minute.notna()

    out = pd.Series(np.nan, index=response.index, dtype=float)
    out = out.mask(response.eq(2), 0.0)
    out = out.mask(response.eq(1) & complete_detail, day * minute)
    out = out.mask(response.isna() & complete_detail, day * minute)
    return out


def _component_with_optional_treatment(
    measurement: pd.Series,
    positive_measurement: pd.Series,
    treatment: pd.Series | None = None,
) -> pd.Series:
    measurement_observed = measurement.notna()
    if treatment is None:
        positive = measurement_observed & positive_measurement
        negative = measurement_observed & ~positive_measurement
    else:
        positive = (measurement_observed & positive_measurement) | treatment.eq(1)
        negative = measurement_observed & ~positive_measurement & treatment.eq(0)
    return pd.Series(
        np.select([positive, negative], [1.0, 0.0], default=np.nan),
        index=measurement.index,
    ).astype("Float64")


def derive_cycle_variables(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()

    sbp_cols = [c for c in [f"sbp_reading_{i}" for i in range(1, 5)] if c in out]
    dbp_cols = [c for c in [f"dbp_reading_{i}" for i in range(1, 5)] if c in out]
    out["mean_sbp"] = out[sbp_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1) if sbp_cols else np.nan
    out["mean_dbp"] = out[dbp_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1) if dbp_cols else np.nan
    out["sbp_reading_count"] = out[sbp_cols].apply(pd.to_numeric, errors="coerce").notna().sum(axis=1) if sbp_cols else 0
    out["dbp_reading_count"] = out[dbp_cols].apply(pd.to_numeric, errors="coerce").notna().sum(axis=1) if dbp_cols else 0
    out["dbp_zero_qc"] = (
        out[dbp_cols].apply(pd.to_numeric, errors="coerce").eq(0).any(axis=1).astype("Int64")
        if dbp_cols else pd.Series(0, index=out.index, dtype="Int64")
    )
    out["bp_implausible_order_qc"] = (
        (out["mean_dbp"].notna() & out["mean_sbp"].notna() & out["mean_dbp"].ge(out["mean_sbp"]))
        .astype("Int64")
    )

    hours = _num(out, "fasting_hours_reported")
    minutes = _num(out, "fasting_minutes_reported")
    out["fasting_duration_hours"] = hours + minutes / 60.0

    anti_q = _questionnaire_treatment(
        out.get("current_antihypertensive_questionnaire", pd.Series(np.nan, index=out.index)),
        out.get("self_reported_hypertension"),
        {2},
    )
    out["antihypertensive_treatment_primary"] = anti_q
    out["antihypertensive_rx_flag"] = _num(out, "antihypertensive_rx_flag").fillna(0)

    insulin = _questionnaire_treatment(
        out.get("current_insulin_use", pd.Series(np.nan, index=out.index)),
        out.get("self_reported_diabetes"),
        {2, 3},
    )
    pills = _questionnaire_treatment(
        out.get("current_diabetes_pills", pd.Series(np.nan, index=out.index)),
        out.get("self_reported_diabetes"),
        {2, 3},
    )
    glucose_rx = _num(out, "glucose_lowering_rx_flag").fillna(0)
    out["glucose_lowering_treatment"] = _combine_any_treatment(insulin, pills, glucose_rx)

    triglyceride_rx = _num(out, "triglyceride_specific_rx_flag")
    if "triglyceride_specific_rx_flag" not in out:
        triglyceride_rx = pd.Series(np.nan, index=out.index)
    out["triglyceride_specific_treatment"] = triglyceride_rx

    sex = _num(out, "sex")
    waist = _num(out, "waist_cm")
    waist_positive = (sex.eq(1) & waist.ge(102)) | (sex.eq(2) & waist.ge(88))
    out["mets_abdominal_obesity"] = _component_with_optional_treatment(
        waist.where(sex.isin([1, 2])), waist_positive
    )

    tg = _num(out, "triglycerides_mg_dl")
    out["mets_elevated_triglycerides"] = _component_with_optional_treatment(
        tg, tg.ge(150), out["triglyceride_specific_treatment"]
    )

    hdl = _num(out, "hdl_mg_dl")
    hdl_positive = (sex.eq(1) & hdl.lt(40)) | (sex.eq(2) & hdl.lt(50))
    out["mets_low_hdl"] = _component_with_optional_treatment(
        hdl.where(sex.isin([1, 2])), hdl_positive
    )

    sbp = _num(out, "mean_sbp")
    dbp = _num(out, "mean_dbp")
    bp_measure = sbp.combine_first(dbp)
    bp_positive = sbp.ge(130) | dbp.ge(85)
    out["mets_elevated_blood_pressure"] = _component_with_optional_treatment(
        bp_measure, bp_positive, out["antihypertensive_treatment_primary"]
    )

    glucose = _num(out, "fasting_glucose_mg_dl")
    out["mets_elevated_fasting_glucose"] = _component_with_optional_treatment(
        glucose, glucose.ge(100), out["glucose_lowering_treatment"]
    )
    out["mets_elevated_fasting_glucose_110_sensitivity"] = _component_with_optional_treatment(
        glucose, glucose.ge(110), out["glucose_lowering_treatment"]
    )

    components = [
        "mets_abdominal_obesity",
        "mets_elevated_triglycerides",
        "mets_low_hdl",
        "mets_elevated_blood_pressure",
        "mets_elevated_fasting_glucose",
    ]
    complete = out[components].notna().all(axis=1)
    out["mets_component_count"] = out[components].sum(axis=1, min_count=5).astype("Float64")
    out["mets_prevalent"] = pd.Series(
        np.where(complete, out["mets_component_count"].ge(3).astype(float), np.nan),
        index=out.index,
        dtype="Float64",
    )

    if {"ever_smoked_100_cigarettes", "current_smoking_frequency"} & set(out.columns):
        ever = _num(out, "ever_smoked_100_cigarettes")
        current = _num(out, "current_smoking_frequency")
        out["smoking_status"] = pd.Series(
            np.select(
                [ever.eq(2), ever.eq(1) & current.eq(3), ever.eq(1) & current.isin([1, 2])],
                ["never", "former", "current"],
                default=None,
            ),
            index=out.index,
            dtype="object",
        )

    vigorous_yes = _num(out, "vigorous_recreation_yes")
    vigorous_days = _num(out, "vigorous_recreation_days_week")
    vigorous_minutes = _num(out, "vigorous_recreation_minutes_day")
    moderate_yes = _num(out, "moderate_recreation_yes")
    moderate_days = _num(out, "moderate_recreation_days_week")
    moderate_minutes = _num(out, "moderate_recreation_minutes_day")

    vigorous_weekly = _activity_minutes_component(
        vigorous_yes, vigorous_days, vigorous_minutes
    )
    moderate_weekly = _activity_minutes_component(
        moderate_yes, moderate_days, moderate_minutes
    )
    both_components_classifiable = vigorous_weekly.notna() & moderate_weekly.notna()
    out["vigorous_recreation_minutes_week"] = vigorous_weekly
    out["moderate_recreation_minutes_week"] = moderate_weekly
    out["ltpa_mvpa_equivalent_min_week"] = (
        2.0 * vigorous_weekly + moderate_weekly
    ).where(both_components_classifiable)
    out["physical_activity_missing_flag"] = (
        ~both_components_classifiable
    ).astype("Int64")
    out["physical_activity_partial_flag"] = (
        vigorous_weekly.notna() ^ moderate_weekly.notna()
    ).astype("Int64")
    out["physical_activity_skip_inconsistency_flag"] = (
        (vigorous_yes.eq(2) & (vigorous_days.notna() | vigorous_minutes.notna()))
        | (moderate_yes.eq(2) & (moderate_days.notna() | moderate_minutes.notna()))
        | (
            vigorous_yes.isna()
            & vigorous_days.notna()
            & vigorous_minutes.notna()
        )
        | (
            moderate_yes.isna()
            & moderate_days.notna()
            & moderate_minutes.notna()
        )
    ).astype("Int64")

    role = out.get("cycle_role", pd.Series("", index=out.index)).astype(str)
    fasting_weight = _num(out, "fasting_weight_2yr")
    out["weight_full_2007_2018"] = fasting_weight / 6.0
    out["weight_development_2007_2014"] = (fasting_weight / 4.0).where(role.eq("development"))
    out["weight_test_2015_2018"] = (fasting_weight / 2.0).where(role.eq("temporal_test"))
    out["weight_sensitivity_2005_2018"] = fasting_weight / 7.0

    cycle = out.get("cycle_id", pd.Series("", index=out.index)).astype(str)
    stratum = out.get("survey_stratum_raw", pd.Series("", index=out.index)).astype(str)
    psu = out.get("survey_psu_raw", pd.Series("", index=out.index)).astype(str)
    out["stratum_pool"] = cycle + "::" + stratum
    out["psu_pool"] = cycle + "::" + stratum + "::" + psu
    return out


def apply_primary_cohort(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = frame.copy()
    flow: list[dict[str, object]] = []

    def restrict(label: str, mask: pd.Series) -> None:
        nonlocal out
        before = len(out)
        out = out[mask.loc[out.index].fillna(False)].copy()
        flow.append({
            "step": len(flow),
            "criterion": label,
            "n_before": before,
            "n_excluded": before - len(out),
            "n_after": len(out),
        })

    flow.append({
        "step": 0,
        "criterion": "harmonized DEMO base",
        "n_before": len(out),
        "n_excluded": 0,
        "n_after": len(out),
    })
    restrict("age >= 20 years", _num(out, "age_years").ge(20))
    restrict("MEC examined", _num(out, "mec_exam_status").eq(2))
    sex = _num(out, "sex")
    pregnancy = _num(out, "pregnancy_status")
    restrict("not known pregnant", ~(sex.eq(2) & pregnancy.eq(1)))
    restrict(
        "reliable Day 1 recall and positive energy",
        _num(out, "day1_recall_status").eq(1) & _num(out, "energy_kcal_day1").gt(0),
    )
    restrict("positive fasting subsample weight", _num(out, "fasting_weight_2yr").gt(0))
    restrict(
        "fasting duration 8 to <24 hours",
        _num(out, "fasting_duration_hours").ge(8) & _num(out, "fasting_duration_hours").lt(24),
    )
    restrict("all five MetS components classifiable", out["mets_prevalent"].notna())
    return out.reset_index(drop=True), pd.DataFrame(flow)
