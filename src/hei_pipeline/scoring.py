from __future__ import annotations

import numpy as np
import pandas as pd


COMPONENTS = {
    "total_fruits": {"max": 5.0, "type": "adequacy", "zero": 0.0, "full": 0.8},
    "whole_fruits": {"max": 5.0, "type": "adequacy", "zero": 0.0, "full": 0.4},
    "total_vegetables": {"max": 5.0, "type": "adequacy", "zero": 0.0, "full": 1.1},
    "greens_and_beans": {"max": 5.0, "type": "adequacy", "zero": 0.0, "full": 0.2},
    "whole_grains": {"max": 10.0, "type": "adequacy", "zero": 0.0, "full": 1.5},
    "dairy": {"max": 10.0, "type": "adequacy", "zero": 0.0, "full": 1.3},
    "total_protein_foods": {"max": 5.0, "type": "adequacy", "zero": 0.0, "full": 2.5},
    "seafood_and_plant_proteins": {"max": 5.0, "type": "adequacy", "zero": 0.0, "full": 0.8},
    "fatty_acids": {"max": 10.0, "type": "adequacy", "zero": 1.2, "full": 2.5},
    "refined_grains": {"max": 10.0, "type": "moderation", "full": 1.8, "zero": 4.3},
    "sodium": {"max": 10.0, "type": "moderation", "full": 1.1, "zero": 2.0},
    "added_sugars": {"max": 10.0, "type": "moderation", "full": 6.5, "zero": 26.0},
    "saturated_fats": {"max": 10.0, "type": "moderation", "full": 8.0, "zero": 16.0},
}

ADEQUACY = [name for name, spec in COMPONENTS.items() if spec["type"] == "adequacy"]
MODERATION = [name for name, spec in COMPONENTS.items() if spec["type"] == "moderation"]


def _adequacy(value: pd.Series, zero: float, full: float, maximum: float) -> pd.Series:
    score = maximum * (value - zero) / (full - zero)
    return score.clip(lower=0, upper=maximum).where(value.notna())


def _moderation(value: pd.Series, full: float, zero: float, maximum: float) -> pd.Series:
    score = maximum * (zero - value) / (zero - full)
    return score.clip(lower=0, upper=maximum).where(value.notna())


def allocate_legumes(frame: pd.DataFrame) -> pd.DataFrame:
    """Allocate legumes to protein components first, then to vegetables.

    FPED total protein already includes legume protein equivalents and FPED total
    vegetables already includes legume cup equivalents. For HEI scoring, legumes
    contribute to Total Protein Foods and Seafood and Plant Proteins until both
    maximum-score standards are met; only the excess contributes to Total
    Vegetables and Greens and Beans.
    """
    out = frame.copy()
    energy = pd.to_numeric(out["energy_kcal"], errors="coerce")
    per1000 = energy / 1000.0

    legume_oz = pd.to_numeric(out["legumes_protein_oz"], errors="coerce").fillna(0.0)
    legume_cup = pd.to_numeric(out["legumes_vegetable_cup"], errors="coerce").fillna(0.0)
    total_protein = pd.to_numeric(out["protein_food_total_oz"], errors="coerce")
    seafood_plant = (
        pd.to_numeric(out["seafood_high_omega3_oz"], errors="coerce").fillna(0.0)
        + pd.to_numeric(out["seafood_low_omega3_oz"], errors="coerce").fillna(0.0)
        + pd.to_numeric(out["soy_oz"], errors="coerce").fillna(0.0)
        + pd.to_numeric(out["nuts_seeds_oz"], errors="coerce").fillna(0.0)
        + legume_oz
    )

    nonlegume_total_protein = (total_protein - legume_oz).clip(lower=0)
    nonlegume_seafood_plant = (seafood_plant - legume_oz).clip(lower=0)

    total_protein_deficit = (2.5 * per1000 - nonlegume_total_protein).clip(lower=0)
    seafood_plant_deficit = (0.8 * per1000 - nonlegume_seafood_plant).clip(lower=0)
    legume_oz_needed_for_protein = pd.concat(
        [total_protein_deficit, seafood_plant_deficit], axis=1
    ).max(axis=1)
    legume_oz_for_protein = np.minimum(legume_oz, legume_oz_needed_for_protein)
    excess_legume_oz = (legume_oz - legume_oz_for_protein).clip(lower=0)

    # Preserve the actual FPED cup-to-ounce relationship for each participant.
    excess_fraction = pd.Series(0.0, index=out.index)
    positive = legume_oz > 0
    excess_fraction.loc[positive] = (
        excess_legume_oz.loc[positive] / legume_oz.loc[positive]
    ).clip(0, 1)
    excess_legume_cup = legume_cup * excess_fraction

    out["legume_oz_for_protein"] = legume_oz_for_protein
    out["legume_cup_for_vegetables"] = excess_legume_cup
    out["hei_total_vegetable_cup"] = (
        pd.to_numeric(out["vegetable_total_cup"], errors="coerce")
        - legume_cup
        + excess_legume_cup
    ).clip(lower=0)
    out["hei_greens_and_beans_cup"] = (
        pd.to_numeric(out["dark_green_vegetable_cup"], errors="coerce").fillna(0.0)
        + excess_legume_cup
    )
    out["hei_total_protein_oz"] = total_protein
    out["hei_seafood_plant_protein_oz"] = seafood_plant
    return out


def build_densities(frame: pd.DataFrame, added_sugar_kcal_per_tsp: float = 16.0) -> pd.DataFrame:
    out = allocate_legumes(frame)
    energy = pd.to_numeric(out["energy_kcal"], errors="coerce")
    per1000 = energy / 1000.0

    out["density_total_fruits"] = out["fruit_total_cup"] / per1000
    out["density_whole_fruits"] = (
        out["fruit_total_cup"] - out["fruit_juice_cup"]
    ).clip(lower=0) / per1000
    out["density_total_vegetables"] = out["hei_total_vegetable_cup"] / per1000
    out["density_greens_and_beans"] = out["hei_greens_and_beans_cup"] / per1000
    out["density_whole_grains"] = out["whole_grain_oz"] / per1000
    out["density_dairy"] = out["dairy_total_cup"] / per1000
    out["density_total_protein_foods"] = out["hei_total_protein_oz"] / per1000
    out["density_seafood_and_plant_proteins"] = (
        out["hei_seafood_plant_protein_oz"] / per1000
    )
    out["ratio_fatty_acids"] = (
        out["monounsaturated_fat_g"] + out["polyunsaturated_fat_g"]
    ) / out["saturated_fat_g"].replace(0, np.nan)
    out["density_refined_grains"] = out["refined_grain_oz"] / per1000
    out["density_sodium"] = (out["sodium_mg"] / 1000.0) / per1000
    out["pct_energy_added_sugars"] = (
        out["added_sugars_tsp"] * added_sugar_kcal_per_tsp / energy * 100.0
    )
    out["pct_energy_saturated_fats"] = (
        out["saturated_fat_g"] * 9.0 / energy * 100.0
    )

    derived = [
        col for col in out.columns
        if col.startswith(("density_", "pct_energy_", "ratio_"))
    ]
    out.loc[energy.le(0) | energy.isna(), derived] = np.nan
    return out


def score_hei2020(frame: pd.DataFrame, added_sugar_kcal_per_tsp: float = 16.0) -> pd.DataFrame:
    out = build_densities(frame, added_sugar_kcal_per_tsp)
    values = {
        "total_fruits": out["density_total_fruits"],
        "whole_fruits": out["density_whole_fruits"],
        "total_vegetables": out["density_total_vegetables"],
        "greens_and_beans": out["density_greens_and_beans"],
        "whole_grains": out["density_whole_grains"],
        "dairy": out["density_dairy"],
        "total_protein_foods": out["density_total_protein_foods"],
        "seafood_and_plant_proteins": out["density_seafood_and_plant_proteins"],
        "fatty_acids": out["ratio_fatty_acids"],
        "refined_grains": out["density_refined_grains"],
        "sodium": out["density_sodium"],
        "added_sugars": out["pct_energy_added_sugars"],
        "saturated_fats": out["pct_energy_saturated_fats"],
    }

    component_cols = []
    for name, spec in COMPONENTS.items():
        if spec["type"] == "adequacy":
            score = _adequacy(values[name], spec["zero"], spec["full"], spec["max"])
        else:
            score = _moderation(values[name], spec["full"], spec["zero"], spec["max"])
        column = f"hei_{name}"
        out[column] = score
        component_cols.append(column)

    out["hei2020_total"] = out[component_cols].sum(
        axis=1, min_count=len(component_cols)
    )
    return add_rule_features(out)


def add_rule_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    severity_cols = []
    state_cols = []

    for name, spec in COMPONENTS.items():
        score_col = f"hei_{name}"
        adherence = (out[score_col] / spec["max"]).clip(0, 1)
        severity_col = f"rule_severity_{name}"
        state_col = f"rule_state_{name}"

        out[severity_col] = 1.0 - adherence
        out[state_col] = np.select(
            [
                out[score_col].ge(spec["max"] - 1e-8),
                out[score_col].le(1e-8),
            ],
            [2, 0],
            default=1,
        ).astype(float)
        out.loc[out[score_col].isna(), state_col] = np.nan

        full_col = f"rule_full_adherence_{name}"
        out[full_col] = np.where(
            out[score_col].isna(),
            np.nan,
            out[score_col].ge(spec["max"] - 1e-8).astype(float),
        )

        severity_cols.append(severity_col)
        state_cols.append(state_col)

    out["adequacy_violation_burden"] = out[
        [f"rule_severity_{name}" for name in ADEQUACY]
    ].sum(axis=1, min_count=len(ADEQUACY))
    out["moderation_violation_burden"] = out[
        [f"rule_severity_{name}" for name in MODERATION]
    ].sum(axis=1, min_count=len(MODERATION))
    out["total_violation_burden"] = out[severity_cols].sum(
        axis=1, min_count=len(severity_cols)
    )
    out["maximum_violation_severity"] = out[severity_cols].max(axis=1)
    out["number_full_adherence_rules"] = out[state_cols].eq(2).sum(axis=1)
    out["number_zero_score_rules"] = out[state_cols].eq(0).sum(axis=1)
    incomplete = out[state_cols].isna().any(axis=1)
    out.loc[
        incomplete,
        ["number_full_adherence_rules", "number_zero_score_rules"],
    ] = np.nan
    return out


def validate_scores(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, spec in COMPONENTS.items():
        column = f"hei_{name}"
        values = pd.to_numeric(frame[column], errors="coerce")
        invalid = values.notna() & (
            values.lt(-1e-8) | values.gt(spec["max"] + 1e-8)
        )
        rows.append({
            "variable": column,
            "minimum": values.min(),
            "maximum": values.max(),
            "missing": int(values.isna().sum()),
            "invalid": int(invalid.sum()),
            "hard_fail": bool(invalid.any()),
        })

    total = pd.to_numeric(frame["hei2020_total"], errors="coerce")
    invalid_total = total.notna() & (total.lt(-1e-8) | total.gt(100 + 1e-8))
    rows.append({
        "variable": "hei2020_total",
        "minimum": total.min(),
        "maximum": total.max(),
        "missing": int(total.isna().sum()),
        "invalid": int(invalid_total.sum()),
        "hard_fail": bool(invalid_total.any()),
    })
    return pd.DataFrame(rows)
