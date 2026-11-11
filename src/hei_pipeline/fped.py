from __future__ import annotations

from pathlib import Path
import re
import numpy as np
import pandas as pd

from .io import read_table


CANONICAL = {
    "FOODCODE": "food_code",
    "MODCODE": "modifier_code",
    "F_TOTAL": "fruit_total_cup",
    "F_CITMLB": "fruit_citmlb_cup",
    "F_OTHER": "fruit_other_cup",
    "F_JUICE": "fruit_juice_cup",
    "V_TOTAL": "vegetable_total_cup",
    "V_DRKGR": "dark_green_vegetable_cup",
    "V_REDOR_TOTAL": "red_orange_vegetable_cup",
    "V_REDOR_TOMATO": "tomato_cup",
    "V_REDOR_OTHER": "other_red_orange_vegetable_cup",
    "V_STARCHY_TOTAL": "starchy_vegetable_cup",
    "V_STARCHY_POTATO": "potato_cup",
    "V_STARCHY_OTHER": "other_starchy_vegetable_cup",
    "V_OTHER": "other_vegetable_cup",
    "V_LEGUMES": "legumes_vegetable_cup",
    "G_TOTAL": "grain_total_oz",
    "G_WHOLE": "whole_grain_oz",
    "G_REFINED": "refined_grain_oz",
    "PF_TOTAL": "protein_food_total_oz",
    "PF_MPS_TOTAL": "meat_poultry_seafood_oz",
    "PF_MEAT": "meat_oz",
    "PF_CUREDMEAT": "cured_meat_oz",
    "PF_ORGAN": "organ_meat_oz",
    "PF_POULT": "poultry_oz",
    "PF_SEAFD_HI": "seafood_high_omega3_oz",
    "PF_SEAFD_LOW": "seafood_low_omega3_oz",
    "PF_EGGS": "eggs_oz",
    "PF_SOY": "soy_oz",
    "PF_NUTSDS": "nuts_seeds_oz",
    "PF_LEGUMES": "legumes_protein_oz",
    "D_TOTAL": "dairy_total_cup",
    "D_MILK": "milk_cup",
    "D_YOGURT": "yogurt_cup",
    "D_CHEESE": "cheese_cup",
    "OILS": "oils_g",
    "SOLID_FATS": "solid_fats_g",
    "ADD_SUGARS": "added_sugars_tsp",
    "A_DRINKS": "alcoholic_drinks",
}

REQUIRED = {
    "food_code", "fruit_total_cup", "fruit_juice_cup",
    "vegetable_total_cup", "dark_green_vegetable_cup",
    "legumes_vegetable_cup", "whole_grain_oz", "refined_grain_oz",
    "protein_food_total_oz", "seafood_high_omega3_oz",
    "seafood_low_omega3_oz", "soy_oz", "nuts_seeds_oz",
    "legumes_protein_oz", "dairy_total_cup", "added_sugars_tsp",
}


def _base_name(column: object) -> str:
    text = str(column).strip().upper()
    return re.sub(r"\s*\(.*?\)\s*$", "", text).strip()


def normalize_fped(frame: pd.DataFrame, cycle: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    renamed = {}
    for col in frame.columns:
        base = _base_name(col)
        if base in CANONICAL:
            renamed[col] = CANONICAL[base]
    out = frame.rename(columns=renamed).copy()
    out["cycle_id"] = cycle

    missing = sorted(REQUIRED - set(out.columns))
    audit = pd.DataFrame([{
        "cycle": cycle,
        "rows": len(out),
        "columns": len(out.columns),
        "missing_required": "|".join(missing),
        "hard_fail": bool(missing),
    }])

    if missing:
        return out, audit

    out["food_code"] = pd.to_numeric(out["food_code"], errors="coerce").astype("Int64")
    if "modifier_code" in out.columns:
        out["modifier_code"] = pd.to_numeric(out["modifier_code"], errors="coerce").fillna(0).astype("Int64")

    measure_cols = [c for c in out.columns if c.endswith(("_cup", "_oz", "_g", "_tsp"))]
    for col in measure_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
        if (out[col].dropna() < 0).any():
            audit.loc[0, "hard_fail"] = True
            audit.loc[0, "missing_required"] += f"|negative:{col}"

    return out, audit


def load_food_reference(path: Path, cycle: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    return normalize_fped(read_table(path), cycle)


def choose_reference_file(folder: Path, tag: str) -> Path:
    candidates = []
    for path in folder.rglob("*"):
        if not path.is_file():
            continue
        name = path.name.lower()
        if name.startswith(f"fped_{tag}".lower()) and path.suffix.lower() in {".xls", ".xlsx", ".csv", ".sas7bdat", ".xpt"}:
            candidates.append(path)
    if not candidates:
        raise FileNotFoundError(f"No usable FPED food-code reference in {folder}")
    priority = {".parquet": 0, ".csv": 1, ".sas7bdat": 2, ".xpt": 3, ".xlsx": 4, ".xls": 5}
    return sorted(candidates, key=lambda p: priority.get(p.suffix.lower(), 99))[0]
