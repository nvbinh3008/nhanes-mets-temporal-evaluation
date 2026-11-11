from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from .config import DictionaryConfig, Settings
from .io import output_path, write_frame
from .utils import normalize_drug_text


_CLASS_SUFFIX_RE = re.compile(r"\s+(?:and\s+)?class\s+members?$", flags=re.I)


def _ingredient_tokens(value: str) -> list[str]:
    """Return deduplicated, boundary-safe ingredient tokens.

    The dictionary is semicolon-delimited. Historical placeholder text such as
    ``ramipril and class members`` is reduced to ``ramipril`` for backward
    compatibility, but v1.2.4 stores explicit ingredient lists.
    """
    tokens: list[str] = []
    for part in str(value).split(";"):
        normalized = normalize_drug_text(part)
        normalized = _CLASS_SUFFIX_RE.sub("", normalized).strip()
        if not normalized:
            continue
        # Slash-delimited alternatives are expanded while preserving the full
        # phrase. This supports legacy entries such as lispro/aspart/glulisine.
        alternatives = [normalized]
        if "/" in normalized:
            pieces = [piece.strip() for piece in normalized.split("/") if piece.strip()]
            if pieces:
                alternatives.extend(pieces)
        for token in alternatives:
            if token and token not in tokens:
                tokens.append(token)
    # Prefer the most specific token as match evidence.
    return sorted(tokens, key=lambda token: (-len(token), token))


def _load_drug_id_map(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame(columns=["rx_generic_drug_id", "flag", "therapeutic_class"])
    frame = pd.read_csv(path, dtype=str).fillna("")
    required = {"rx_generic_drug_id", "flag"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Drug ID map is missing columns: {sorted(missing)}")
    return frame


def _deduplicate_matches(
    matches: list[tuple[str, str, str, str]]
) -> list[tuple[str, str, str, str]]:
    seen: set[tuple[str, str, str, str]] = set()
    out: list[tuple[str, str, str, str]] = []
    for item in matches:
        if item not in seen:
            out.append(item)
            seen.add(item)
    return out


def _route_excluded(flag: str, normalized_name: str) -> bool:
    """Prevent route-specific false positives in medication-class auditing."""
    if flag == "antihypertensive" and re.search(
        r"\b(?:ophthalmic|intraocular)\b", normalized_name
    ):
        return True
    return False


def map_medications(
    rx_rows: pd.DataFrame,
    dictionary: DictionaryConfig,
    settings: Settings,
    cycle: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if rx_rows.empty:
        return pd.DataFrame(columns=["seqn"]), pd.DataFrame()

    rx = rx_rows.copy()
    all_seqn = rx[["seqn"]].drop_duplicates().copy()
    use_numeric_all = pd.to_numeric(
        rx.get("rx_use_past_month", pd.Series(float("nan"), index=rx.index)),
        errors="coerce",
    )
    use_summary = (
        pd.DataFrame({"seqn": rx["seqn"], "use": use_numeric_all})
        .groupby("seqn", dropna=False)["use"]
        .agg(
            lambda x: 1.0
            if x.eq(1).any()
            else (0.0 if x.notna().any() and x.dropna().eq(2).all() else float("nan"))
        )
        .rename("rxq_any_prescription")
        .reset_index()
    )
    all_seqn = all_seqn.merge(use_summary, on="seqn", how="left", validate="one_to_one")
    all_seqn["rxq_record_present"] = 1

    rx["rx_generic_name_normalized"] = rx.get(
        "rx_generic_name", pd.Series("", index=rx.index)
    ).map(normalize_drug_text)
    rx["rx_generic_drug_id_normalized"] = rx.get(
        "rx_generic_drug_id", pd.Series("", index=rx.index)
    ).fillna("").astype(str).str.strip()
    if "rx_use_past_month" in rx:
        active = pd.to_numeric(rx["rx_use_past_month"], errors="coerce").eq(1)
        rx = rx[active | rx["rx_use_past_month"].isna()].copy()

    rules = dictionary.medication_map.copy()
    rules["tokens"] = rules["representative_generic_names"].map(_ingredient_tokens)
    id_map = _load_drug_id_map(settings.drug_id_map)
    id_lookup: dict[str, list[tuple[str, str]]] = {}
    for row in id_map.to_dict(orient="records"):
        key = str(row["rx_generic_drug_id"]).strip()
        if key:
            id_lookup.setdefault(key, []).append(
                (str(row["flag"]), str(row.get("therapeutic_class", "")))
            )

    matches: list[dict[str, object]] = []
    for _, row in rx.iterrows():
        drug_id = str(row.get("rx_generic_drug_id_normalized", ""))
        name = str(row.get("rx_generic_name_normalized", ""))
        row_matches: list[tuple[str, str, str, str]] = []

        for flag, therapeutic_class in id_lookup.get(drug_id, []):
            row_matches.append(
                (str(flag), str(therapeutic_class), "exact_drug_id", drug_id)
            )

        if not row_matches and name:
            for rule in rules.to_dict(orient="records"):
                rule_flag = str(rule["flag"])
                if _route_excluded(rule_flag, name):
                    continue
                tokens = list(rule["tokens"])
                exact = next((token for token in tokens if name == token), None)
                if exact is not None:
                    row_matches.append(
                        (
                            str(rule["flag"]),
                            str(rule["therapeutic_class"]),
                            "exact_generic_name",
                            exact,
                        )
                    )
                    continue

                token = next(
                    (
                        candidate
                        for candidate in tokens
                        if re.search(
                            rf"(?<![a-z0-9]){re.escape(candidate)}(?![a-z0-9])",
                            name,
                        )
                    ),
                    None,
                )
                if token is not None:
                    row_matches.append(
                        (
                            str(rule["flag"]),
                            str(rule["therapeutic_class"]),
                            "generic_ingredient_token",
                            token,
                        )
                    )

        row_matches = _deduplicate_matches(row_matches)
        if not row_matches:
            row_matches.append(("unmapped", "", "unmapped", ""))

        for flag, therapeutic_class, method, evidence in row_matches:
            matches.append(
                {
                    "cycle_id": cycle,
                    "seqn": row["seqn"],
                    "rx_generic_drug_id": drug_id,
                    "rx_generic_name": row.get("rx_generic_name", ""),
                    "rx_generic_name_normalized": name,
                    "flag": flag,
                    "therapeutic_class": therapeutic_class,
                    "match_method": method,
                    "match_evidence": evidence,
                }
            )

    match_frame = pd.DataFrame(matches)
    mapped = match_frame[match_frame["flag"].ne("unmapped")].copy()
    if mapped.empty:
        pivot = pd.DataFrame(columns=["seqn"])
    else:
        pivot = (
            mapped.assign(value=1)
            .pivot_table(
                index="seqn",
                columns="flag",
                values="value",
                aggfunc="max",
                fill_value=0,
            )
            .reset_index()
        )
    flags = all_seqn.merge(pivot, on="seqn", how="left", validate="one_to_one")
    for flag in sorted(set(rules["flag"])):
        if flag not in flags:
            flags[flag] = 0
        flags[flag] = flags[flag].fillna(0).astype(int)
    unmapped_counts = (
        match_frame[match_frame["flag"].eq("unmapped")]
        .groupby("seqn")
        .size()
        .rename("medication_unmapped_row_count")
        .reset_index()
    )
    flags = flags.merge(unmapped_counts, on="seqn", how="left", validate="one_to_one")
    flags["medication_unmapped_row_count"] = (
        flags["medication_unmapped_row_count"].fillna(0).astype(int)
    )
    flags = flags.rename(
        columns={
            "glucose_lowering": "glucose_lowering_rx_flag",
            "triglyceride_specific": "triglyceride_specific_rx_flag",
            "other_lipid_lowering": "other_lipid_lowering_rx_flag",
            "antihypertensive": "antihypertensive_rx_flag",
        }
    )
    return flags, match_frame


def medication_mapping_summary(matches: pd.DataFrame) -> pd.DataFrame:
    """Compact audit table by mapped flag/class/method."""
    columns = [
        "cycle_id",
        "flag",
        "therapeutic_class",
        "match_method",
        "rows",
        "participants",
        "generic_names",
    ]
    if matches.empty:
        return pd.DataFrame(columns=columns)
    return (
        matches.groupby(
            ["cycle_id", "flag", "therapeutic_class", "match_method"],
            dropna=False,
        )
        .agg(
            rows=("seqn", "size"),
            participants=("seqn", "nunique"),
            generic_names=("rx_generic_name_normalized", "nunique"),
        )
        .reset_index()
        .sort_values(["cycle_id", "flag", "therapeutic_class", "match_method"])
        .reset_index(drop=True)
    )


def unmapped_medication_summary(matches: pd.DataFrame) -> pd.DataFrame:
    """Unique unresolved names for investigator review, not an error count.

    Most rows in this table are unrelated medicines; only candidate target-class
    ingredients should be promoted into the frozen dictionary.
    """
    columns = [
        "cycle_id",
        "rx_generic_drug_id",
        "rx_generic_name_normalized",
        "rows",
        "participants",
    ]
    if matches.empty:
        return pd.DataFrame(columns=columns)
    unmapped = matches[matches["flag"].eq("unmapped")].copy()
    if unmapped.empty:
        return pd.DataFrame(columns=columns)
    return (
        unmapped.groupby(
            ["cycle_id", "rx_generic_drug_id", "rx_generic_name_normalized"],
            dropna=False,
        )
        .agg(rows=("seqn", "size"), participants=("seqn", "nunique"))
        .reset_index()
        .sort_values(["cycle_id", "rows"], ascending=[True, False])
        .reset_index(drop=True)
    )


def write_medication_outputs(
    flags: pd.DataFrame,
    matches: pd.DataFrame,
    settings: Settings,
    cycle: str,
) -> tuple[Path, Path]:
    base = settings.processed_dir / cycle.replace("-", "_")
    flag_path = output_path(base, "medication_flags", settings.output_format)
    match_path = output_path(base, "medication_matches", settings.output_format)
    write_frame(flags, flag_path, settings.output_format)
    write_frame(matches, match_path, settings.output_format)
    return flag_path, match_path
