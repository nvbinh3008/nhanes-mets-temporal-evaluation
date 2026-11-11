from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import yaml

from .exceptions import ConfigError


REQUIRED_SHEETS = ("Cycle_File_Matrix", "Variable_Long")
OPTIONAL_SHEETS = (
    "Medication_Map",
    "Predictor_Blacklist",
    "QA_Checks",
    "Derived_Rules",
    "FPED_HEI",
)

CYCLE_FILE_COLUMNS = {
    "cycle",
    "suffix",
    "cycle_role",
    "domain",
    "file_stem",
    "xpt_file",
    "codebook_url",
    "data_url",
    "status",
    "notes",
}

VARIABLE_LONG_COLUMNS = {
    "cycle",
    "suffix",
    "cycle_role",
    "domain",
    "file_name",
    "original_variable",
    "normalized_variable",
    "unit_or_coding",
    "valid_condition",
    "domain_weight",
    "final_analysis_weight",
    "analysis_role",
    "blacklist_status",
    "codebook_url",
    "notes",
    "protocol_status",
}


@dataclass(frozen=True)
class Settings:
    root: Path
    dictionary: Path
    raw_dir: Path
    staged_dir: Path
    processed_dir: Path
    reports_dir: Path
    logs_dir: Path
    output_format: str
    default_roles: tuple[str, ...]
    default_statuses: tuple[str, ...]
    optional_statuses: tuple[str, ...]
    one_to_many_stems: tuple[str, ...]
    person_level_stems: tuple[str, ...]
    strict: bool
    retries: int
    timeout_seconds: int
    chunk_size_bytes: int
    user_agent: str
    drug_id_map: Path | None

    @classmethod
    def from_yaml(cls, path: Path, dictionary_override: Path | None = None) -> "Settings":
        path = path.resolve()
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        root = (path.parent / raw.get("root", ".")).resolve()
        paths = raw.get("paths", {})
        download = raw.get("download", {})
        selection = raw.get("selection", {})
        dictionary = dictionary_override or (root / raw.get("dictionary", "NHANES_cycle_harmonization_dictionary_v1_2.xlsx"))
        drug_id_map_value = raw.get("medication", {}).get("drug_id_map")
        drug_id_map = (root / drug_id_map_value).resolve() if drug_id_map_value else None
        return cls(
            root=root,
            dictionary=Path(dictionary).resolve(),
            raw_dir=(root / paths.get("raw", "data/raw")).resolve(),
            staged_dir=(root / paths.get("staged", "data/staged")).resolve(),
            processed_dir=(root / paths.get("processed", "data/processed")).resolve(),
            reports_dir=(root / paths.get("reports", "reports")).resolve(),
            logs_dir=(root / paths.get("logs", "logs")).resolve(),
            output_format=str(raw.get("output_format", "parquet")).lower(),
            default_roles=tuple(selection.get("roles", ["development", "temporal_test"])),
            default_statuses=tuple(selection.get("statuses", ["required_primary"])),
            optional_statuses=tuple(selection.get("optional_statuses", ["blacklist_audit_only", "sensitivity_only"])),
            one_to_many_stems=tuple(raw.get("structure", {}).get("one_to_many_stems", ["DR1IFF", "DR2IFF", "RXQ_RX"])),
            person_level_stems=tuple(raw.get("structure", {}).get(
                "person_level_stems",
                ["DEMO", "DR1TOT", "DR2TOT", "BMX", "BPX", "BPQ", "DIQ", "TRIGLY", "HDL", "GLU", "FASTQX", "PAQ", "SMQ", "GHB", "INS", "TCHOL"],
            )),
            strict=bool(raw.get("strict", True)),
            retries=int(download.get("retries", 4)),
            timeout_seconds=int(download.get("timeout_seconds", 90)),
            chunk_size_bytes=int(download.get("chunk_size_bytes", 1024 * 1024)),
            user_agent=str(download.get("user_agent", "NHANES-rule-HEI-ETL/0.1 (research use)")),
            drug_id_map=drug_id_map,
        )


@dataclass
class DictionaryConfig:
    cycle_files: pd.DataFrame
    variables_long: pd.DataFrame
    medication_map: pd.DataFrame
    predictor_blacklist: pd.DataFrame
    qa_checks: pd.DataFrame
    derived_rules: pd.DataFrame
    fped_hei: pd.DataFrame

    @classmethod
    def load(cls, path: Path) -> "DictionaryConfig":
        if not path.exists():
            raise ConfigError(f"Dictionary workbook not found: {path}")
        try:
            excel = pd.ExcelFile(path, engine="openpyxl")
            missing_required = [name for name in REQUIRED_SHEETS if name not in excel.sheet_names]
            if missing_required:
                raise ConfigError(
                    "Workbook is missing required sheets: " + ", ".join(missing_required)
                )
            sheet_names = [
                name for name in REQUIRED_SHEETS + OPTIONAL_SHEETS
                if name in excel.sheet_names
            ]
            loaded = pd.read_excel(excel, sheet_name=sheet_names, dtype=object)
        except ConfigError:
            raise
        except Exception as exc:
            raise ConfigError(f"Could not read dictionary workbook {path}: {exc}") from exc

        def tidy(name: str) -> pd.DataFrame:
            frame = loaded.get(name, pd.DataFrame()).copy()
            frame.columns = [str(col).strip() for col in frame.columns]
            frame = frame.dropna(how="all")
            for col in frame.columns:
                if frame[col].dtype == object:
                    frame[col] = frame[col].map(lambda x: x.strip() if isinstance(x, str) else x)
            return frame.fillna("")

        config = cls(
            cycle_files=tidy("Cycle_File_Matrix"),
            variables_long=tidy("Variable_Long"),
            medication_map=tidy("Medication_Map"),
            predictor_blacklist=tidy("Predictor_Blacklist"),
            qa_checks=tidy("QA_Checks"),
            derived_rules=tidy("Derived_Rules"),
            fped_hei=tidy("FPED_HEI"),
        )
        config._normalize()
        return config

    def _normalize(self) -> None:
        for frame in (self.cycle_files, self.variables_long):
            for col in ("cycle", "suffix", "cycle_role"):
                if col in frame:
                    frame[col] = frame[col].astype(str).str.strip()
        if "xpt_file" in self.cycle_files:
            self.cycle_files["xpt_file"] = self.cycle_files["xpt_file"].astype(str).str.strip().str.upper()
        if "file_stem" in self.cycle_files:
            self.cycle_files["file_stem"] = self.cycle_files["file_stem"].astype(str).str.strip().str.upper()
        if "file_name" in self.variables_long:
            self.variables_long["file_name"] = self.variables_long["file_name"].astype(str).str.strip().str.upper()
        if "original_variable" in self.variables_long:
            self.variables_long["original_variable"] = self.variables_long["original_variable"].astype(str).str.strip().str.upper()
        if "normalized_variable" in self.variables_long:
            self.variables_long["normalized_variable"] = self.variables_long["normalized_variable"].astype(str).str.strip()

    def audit(self) -> pd.DataFrame:
        rows: list[dict[str, object]] = []

        def add(check: str, severity: str, passed: bool, detail: str) -> None:
            rows.append({"check": check, "severity": severity, "passed": bool(passed), "detail": detail})

        missing_cf = sorted(CYCLE_FILE_COLUMNS - set(self.cycle_files.columns))
        missing_vl = sorted(VARIABLE_LONG_COLUMNS - set(self.variables_long.columns))
        add("Cycle_File_Matrix columns", "hard", not missing_cf, "missing: " + ", ".join(missing_cf) if missing_cf else "all required columns present")
        add("Variable_Long columns", "hard", not missing_vl, "missing: " + ", ".join(missing_vl) if missing_vl else "all required columns present")

        if missing_cf or missing_vl:
            return pd.DataFrame(rows)

        duplicate_files = self.cycle_files.duplicated(["cycle", "xpt_file"], keep=False)
        add(
            "Unique cycle-file rows",
            "hard",
            not duplicate_files.any(),
            f"{int(duplicate_files.sum())} duplicate rows",
        )

        active = self.variables_long[
            self.variables_long["file_name"].ne("") & self.variables_long["original_variable"].ne("")
        ].copy()
        duplicate_mappings = active.duplicated(
            ["cycle", "file_name", "normalized_variable"], keep=False
        )
        add(
            "Unique active cycle-file-canonical mappings",
            "hard",
            not duplicate_mappings.any(),
            f"{int(duplicate_mappings.sum())} duplicate rows",
        )

        file_keys = set(zip(self.cycle_files["cycle"], self.cycle_files["xpt_file"]))
        unresolved = active[
            ~active.apply(lambda r: (r["cycle"], r["file_name"]) in file_keys, axis=1)
        ]
        add(
            "Variable-to-file referential integrity",
            "hard",
            unresolved.empty,
            f"{len(unresolved)} active mappings reference no Cycle_File_Matrix row",
        )

        roles_by_cycle = self.cycle_files.groupby("cycle")["cycle_role"].nunique()
        add(
            "One role per cycle",
            "hard",
            bool((roles_by_cycle == 1).all()),
            f"cycles with >1 role: {roles_by_cycle[roles_by_cycle > 1].index.tolist()}",
        )

        suffix_by_cycle = self.cycle_files.groupby("cycle")["suffix"].nunique()
        add(
            "One suffix per cycle",
            "hard",
            bool((suffix_by_cycle == 1).all()),
            f"cycles with >1 suffix: {suffix_by_cycle[suffix_by_cycle > 1].index.tolist()}",
        )

        inactive = self.variables_long[
            self.variables_long["file_name"].eq("") | self.variables_long["original_variable"].eq("")
        ]
        inactive_ok = inactive["valid_condition"].astype(str).str.contains(
            "not available|not harmonized", case=False, regex=True
        )
        add(
            "Blank mappings explicitly unavailable",
            "hard",
            bool(inactive_ok.all()),
            f"{int((~inactive_ok).sum())} blank mappings lack an unavailable/not-harmonized declaration",
        )

        url_ok = self.cycle_files.loc[
            ~self.cycle_files["status"].eq("not_harmonized"), "data_url"
        ].astype(str).str.startswith("https://")
        add(
            "HTTPS data URLs",
            "hard",
            bool(url_ok.all()),
            f"{int((~url_ok).sum())} downloadable rows lack an HTTPS URL",
        )

        return pd.DataFrame(rows)

    def assert_valid(self) -> pd.DataFrame:
        report = self.audit()
        failures = report[(report["severity"] == "hard") & (~report["passed"])]
        if not failures.empty:
            detail = "; ".join(f"{r.check}: {r.detail}" for r in failures.itertuples())
            raise ConfigError(f"Dictionary configuration failed hard checks: {detail}")
        return report

    def file_plan(
        self,
        roles: Iterable[str],
        statuses: Iterable[str],
        cycles: Iterable[str] | None = None,
    ) -> pd.DataFrame:
        frame = self.cycle_files.copy()
        roles_set = {str(x) for x in roles}
        statuses_set = {str(x) for x in statuses}
        frame = frame[frame["cycle_role"].isin(roles_set) & frame["status"].isin(statuses_set)]
        if cycles:
            cycle_set = {str(x) for x in cycles}
            frame = frame[frame["cycle"].isin(cycle_set)]
        return frame.sort_values(["cycle", "file_stem"]).reset_index(drop=True)

    def expected_variables(self, cycle: str, file_name: str) -> pd.DataFrame:
        frame = self.variables_long
        return frame[
            frame["cycle"].eq(cycle)
            & frame["file_name"].eq(file_name.upper())
            & frame["original_variable"].ne("")
        ].copy()

    def cycle_role(self, cycle: str) -> str:
        values = self.cycle_files.loc[self.cycle_files["cycle"].eq(cycle), "cycle_role"].unique()
        if len(values) != 1:
            raise ConfigError(f"Could not resolve a unique role for cycle {cycle}")
        return str(values[0])

    def cycles_for_roles(self, roles: Iterable[str]) -> list[str]:
        roles_set = set(roles)
        frame = self.cycle_files[self.cycle_files["cycle_role"].isin(roles_set)]
        return sorted(frame["cycle"].unique().tolist())
