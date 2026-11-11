from __future__ import annotations

import argparse
import json
from pathlib import Path

from nhanes_etl.cli import main as etl_main
from hei_pipeline.pipeline import run as run_hei


def _parse_cycles(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def preflight(root: Path) -> dict:
    checks = {
        "project_root": str(root.resolve()),
        "dictionary_exists": (
            root / "NHANES_cycle_harmonization_dictionary_v1_2_4.xlsx"
        ).exists(),
        "etl_config_exists": (root / "config/pipeline.yaml").exists(),
        "hei_config_exists": (root / "config/hei_pipeline.yaml").exists(),
        "raw_directory_exists": (root / "data/raw").exists(),
        "fped_directory_exists": (root / "data/fped").exists(),
        "staged_directory_exists": (root / "data/staged").exists(),
        "processed_directory_exists": (root / "data/processed").exists(),
    }
    checks["fped_cycle_directories"] = {
        cycle: (root / "data/fped" / cycle).exists()
        for cycle in (
            "2007_2008",
            "2009_2010",
            "2011_2012",
            "2013_2014",
            "2015_2016",
            "2017_2018",
        )
    }
    checks["ready_for_etl"] = bool(
        checks["dictionary_exists"] and checks["etl_config_exists"]
    )
    checks["ready_for_hei"] = bool(
        checks["hei_config_exists"]
        and checks["fped_directory_exists"]
        and checks["staged_directory_exists"]
        and checks["processed_directory_exists"]
        and all(checks["fped_cycle_directories"].values())
    )
    return checks


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nhanes-full",
        description=(
            "Run the NHANES ETL and FPED/HEI-2020 stages in a single workflow."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("preflight")

    etl = sub.add_parser("run-etl")
    etl.add_argument("--cycles", default=None)
    etl.add_argument("--download", action="store_true")
    etl.add_argument("--workers", type=int, default=4)

    hei = sub.add_parser("run-hei")
    hei.add_argument("--cycles", default=None)
    hei.add_argument("--extract-first", action="store_true")
    hei.add_argument("--allow-self-extracting-exe", action="store_true")

    full = sub.add_parser("run-all")
    full.add_argument("--cycles", default=None)
    full.add_argument("--download", action="store_true")
    full.add_argument("--workers", type=int, default=4)
    full.add_argument("--extract-first", action="store_true")
    full.add_argument("--allow-self-extracting-exe", action="store_true")
    return parser


def _run_etl(cycles: str | None, download: bool, workers: int) -> None:
    common = []
    if cycles:
        common += ["--cycles", cycles]

    etl_main(["config-audit", *common])
    etl_main(["plan", *common])

    if download:
        etl_main([
            "download",
            *common,
            "--workers",
            str(workers),
        ])

    etl_main(["schema-audit", *common])
    etl_main([
        "run",
        *common,
        "--skip-download",
        "--skip-schema-audit",
    ])


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path.cwd()

    if args.command == "preflight":
        result = preflight(root)
        print(json.dumps(result, indent=2))
        return 0

    if args.command == "run-etl":
        _run_etl(args.cycles, args.download, args.workers)
        return 0

    if args.command == "run-hei":
        result = run_hei(
            "config/hei_pipeline.yaml",
            cycles_requested=_parse_cycles(args.cycles),
            extract_first=args.extract_first,
            allow_self_extracting_exe=args.allow_self_extracting_exe,
        )
        print(json.dumps(result, indent=2))
        return 0

    _run_etl(args.cycles, args.download, args.workers)
    result = run_hei(
        "config/hei_pipeline.yaml",
        cycles_requested=_parse_cycles(args.cycles),
        extract_first=args.extract_first,
        allow_self_extracting_exe=args.allow_self_extracting_exe,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
