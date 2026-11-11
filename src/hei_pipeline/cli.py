from __future__ import annotations

import argparse
import json
from pathlib import Path

from .archive import extract_archives
from .config import load_config, resolve
from .discovery import discover_fped
from .pipeline import run


def _parse_cycles(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hei-pipeline",
        description="Cycle-aware FPED linkage and HEI-2020 scoring.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    for name in ("discover", "extract", "run"):
        command = sub.add_parser(name)
        command.add_argument(
            "--config",
            default="config/hei_pipeline.yaml",
        )
        command.add_argument(
            "--cycles",
            default=None,
            help="Comma-separated cycles, e.g. 2007-2008,2009-2010",
        )

    extract = sub.choices["extract"]
    extract.add_argument(
        "--allow-self-extracting-exe",
        action="store_true",
        help="Use 7-Zip as an archive reader; EXE files are never executed.",
    )

    run_parser = sub.choices["run"]
    run_parser.add_argument("--extract-first", action="store_true")
    run_parser.add_argument(
        "--allow-self-extracting-exe",
        action="store_true",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    root = Path(cfg["project_root"]).resolve()
    fped_root = resolve(root, cfg["fped_root"])
    report_root = resolve(root, cfg["report_root"])
    report_root.mkdir(parents=True, exist_ok=True)

    cycles = cfg["cycles"]
    requested = _parse_cycles(args.cycles)
    if requested:
        cycles = {cycle: cycles[cycle] for cycle in requested}

    if args.command == "discover":
        report = discover_fped(fped_root, cycles)
        report.to_csv(
            report_root / "fped_archive_discovery.csv",
            index=False,
        )
        print(report.to_string(index=False))
        return 0

    if args.command == "extract":
        extracted_root = resolve(root, cfg["fped_extracted_root"])
        report = extract_archives(
            fped_root,
            extracted_root,
            allow_self_extracting_exe=args.allow_self_extracting_exe,
        )
        report.to_csv(
            report_root / "fped_extraction_report.csv",
            index=False,
        )
        print(report.to_string(index=False))
        return 0

    result = run(
        args.config,
        cycles_requested=requested,
        extract_first=args.extract_first,
        allow_self_extracting_exe=args.allow_self_extracting_exe,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
