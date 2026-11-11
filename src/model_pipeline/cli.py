from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config, resolve_path, smoke_config
from .data import preflight_development
from .utils import json_dump, utc_now_iso


def _representations(args, config: dict) -> list[str]:
    values = args.representations or config["data"]["representations"]
    return [value.upper() for value in values]



def _view(args, config: dict) -> str:
    if args.view in config["data"]["views"]:
        return config["data"]["views"][args.view]
    return args.view


def run_preflight(config: dict, view: str, representations: list[str]) -> dict:
    report_root = resolve_path(config, "report_root") / "preflight" / view
    report_root.mkdir(parents=True, exist_ok=True)
    audit = preflight_development(config, view, representations)
    audit.to_csv(report_root / "development_representation_audit.csv", index=False)
    hard_failures = int(audit["hard_fail"].sum())
    summary = {
        "generated": utc_now_iso(),
        "view": view,
        "representations": representations,
        "rows": int(len(audit)),
        "hard_failures": hard_failures,
        "temporal_rows_read": False,
        "audit_path": str(report_root / "development_representation_audit.csv"),
    }
    json_dump(summary, report_root / "preflight_summary.json")
    if hard_failures:
        raise RuntimeError(f"Preflight found {hard_failures} hard failures")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nhanes-model",
        description=(
            "Inspect modeling commands. RC2 disables historical inference and fitting. "
            "See BOOTSTRAP_PROVENANCE.md for the corrected Phase B outputs."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--config",
        default="config/modeling.yaml",
        help="Path to modeling YAML configuration.",
    )
    common.add_argument(
        "--view",
        default="primary",
        help="primary, sensitivity, day1_primary, or two_day_average",
    )
    common.add_argument(
        "--representations",
        nargs="+",
        choices=["R0", "R1", "R2", "R3", "R4", "R5"],
    )
    common.add_argument(
        "--algorithms",
        nargs="+",
        choices=["logistic", "boost_unconstrained", "boost_monotonic"],
    )
    common.add_argument(
        "--smoke",
        action="store_true",
        help="Historical option. This does not enable model fitting or bootstrap in RC2.",
    )

    subparsers.add_parser("preflight", parents=[common])

    nested = subparsers.add_parser("nested-validate", parents=[common], help="Disabled in RC2 (historical intervals).")
    nested.add_argument(
        "--no-bootstrap",
        action="store_true",
        help="Set bootstrap replicates to zero for debugging only.",
    )

    subparsers.add_parser("fit-final", parents=[common], help="Disabled in RC2 (frozen model is archived).")
    subparsers.add_parser("run-development", parents=[common], help="Disabled in RC2 (historical workflow).")

    temporal = subparsers.add_parser("temporal-evaluate", help="Disabled in RC2 (historical intervals).")
    temporal.add_argument(
        "--config",
        default="config/modeling.yaml",
        help="Path to modeling YAML configuration.",
    )
    temporal.add_argument(
        "--view",
        default="primary",
        help="primary, sensitivity, day1_primary, or two_day_average",
    )
    temporal.add_argument(
        "--resume-interrupted",
        action="store_true",
        help="Resume only after confirming a prior temporal run did not complete.",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command != "preflight":
        parser.error(
            f"{args.command} is disabled in RC2. The original modeling workflow used "
            "a HISTORICAL / SUPERSEDED 500-replicate unrescaled PSU bootstrap. "
            "For the corrected 2,000-replicate rescaled Phase B intervals and "
            "restricted input requirements, see BOOTSTRAP_PROVENANCE.md."
        )
    config = load_config(args.config)
    if args.smoke:
        config = smoke_config(config)
    view = _view(args, config)
    result = run_preflight(config, view, _representations(args, config))
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
