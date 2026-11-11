from __future__ import annotations

import argparse
import sys
from pathlib import Path

import requests

from .audit import audit_plan, schema_has_hard_fail
from .config import DictionaryConfig, Settings
from .downloader import download_plan, download_has_failures
from .io import write_report
from .pipeline import run_pipeline
from .transform import stage_plan
from .utils import ensure_dir, parse_csv_list


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--settings", type=Path, default=Path("config/pipeline.yaml"))
    parser.add_argument("--dictionary", type=Path, default=None)
    parser.add_argument("--roles", default=None, help="Comma-separated cycle roles")
    parser.add_argument("--statuses", default=None, help="Comma-separated Cycle_File_Matrix statuses")
    parser.add_argument("--cycles", default=None, help="Comma-separated cycles, e.g. 2007-2008,2009-2010")
    parser.add_argument("--include-optional", action="store_true")
    parser.add_argument("--verbose", action="store_true")


def _resolve(args: argparse.Namespace) -> tuple[Settings, DictionaryConfig, list[str], list[str], list[str] | None]:
    settings = Settings.from_yaml(args.settings, args.dictionary)
    dictionary = DictionaryConfig.load(settings.dictionary)
    roles = parse_csv_list(args.roles) or list(settings.default_roles)
    statuses = parse_csv_list(args.statuses) or list(settings.default_statuses)
    if args.include_optional:
        statuses = list(dict.fromkeys(statuses + list(settings.optional_statuses)))
    cycles = parse_csv_list(args.cycles) or None
    return settings, dictionary, roles, statuses, cycles


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nhanes-etl",
        description="Configuration-driven NHANES download, schema audit, and cycle-wise ETL.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("config-audit", help="Validate Variable_Long and Cycle_File_Matrix")
    _common(p)

    p = sub.add_parser("plan", help="Print and save the selected file plan")
    _common(p)

    p = sub.add_parser("download", help="Download selected NHANES XPT files")
    _common(p)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--workers", type=int, default=4)

    p = sub.add_parser("schema-audit", help="Audit downloaded XPT files")
    _common(p)

    p = sub.add_parser("stage", help="Canonicalize selected XPT files cycle by cycle")
    _common(p)

    p = sub.add_parser("doctor", help="Test the first selected NHANES URL and report network/TLS/proxy details")
    _common(p)

    p = sub.add_parser("run", help="Run config audit, optional download, schema audit, and ETL")
    _common(p)
    p.add_argument("--skip-download", action="store_true")
    p.add_argument("--skip-schema-audit", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--workers", type=int, default=4)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings, dictionary, roles, statuses, cycles = _resolve(args)
    ensure_dir(settings.reports_dir)
    plan = dictionary.file_plan(roles, statuses, cycles)

    if args.command == "config-audit":
        report = dictionary.audit()
        write_report(report, settings.reports_dir / "config_audit.csv")
        print(report.to_string(index=False))
        dictionary.assert_valid()
        return 0

    if args.command == "plan":
        write_report(plan, settings.reports_dir / "selected_file_plan.csv")
        print(plan.to_string(index=False))
        return 0

    if args.command == "download":
        dictionary.assert_valid()
        manifest = download_plan(plan, settings, args.overwrite, args.workers)
        write_report(manifest, settings.reports_dir / "download_manifest.csv")
        display_cols = ["cycle", "xpt_file", "status", "http_status", "bytes", "error"]
        print(manifest[display_cols].to_string(index=False))
        if settings.strict and download_has_failures(manifest):
            failed = manifest[manifest["status"].eq("failed")]
            examples = " | ".join(
                f"{row.xpt_file}: {row.error}" for row in failed.head(3).itertuples()
            )
            raise RuntimeError(
                f"{len(failed)} download(s) failed. First errors: {examples}. "
                "Inspect reports/download_manifest.csv"
            )
        return 0

    if args.command == "doctor":
        dictionary.assert_valid()
        if plan.empty:
            raise RuntimeError("Selected file plan is empty")
        row = plan.iloc[0]
        url = str(row["data_url"])
        print(f"Testing: {url}")
        try:
            response = requests.get(
                url,
                stream=True,
                timeout=settings.timeout_seconds,
                headers={"User-Agent": settings.user_agent, "Accept": "*/*"},
            )
            print(f"HTTP status: {response.status_code}")
            print(f"Final URL: {response.url}")
            print(f"Content-Type: {response.headers.get('Content-Type', '')}")
            print(f"Content-Length: {response.headers.get('Content-Length', '')}")
            response.raise_for_status()
            first = next(response.iter_content(chunk_size=64), b"")
            if not first:
                raise RuntimeError("Server returned an empty response body")
            if b"<html" in first.lower():
                raise RuntimeError("Server returned HTML instead of an XPT binary")
            print("Network preflight passed: binary content was received.")
            return 0
        except requests.exceptions.SSLError as exc:
            raise RuntimeError(
                "TLS certificate validation failed. Update certifi or configure the "
                "organization's trusted CA via REQUESTS_CA_BUNDLE; do not disable verification. "
                f"Original error: {exc}"
            ) from exc
        except requests.exceptions.ProxyError as exc:
            raise RuntimeError(
                "Proxy connection failed. Check HTTPS_PROXY/HTTP_PROXY and institutional firewall settings. "
                f"Original error: {exc}"
            ) from exc
        except requests.exceptions.ConnectionError as exc:
            raise RuntimeError(
                "Could not connect to the CDC host. Check DNS, firewall, VPN, and Internet access. "
                f"Original error: {exc}"
            ) from exc

    if args.command == "schema-audit":
        dictionary.assert_valid()
        files, variables = audit_plan(plan, dictionary, settings)
        write_report(files, settings.reports_dir / "schema_file_audit.csv")
        write_report(variables, settings.reports_dir / "schema_variable_audit.csv")
        print(files.to_string(index=False))
        if settings.strict and schema_has_hard_fail(files, variables):
            failed_files = files.loc[files["hard_fail"].astype(bool)].copy()
            failed_vars = variables.loc[variables["hard_fail"].astype(bool)].copy()
            if not failed_files.empty:
                print("\nHard file failures (first 20):")
                cols = [c for c in ["cycle", "file", "detail"] if c in failed_files.columns]
                print(failed_files[cols].head(20).to_string(index=False))
            if not failed_vars.empty:
                print("\nHard variable failures (first 20):")
                cols = [c for c in [
                    "cycle", "file", "original_variable", "normalized_variable",
                    "present", "type_valid", "invalid_count"
                ] if c in failed_vars.columns]
                print(failed_vars[cols].head(20).to_string(index=False))
            raise RuntimeError(
                "Schema hard failures found. Review reports/schema_file_audit.csv "
                "and reports/schema_variable_audit.csv. Do not run with "
                "--skip-schema-audit until these failures are resolved."
            )
        return 0

    if args.command == "stage":
        dictionary.assert_valid()
        manifest = stage_plan(plan, dictionary, settings)
        write_report(manifest, settings.reports_dir / "staging_manifest.csv")
        print(manifest.to_string(index=False))
        return 0

    if args.command == "run":
        results = run_pipeline(
            dictionary_path=settings.dictionary,
            settings_path=args.settings,
            roles=roles,
            statuses=statuses,
            cycles=cycles,
            do_download=not args.skip_download,
            do_audit=not args.skip_schema_audit,
            overwrite=args.overwrite,
            workers=args.workers,
            verbose=args.verbose,
        )
        print(results.to_string(index=False))
        return 0

    return 2


if __name__ == "__main__":
    sys.exit(main())
