"""Format audited aggregate inputs as the displayed Paper 3 Tables 3 and 5."""
import argparse
import csv
from decimal import Decimal
import io
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results" / "aggregate_tables"
VIEWS = {"day1_primary": "Day 1", "two_day_average": "Two-day"}
PERIODS = {"2015-2016": "2015--2016", "2017-2018": "2017--2018", "temporal": "Pooled"}
PRIMARY = ("prevalence", "mean_prediction", "log_loss", "brier", "roc_auc", "calibration_intercept", "calibration_slope")
DRIFT = {
    "prevalence": "Prevalence", "log_loss": "Log loss", "null_log_loss": "Null log loss",
    "log_loss_skill": "Log-loss skill", "brier": "Brier score", "null_brier": "Null Brier",
    "brier_skill": "Brier skill", "roc_auc": "AUROC",
}


def read(name):
    with (DATA / name).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def decimal3(value, signed=False):
    number = Decimal(value)
    if not number.is_finite():
        raise ValueError(f"Non-finite published metric: {value}")
    return format(number, "+.3f" if signed else ".3f")


def unique_index(rows, keys):
    result = {}
    for row in rows:
        key = tuple(row[k] for k in keys)
        if key in result:
            raise ValueError(f"Duplicate audited aggregate row: {key}")
        result[key] = row
    return result


def table3():
    audit = [r for r in read("CORRECTED_ALL_METRICS_AND_CI.csv")
             if r["representation"] == "R5" and r["algorithm"] == "boost_monotonic"
             and r["group_variable"] == r["group"] == "all" and r["suppressed"] == "False"
             and r["scope"] in PERIODS and r["metric"] in PRIMARY]
    index = unique_index(audit, ("view", "scope", "metric"))
    output = []
    for view, display in VIEWS.items():
        for scope, period in PERIODS.items():
            source = [index[view, scope, metric] for metric in PRIMARY]
            sizes = {r["n"] for r in source}
            if len(sizes) != 1:
                raise ValueError(f"Inconsistent sample sizes in {(view, scope)}")
            values = [decimal3(r["estimate"]) for r in source]
            output.append([display, period, f"{int(sizes.pop()):,}", *values[:5], "/".join(values[5:])])
    return ["View", "Period", "n", "MetS prev.", "Mean pred.", "Log loss", "Brier", "AUROC", "Intercept/slope"], output


def table5():
    index = unique_index(read("CORRECTED_DRIFT_AND_SKILL_INTERVALS.csv"), ("view", "metric"))
    output = []
    for view, display in VIEWS.items():
        for metric, label in DRIFT.items():
            if view == "two_day_average" and metric in {"prevalence", "null_log_loss", "null_brier"}:
                continue
            row = index[view, metric]
            delta, low, high = (decimal3(row[k], True) for k in ("difference", "ci_lower", "ci_upper"))
            output.append([display, label, decimal3(row["development"]), decimal3(row["temporal"]), f"{delta} ({low} to {high})"])
    return ["View", "Metric", "Development", "Temporal", "Difference (95% interval)"], output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check", action="store_true", help="Compare against the audited display fixtures.")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.is_relative_to(DATA.resolve()) or output.is_relative_to((ROOT / "tests").resolve()):
        parser.error("Choose an output directory outside the audited inputs and reference fixtures.")
    for name, renderer in [("table3.csv", table3), ("table5.csv", table5)]:
        header, rows = renderer()
        buf = io.StringIO(newline="")
        writer = csv.writer(buf, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)
        content = buf.getvalue()
        if args.check and content != (ROOT / "tests" / "expected_tables" / name).read_text(encoding="utf-8"):
            raise SystemExit(f"{name} differs from the audited manuscript display")
        output.mkdir(parents=True, exist_ok=True)
        (output / name).write_text(content, encoding="utf-8", newline="\n")
        print(f"{'PASS' if args.check else 'WROTE'} {name}: {len(rows)} rows")


if __name__ == "__main__":
    main()
