"""Check the byte count and SHA-256 of each file in the release manifest."""
import argparse
import csv
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = "RELEASE_MANIFEST_SHA256.csv"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict", action="store_true", help="Require a pristine checkout, excluding .git.")
    args = parser.parse_args()
    with (ROOT / MANIFEST).open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit("Release manifest is empty")
    seen = set()
    for row in rows:
        rel = row["path"]
        path = ROOT / rel
        if rel in seen or not path.resolve().is_relative_to(ROOT) or path.is_symlink():
            raise SystemExit(f"Invalid manifest path: {rel}")
        seen.add(rel)
        if not path.is_file():
            raise SystemExit(f"Missing release file: {rel}")
        with path.open("rb") as f:
            digest = hashlib.file_digest(f, "sha256").hexdigest()
        if path.stat().st_size != int(row["bytes"]) or digest != row["sha256"]:
            raise SystemExit(f"Release verification failed: {rel}")
    if args.strict:
        actual = {p.relative_to(ROOT).as_posix() for p in ROOT.rglob("*")
                  if (p.is_file() or p.is_symlink()) and ".git" not in p.relative_to(ROOT).parts}
        extra = actual - seen - {MANIFEST}
        if extra:
            raise SystemExit(f"Files absent from release manifest: {sorted(extra)}")
    print(f"PASS: {len(rows)} release SHA-256 entries")


if __name__ == "__main__":
    main()
