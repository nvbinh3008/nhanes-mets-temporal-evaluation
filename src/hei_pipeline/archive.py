from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import zipfile
import pandas as pd


def _find_7zip() -> str | None:
    for name in ("7z", "7zz", "7za"):
        found = shutil.which(name)
        if found:
            return found
    common = []  # Portable public candidate: discover 7-Zip through PATH.
    for path in common:
        if path.exists():
            return str(path)
    return None


def extract_archives(
    fped_root: Path,
    output_root: Path,
    allow_self_extracting_exe: bool = False,
) -> pd.DataFrame:
    """Safely extract ZIP files and optionally self-extracting EXE archives.

    EXE files are never executed. If 7-Zip is available and
    ``allow_self_extracting_exe`` is true, 7-Zip is used as an archive reader.
    """
    output_root.mkdir(parents=True, exist_ok=True)
    seven_zip = _find_7zip()
    rows = []

    for path in sorted(fped_root.rglob("*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix not in {".zip", ".exe"}:
            continue

        relative_parent = path.parent.relative_to(fped_root)
        target = output_root / relative_parent / path.stem
        target.mkdir(parents=True, exist_ok=True)

        status = "skipped"
        detail = ""
        extracted_files = 0

        try:
            if suffix == ".zip":
                with zipfile.ZipFile(path) as archive:
                    archive.extractall(target)
                    extracted_files = len([n for n in archive.namelist() if not n.endswith("/")])
                status = "extracted"
            elif allow_self_extracting_exe and seven_zip:
                result = subprocess.run(
                    [seven_zip, "x", "-y", f"-o{target}", str(path)],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                status = "extracted" if result.returncode == 0 else "failed"
                detail = (result.stderr or result.stdout)[-2000:]
                extracted_files = sum(1 for p in target.rglob("*") if p.is_file())
            elif suffix == ".exe":
                status = "skipped"
                detail = (
                    "Self-extracting archive was not executed. Install 7-Zip and "
                    "rerun with --allow-self-extracting-exe, or extract manually."
                )
        except Exception as exc:
            status = "failed"
            detail = repr(exc)

        rows.append({
            "archive": str(path),
            "archive_type": suffix,
            "target": str(target),
            "status": status,
            "extracted_files": extracted_files,
            "detail": detail,
        })

    return pd.DataFrame(rows)
