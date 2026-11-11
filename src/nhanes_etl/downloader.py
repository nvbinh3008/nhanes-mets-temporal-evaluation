from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .config import Settings
from .exceptions import DownloadError
from .utils import cycle_slug, ensure_dir, sha256_file, utc_now_iso


def build_session(settings: Settings) -> requests.Session:
    retry = Retry(
        total=settings.retries,
        connect=settings.retries,
        read=settings.retries,
        status=settings.retries,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET", "HEAD"}),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=8, pool_maxsize=8)
    session = requests.Session()
    session.mount("https://", adapter)
    session.headers.update({"User-Agent": settings.user_agent, "Accept": "*/*"})
    return session


def _download_one(row: dict[str, object], settings: Settings, overwrite: bool) -> dict[str, object]:
    cycle = str(row["cycle"])
    xpt_file = str(row["xpt_file"]).upper()
    url = str(row["data_url"])
    target = ensure_dir(settings.raw_dir / cycle_slug(cycle)) / xpt_file
    started = utc_now_iso()
    result: dict[str, object] = {
        "cycle": cycle,
        "cycle_role": row.get("cycle_role", ""),
        "file_stem": row.get("file_stem", ""),
        "xpt_file": xpt_file,
        "url": url,
        "target": str(target),
        "started_utc": started,
        "finished_utc": "",
        "status": "",
        "http_status": "",
        "bytes": 0,
        "sha256": "",
        "error": "",
    }

    if target.exists() and not overwrite:
        result.update(
            {
                "status": "cached",
                "bytes": target.stat().st_size,
                "sha256": sha256_file(target),
                "finished_utc": utc_now_iso(),
            }
        )
        return result

    part = target.with_suffix(target.suffix + ".part")
    if part.exists():
        part.unlink()

    session = build_session(settings)
    try:
        with session.get(url, stream=True, timeout=settings.timeout_seconds) as response:
            result["http_status"] = response.status_code
            if response.status_code != 200:
                raise DownloadError(f"HTTP {response.status_code} for {url}")
            content_type = response.headers.get("Content-Type", "")
            if "text/html" in content_type.lower():
                raise DownloadError(f"Expected XPT but received HTML from {url}")
            ensure_dir(part.parent)
            size = 0
            with part.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=settings.chunk_size_bytes):
                    if chunk:
                        handle.write(chunk)
                        size += len(chunk)
                handle.flush()
                os.fsync(handle.fileno())
            if size < 512:
                raise DownloadError(f"Downloaded file is unexpectedly small ({size} bytes): {url}")
            part.replace(target)
            result.update(
                {
                    "status": "downloaded",
                    "bytes": size,
                    "sha256": sha256_file(target),
                    "finished_utc": utc_now_iso(),
                }
            )
    except Exception as exc:
        if part.exists():
            part.unlink()
        result.update({"status": "failed", "error": str(exc), "finished_utc": utc_now_iso()})
    finally:
        session.close()
    return result


def download_plan(
    plan: pd.DataFrame,
    settings: Settings,
    overwrite: bool = False,
    workers: int = 4,
) -> pd.DataFrame:
    if plan.empty:
        return pd.DataFrame(
            columns=[
                "cycle", "cycle_role", "file_stem", "xpt_file", "url", "target",
                "started_utc", "finished_utc", "status", "http_status", "bytes",
                "sha256", "error",
            ]
        )

    records = plan.to_dict(orient="records")
    results: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {
            pool.submit(_download_one, row, settings, overwrite): row for row in records
        }
        for future in as_completed(futures):
            results.append(future.result())

    manifest = pd.DataFrame(results).sort_values(["cycle", "xpt_file"]).reset_index(drop=True)
    return manifest


def download_has_failures(manifest: pd.DataFrame) -> bool:
    return bool(not manifest.empty and manifest["status"].eq("failed").any())
