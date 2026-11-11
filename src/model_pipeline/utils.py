from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tree(root: Path, suffixes: tuple[str, ...] = (".py", ".yaml", ".toml")) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()



def config_fingerprint(config: dict) -> str:
    public = {key: value for key, value in config.items() if not key.startswith("_")}
    payload = json.dumps(public, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def json_dump(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def json_load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_weights(weights):
    weights = weights.astype(float)
    positive = weights > 0
    if not positive.all():
        raise ValueError("All analysis weights must be positive.")
    return weights / weights.mean()
