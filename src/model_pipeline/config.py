from __future__ import annotations

from pathlib import Path
import copy
import yaml


def load_config(path: str | Path) -> dict:
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["_config_path"] = str(path)
    config["_project_root"] = str(path.parent.parent.resolve())
    return config


def resolve_path(config: dict, key: str) -> Path:
    root = Path(config["_project_root"])
    value = Path(config["paths"][key])
    return value if value.is_absolute() else root / value


def smoke_config(config: dict) -> dict:
    cfg = copy.deepcopy(config)
    cfg["cross_validation"]["outer_splits"] = 2
    cfg["cross_validation"]["inner_splits"] = 2
    cfg["cross_validation"]["calibration_splits"] = 2
    cfg["cross_validation"]["calibration_methods"] = ["none", "sigmoid"]
    cfg["models"]["logistic"]["candidates"] = [{"C": 1.0}]
    cfg["models"]["boost"]["candidates"] = [
        {
            "learning_rate": 0.08,
            "max_iter": 40,
            "max_leaf_nodes": 7,
            "min_samples_leaf": 20,
            "l2_regularization": 1.0,
        }
    ]
    cfg["uncertainty"]["replicates"] = 20
    cfg["_smoke_mode"] = True
    return cfg
