from __future__ import annotations
import os
from pathlib import Path
from typing import Any
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CFG_PATH = PROJECT_ROOT / "configs" / "default.yaml"


class Config(dict):
    """dict-with-attribute-access wrapper."""

    def __getattr__(self, name: str) -> Any:
        try:
            v = self[name]
        except KeyError as e:
            raise AttributeError(name) from e
        if isinstance(v, dict) and not isinstance(v, Config):
            v = Config(v)
            self[name] = v
        return v

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


def load_config(path: str | os.PathLike | None = None) -> Config:
    p = Path(path) if path else DEFAULT_CFG_PATH
    with open(p, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    cfg = Config(raw)
    cfg["_project_root"] = str(PROJECT_ROOT)
    return cfg


def resolve_path(cfg: Config, p: str | os.PathLike) -> Path:
    pth = Path(p)
    if pth.is_absolute():
        return pth
    return (PROJECT_ROOT / pth).resolve()


def ensure_dir(p: str | os.PathLike) -> Path:
    pth = Path(p)
    pth.mkdir(parents=True, exist_ok=True)
    return pth
