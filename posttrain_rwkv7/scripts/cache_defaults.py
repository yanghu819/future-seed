#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path
from typing import MutableMapping


def future_seed_cache_root() -> Path:
    override = os.environ.get("FUTURE_SEED_CACHE_ROOT")
    if override:
        return Path(override).expanduser()
    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache:
        return Path(xdg_cache).expanduser() / "future-seed"
    return Path.home() / ".cache" / "future-seed"


def apply_cache_env(env: MutableMapping[str, str] | None = None) -> MutableMapping[str, str]:
    target = os.environ if env is None else env
    root = future_seed_cache_root()
    defaults = {
        "TORCH_EXTENSIONS_DIR": root / "torch_extensions",
        "HF_HOME": root / "hf",
        "HF_DATASETS_CACHE": root / "hf_datasets",
        "TRANSFORMERS_CACHE": root / "hf_transformers",
    }
    for key, path in defaults.items():
        target.setdefault(key, str(path))
    return target


def ensure_cache_dirs(env: MutableMapping[str, str] | None = None) -> None:
    target = os.environ if env is None else env
    for key in ("TORCH_EXTENSIONS_DIR", "HF_HOME", "HF_DATASETS_CACHE", "TRANSFORMERS_CACHE"):
        value = target.get(key)
        if value:
            Path(value).expanduser().mkdir(parents=True, exist_ok=True)
