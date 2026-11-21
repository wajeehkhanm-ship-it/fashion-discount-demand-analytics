"""Load `config/settings.yaml` and resolve project paths."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

SETTINGS_RELPATH = Path("config") / "settings.yaml"


def find_project_root(start: Path | None = None) -> Path:
    """Return the repo root: `$FDDA_ROOT` if set, else the first parent holding the settings."""
    env_root = os.getenv("FDDA_ROOT")
    if env_root:
        return Path(env_root).resolve()
    here = (start or Path.cwd()).resolve()
    for candidate in (here, *here.parents):
        if (candidate / SETTINGS_RELPATH).exists():
            return candidate
    # Fall back to the package location (src/fdda/config.py -> repo root).
    pkg_root = Path(__file__).resolve().parents[2]
    if (pkg_root / SETTINGS_RELPATH).exists():
        return pkg_root
    raise FileNotFoundError(f"Could not find {SETTINGS_RELPATH} above {here}")


def load_settings(path: Path | None = None) -> dict[str, Any]:
    """Read the YAML settings and attach the repo root under the `_root` key."""
    root = find_project_root() if path is None else Path(path).resolve().parents[1]
    settings_path = path or root / SETTINGS_RELPATH
    with open(settings_path, encoding="utf-8") as fh:
        settings: dict[str, Any] = yaml.safe_load(fh)
    settings["_root"] = root
    return settings


def resolve_path(settings: dict[str, Any], key: str) -> Path:
    """Absolute path for `settings['paths'][key]`."""
    return Path(settings["_root"]) / settings["paths"][key]


def data_mode(settings: dict[str, Any]) -> str:
    """'dev' or 'full'. `$FDDA_MODE` overrides `dev_mode` (e.g. `FDDA_MODE=full make analysis`)."""
    env = os.getenv("FDDA_MODE", "").strip().lower()
    if env in {"dev", "full"}:
        return env
    if env:
        raise ValueError(f"FDDA_MODE must be 'dev' or 'full', got {env!r}")
    return "dev" if settings.get("dev_mode", True) else "full"
