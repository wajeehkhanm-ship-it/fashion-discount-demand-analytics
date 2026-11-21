"""Download raw data from Kaggle into `data/raw/` (never committed).

Credentials come from `.env` (KAGGLE_USERNAME / KAGGLE_KEY) or `~/.kaggle/kaggle.json`.
The H&M download only works after you accept the competition rules once on Kaggle.

Usage: `python -m fdda.data.download [--only hm|myntra] [--force]`
"""

from __future__ import annotations

import argparse
import os
import zipfile
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from fdda.config import load_settings, resolve_path
from fdda.log import get_logger

log = get_logger(__name__)

HM_RULES_URL = (
    "https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/rules"
)


def _has_credentials() -> bool:
    """True if Kaggle credentials are available via env vars or kaggle.json."""
    env_ok = bool(os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY"))
    token_ok = bool(os.getenv("KAGGLE_API_TOKEN"))
    file_ok = (Path.home() / ".kaggle" / "kaggle.json").exists()
    return env_ok or token_ok or file_ok


def get_kaggle_api(root: Path) -> Any:
    """Load `.env`, check credentials and return an authenticated KaggleApi.

    The `kaggle` package authenticates at import time, so it is imported only
    after credentials are confirmed.
    """
    load_dotenv(root / ".env")
    if not _has_credentials():
        raise RuntimeError(
            "No Kaggle credentials found. Copy .env.example to .env and fill in "
            "KAGGLE_API_TOKEN (or KAGGLE_USERNAME + KAGGLE_KEY) from "
            "kaggle.com -> Settings -> API -> Create New Token."
        )
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    return api


def _unzip_all(folder: Path) -> None:
    """Extract and delete every .zip in `folder`."""
    for zpath in folder.glob("*.zip"):
        log.info("Unzipping %s", zpath.name)
        with zipfile.ZipFile(zpath) as zf:
            zf.extractall(folder)
        zpath.unlink()


def download_hm(api: Any, out_dir: Path, competition: str, files: list[str], force: bool) -> None:
    """Download the listed H&M competition files (tabular only, no images)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in files:
        if (out_dir / name).exists() and not force:
            log.info("H&M %s already present, skipping (use --force to re-download)", name)
            continue
        log.info("Downloading H&M %s ...", name)
        try:
            api.competition_download_file(competition, name, path=str(out_dir), force=force)
        except Exception as exc:  # kaggle raises plain ApiException/HTTPError
            if "401" in str(exc) or "403" in str(exc):
                raise RuntimeError(
                    "Kaggle refused the H&M download (401/403). "
                    f"Accept the competition rules once at {HM_RULES_URL}"
                ) from exc
            raise
        _unzip_all(out_dir)


def download_myntra(api: Any, out_dir: Path, dataset: str, force: bool) -> None:
    """Download and unzip the Myntra products dataset."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if any(out_dir.glob("*.csv")) and not force:
        log.info("Myntra CSV already present, skipping (use --force to re-download)")
        return
    log.info("Downloading Myntra dataset %s ...", dataset)
    api.dataset_download_files(dataset, path=str(out_dir), unzip=True, force=force)
    _unzip_all(out_dir)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", choices=["hm", "myntra"], help="download one dataset only")
    parser.add_argument("--force", action="store_true", help="re-download existing files")
    args = parser.parse_args(argv)

    settings = load_settings()
    raw = resolve_path(settings, "raw")
    api = get_kaggle_api(settings["_root"])
    ds = settings["datasets"]
    if args.only in (None, "hm"):
        download_hm(api, raw / "hm", ds["hm"]["competition"], ds["hm"]["files"], args.force)
    if args.only in (None, "myntra"):
        download_myntra(api, raw / "myntra", ds["myntra"]["dataset"], args.force)
    for path in sorted(raw.rglob("*.csv")):
        log.info("raw file: %s (%.1f MB)", path.relative_to(raw), path.stat().st_size / 1e6)


if __name__ == "__main__":
    main()
