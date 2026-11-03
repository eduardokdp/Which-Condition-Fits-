"""Load raw posts into one DataFrame with columns: community, author, date, post."""

from __future__ import annotations

import urllib.request
from pathlib import Path

import pandas as pd

from src.cmn.config import resolve
from src.cmn.io import get_logger

log = get_logger()
_KEEP = {"subreddit", "author", "date", "post"}


def download_low2020(cfg: dict) -> list[Path]:
    """Download the per-community CSV files of Low et al. (2020) from Zenodo (PDDL 1.0)."""
    raw_dir = resolve(cfg["data"]["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for community, fname in cfg["data"]["files"].items():
        dest = raw_dir / fname
        if dest.exists() and dest.stat().st_size > 0:
            log.info("found %s, skipping download", dest.name)
        else:
            url = cfg["data"]["zenodo_url"].format(file=fname)
            log.info("downloading %s -> %s", url, dest)
            tmp = dest.with_suffix(".part")
            urllib.request.urlretrieve(url, tmp)
            tmp.rename(dest)
        paths.append(dest)
    return paths


def _standardise(df: pd.DataFrame, community: str | None = None) -> pd.DataFrame:
    df = df[[c for c in df.columns if c in _KEEP]].copy()
    if "post" not in df.columns:
        raise ValueError("Input data must contain a 'post' column.")
    if community is not None:
        df["subreddit"] = community
    if "subreddit" not in df.columns:
        raise ValueError("Input data must contain a 'subreddit' column (the source community).")
    if "author" not in df.columns:
        df["author"] = [f"anon_{i}" for i in range(len(df))]
    if "date" not in df.columns:
        df["date"] = ""
    df = df.rename(columns={"subreddit": "community"})
    df["community"] = df["community"].astype(str).str.lower().str.strip()
    return df[["community", "author", "date", "post"]]


def load_raw(cfg: dict) -> pd.DataFrame:
    src = cfg["data"]["source"]
    if src == "low2020":
        raw_dir = resolve(cfg["data"]["raw_dir"])
        frames = []
        for community, fname in cfg["data"]["files"].items():
            path = raw_dir / fname
            if not path.exists():
                raise FileNotFoundError(
                    f"{path} not found. Run: python -m src.main download --config {cfg['_config_path']}"
                )
            df = pd.read_csv(path, usecols=lambda c: c in _KEEP, dtype=str)
            frames.append(_standardise(df, community))
            log.info("loaded %-12s %7d posts", community, len(df))
        return pd.concat(frames, ignore_index=True)
    if src == "csv":
        path = resolve(cfg["data"]["csv_path"])
        df = pd.read_csv(path, dtype=str)
        log.info("loaded %s (%d posts)", path.name, len(df))
        return _standardise(df)
    raise ValueError(f"Unknown data.source '{src}' (expected 'low2020' or 'csv').")
