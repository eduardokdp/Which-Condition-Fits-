"""Clean posts, build weak multi-label targets, mask condition names, and split.

Weak multi-label supervision (Section 3 of the report, Eq. 2):
    y_d(s) = 1  if s was posted in the community linked to condition d,
             or s contains a first-person statement of a diagnosis of d;
    y_d(s) = 0  otherwise.
"""

from __future__ import annotations

import html
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.cmn.config import resolve
from src.cmn.io import get_logger, write_json
from src.dataprep.lexicon import mask_condition_names, self_reported_conditions
from src.dataprep.load import load_raw

log = get_logger()
_URL = re.compile(r"https?://\S+|www\.\S+")
_MD_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")           # [text](url) -> text
_MD_MARK = re.compile(r"(?m)^\s{0,3}(?:#{1,6}\s+|>\s?|[-*+]\s+)")  # headings, quotes, list bullets
_MD_EMPH = re.compile(r"(\*\*|__|~~|\*)(?=\S)(.+?)(?<=\S)\1")   # **bold**, __bold__, ~~strike~~, *it*
_WS = re.compile(r"[ \t\r\f\v]+")
_UNKNOWN_AUTHORS = {"", "nan", "[deleted]", "[removed]"}


def clean_text(text: str) -> str:
    """Normalise a Reddit post: HTML entities, Markdown, URLs, quotes, and whitespace."""
    text = html.unescape(str(text))
    text = text.replace("\u200b", " ").replace("\u2019", "'").replace("\u2018", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = _MD_LINK.sub(r"\1", text)
    text = _URL.sub(" ", text)
    text = _MD_MARK.sub("", text)
    text = _MD_EMPH.sub(r"\2", text)
    text = _WS.sub(" ", text)
    text = re.sub(r" *\n[ \n]*", "\n", text)
    return text.strip()


def is_bot(author: str) -> bool:
    """AutoModerator and accounts whose name marks them as bots (e.g. 'reminder_bot', 'MyBot')."""
    a = str(author).strip().lower()
    return a == "automoderator" or bool(re.search(r"(^|[_\-])bot($|[_\-])|bot$", a))


def _n_tokens(text: str) -> int:
    return len(text.split())


def build_dataset(cfg: dict) -> pd.DataFrame:
    dcfg, conditions, seed = cfg["data"], cfg["conditions"], cfg["seed"]
    c2d = dcfg["community_to_condition"]
    df = load_raw(cfg)
    df = df[df["community"].isin(c2d.keys())].copy()
    df = df.dropna(subset=["post"])
    df["post"] = df["post"].map(clean_text)
    df = df[~df["post"].str.lower().isin({"[removed]", "[deleted]", ""})]
    df = df[~df["author"].fillna("").map(is_bot)]
    df["n_tokens"] = df["post"].map(_n_tokens)
    df = df[(df["n_tokens"] >= dcfg["min_tokens"]) & (df["n_tokens"] <= dcfg["max_tokens"])]

    # seeded shuffle, then one post per known author across all communities (prevents author
    # leakage across splits); posts by deleted or unknown accounts cannot be linked and are kept
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    if dcfg.get("one_post_per_author", True):
        known = ~df["author"].fillna("").astype(str).str.strip().str.lower().isin(_UNKNOWN_AUTHORS)
        df = df[~(known & df.duplicated(subset="author", keep="first"))]
    df = df.drop_duplicates(subset="post", keep="first")

    # balanced cap per community (deterministic: first rows of the seeded shuffle)
    cap = dcfg.get("max_posts_per_community")
    if cap:
        df = df.groupby("community", sort=True, group_keys=False).head(cap)

    # weak multi-label targets
    for d in conditions:
        df[d] = 0
    for idx, row in df.iterrows():
        df.at[idx, c2d[row["community"]]] = 1
    reported = df["post"].map(lambda t: self_reported_conditions(t, conditions))
    for d in conditions:
        df[d] = np.maximum(df[d].to_numpy(), reported.map(lambda s, d=d: int(d in s)).to_numpy())
    df["n_self_reported"] = reported.map(len)

    # mask condition names so the model must use symptom language
    df["text"] = df["post"].map(lambda t: mask_condition_names(t, conditions))
    df = df[df["text"].map(_n_tokens) >= dcfg["min_tokens"]]

    df = df.sort_values(["community", "author"]).reset_index(drop=True)
    df.insert(0, "id", [f"{c}_{i:05d}" for i, c in enumerate(df["community"])])
    return df[["id", "community", "author", "date", "post", "text", "n_self_reported", *conditions]]


def _strat_key(df: pd.DataFrame, conditions: list[str], min_count: int = 6) -> pd.Series:
    """Label-combination key for stratified splitting. Combinations with fewer than `min_count`
    members are pooled as 'rare'; if that pool is still too small to split, it joins the most
    frequent combination, so train_test_split never sees a class with a single member."""
    key = df[conditions].astype(str).agg("".join, axis=1)
    key = key.where(key.map(key.value_counts()) >= min_count, "rare")
    counts = key.value_counts()
    if counts.get("rare", 0) and counts["rare"] < 2:
        key = key.replace("rare", counts.drop("rare").idxmax())
    return key


def split_dataset(df: pd.DataFrame, cfg: dict) -> dict[str, pd.DataFrame]:
    s, seed, conditions = cfg["data"]["split"], cfg["seed"], cfg["conditions"]
    train, rest = train_test_split(df, train_size=s["train"], random_state=seed,
                                   stratify=_strat_key(df, conditions))
    val_share = s["val"] / (s["val"] + s["test"])
    val, test = train_test_split(rest, train_size=val_share, random_state=seed,
                                 stratify=_strat_key(rest, conditions, min_count=2))
    return {k: v.sort_values("id").reset_index(drop=True) for k, v in
            {"train": train, "val": val, "test": test}.items()}


def describe(splits: dict[str, pd.DataFrame], conditions: list[str]) -> dict:
    out = {}
    for name, part in splits.items():
        y = part[conditions].to_numpy()
        combos = part[conditions].astype(str).agg("".join, axis=1).value_counts().to_dict()
        out[name] = {
            "n": int(len(part)),
            "label_counts": {d: int(part[d].sum()) for d in conditions},
            "label_cardinality": round(float(y.sum(1).mean()), 4),
            "multi_label_share": round(float((y.sum(1) > 1).mean()), 4),
            "label_combinations": {"".join(k) if isinstance(k, tuple) else k: int(v) for k, v in combos.items()},
        }
    out["combination_key_order"] = conditions
    return out


def prepare(cfg: dict) -> dict[str, pd.DataFrame]:
    df = build_dataset(cfg)
    splits = split_dataset(df, cfg)
    out_dir = resolve(cfg["data"]["preprocessed_dir"]) / cfg["run_name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, part in splits.items():
        part.to_csv(out_dir / f"{name}.csv", index=False)
    stats = describe(splits, cfg["conditions"])
    write_json(stats, out_dir / "data_card.json")
    for name in ("train", "val", "test"):
        log.info("%-5s n=%5d  labels=%s  multi-label share=%.3f", name, stats[name]["n"],
                 stats[name]["label_counts"], stats[name]["multi_label_share"])
    return splits


def load_splits(cfg: dict) -> dict[str, pd.DataFrame]:
    in_dir = resolve(cfg["data"]["preprocessed_dir"]) / cfg["run_name"]
    if not (in_dir / "train.csv").exists():
        raise FileNotFoundError(f"No preprocessed data in {in_dir}. Run the 'prepare' step first.")
    return {k: pd.read_csv(in_dir / f"{k}.csv", dtype={"text": str, "post": str},
                           keep_default_na=False)
            for k in ("train", "val", "test")}
