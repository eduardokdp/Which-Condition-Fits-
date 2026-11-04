"""Rebuild data/raw/reddit_public_sample.csv from two public GitHub repositories.

The quickstart runs on real Reddit posts that other researchers have already published on GitHub:

* r/ADHD       -- premDelaprem/subbreddit-classification (MIT licence), 1,000 most recent posts as of
                  5 January 2024, file data/scraped_reddit_data.csv
* r/Anxiety,   -- rishav-dev/MentalHealthResearch-SocialMedia (no licence file; posts collected with
  r/depression    PRAW in 2025), files reddit_posts.csv and reddit_mental_health_anxiety_posts.csv

Both repositories are pinned to the commits below so the sample is reproducible. Only the subreddit
and the post text (title + body) are kept: no user names, post IDs, URLs, or scores.

    python scripts/build_public_sample.py          # needs git and network access to github.com
"""

from __future__ import annotations

import argparse
import re
import subprocess
import tempfile
from pathlib import Path

import pandas as pd

SOURCES = {
    "premDelaprem/subbreddit-classification": "aa08428fcc8c71cedb753dc5708ddede6c97442b",
    "rishav-dev/MentalHealthResearch-SocialMedia": "ce62d48523990958c78ab27f2eb293e9da750a7e",
}
ROOT = Path(__file__).resolve().parents[1]


def fetch(repo: str, commit: str, dest: Path) -> Path:
    """Shallow-fetch one pinned commit of a public GitHub repository."""
    path = dest / repo.replace("/", "__")
    path.mkdir(parents=True, exist_ok=True)
    run = lambda *a: subprocess.run(["git", *a], cwd=path, check=True, capture_output=True)  # noqa: E731
    run("init", "-q")
    run("remote", "add", "origin", f"https://github.com/{repo}.git")
    run("fetch", "-q", "--depth", "1", "origin", commit)
    run("checkout", "-q", "FETCH_HEAD")
    return path


_USER = re.compile(r"(?<![\w/])/?u/[A-Za-z0-9_-]{3,}")
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")


def _post(title, body) -> str:
    """Title and body as one description, with Reddit user names and e-mail addresses removed."""
    title = "" if pd.isna(title) else str(title).strip()
    body = "" if pd.isna(body) else str(body).strip()
    text = f"{title}\n{body}".strip()
    return _EMAIL.sub("[email]", _USER.sub("u/[user]", text))


def build(workdir: Path) -> pd.DataFrame:
    adhd_repo = fetch("premDelaprem/subbreddit-classification", SOURCES["premDelaprem/subbreddit-classification"], workdir)
    mh_repo = fetch("rishav-dev/MentalHealthResearch-SocialMedia", SOURCES["rishav-dev/MentalHealthResearch-SocialMedia"], workdir)

    a = pd.read_csv(adhd_repo / "data/scraped_reddit_data.csv")
    a = a[a["subreddit"].str.lower() == "adhd"].dropna(subset=["post"])
    adhd = pd.DataFrame({"subreddit": "adhd",
                         "post": [_post(t, b) for t, b in zip(a["title"], a["post"])],
                         "source": "premDelaprem/subbreddit-classification@aa08428"})

    r1 = pd.read_csv(mh_repo / "reddit_posts.csv").rename(columns={"post_id": "pid", "text": "body"})
    r2 = pd.read_csv(mh_repo / "reddit_mental_health_anxiety_posts.csv").rename(columns={"id": "pid", "selftext": "body"})
    r = pd.concat([r1[["pid", "subreddit", "title", "body"]], r2[["pid", "subreddit", "title", "body"]]])
    r = r[r["subreddit"].isin(["Anxiety", "depression"])].drop_duplicates(subset="pid").dropna(subset=["body"])
    mh = pd.DataFrame({"subreddit": r["subreddit"].str.lower().to_numpy(),
                       "post": [_post(t, b) for t, b in zip(r["title"], r["body"])],
                       "source": "rishav-dev/MentalHealthResearch-SocialMedia@ce62d48"})

    out = pd.concat([adhd, mh], ignore_index=True)
    out = out[out["post"].str.split().str.len() > 0].drop_duplicates(subset="post", keep="first")
    return out.reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "data/raw/reddit_public_sample.csv"))
    args = ap.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        df = build(Path(tmp))
    df.to_csv(args.out, index=False)
    print(f"wrote {len(df)} posts to {args.out}: {df['subreddit'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
