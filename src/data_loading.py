"""Loading the raw contest files into two tidy tables.

- `load_captions` -> one row per rated caption entry (~2.3M rows), cached as parquet.
- `load_contests` -> one row per contest, with the cartoon metadata flattened.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

RATING_COLUMNS = ["not_funny", "somewhat_funny", "funny"]


def find_data_dir() -> Path:
    """Return the dataset folder, whether we run from m2/, the project root or above."""
    for p in [Path("../newyorker_caption_contest"),
              Path("newyorker_caption_contest"),
              Path("ADA_PROJECT/newyorker_caption_contest")]:
        if (p / "contests.json").exists():
            return p
    raise FileNotFoundError("Could not find the newyorker_caption_contest folder.")


def load_captions(data_dir: Path, cache: Path | None = None) -> pd.DataFrame:
    """Concatenate all `data/<id>.csv` files into one DataFrame.

    The `rank` column only exists in 148 older files, so we drop it and recompute
    a rank later. Integer counts are downcast to save memory, and the result is
    cached to parquet so the notebook re-runs in a couple of seconds.
    """
    if cache is not None and Path(cache).exists():
        return pd.read_parquet(cache)

    frames = []
    for path in sorted(Path(data_dir, "data").glob("*.csv")):
        df = pd.read_csv(path).drop(columns="rank", errors="ignore")
        df.insert(0, "contest_id", int(path.stem))
        frames.append(df)
    captions = pd.concat(frames, ignore_index=True)

    captions["contest_id"] = captions["contest_id"].astype("int16")
    for col in ["votes"] + RATING_COLUMNS:
        captions[col] = captions[col].astype("int32")
    captions["mean"] = captions["mean"].astype("float32")
    captions["precision"] = captions["precision"].astype("float32")

    if cache is not None:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        captions.to_parquet(cache, index=False)
    return captions


def _clean_location(location: str) -> str:
    """'A living room' / 'the living room' -> 'living room'."""
    location = location.lower().strip()
    for article in ("the ", "a ", "an "):
        if location.startswith(article):
            return location[len(article):]
    return location


def load_contests(data_dir: Path) -> pd.DataFrame:
    """One row per contest with the (optional) Hessel et al. metadata flattened."""
    with open(Path(data_dir, "contests.json")) as f:
        raw = json.load(f)

    rows = []
    for c in raw:
        meta = c["metadata"] or {}
        rows.append({
            "contest_id": c["contest_id"],
            "has_metadata": c["metadata"] is not None,
            "location": _clean_location(meta["image_locations"][0]) if meta.get("image_locations") else None,
            "description": meta.get("image_descriptions", [None])[0],
            "uncanny": meta.get("image_uncanny_descriptions", [None])[0],
            "entities": [e.rsplit("/", 1)[-1] for e in meta.get("entities", [])],
            "n_descriptions": len(meta.get("image_descriptions", [])),
        })
    return pd.DataFrame(rows).set_index("contest_id")
