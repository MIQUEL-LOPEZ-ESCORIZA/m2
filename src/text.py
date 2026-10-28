"""Text normalisation, simple caption features and TF-IDF similarity."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

_QUOTES = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"', "…": "..."})
_NON_WORD = re.compile(r"[^a-z0-9' ]+")
_SPACES = re.compile(r"\s+")


def normalize(text: pd.Series) -> pd.Series:
    """Lower-case, unify quotes, drop punctuation and extra spaces.

    "Who farted?" and "who farted!" become the same string "who farted".
    """
    text = text.fillna("").str.translate(_QUOTES).str.lower()
    text = text.str.replace(_NON_WORD, " ", regex=True)
    return text.str.replace(_SPACES, " ", regex=True).str.strip()


def add_text_features(captions: pd.DataFrame) -> pd.DataFrame:
    """A few cheap, interpretable surface features of a caption."""
    raw = captions["caption"].fillna("")
    captions["caption_norm"] = normalize(raw)
    captions["n_words"] = raw.str.split().str.len().astype("int16")
    captions["n_chars"] = raw.str.len().astype("int16")
    captions["is_question"] = raw.str.rstrip().str.endswith("?")
    captions["is_exclamation"] = raw.str.rstrip().str.endswith("!")
    captions["has_quote"] = raw.str.contains(r"[\"“”]", regex=True)
    captions["second_person"] = captions["caption_norm"].str.contains(r"\byou\b|\byour\b", regex=True)
    return captions


def fit_tfidf(texts: pd.Series, **kwargs) -> tuple:
    """Word uni+bi-gram TF-IDF (L2-normalised, so a dot product is a cosine similarity)."""
    params = dict(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    params.update(kwargs)
    vec = TfidfVectorizer(**params)
    return vec, vec.fit_transform(texts)


def idea_crowd_size(contest: pd.DataFrame, query_idx: np.ndarray, threshold: float = 0.4) -> np.ndarray:
    """For the captions at positions `query_idx`, count how many *other* captions of the
    same contest have a TF-IDF cosine similarity >= threshold (a proxy for how many
    people had the same joke idea). Empty-vector captions get NaN.
    """
    _, X = fit_tfidf(contest["caption_norm"])
    sims = (X[query_idx] @ X.T).toarray()
    counts = (sims >= threshold).sum(axis=1) - 1  # remove the caption itself
    empty = np.asarray(X[query_idx].sum(axis=1)).ravel() == 0
    return np.where(empty, np.nan, counts)


def nearest_captions(contest: pd.DataFrame, position: int, k: int = 10) -> pd.DataFrame:
    """The k captions most similar to the caption at `position` within one contest."""
    _, X = fit_tfidf(contest["caption_norm"])
    sims = (X[position] @ X.T).toarray().ravel()
    order = np.argsort(-sims)[1:k + 1]
    out = contest.iloc[order][["caption", "mean", "votes", "rank"]].copy()
    out.insert(0, "similarity", sims[order].round(2))
    return out


def crowd_sizes(captions: pd.DataFrame, n_top: int = 10, n_random: int = 200,
                threshold: float = 0.4, seed: int = 0) -> pd.DataFrame:
    """Idea crowd size for the top `n_top` captions and `n_random` random others, in every contest."""
    rng = np.random.default_rng(seed)
    rows = []
    for cid, c in captions.groupby("contest_id"):
        c = c.sort_values("rank").reset_index(drop=True)
        others = rng.choice(np.arange(n_top, len(c)), min(n_random, len(c) - n_top), replace=False)
        query = np.r_[np.arange(n_top), others]
        rows.append(pd.DataFrame({"contest_id": cid, "rank": c["rank"].to_numpy()[query],
                                  "z": c["z"].to_numpy()[query],
                                  "n_words": c["n_words"].to_numpy()[query],
                                  "crowd": idea_crowd_size(c, query, threshold)}))
    return pd.concat(rows, ignore_index=True)
