"""A deliberately simple, text-only baseline "humour judge" (feasibility check for RQ3).

The final project will need a judge that predicts the audience's rating of a caption
(including AI-generated ones). Before building anything multimodal, we check how far
the caption text alone gets us, evaluated on *held-out contests* only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold


def training_sample(captions: pd.DataFrame, n_top: int = 50, n_random: int = 400,
                    seed: int = 0) -> pd.DataFrame:
    """Per contest: the top `n_top` captions, ranks 1000-1009 and `n_random` random ones.

    Keeping the top and the #1000-1009 block lets us reproduce the paper's
    "top 10 vs #1000-1009" pairwise evaluation.
    """
    parts = []
    for _, df in captions.groupby("contest_id"):
        keep = df[(df["rank"] <= n_top) | df["rank"].between(1000, 1009)]
        rest = df.drop(keep.index)
        parts += [keep, rest.sample(min(n_random, len(rest)), random_state=seed)]
    return pd.concat(parts)


def pairwise_accuracy(df: pd.DataFrame, pred: str = "pred") -> float:
    """Share of (top-10, #1000-1009) pairs from the same contest where the
    top-10 caption gets the higher predicted score (0.5 = chance).
    Contests with fewer than 1000 captions have no such pairs and are skipped."""
    hits, total = 0, 0
    for _, c in df.groupby("contest_id"):
        top = c.loc[c["rank"] <= 10, pred].to_numpy()
        low = c.loc[c["rank"].between(1000, 1009), pred].to_numpy()
        if len(top) and len(low):
            hits += (top[:, None] > low[None, :]).sum()
            total += len(top) * len(low)
    return hits / total if total else np.nan


def cross_validated_judge(sample: pd.DataFrame, n_splits: int = 5, alpha: float = 3.0) -> pd.DataFrame:
    """Out-of-fold predictions of the within-contest z-score from TF-IDF features.

    GroupKFold keeps every contest entirely in train or test, so the model is always
    judged on cartoons it has never seen (the setting AI captions will be judged in).
    """
    sample = sample.copy()
    sample["pred"] = np.nan
    folds = GroupKFold(n_splits=n_splits).split(sample, groups=sample["contest_id"])
    for train, test in folds:
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True)
        X_train = vec.fit_transform(sample["caption"].iloc[train])
        model = Ridge(alpha=alpha).fit(X_train, sample["z"].iloc[train])
        sample.iloc[test, sample.columns.get_loc("pred")] = model.predict(
            vec.transform(sample["caption"].iloc[test]))
    return sample


def fit_judge(sample: pd.DataFrame, alpha: float = 3.0, min_contests: int = 50) -> pd.Series:
    """Fit the baseline on the whole sample and return the word coefficients
    (positive = associated with a higher within-contest score).

    Only terms used in at least `min_contests` different contests are kept, so the
    coefficients describe general style rather than words tied to one cartoon.
    """
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True)
    X = vec.fit_transform(sample["caption"])
    # Number of distinct contests in which each term appears (contest one-hot @ term presence).
    codes = pd.factorize(sample["contest_id"])[0]
    onehot = sparse.csr_matrix((np.ones(len(codes)), (codes, np.arange(len(codes)))))
    n_contests = np.asarray(((onehot @ (X > 0).astype(float)) > 0).sum(axis=0)).ravel()
    keep = np.flatnonzero(n_contests >= min_contests)
    model = Ridge(alpha=alpha).fit(X[:, keep], sample["z"])
    return pd.Series(model.coef_, index=vec.get_feature_names_out()[keep]).sort_values()
