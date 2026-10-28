"""Turning raw ratings into comparable scores.

Raw means are not directly comparable: contests differ in difficulty, and captions
with few votes have noisy means. We therefore add:

- `rank` / `rank_pct`: observed ordering inside a contest (1 = best);
- `z`: the mean standardised within its contest;
- `eb_mean`: an empirical-Bayes (shrunk) mean that pulls noisy captions toward the contest mean.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def add_scores(captions: pd.DataFrame) -> pd.DataFrame:
    """Add rank, within-contest z-score and empirical-Bayes mean.

    Empirical Bayes (normal-normal model), done separately for every contest c:
        x_i | mu_i ~ N(mu_i, s_i^2)      (s_i = reported `precision`, a standard error)
        mu_i       ~ N(m_c, tau_c^2)     (true funniness of the captions of contest c)
    Method of moments: m_c = mean(x), tau_c^2 = max(var(x) - mean(s^2), eps).
    Posterior mean: eb_i = w_i x_i + (1 - w_i) m_c with w_i = tau_c^2 / (tau_c^2 + s_i^2).
    """
    g = captions.groupby("contest_id")["mean"]
    captions["rank"] = g.rank(ascending=False, method="first").astype("int32")
    captions["rank_pct"] = (captions["rank"] / g.transform("size")).astype("float32")
    captions["z"] = ((captions["mean"] - g.transform("mean")) / g.transform("std")).astype("float32")

    s2 = captions["precision"].astype(float) ** 2
    m = g.transform("mean")
    tau2 = (g.transform("var") - s2.groupby(captions["contest_id"]).transform("mean")).clip(lower=1e-4)
    w = tau2 / (tau2 + s2)
    captions["eb_weight"] = w.astype("float32")
    captions["eb_mean"] = (w * captions["mean"] + (1 - w) * m).astype("float32")
    return captions


def aggregate_text(captions: pd.DataFrame, key: str = "caption_norm") -> pd.DataFrame:
    """One row per (contest, normalised wording).

    Identical wording submitted several times in one contest gets several rows with
    different ratings; we average them (equal weight) instead of keeping the best one.
    """
    return (captions.groupby(["contest_id", key], as_index=False, observed=True)
            .agg(mean=("mean", "mean"), z=("z", "mean"), votes=("votes", "sum"),
                 n_entries=("mean", "size")))


def _icc_from_stats(n: np.ndarray, s1: np.ndarray, s2: np.ndarray) -> tuple:
    """ICC from per-group count n, sum of y (s1) and sum of y^2 (s2)."""
    G, N = len(n), n.sum()
    grand = s1.sum() / N
    ssb = (s1 ** 2 / n).sum() - N * grand ** 2
    ssw = s2.sum() - (s1 ** 2 / n).sum()
    msb, msw = ssb / (G - 1), ssw / (N - G)
    n0 = (N - (n ** 2).sum() / N) / (G - 1)
    s_a2 = max((msb - msw) / n0, 0.0)
    return s_a2, msw, s_a2 / (s_a2 + msw)


def _group_stats(df: pd.DataFrame, group: str, value: str) -> tuple:
    y = df[value].astype(float)
    g = pd.DataFrame({"y": y, "y2": y ** 2}).groupby(df[group].to_numpy())
    agg = g.agg(n=("y", "size"), s1=("y", "sum"), s2=("y2", "sum"))
    return agg["n"].to_numpy(float), agg["s1"].to_numpy(), agg["s2"].to_numpy()


def icc_oneway(df: pd.DataFrame, group: str, value: str) -> dict:
    """Intra-class correlation from a one-way random-effects ANOVA.

    Model: y_gj = mu + a_g + e_gj with a_g ~ (0, s_a^2), e_gj ~ (0, s_e^2).
    ICC = s_a^2 / (s_a^2 + s_e^2) = share of the variance explained by the group.
    For unbalanced groups we use the usual n0 correction:
        MSB = sum n_g (ybar_g - ybar)^2 / (G - 1),  MSW = sum (y - ybar_g)^2 / (N - G)
        n0  = (N - sum n_g^2 / N) / (G - 1),       s_a^2 = (MSB - MSW) / n0
    """
    n, s1, s2 = _group_stats(df, group, value)
    s_a2, msw, icc = _icc_from_stats(n, s1, s2)
    return {"groups": len(n), "observations": int(n.sum()), "between_var": s_a2,
            "within_var": msw, "icc": icc}


def bootstrap_icc(df: pd.DataFrame, group: str, value: str,
                  n_boot: int = 1000, seed: int = 0) -> tuple:
    """95% percentile CI for the ICC, resampling whole groups (cluster bootstrap)."""
    rng = np.random.default_rng(seed)
    n, s1, s2 = _group_stats(df, group, value)
    stats = []
    for _ in range(n_boot):
        i = rng.integers(0, len(n), len(n))
        stats.append(_icc_from_stats(n[i], s1[i], s2[i])[2])
    return tuple(np.percentile(stats, [2.5, 97.5]))
