"""Data-quality flags. We flag problems instead of silently repairing the ratings."""
from __future__ import annotations

import numpy as np
import pandas as pd

# The paper's crowdsourced rating system starts with contest 530.
FIRST_CROWDSOURCED_CONTEST = 530


def add_quality_flags(captions: pd.DataFrame) -> pd.DataFrame:
    """Add boolean columns describing the consistency of each row.

    - `counts_ok`  : the three rating counts sum to `votes`
    - `mean_ok`    : the reported mean equals the mean recomputed from the counts, and lies in [1, 3]
    - `precision_ok`: the reported standard error is finite (it is `inf` for single-vote rows)
    - `crowdsourced`: contest id >= 530 (era of the bandit-based rating system)
    - `valid`      : all of the above, plus a non-missing caption
    """
    n1, n2, n3 = captions["not_funny"], captions["somewhat_funny"], captions["funny"]
    total = n1 + n2 + n3
    recomputed = (n1 + 2 * n2 + 3 * n3) / total.replace(0, np.nan)

    captions["counts_ok"] = total == captions["votes"]
    captions["mean_ok"] = ((recomputed - captions["mean"]).abs() < 1e-4) & captions["mean"].between(1, 3)
    captions["precision_ok"] = np.isfinite(captions["precision"]) & (captions["votes"] > 1)
    captions["crowdsourced"] = captions["contest_id"] >= FIRST_CROWDSOURCED_CONTEST
    captions["valid"] = (captions["counts_ok"] & captions["mean_ok"] & captions["precision_ok"]
                         & captions["crowdsourced"] & captions["caption"].notna())
    return captions


def quality_report(captions: pd.DataFrame) -> pd.DataFrame:
    """Number and share of rows failing each check."""
    checks = {
        "missing caption text": captions["caption"].isna(),
        "counts do not sum to votes": ~captions["counts_ok"],
        "mean differs from counts / outside [1,3]": ~captions["mean_ok"],
        "infinite precision (1 vote)": ~captions["precision_ok"],
        "pre-crowdsourcing contest (< 530)": ~captions["crowdsourced"],
        "exact duplicate record": captions.duplicated(
            ["contest_id", "caption", "mean", "precision", "votes"]),
        "any problem (not valid)": ~captions["valid"],
    }
    report = pd.Series({k: int(v.sum()) for k, v in checks.items()}).to_frame("rows")
    report["share"] = (report["rows"] / len(captions)).map("{:.2%}".format)
    return report
