"""Offline metrics over the users who buy in the label window."""
from __future__ import annotations

import numpy as np
import pandas as pd

K = 20


def topk(scored: pd.DataFrame, score: str, k: int = K, ascending: bool = False) -> pd.DataFrame:
    """The k best rows per user by ``score``, with their position 1..k."""
    s = scored.sort_values(["user", score], ascending=[True, ascending], kind="mergesort")
    s = s.groupby("user", sort=False).head(k).copy()
    s["pos"] = s.groupby("user", sort=False).cumcount() + 1
    return s[["user", "item", "pos"]]


def metrics(recs: pd.DataFrame, truth: pd.DataFrame, users: pd.Series, k: int = K) -> dict:
    """recall@k (share of a user's bought items in the list), hit@k (at least one) and
    NDCG@k with binary relevance, averaged over ``users``. A user with no recommendations
    counts as zero, so a generator that covers fewer users is not flattered."""
    n_true = truth.groupby("user").size()
    hits = recs.merge(truth, on=["user", "item"])
    hits = hits[hits["pos"] <= k]
    per_user = hits.groupby("user").agg(h=("pos", "size"),
                                        dcg=("pos", lambda p: float((1 / np.log2(p + 1)).sum())))
    idx = pd.Index(users.unique())
    nt = n_true.reindex(idx).fillna(0)
    h = per_user["h"].reindex(idx).fillna(0)
    dcg = per_user["dcg"].reindex(idx).fillna(0)
    ideal = nt.clip(upper=k).map(lambda n: float((1 / np.log2(np.arange(2, int(n) + 2))).sum()) if n else 1.0)
    return {
        "users": int(len(idx)),
        f"recall@{k}": round(float((h / nt.clip(lower=1)).mean()), 4),
        f"hit@{k}": round(float((h > 0).mean()), 4),
        f"ndcg@{k}": round(float((dcg / ideal).mean()), 4),
    }


def candidate_recall(cands: pd.DataFrame, truth: pd.DataFrame, users: pd.Series) -> float:
    """Share of bought (user, item) pairs that appear anywhere in the candidates: the
    ceiling any ranker over these candidates can reach."""
    t = truth[truth["user"].isin(users)]
    return round(len(t.merge(cands[["user", "item"]].drop_duplicates(), on=["user", "item"])) / len(t), 4)
