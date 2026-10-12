"""End to end: candidates -> features -> LightGBM ranker -> offline evaluation -> results.json.

Stage ``valid`` (history to Nov 13, labels Nov 18-23) trains the ranker; stage ``test``
(history to Nov 23, labels Nov 24-30) is only evaluated. Each stage samples SAMPLE_USERS
buyers with history (fixed seed) so the feature table fits in memory; the metrics are
averages over those users, and every method is scored on the same ones.
"""
from __future__ import annotations

import json
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")  # implicit ALS runs its own threads
import time
from pathlib import Path

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd

from src import candidates, data, evaluate, features

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "reports" / "results.json"
SAMPLE_USERS = 50_000
SEED = 42
STAGES = {"valid": "2019-11-13", "test": "2019-11-23"}


def stage(con, name: str) -> dict:
    history_end = STAGES[name]
    start, end = data.WINDOWS[name]
    t0 = time.perf_counter()
    tg = candidates.targets(con, history_end, start, end)
    warm = tg[tg["warm"]].sample(n=min(SAMPLE_USERS, int(tg["warm"].sum())), random_state=SEED)
    truth = candidates.labels(con, start, end)
    cands = candidates.generate(con, history_end, warm)
    table = features.candidate_table(cands)
    feats = features.build(con, table, history_end)
    feats = feats.merge(truth.assign(label=1), on=["user", "item"], how="left")
    feats["label"] = feats["label"].fillna(0).astype(int)
    return {"targets": tg, "warm": warm, "truth": truth, "cands": cands, "feats": feats,
            "seconds": round(time.perf_counter() - t0, 1)}


def train_ranker(feats: pd.DataFrame) -> lgb.LGBMRanker:
    useful = feats.groupby("user")["label"].transform("max") > 0  # groups with no positive teach nothing
    f = feats[useful].sort_values("user")
    model = lgb.LGBMRanker(objective="lambdarank", n_estimators=400, learning_rate=0.05,
                           num_leaves=63, min_child_samples=50, subsample=0.8, subsample_freq=1,
                           colsample_bytree=0.8, random_state=SEED, verbose=-1)
    model.fit(f[features.FEATURES], f["label"], group=f.groupby("user", sort=False).size().to_numpy(),
              eval_at=[evaluate.K])
    return model


def cold_start(con, test: dict) -> dict:
    """Buyers with no history get the last week's best sellers; the only other signal is
    what they view before buying, which is reported as a separate, session-based baseline."""
    tg = test["targets"]
    cold = tg[~tg["warm"]]["user"]
    week = con.execute(f"""
        SELECT item, row_number() OVER (ORDER BY count(*) DESC) AS pos FROM events
        WHERE is_purchase AND ts::DATE BETWEEN DATE '{STAGES['test']}' - INTERVAL 6 DAY AND '{STAGES['test']}'
        GROUP BY 1 QUALIFY pos <= {evaluate.K}
    """).df()
    recs = pd.DataFrame({"user": np.repeat(cold.to_numpy(), len(week)),
                         "item": np.tile(week["item"].to_numpy(), len(cold)),
                         "pos": np.tile(week["pos"].to_numpy(), len(cold))})
    return {"share_of_buyers": round(len(cold) / len(tg), 4),
            "popular": evaluate.metrics(recs, test["truth"], cold)}


def main() -> dict:
    con = duckdb.connect()
    data.events(con)
    valid = stage(con, "valid")
    test = stage(con, "test")
    model = train_ranker(valid["feats"])

    tf = test["feats"].copy()
    tf["score"] = model.predict(tf[features.FEATURES])
    users = test["warm"]["user"]
    methods = {}
    for src in ("recent", "coview", "als", "popular"):
        recs = evaluate.topk(tf[tf[f"rank_{src}"] < features.MISSING_RANK], f"rank_{src}", ascending=True)
        methods[src] = evaluate.metrics(recs, test["truth"], users)
    methods["ranker"] = evaluate.metrics(evaluate.topk(tf, "score"), test["truth"], users)

    truth_u = test["truth"][test["truth"]["user"].isin(users)]
    seen = con.execute(f"""SELECT DISTINCT user, item FROM events WHERE ts::DATE <= '{STAGES['test']}'""").df()
    never_seen = 1 - len(truth_u.merge(seen, on=["user", "item"])) / len(truth_u)

    importance = pd.Series(model.booster_.feature_importance("gain"), index=features.FEATURES)
    results = {
        "events": con.execute("SELECT count(*), sum(is_purchase::INT) FROM events").fetchone(),
        "test_window": data.WINDOWS["test"],
        "test_buyers": int(len(test["targets"])),
        "evaluated_users": int(len(users)),
        "bought_pairs_never_viewed_before": round(float(never_seen), 4),
        "candidate_recall": {
            **{s: evaluate.candidate_recall(test["cands"][s], test["truth"], users) for s in test["cands"]},
            "union": evaluate.candidate_recall(tf, test["truth"], users),
        },
        "candidates_per_user": round(len(tf) / len(users), 1),
        "methods": methods,
        "cold_start": cold_start(con, test),
        "importance_gain_share": (importance / importance.sum()).round(4).sort_values(ascending=False).to_dict(),
        "seconds": {"valid": valid["seconds"], "test": test["seconds"]},
    }
    results["events"] = {"events": int(results["events"][0]), "purchases": int(results["events"][1])}
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
