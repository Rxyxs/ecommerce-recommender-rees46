"""Metrics and candidate generators on hand-made events small enough to check by eye."""
from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd
import pytest

from src import candidates, evaluate, features


def test_metrics_on_known_lists():
    truth = pd.DataFrame({"user": [1, 1, 2], "item": [10, 11, 20]})
    recs = pd.DataFrame({"user": [1, 1, 2], "item": [10, 99, 21], "pos": [1, 2, 1]})
    m = evaluate.metrics(recs, truth, pd.Series([1, 2]), k=2)
    assert m["recall@2"] == 0.25          # user 1: 1 of 2, user 2: 0 of 1
    assert m["hit@2"] == 0.5
    ideal_1 = 1 + 1 / np.log2(3)
    assert m["ndcg@2"] == round((1 / ideal_1 + 0) / 2, 4)


def test_users_without_recommendations_count_as_zero():
    truth = pd.DataFrame({"user": [1, 2], "item": [10, 20]})
    recs = pd.DataFrame({"user": [1], "item": [10], "pos": [1]})
    assert evaluate.metrics(recs, truth, pd.Series([1, 2]))["hit@20"] == 0.5


def test_topk_orders_and_cuts():
    s = pd.DataFrame({"user": [1, 1, 1], "item": [1, 2, 3], "score": [0.1, 0.9, 0.5]})
    assert evaluate.topk(s, "score", k=2)["item"].tolist() == [2, 3]


@pytest.fixture()
def con():
    ts = pd.to_datetime
    ev = pd.DataFrame({
        "ts": ts(["2019-11-10 10:00", "2019-11-10 10:05", "2019-11-11 09:00", "2019-11-12 12:00",
                  "2019-11-12 12:01", "2019-11-13 08:00", "2019-11-19 10:00"]),
        "is_purchase": [False, False, False, False, True, False, True],
        "item": [1, 2, 3, 1, 2, 4, 3],
        "user": [7, 7, 7, 8, 8, 8, 7],
        "category_id": 1, "top_category": "electronics", "brand": "x", "price": 10.0,
        "session": ["a", "a", "b", "c", "c", "d", "e"],
    })
    c = duckdb.connect()
    c.register("ev", ev)
    c.execute("CREATE VIEW events AS SELECT * FROM ev")
    return c


def test_targets_and_recent(con):
    tg = candidates.targets(con, "2019-11-13", "2019-11-18", "2019-11-23")
    assert tg.values.tolist() == [[7, True]]
    candidates._register_targets(con, tg)
    rec = candidates.recent(con, "2019-11-13")
    assert rec.sort_values("rank")["item"].tolist() == [3, 2, 1]   # most recent view first


def test_coview_links_items_from_shared_sessions(con):
    tg = pd.DataFrame({"user": [7]})
    candidates._register_targets(con, tg)
    cv = candidates.coview(con, "2019-11-13")
    assert set(cv["item"]) == {1, 2}   # sessions a and c pair items 1 and 2


def test_candidate_table_marks_missing_sources():
    a = pd.DataFrame({"user": [1], "item": [5], "rank": [1]})
    b = pd.DataFrame({"user": [1], "item": [6], "rank": [2], "als_score": [0.3]})
    t = features.candidate_table({"recent": a, "als": b})
    row = t.set_index("item").loc[5]
    assert row["rank_als"] == features.MISSING_RANK and row["n_sources"] == 1
