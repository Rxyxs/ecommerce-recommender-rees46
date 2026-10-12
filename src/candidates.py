"""Candidate generation: for each target user, a few hundred items worth ranking.

Every generator sees only the history (events before ``history_end``); the labels are the
items each user buys in the window that follows. Four generators, each a baseline on its own:

- ``recent``: items the user viewed, most recent first. People buy what they looked at.
- ``coview``: items viewed in the same sessions as the user's last viewed items.
- ``als``: implicit-feedback matrix factorisation on view and purchase counts.
- ``popular``: best sellers of the last week, overall and in the user's main category.
"""
from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd
import scipy.sparse as sp

RECENT_DAYS = 30
COVIEW_DAYS = 14
COVIEW_MAX_SESSION_ITEMS = 30
COVIEW_NEIGHBOURS = 20
SEED_ITEMS = 5
N_PER_SOURCE = 50
ALS_FACTORS = 64
ALS_DAYS = 30


def labels(con, start: str, end: str) -> pd.DataFrame:
    """Distinct (user, item) bought in the window."""
    return con.execute(f"""
        SELECT DISTINCT user, item FROM events
        WHERE is_purchase AND ts::DATE BETWEEN '{start}' AND '{end}'
    """).df()


def targets(con, history_end: str, start: str, end: str) -> pd.DataFrame:
    """Users who buy in the window, split into those with history (warm) and without (cold)."""
    return con.execute(f"""
        WITH buyers AS (SELECT DISTINCT user FROM events
                        WHERE is_purchase AND ts::DATE BETWEEN '{start}' AND '{end}'),
             seen AS (SELECT DISTINCT user FROM events WHERE ts::DATE <= '{history_end}')
        SELECT b.user, s.user IS NOT NULL AS warm FROM buyers b LEFT JOIN seen s USING (user)
    """).df()


def _register_targets(con, users: pd.DataFrame) -> None:
    con.register("target_users_df", users[["user"]])
    con.execute("CREATE OR REPLACE TEMP TABLE target_users AS SELECT user FROM target_users_df")


def recent(con, history_end: str) -> pd.DataFrame:
    return con.execute(f"""
        WITH h AS (
            SELECT e.user, e.item, count(*) AS n_views, max(e.ts) AS last_ts
            FROM events e JOIN target_users USING (user)
            WHERE NOT e.is_purchase
              AND e.ts::DATE BETWEEN DATE '{history_end}' - INTERVAL {RECENT_DAYS} DAY AND '{history_end}'
            GROUP BY 1, 2
        )
        SELECT user, item, row_number() OVER (PARTITION BY user ORDER BY last_ts DESC) AS rank
        FROM h QUALIFY rank <= {N_PER_SOURCE}
    """).df()


def coview(con, history_end: str) -> pd.DataFrame:
    """Item-item co-view counts from sessions in the last COVIEW_DAYS, normalised by the
    popularity of the neighbour (so a best seller is not every item's neighbour)."""
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE sess AS
        WITH s AS (
            SELECT DISTINCT session, item FROM events
            WHERE ts::DATE BETWEEN DATE '{history_end}' - INTERVAL {COVIEW_DAYS} DAY AND '{history_end}'
        )
        SELECT s.* FROM s
        JOIN (SELECT session FROM s GROUP BY 1
              HAVING count(*) BETWEEN 2 AND {COVIEW_MAX_SESSION_ITEMS}) USING (session)
    """)
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE neighbours AS
        WITH pairs AS (
            SELECT a.item AS item, b.item AS nb, count(*) AS n
            FROM sess a JOIN sess b ON a.session = b.session AND a.item <> b.item
            GROUP BY 1, 2
        ), pop AS (SELECT item, count(*) AS n_sess FROM sess GROUP BY 1)
        SELECT pairs.item, pairs.nb, pairs.n / sqrt(p.n_sess) AS score FROM pairs JOIN pop p ON p.item = pairs.nb
        QUALIFY row_number() OVER (PARTITION BY pairs.item ORDER BY score DESC) <= {COVIEW_NEIGHBOURS}
    """)
    return con.execute(f"""
        WITH seeds AS (
            SELECT user, item, max(ts) AS last_ts FROM events JOIN target_users USING (user)
            WHERE ts::DATE <= '{history_end}' GROUP BY 1, 2
            QUALIFY row_number() OVER (PARTITION BY user ORDER BY last_ts DESC) <= {SEED_ITEMS}
        ), scored AS (
            SELECT s.user, n.nb AS item, sum(n.score) AS score
            FROM seeds s JOIN neighbours n ON n.item = s.item GROUP BY 1, 2
        )
        SELECT user, item, score AS coview_score,
               row_number() OVER (PARTITION BY user ORDER BY score DESC) AS rank
        FROM scored QUALIFY rank <= {N_PER_SOURCE}
    """).df()


def popular(con, history_end: str) -> pd.DataFrame:
    """Top sellers of the last 7 days, overall and inside the user's most viewed category."""
    return con.execute(f"""
        WITH week AS (
            SELECT item, top_category, count(*) AS n FROM events
            WHERE is_purchase AND ts::DATE BETWEEN DATE '{history_end}' - INTERVAL 6 DAY AND '{history_end}'
            GROUP BY 1, 2
        ), global AS (
            SELECT item, row_number() OVER (ORDER BY n DESC) AS rank FROM week QUALIFY rank <= {N_PER_SOURCE}
        ), in_cat AS (
            SELECT item, top_category, row_number() OVER (PARTITION BY top_category ORDER BY n DESC) AS rank
            FROM week QUALIFY rank <= {N_PER_SOURCE}
        ), main_cat AS (
            SELECT user, arg_max(top_category, n) AS top_category FROM (
                SELECT user, top_category, count(*) AS n FROM events JOIN target_users USING (user)
                WHERE ts::DATE <= '{history_end}' GROUP BY 1, 2) GROUP BY 1
        )
        SELECT t.user, g.item, g.rank FROM target_users t CROSS JOIN global g
        UNION ALL
        SELECT m.user, c.item, c.rank FROM main_cat m JOIN in_cat c USING (top_category)
    """).df().sort_values(["user", "rank"]).drop_duplicates(["user", "item"])


def als(con, history_end: str, users: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """ALS on the last ALS_DAYS of views (weight 1) and purchases (weight 5)."""
    from implicit.als import AlternatingLeastSquares

    inter = con.execute(f"""
        SELECT user, item, sum(CASE WHEN is_purchase THEN 5 ELSE 1 END)::FLOAT AS w FROM events
        WHERE ts::DATE BETWEEN DATE '{history_end}' - INTERVAL {ALS_DAYS} DAY AND '{history_end}'
        GROUP BY 1, 2
    """).df()
    u_codes, u_index = pd.factorize(inter["user"])
    i_codes, i_index = pd.factorize(inter["item"])
    mat = sp.csr_matrix((np.log1p(inter["w"].to_numpy()), (u_codes, i_codes)),
                        shape=(len(u_index), len(i_index)))
    model = AlternatingLeastSquares(factors=ALS_FACTORS, regularization=0.05, iterations=15,
                                    random_state=seed)
    model.fit(mat, show_progress=False)
    pos = pd.Index(u_index).get_indexer(users["user"])
    known = users["user"].to_numpy()[pos >= 0]
    rows = pos[pos >= 0]
    out = []
    for start in range(0, len(rows), 50_000):
        chunk = rows[start:start + 50_000]
        ids, scores = model.recommend(chunk, mat[chunk], N=N_PER_SOURCE, filter_already_liked_items=False)
        out.append(pd.DataFrame({
            "user": np.repeat(known[start:start + 50_000], N_PER_SOURCE),
            "item": i_index.to_numpy()[ids.ravel()],
            "als_score": scores.ravel(),
            "rank": np.tile(np.arange(1, N_PER_SOURCE + 1), len(chunk)),
        }))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["user", "item", "als_score", "rank"])


def generate(con, history_end: str, users: pd.DataFrame) -> dict[str, pd.DataFrame]:
    _register_targets(con, users)
    return {
        "recent": recent(con, history_end),
        "coview": coview(con, history_end),
        "als": als(con, history_end, users),
        "popular": popular(con, history_end),
    }
