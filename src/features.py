"""One row per (user, candidate item) with what the ranker learns from, all from history."""
from __future__ import annotations

import pandas as pd

MISSING_RANK = 999
FEATURES = [
    "rank_recent", "rank_coview", "rank_als", "rank_popular", "n_sources",
    "coview_score", "als_score",
    "ui_views", "ui_views_7d", "ui_days_since_view", "ui_bought_before", "ui_in_last_session",
    "item_views_7d", "item_purchases_7d", "item_conversion_30d", "item_price", "item_price_vs_user",
    "user_events_30d", "user_purchases", "user_items_viewed", "same_main_category",
]


def candidate_table(cands: dict[str, pd.DataFrame]) -> pd.DataFrame:
    parts = []
    for name, df in cands.items():
        cols = ["user", "item", "rank"] + [c for c in ("coview_score", "als_score") if c in df]
        parts.append(df[cols].rename(columns={"rank": f"rank_{name}"}))
    out = parts[0]
    for p in parts[1:]:
        out = out.merge(p, on=["user", "item"], how="outer")
    rank_cols = [f"rank_{n}" for n in cands]
    out["n_sources"] = out[rank_cols].notna().sum(axis=1)
    out[rank_cols] = out[rank_cols].fillna(MISSING_RANK)
    return out


def build(con, cands: pd.DataFrame, history_end: str) -> pd.DataFrame:
    con.register("cand_df", cands)
    h = f"ts::DATE <= '{history_end}'"
    week = f"ts::DATE BETWEEN DATE '{history_end}' - INTERVAL 6 DAY AND '{history_end}'"
    month = f"ts::DATE BETWEEN DATE '{history_end}' - INTERVAL 29 DAY AND '{history_end}'"
    return con.execute(f"""
        WITH cu AS (SELECT DISTINCT user FROM cand_df),
        hist AS (SELECT e.* FROM events e JOIN cu USING (user) WHERE {h}),
        ui AS (
            SELECT user, item,
                   count(*) FILTER (WHERE NOT is_purchase) AS ui_views,
                   count(*) FILTER (WHERE NOT is_purchase AND {week}) AS ui_views_7d,
                   date_diff('day', max(ts) FILTER (WHERE NOT is_purchase), DATE '{history_end}')
                       AS ui_days_since_view,
                   bool_or(is_purchase)::INT AS ui_bought_before
            FROM hist GROUP BY 1, 2
        ),
        last_sess AS (
            SELECT user, arg_max(session, ts) AS session FROM hist GROUP BY 1
        ),
        ls_items AS (
            SELECT DISTINCT h.user, h.item FROM hist h JOIN last_sess l USING (user, session)
        ),
        it AS (
            SELECT item,
                   count(*) FILTER (WHERE NOT is_purchase AND {week}) AS item_views_7d,
                   count(*) FILTER (WHERE is_purchase AND {week}) AS item_purchases_7d,
                   count(*) FILTER (WHERE is_purchase AND {month})
                       / greatest(count(*) FILTER (WHERE NOT is_purchase AND {month}), 1) AS item_conversion_30d,
                   median(price) AS item_price,
                   arg_max(top_category, ts) AS item_category
            FROM events WHERE {month} AND item IN (SELECT DISTINCT item FROM cand_df) GROUP BY 1
        ),
        us AS (
            SELECT user,
                   count(*) FILTER (WHERE {month}) AS user_events_30d,
                   count(*) FILTER (WHERE is_purchase) AS user_purchases,
                   count(DISTINCT item) AS user_items_viewed,
                   median(price) AS user_median_price,
                   mode(top_category) AS user_category
            FROM hist GROUP BY 1
        )
        SELECT c.*,
               coalesce(ui.ui_views, 0) AS ui_views, coalesce(ui.ui_views_7d, 0) AS ui_views_7d,
               coalesce(ui.ui_days_since_view, 99) AS ui_days_since_view,
               coalesce(ui.ui_bought_before, 0) AS ui_bought_before,
               (ls.item IS NOT NULL)::INT AS ui_in_last_session,
               coalesce(it.item_views_7d, 0) AS item_views_7d,
               coalesce(it.item_purchases_7d, 0) AS item_purchases_7d,
               coalesce(it.item_conversion_30d, 0) AS item_conversion_30d,
               it.item_price,
               it.item_price / nullif(us.user_median_price, 0) AS item_price_vs_user,
               coalesce(us.user_events_30d, 0) AS user_events_30d,
               coalesce(us.user_purchases, 0) AS user_purchases,
               coalesce(us.user_items_viewed, 0) AS user_items_viewed,
               (it.item_category = us.user_category)::INT AS same_main_category
        FROM cand_df c
        LEFT JOIN ui USING (user, item)
        LEFT JOIN ls_items ls USING (user, item)
        LEFT JOIN it USING (item)
        LEFT JOIN us USING (user)
    """).df()
