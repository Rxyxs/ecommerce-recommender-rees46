"""REES46 events -> one compact Parquet of views and purchases, and the time split.

Source: *eCommerce behavior data from multi category store* (Kaggle, mkechinov; REES46 /
Open CDP), October-November 2019, license "copyright-authors": downloaded by each user,
never committed, only aggregates published.

Two facts about this data, measured in the companion lakehouse project
(github.com/Rxyxs/ecommerce-lakehouse-duckdb), decide what is used:

- **Cart events are not usable as a signal.** Their tracking changes six times in two
  months (0.8 to 2.9 carts per purchase from one day to the next), so a model trained
  on carts would learn the tracking. Only views and purchases are kept.
- **Purchases are missing on November 14-17** (48 busy hours with none recorded). Those
  days are left out of every window, so no split is evaluated on missing labels.

Split, all by time:

    train   Oct 01 - Nov 13   history used to build candidates and features
    valid   Nov 18 - Nov 23   labels for training the ranker
    test    Nov 24 - Nov 30   labels for the final evaluation only

For the test, the history is everything up to Nov 23 (train + valid), the way a model
retrained the day before would see it.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
EVENTS = ROOT / "data" / "events.parquet"
DATASET = "mkechinov/ecommerce-behavior-data-from-multi-category-store"
FILES = ["2019-Oct.csv", "2019-Nov.csv"]

GAP = ("2019-11-14", "2019-11-17")
WINDOWS = {
    "train": ("2019-10-01", "2019-11-13"),
    "valid": ("2019-11-18", "2019-11-23"),
    "test": ("2019-11-24", "2019-11-30"),
}


def download(raw: Path = RAW) -> None:
    if all((raw / f).exists() for f in FILES):
        return
    raw.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, "-m", "kaggle.cli", "datasets", "download", "-d", DATASET,
                    "-p", str(raw), "--unzip"], check=True)


def build_events(raw: Path = RAW, out: Path = EVENTS) -> dict:
    """Views and purchases with a non-null session, exact duplicates removed, sorted by time."""
    files = ", ".join(f"'{(raw / f).as_posix()}'" for f in FILES)
    con = duckdb.connect()
    con.execute(f"""
        COPY (
            SELECT DISTINCT
                   event_time::TIMESTAMP AS ts,
                   event_type = 'purchase' AS is_purchase,
                   product_id::INTEGER AS item,
                   user_id::INTEGER AS user,
                   category_id,
                   COALESCE(split_part(category_code, '.', 1), 'uncoded') AS top_category,
                   brand,
                   CASE WHEN price > 0 THEN price END AS price,
                   user_session AS session
            FROM read_csv([{files}], header=true, timestampformat='%Y-%m-%d %H:%M:%S UTC')
            WHERE event_type IN ('view', 'purchase') AND user_session IS NOT NULL
              AND event_time::DATE NOT BETWEEN '{GAP[0]}' AND '{GAP[1]}'
            ORDER BY ts
        ) TO '{out.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 1000000)
    """)
    row = con.execute(f"""
        SELECT count(*), sum(is_purchase::INT), count(DISTINCT user), count(DISTINCT item)
        FROM '{out.as_posix()}'
    """).fetchone()
    return {"events": row[0], "purchases": row[1], "users": row[2], "items": row[3]}


def events(con: duckdb.DuckDBPyConnection, path: Path = EVENTS) -> None:
    """Registers the ``events`` view on ``con``."""
    con.execute(f"CREATE OR REPLACE VIEW events AS SELECT * FROM '{path.as_posix()}'")


if __name__ == "__main__":
    download()
    print(build_events())
