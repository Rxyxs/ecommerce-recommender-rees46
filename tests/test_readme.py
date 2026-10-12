"""Every figure quoted in both READMEs must come from ``reports/results.json``."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
R = json.loads((ROOT / "reports" / "results.json").read_text(encoding="utf-8"))


def _f(x: float, d: int, es: bool) -> str:
    s = f"{x:.{d}f}"
    return s.replace(".", ",") if es else s


def _pct(x: float, es: bool) -> str:
    return _f(x * 100, 1, es) + "%"


def _int(n: int, es: bool) -> str:
    s = f"{n:,}"
    return s.replace(",", ".") if es else s


def expected(es: bool) -> list[str]:
    m = R["methods"]
    best = max((k for k in m if k != "ranker"), key=lambda k: m[k]["recall@20"])
    rec_gain = m["ranker"]["recall@20"] / m[best]["recall@20"] - 1
    ndcg_gain = m["ranker"]["ndcg@20"] / m[best]["ndcg@20"] - 1
    imp = R["importance_gain_share"]
    out = [
        _int(R["events"]["events"], es), _int(R["events"]["purchases"], es),
        _int(R["test_buyers"], es), _int(R["evaluated_users"], es),
        _pct(R["bought_pairs_never_viewed_before"], es), _pct(R["candidate_recall"]["union"], es),
        f"{R['candidates_per_user']:.0f} ", f"{rec_gain * 100:.0f}%", f"{ndcg_gain * 100:.0f}%",
        _pct(imp["ui_views"], es), _pct(imp["item_purchases_7d"], es), _pct(imp["als_score"], es),
        _pct(R["cold_start"]["share_of_buyers"], es), _int(R["cold_start"]["popular"]["users"], es),
        _f(R["cold_start"]["popular"]["recall@20"], 3, es),
    ]
    for k in m:
        out += [_f(m[k][x], 3, es) for x in ("recall@20", "hit@20", "ndcg@20")]
    return out


@pytest.mark.parametrize("name,es", [("README.md", False), ("README.es.md", True)])
def test_readme_numbers(name, es):
    text = (ROOT / name).read_text(encoding="utf-8")
    missing = [s for s in expected(es) if s not in text]
    assert not missing, f"{name} does not quote {missing}"
