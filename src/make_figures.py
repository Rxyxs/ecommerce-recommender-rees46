"""README figures, drawn only from ``reports/results.json``."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "outputs" / "figures"
INK, GRID, MAIN, ALT, MUTED = "#2B2B2B", "#D9D9D9", "#2a78d6", "#eb6834", "#9aa3ad"
LABELS = {"recent": "recently viewed", "coview": "co-viewed", "als": "ALS",
          "popular": "popular", "ranker": "LightGBM ranker"}


def _style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def methods(r: dict, path: Path):
    m = r["methods"]
    names = ["popular", "als", "coview", "recent", "ranker"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    for ax, metric in zip(axes, ["recall@20", "ndcg@20"]):
        vals = [m[n][metric] for n in names]
        ax.barh([LABELS[n] for n in names], vals, color=[MUTED] * 4 + [MAIN], height=0.6)
        for y, v in enumerate(vals):
            ax.text(v + 0.005, y, f"{v:.3f}", va="center", fontsize=9, color=INK)
        ax.set_title(metric, loc="left", fontsize=11, color=INK)
        ax.set_xlim(0, max(vals) * 1.2)
        _style(ax)
    fig.suptitle(f"Test week (Nov 24-30), {r['evaluated_users']:,} buyers with history",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def ceiling(r: dict, path: Path):
    c = r["candidate_recall"]
    names = ["recent", "als", "coview", "popular", "union"]
    labels = [LABELS.get(n, "all four together") for n in names]
    fig, ax = plt.subplots(figsize=(9, 3.4))
    vals = [c[n] for n in names]
    ax.barh(labels, vals, color=[MUTED] * 4 + [ALT], height=0.6)
    for y, v in enumerate(vals):
        ax.text(v + 0.005, y, f"{v:.0%}", va="center", fontsize=9, color=INK)
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of bought (user, item) pairs found among the candidates")
    ax.set_title("Candidate recall: the ceiling any ranker can reach", loc="left", fontsize=11, color=INK)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def importance(r: dict, path: Path):
    imp = list(r["importance_gain_share"].items())[:10][::-1]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.barh([k for k, _ in imp], [v * 100 for _, v in imp], color=MAIN, height=0.6)
    ax.set_xlabel("% of total gain")
    ax.set_title("What the ranker relies on (top 10 features)", loc="left", fontsize=11, color=INK)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def make_all():
    FIG.mkdir(parents=True, exist_ok=True)
    r = json.loads((ROOT / "reports" / "results.json").read_text(encoding="utf-8"))
    methods(r, FIG / "methods.png")
    ceiling(r, FIG / "candidate_recall.png")
    importance(r, FIG / "importance.png")


if __name__ == "__main__":
    make_all()
