"""CLV-by-side summary: graded vs excluded rows, clustering, and the report on a synthetic opinion log."""

import csv
from math import isnan

from market import clv_summary as cs
from market.clv import EXACT, MISSING, SHIFTED, UNPRICED

COLS = ["opinion_id", "game_id", "market", "side", "model_version", "clv_pct", "clv_method"]


def row(i, game, side, clv, method=EXACT, lane="sidequest-nb-1.2", market="player_rebounds"):
    return dict(opinion_id=i, game_id=game, market=market, side=side, model_version=lane,
                clv_pct="" if clv is None else str(clv), clv_method=method)


def test_only_priced_closes_enter_the_mean_and_counts_are_reported():
    rows = [row("a", "g1", "Under", 0.02), row("b", "g2", "Under", -0.01), row("c", "g3", "Under", 0.04, SHIFTED),
            row("d", "g4", "Under", None, MISSING), row("e", "g5", "Under", None, UNPRICED),
            row("f", "g6", "Under", None, "")]
    g = cs.summarize(rows, ("side",))[0]
    assert (g["n_logged"], g["n_closed"], g["n_missing"], g["n_graded"], g["n_exact"]) == (6, 5, 2, 3, 2)
    assert abs(g["mean_clv"] - (0.02 - 0.01 + 0.04) / 3) < 1e-12


def test_sides_are_summarized_separately():
    rows = [row("a", "g1", "Under", 0.03), row("b", "g2", "Under", 0.01),
            row("c", "g3", "Over", -0.02), row("d", "g4", "Over", -0.04)]
    out = {g["key"]: g for g in cs.summarize(rows, ("side",))}
    assert out[("Under",)]["mean_clv"] > 0 > out[("Over",)]["mean_clv"]


def test_clustered_se_is_wider_than_iid_when_one_game_drives_the_rows():
    vals = [0.05, 0.05, 0.05, -0.05, -0.05, -0.05]
    clusters = ["g1", "g1", "g1", "g2", "g2", "g2"]
    m, se = cs.cluster_mean_se(vals, clusters)
    iid = (sum((v - m) ** 2 for v in vals) / (len(vals) - 1) / len(vals)) ** 0.5
    assert abs(m) < 1e-12 and se > iid


def test_single_game_has_no_clustered_se():
    _, se = cs.cluster_mean_se([0.01, 0.02], ["g1", "g1"])
    assert isnan(se)


def test_report_reads_the_csv_and_filters_by_lane(tmp_path):
    p = tmp_path / "opinion_log.csv"
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        w.writeheader()
        w.writerows([row("a", "g1", "Under", 0.02), row("b", "g2", "Over", -0.01),
                     row("c", "g3", "Under", 0.03, lane="market-baseline-v0")])
    md = cs.report(p, model_version_prefix="sidequest")
    assert "CLV by side" in md and "market-baseline-v0" not in md and "| Under |" in md
