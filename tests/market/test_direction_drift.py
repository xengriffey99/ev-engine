"""Directional drift / saturation checks, plus regression tests on direction-aware selection (spec section 12).

Numbers come from the 11/3 direction audit (ledger 10/30 grade; 10/30 REB+AST pool transcribed from SCREEN_1030).
"""

import random
from datetime import datetime, timezone

import numpy as np
import pytest

from market import direction_drift as dd
from market import selection as sel
from market.devig import OneWayMarketError
from market.quotes import Quote
from props import engine as eng


def sides(n_under, n_over):
    xs = ["Under"] * n_under + ["Over"] * n_over
    random.Random(7).shuffle(xs)
    return xs


# ---- Under share vs null -----------------------------------------------------------------------------------
def test_balanced_pool_is_ok():
    assert dd.under_share_test(sides(50, 50)).verdict == "OK"


def test_17_of_17_unders_is_saturated():
    # ledger: every lean since 10/28 (17) is an Under
    r = dd.under_share_test(sides(17, 0))
    assert r.verdict == "SATURATED" and r.p_value < 1e-4


def test_low_n_is_never_ok():
    assert dd.under_share_test(sides(3, 0)).verdict == "LOW_N"


def test_lean_list_is_drift_vs_50_but_not_vs_the_pool_null():
    s = sides(34, 10)  # 11/2 REB/AST lean list: 34 of 44 Unders
    assert dd.under_share_test(s, null_share=0.5).verdict == "DRIFT"
    assert dd.under_share_test(s, null_share=0.63).verdict == "OK"  # the 10/30 pool was 63% Under


def test_unknown_side_label_raises():
    with pytest.raises(ValueError):
        dd.under_share_test(["Under", "under"])


def test_clustered_se_exceeds_iid_when_games_are_correlated():
    ss = ["Under"] * 9 + ["Over"] * 9 + ["Under"] * 9 + ["Over"] * 9
    cl = ["g1"] * 9 + ["g2"] * 9 + ["g3"] * 9 + ["g4"] * 9
    p, se = dd.clustered_share(ss, cl)
    assert p == 0.5 and se > 3 * (p * (1 - p) / len(ss)) ** 0.5


# ---- Funnel ------------------------------------------------------------------------------------------------
def test_funnel_flags_stage_that_jumps_vs_pool():
    st = {"pool": sides(114, 67), "P>=61%": sides(42, 9), "card": sides(4, 0)}  # 10/30 REB+AST: 63% -> 82% -> 4/4
    rep = dd.funnel_report(st)
    assert rep[1]["flag"] == "ASYMMETRIC_STAGE"
    assert rep[2]["flag"] == "LOW_N"  # 4 of 4 Unders cannot be waved through as ok


def test_funnel_balanced_has_no_flags():
    st = {"pool": sides(60, 60), "gate": sides(30, 30), "card": sides(5, 5)}
    assert all(r["flag"] == "" for r in dd.funnel_report(st))


# ---- Pool symmetry -----------------------------------------------------------------------------------------
def test_pool_with_one_sided_candidates_is_reported():
    c = [dict(game="g", player="a", market="reb", line=4.5, side="Over"),
         dict(game="g", player="a", market="reb", line=4.5, side="Under"),
         dict(game="g", player="b", market="reb", line=2.5, side="Under")]
    assert dd.missing_side_keys(c) == [("g", "b", "reb", 2.5)]


# ---- Baselines ---------------------------------------------------------------------------------------------
def test_model_vs_always_under_10_30_is_not_distinguishable():
    r = dd.baseline_gap(40, 77)  # model Overs 40-37 are the only rows that differ from always-Under
    assert r["diff"] == 3 and r["verdict"] == "NOT_DISTINGUISHABLE" and abs(r["z"] - 0.34) < 0.01


def test_baseline_gap_detects_real_separation():
    assert dd.baseline_gap(60, 77)["verdict"] == "DISTINGUISHABLE"


def test_baseline_gap_with_no_disagreement_is_low_n():
    assert dd.baseline_gap(0, 0)["verdict"] == "LOW_N"


# ---- Mean offset -------------------------------------------------------------------------------------------
def test_persistent_negative_mean_offset_is_drift():
    gap = np.random.default_rng(1).normal(-0.18, 0.9, 181).tolist()  # 10/30 REB+AST offset at VM 1.3
    assert dd.mean_offset_report(gap, t_crit=2.5)["verdict"] == "DRIFT"


def test_zero_mean_offset_is_ok():
    assert dd.mean_offset_report(np.random.default_rng(2).normal(0.0, 0.9, 200).tolist())["verdict"] == "OK"


def test_mean_offset_small_sample_is_low_n():
    assert dd.mean_offset_report([-0.2] * 10)["verdict"] == "LOW_N"


# ---- Skew null ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("line,lo,hi", [(2.5, 0.55, 0.57), (4.5, 0.54, 0.55), (6.5, 0.53, 0.545), (12.5, 0.52, 0.53)])
def test_mean_on_line_gives_under_above_50_at_vm13(line, lo, hi):
    assert lo <= dd.p_under_mean_on_line(line, 1.3) <= hi


def test_dispersion_widens_the_skew():
    for line in (3.5, 6.5, 10.5):
        assert dd.p_under_mean_on_line(line, 1.6) > dd.p_under_mean_on_line(line, 1.3) > dd.p_under_mean_on_line(line, 1.15)


# ---- Direction block ---------------------------------------------------------------------------------------
def test_direction_block_reports_every_check_with_a_verdict():
    md = dd.direction_block_md(
        "10/30 REB+AST", sides(114, 67), selected_sides=sides(4, 0),
        stages={"pool": sides(114, 67), "P>=61%": sides(42, 9), "card": sides(4, 0)},
        baseline=(40, 77), mean_gaps=np.random.default_rng(3).normal(-0.18, 0.9, 181).tolist())
    for needle in ("Pool Under share", "Selected Under share vs pool null", "NOT_DISTINGUISHABLE", "ASYMMETRIC_STAGE",
                   "Mean offset"):
        assert needle in md


def test_direction_block_with_optional_parts_missing_does_not_crash():
    md = dd.direction_block_md("pool only", sides(5, 5))
    assert "Pool Under share" in md and "Funnel" not in md


def test_direction_block_flags_saturated_card():
    assert "SATURATED" in dd.direction_block_md("card", sides(60, 40), selected_sides=sides(17, 0))


def test_direction_block_low_n_selection_never_prints_nan():
    md = dd.direction_block_md("card", sides(60, 40), selected_sides=sides(3, 0))
    assert "LOW_N" in md and "nan" not in md


# ---- Regression: direction-aware snapshot selection (NBA_CONTEXT section 12) --------------------------------
def _q(book, side, dec, line=5.5, player="X Player"):
    t = datetime(2026, 11, 3, 17, 0, tzinfo=timezone.utc)
    return Quote(snapshot_id="s1", fetched_at=t, event_id="e1", commence_time=t, home_team="H", away_team="A",
                 book=book, market="player_rebounds", side=side, line=line, price_decimal=dec, player=player,
                 market_last_update=t)


KW = dict(event_id="e1", market="player_rebounds", line=5.5, player="X Player")


def test_best_price_is_direction_aware():
    qs = [_q("draftkings", "Over", 2.10), _q("fanduel", "Over", 1.95),
          _q("draftkings", "Under", 1.70), _q("fanduel", "Under", 1.80)]
    pick = sel.best_price(qs, side="Under", **KW)
    assert (pick.side, pick.book, pick.price_decimal) == ("Under", "fanduel", 1.80)
    assert sel.best_price(qs, side="Over", **KW).price_decimal == 2.10


def test_best_price_never_falls_back_to_the_other_side():
    with pytest.raises(sel.NoQuoteError):
        sel.best_price([_q("draftkings", "Over", 2.10)], side="Under", **KW)


def test_one_way_market_never_yields_a_reference():
    qs = [_q("draftkings", "Over", 2.10), _q("fanduel", "Over", 1.95)]
    with pytest.raises(OneWayMarketError):
        sel.reference_prob(qs, side="Over", **KW)


def test_reference_probs_of_both_sides_sum_to_one():
    qs = [_q("draftkings", "Over", 1.87), _q("draftkings", "Under", 1.95),
          _q("fanduel", "Over", 1.91), _q("fanduel", "Under", 1.91)]
    po = sel.reference_prob(qs, side="Over", **KW).p
    pu = sel.reference_prob(qs, side="Under", **KW).p
    assert abs(po + pu - 1.0) < 1e-9


def test_prob_over_push_handling_is_symmetric():
    pts = np.array([1, 2, 3, 3, 4, 5, 5, 6, 7, 8])
    under_nopush = float((pts < 3).mean()) / (1 - float((pts == 3).mean()))
    assert abs(eng.prob_over(pts, 3) + under_nopush - 1.0) < 1e-12


def test_restrictions_only_ever_remove_unders():
    assert eng.restriction("player_rebounds", "Under", 10.0, False) is not None
    assert eng.restriction("player_rebounds", "Over", 10.0, False) is None
