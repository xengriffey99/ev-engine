import numpy as np
import pandas as pd
import pytest

from nba_sim import backtest, garbage, ledger, market, montecarlo as mc, pricing
from nba_sim.kalman import LeagueKalman, TeamState, fuse


def test_breakeven_table_matches_context_doc():
    assert round(pricing.implied_prob(-110), 4) == 0.5238
    assert round(pricing.implied_prob(-105), 4) == 0.5122
    assert pricing.implied_prob(100) == 0.5


def test_devig_sums_to_one():
    for a, b in [(-110, -110), (-179, 152), (-300, 240)]:
        p, q = pricing.devig(a, b)
        assert abs(p + q - 1) < 1e-9


def test_ev_and_quarter_kelly():
    # p=0.55 at -110: EV = 0.55*1.9091 - 1 = +5.0%; full Kelly 5.5%; quarter = 1.38u
    assert abs(pricing.ev_pct(0.55, -110) - 0.0500) < 1e-3
    assert pricing.kelly_units(0.55, -110) == pytest.approx(1.38, abs=0.01)
    assert pricing.kelly_units(0.80, +200) == 2.5          # cap
    assert pricing.kelly_units(0.45, -110) == 0.0          # negative edge


def test_simulation_moments_and_ot_calibration():
    sim = mc.simulate(3.0, 225.0, n=40_000)
    s = mc.summarize(sim)
    assert abs(s["exp_margin"] - 3.0) < 0.3
    assert abs(s["exp_total"] - 225.0) < 0.3                # OT-calibrated full-game mean
    assert 12.0 < s["sd_margin"] < 13.6
    assert 17.5 < s["sd_total"] < 19.8
    assert (sim["margin"] != 0).all()                        # no ties after OT


def test_rlm_over_public_total_drops():
    r = market.detect_rlm(222.5, 221.75, 0.75, public_side_sign=-1)   # public Over, total fell
    assert r["flag"]
    r2 = market.detect_rlm(222.5, 223.5, 0.75, public_side_sign=-1)   # moved against Over bettors
    assert not r2["flag"] and not r2["sub_threshold"]


def test_sharp_handle_rule():
    assert market.sharp_handle(0.62, 0.35)
    assert not market.sharp_handle(0.66, 0.51)


def test_key_numbers_favourite_flip():
    assert market.key_numbers_crossed(-2.5, 1.5) == [1.5]
    assert market.key_numbers_crossed(-1.5, -3.0) == [2.5]


def test_fusion_weights():
    f = fuse(0.0, 4.0, sd_model=4.0, sd_market=1.5)
    assert abs(f["w_market"] - 16 / (16 + 2.25)) < 1e-9


def test_kalman_update_moves_toward_result():
    k = LeagueKalman({"A": TeamState(0, 16), "B": TeamState(0, 16)}, hca=0)
    k.update("A", "B", 20.0)
    assert k.s["A"].theta > 0 > k.s["B"].theta and k.s["A"].var < 16


def test_garbage_filter_drops_blowout_q4():
    df = pd.DataFrame({"period": [4, 4, 2], "seconds_left_game": [120, 600, 1800], "home_lead": [25, 2, 25]})
    out = garbage.filter_garbage(df)
    assert out.attrs["garbage_rows_dropped"] == 1


def test_ledger_grading_and_roi():
    e = {"market": "spread", "side": "home", "line": 3.5, "price": -105, "stake": 1.0}
    w = ledger.grade(e, 100, 102)       # home lost by 2, +3.5 covers
    assert w["status"] == "win" and ledger.pl_units(w) == pytest.approx(0.95, abs=0.01)


def test_backtest_metrics_on_synthetic_calibrated_model():
    rng = np.random.default_rng(1)
    p = rng.uniform(0.35, 0.65, 4000)
    y = (rng.uniform(size=4000) < p).astype(int)
    recs = [{"market": "spread", "p_model": float(pi), "p_market_novig": 0.5, "outcome": int(yi)} for pi, yi in zip(p, y)]
    res = backtest.evaluate(recs)
    assert res["brier_skill_vs_market"] > 0 and res["ece"] < 0.03
