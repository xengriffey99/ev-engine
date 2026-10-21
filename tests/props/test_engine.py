import numpy as np
import pytest

from props.engine import (EngineParams, MarketEnv, PlayerIn, allocate_minutes, healthy_allocation,
                          prob_over, restriction, simulate_game, summarize)


def roster(team, star_status="available"):
    # 13 players: a star, four starters, bench; projected MPG sums past 240 like the D&T table.
    spec = [("Star", 35, 34, 10, .37, 8, .85), ("S2", 32, 22, 6, .36, 3, .8), ("S3", 31, 20, 5, .35, 3, .8),
            ("S4", 30, 18, 2, .33, 4, .7), ("S5", 29, 16, 1, .30, 3, .6)]
    spec += [(f"B{i}", 24 - 2 * i, 18, 5, .35, 2.5, .75) for i in range(8)]
    out = []
    for i, (nm, mpg, p100, fg3a, fg3p, fta, ftp) in enumerate(spec):
        out.append(PlayerIn(f"{team}-{nm}", team, mpg, 0.9 if i < 5 else 0.1, p100, 100.0, fg3a, fg3p,
                            fta, ftp, star_status if nm == "Star" else "available"))
    return out


P = EngineParams(n_sims=8000)


def test_minutes_fill_240_and_absences_flow_to_teammates():
    mpg = np.array([pl.mpg for pl in roster("H")], dtype=float)
    full = healthy_allocation(mpg, 240)
    assert full.sum() == pytest.approx(240)
    active = np.ones_like(mpg, dtype=bool)
    active[0] = False
    m = allocate_minutes(mpg, active, P)
    assert m[0] == 0 and m.sum() == pytest.approx(240)
    assert (m[1:5] >= full[1:5] - 1e-9).all() and (m[1:5] <= np.minimum(42, 1.15 * full[1:5]) + 1e-9).all()
    assert m[5:].sum() > full[5:].sum()  # next men up gain the rest


def test_team_points_match_the_market_environment():
    env = MarketEnv("H", "A", mu_margin=4.0, mu_total=230.0)
    sim = simulate_game(env, roster("H"), roster("A"), P)
    home = sum(sp["points"].mean() for sp in sim["players"].values() if sp["team"] == "H")
    assert home == pytest.approx((230 + 4) / 2, abs=1.0)
    mins = sum(sp["minutes"] for sp in sim["players"].values() if sp["team"] == "H")
    assert np.median(mins) == pytest.approx(240, abs=1e-6)


def test_usage_cascade_raises_teammates_when_the_star_sits():
    env = MarketEnv("H", "A", 0.0, 228.0)
    base = simulate_game(env, roster("H"), roster("A"), P)["players"]
    out = simulate_game(env, roster("H", "out"), roster("A"), P)["players"]
    assert out["H-Star"]["points"].mean() == 0
    assert out["H-S2"]["points"].mean() > base["H-S2"]["points"].mean() + 1.0


def test_questionable_star_plays_about_half_the_time():
    sim = simulate_game(MarketEnv("H", "A", 0.0, 228.0), roster("H", "questionable"), roster("A"), P)
    assert summarize(sim["players"]["H-Star"])["p_play"] == pytest.approx(0.5, abs=0.02)


def test_points_are_right_skewed_and_overdispersed_but_not_absurd():
    s = summarize(simulate_game(MarketEnv("H", "A", 0.0, 228.0), roster("H"), roster("A"), P)["players"]["H-Star"])
    assert s["mean"] > s["median"] - 0.5  # skew: mean at or above median
    assert 0.25 < s["sd"] / s["mean"] < 0.45


def test_prob_over_is_monotone_and_no_push():
    pts = np.array([10, 12, 12, 14, 20])
    assert prob_over(pts, 11.5) == pytest.approx(4 / 5)
    assert prob_over(pts, 12) == pytest.approx(2 / 3)  # pushes removed
    assert prob_over(pts, 9.5) > prob_over(pts, 13.5) > prob_over(pts, 25)


def test_blowout_cuts_starter_minutes():
    close = simulate_game(MarketEnv("H", "A", 0.0, 228.0, sd_margin=1.0), roster("H"), roster("A"), P)
    rout = simulate_game(MarketEnv("H", "A", 30.0, 228.0, sd_margin=1.0), roster("H"), roster("A"), P)
    assert rout["players"]["H-Star"]["minutes"].mean() < close["players"]["H-Star"]["minutes"].mean() - 4


def test_2026_27_restrictions():
    assert restriction("player_points", "Under", 10.5, False) is not None
    assert restriction("player_points", "Over", 10.5, False) is None
    assert restriction("player_points", "Under", 30.0, False) is None
    assert restriction("player_points", "Over", 30.0, True) is not None  # two-way player
    assert restriction("player_points", "Under", 30.0, False, is_main_line=False) is not None
    assert restriction("player_turnovers", "Over", 30.0, False) is not None
