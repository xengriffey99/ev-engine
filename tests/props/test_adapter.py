import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "market"))
from factories import AWAY, HOME, game, q  # noqa: E402

from props.adapter import build_prop_opinions, main_lines, market_env  # noqa: E402
from props.engine import PlayerIn  # noqa: E402


def fake_sim(points, name="Jayson Tatum", minutes=34.0, two_way=False):
    pl = PlayerIn(name, "BOS", 34, 0.9, 30, 100, 8, 0.37, 7, 0.85, two_way=two_way)
    n = len(points)
    return {"players": {name: {"team": "BOS", "player": pl, "points": np.asarray(points),
                               "minutes": np.full(n, minutes), "active": np.ones(n, dtype=bool)}}}


def board():
    # Main line 26.5 at four books; an alternate 30.5 with an over-only book.
    return [
        q("fanduel", "Over", 26.5, -110), q("fanduel", "Under", 26.5, -110),
        q("draftkings", "Over", 26.5, -115), q("draftkings", "Under", 26.5, -105),
        q("betmgm", "Over", 26.5, -112), q("betmgm", "Under", 26.5, -108),
        q("novig", "Over", 26.5, +120), q("novig", "Under", 26.5, -102),  # big Over price, normal Under
        q("fanduel", "Over", 30.5, +150), q("fanduel", "Under", 30.5, -190),
        q("ballybet", "Over", 24.5, -150),  # one-way: never a main line
    ]


def test_main_line_is_the_most_books_two_way_line():
    assert main_lines(board(), "evt1", "player_points") == {"Jayson Tatum": 26.5}


def test_under_is_never_priced_off_the_overs_best_odds():
    rng = np.random.default_rng(1)
    sim = fake_sim(rng.poisson(22, 20000))  # model loves the Under
    rows, _ = build_prop_opinions(sim, board(), "evt1", books=frozenset({"novig", "draftkings"}))
    op = rows[0].opinion
    assert op.side == "Under"
    assert op.book in {"novig", "draftkings"} and op.price_american < 0  # not novig's +120 Over price
    assert op.p_market_novig < 0.5 + 0.05 and op.ref_quality in ("consensus_retail", "single_retail")


def test_restricted_under_and_hold_are_passes():
    rng = np.random.default_rng(2)
    low_min = fake_sim(rng.poisson(22, 20000), minutes=9.0)
    rows, _ = build_prop_opinions(low_min, board(), "evt1", books=frozenset({"novig", "draftkings"}))
    assert rows[0].opinion.pass_reason == "restricted" and "minutes" in rows[0].opinion.notes
    held, _ = build_prop_opinions(fake_sim(rng.poisson(30, 20000)), board(), "evt1",
                                  books=frozenset({"novig", "draftkings"}), hold=True)
    assert held[0].opinion.pass_reason == "model_hold"


def test_market_env_centre_from_devigged_pinnacle():
    qs = [game("pinnacle", HOME, -4.5, -110), game("pinnacle", AWAY, 4.5, -110),
          game("pinnacle", "Over", 230.5, -110, market="totals"), game("pinnacle", "Under", 230.5, -110, market="totals")]
    env, info = market_env(qs, "evt1", "BOS", "NYK")
    assert env.mu_margin == pytest.approx(4.5) and env.mu_total == pytest.approx(230.5)
    qs[0] = game("pinnacle", HOME, -4.5, -130)  # home favoured beyond the line
    qs[1] = game("pinnacle", AWAY, 4.5, +110)
    env2, _ = market_env(qs, "evt1", "BOS", "NYK")
    assert env2.mu_margin > 4.5 + 1.0
