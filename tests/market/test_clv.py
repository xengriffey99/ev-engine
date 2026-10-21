from math import erf, sqrt

import pytest
from factories import AWAY, HOME, game, q

from market.clv import (
    EXACT,
    MISSING,
    SHIFTED,
    UNPRICED,
    bet_threshold,
    clv_exact,
    compute_clv,
    shift_prob,
    win_push,
)


def Phi(x):
    return 0.5 * (1 + erf(x / sqrt(2)))


def test_clv_exact_definition():
    # p_close 0.55 at -110 -> 0.55 * 1.90909 - 1 = +5.0%
    assert clv_exact(0.55, -110) == pytest.approx(0.05, abs=1e-12)
    # Fair close at the price taken -> zero CLV
    assert clv_exact(1 / 1.90909090909, -110) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("line", [220.5, 220.0])
def test_shift_is_identity_at_same_line(line):
    for p in (0.3, 0.5, 0.62):
        got = shift_prob(p, line, line, market="totals", side="Over", sigma=18.0)
        assert got == pytest.approx(p, abs=1e-9)


def test_shift_half_point_lines_matches_closed_form():
    # Close Over 222.5 at 50% -> mu = 222.5; P(Over 220.5) = Phi(2/18)
    got = shift_prob(0.5, 222.5, 220.5, market="totals", side="Over", sigma=18.0)
    assert got == pytest.approx(Phi(2 / 18), abs=1e-9)
    clv = clv_exact(got, -110)
    assert clv == pytest.approx(Phi(2 / 18) * (1 + 100 / 110) - 1, abs=1e-9)
    assert clv > 0  # bettor beat the move


def test_shift_integer_bet_line_handles_push():
    # Close Over 220.5 at 50% -> mu = 220.5. Bet Over 220 (push on exactly 220).
    p_win = 1 - Phi(0.0)
    p_push = Phi(0.0) - Phi(-1 / 18)
    expected = p_win / (1 - p_push)
    got = shift_prob(0.5, 220.5, 220.0, market="totals", side="Over", sigma=18.0)
    assert got == pytest.approx(expected, abs=1e-9)
    w, p = win_push(220.0, 220.5, 18.0)
    assert p > 0 and w == pytest.approx(p_win)


def test_shift_under_frame():
    # Bet Under 220.5, close Under 218.5 at 50%: market moved toward the under,
    # so Under 220.5 is MORE likely than 50%.
    got = shift_prob(0.5, 218.5, 220.5, market="totals", side="Under", sigma=18.0)
    assert got == pytest.approx(Phi(2 / 18), abs=1e-9)


def test_shift_spread_frame():
    # Bet favorite -4.5; closed -6.5 at 50% -> P(cover -4.5) = Phi(2/12)
    got = shift_prob(0.5, -6.5, -4.5, market="spreads", side=HOME, sigma=12.0)
    assert got == pytest.approx(Phi(2 / 12), abs=1e-9)
    # Dog +4.5 that closed +3.5 at 50%: the dog bettor got the better number
    got_dog = shift_prob(0.5, 3.5, 4.5, market="spreads", side=AWAY, sigma=12.0)
    assert got_dog == pytest.approx(Phi(1 / 12), abs=1e-9)


def test_bet_threshold_rejects_props():
    with pytest.raises(ValueError):
        bet_threshold("player_points", "Over", 24.5)


# --- compute_clv against a close snapshot ----------------------------------------
def _sides_close(line=4.5, home_px=-110, away_px=-110, book="pinnacle"):
    return [
        game(book, HOME, -line, home_px, snapshot_id="close"),
        game(book, AWAY, line, away_px, snapshot_id="close"),
    ]


def test_compute_clv_exact_line_uses_bet_side():
    close = _sides_close(4.5, -120, +100)  # money came on the HOME side
    res = compute_clv(close, event_id="evt1", market="spreads", side=AWAY, bet_line=4.5, bet_price_american=-105)
    assert res.clv_method == EXACT
    assert res.ref_quality == "sharp"
    assert res.p_close_novig < 0.5
    assert res.clv_pct == pytest.approx(res.p_close_novig * (1 + 100 / 105) - 1, abs=1e-12)
    assert res.clv_pct < 0  # bet the side the market moved against
    assert res.close_price == pytest.approx(100.0)


def test_compute_clv_shifts_when_line_moves():
    close = _sides_close(3.5)  # AWAY closed +3.5 (we got +4.5)
    res = compute_clv(close, event_id="evt1", market="spreads", side=AWAY, bet_line=4.5, bet_price_american=-110)
    assert res.clv_method == SHIFTED
    assert res.close_line == 3.5
    assert res.p_close_novig == pytest.approx(Phi(1 / 12), abs=1e-6)
    assert res.clv_pct > 0


def test_compute_clv_props_line_moved_is_unpriced():
    close = [
        q("fanduel", "Over", 25.5, -110, snapshot_id="close"),
        q("fanduel", "Under", 25.5, -110, snapshot_id="close"),
    ]
    res = compute_clv(close, event_id="evt1", market="player_points", side="Under", bet_line=24.5,
                      bet_price_american=-110, player="Jayson Tatum")
    assert res.clv_method == UNPRICED
    assert res.clv_pct is None and res.p_close_novig is None
    assert res.close_line == 25.5  # still recorded


def test_compute_clv_under_prop_exact_uses_under_close():
    close = [
        q("fanduel", "Over", 24.5, -140, snapshot_id="close"),
        q("fanduel", "Under", 24.5, +115, snapshot_id="close"),
    ]
    res = compute_clv(close, event_id="evt1", market="player_points", side="Under", bet_line=24.5,
                      bet_price_american=-110, player="Jayson Tatum")
    assert res.clv_method == EXACT
    assert res.p_close_novig < 0.5  # the Under's no-vig prob, not the Over's
    assert res.clv_pct < 0
    assert res.ref_quality == "single_retail"


def test_compute_clv_missing_close():
    res = compute_clv([], event_id="evt1", market="spreads", side=AWAY, bet_line=4.5, bet_price_american=-110)
    assert res.clv_method == MISSING and res.clv_pct is None


def test_compute_clv_stale_close_is_not_used():
    close = [
        game("pinnacle", HOME, -4.5, -110, snapshot_id="close", age_min=45),
        game("pinnacle", AWAY, 4.5, -110, snapshot_id="close", age_min=45),
    ]
    res = compute_clv(close, event_id="evt1", market="spreads", side=AWAY, bet_line=4.5, bet_price_american=-110)
    assert res.clv_method == MISSING
