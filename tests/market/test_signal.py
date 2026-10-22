import pytest

from market.signal import signal_stats


def _row(adv_p, move, price=-110):
    return {"market": "spreads", "price_american": price, "p_market_novig": adv_p,
            "p_close_novig": adv_p + move}


def test_signal_detects_negative_relationship():
    rows = [_row(0.5 + 0.01 * i, -0.01 * i * 0.8) for i in range(-3, 6)]
    s = signal_stats(rows)
    assert s["n"] == 9 and s["r"] < -0.9 and s["slope"] < 0


def test_signal_ignores_props_and_unclosed_rows():
    rows = [{"market": "player_points", "price_american": -110, "p_market_novig": .5, "p_close_novig": .5},
            {"market": "spreads", "price_american": -110, "p_market_novig": .5, "p_close_novig": None}]
    assert signal_stats(rows)["n"] == 0
