import importlib

import pytest

from market import config as C


def test_fetch_lists_stay_at_one_region():
    assert len(C.SIDES_TOTALS_BOOKS) <= 10 and len(set(C.SIDES_TOTALS_BOOKS)) == len(C.SIDES_TOTALS_BOOKS)
    assert len(C.PROPS_BOOKS) <= 10 and len(set(C.PROPS_BOOKS)) == len(C.PROPS_BOOKS)


def test_state_profiles():
    me, ma = C.BETTABLE_BY_STATE["ME"], C.BETTABLE_BY_STATE["MA"]
    assert "kalshi" in me and "kalshi" not in ma  # MA court injunction on sports contracts
    assert "fanduel" not in me and "fanduel" in ma
    assert "pinnacle" not in me | ma and "lowvig" not in me | ma  # reference-only books


def test_every_free_tier_bettable_book_is_fetched():
    free_tier_paid_only = {"williamhill_us", "fanatics"}
    for state, books in C.BETTABLE_BY_STATE.items():
        for b in books - free_tier_paid_only:
            assert b in C.SIDES_TOTALS_BOOKS, (state, b)


def test_bet_state_env(monkeypatch):
    monkeypatch.setenv("NBA_EV_BET_STATE", "ma")
    mod = importlib.reload(C)
    assert mod.BET_STATE == "MA" and mod.BETTABLE_BOOKS == mod.BETTABLE_BY_STATE["MA"]
    monkeypatch.setenv("NBA_EV_BET_STATE", "NH")
    with pytest.raises(ValueError):
        importlib.reload(C)
    monkeypatch.delenv("NBA_EV_BET_STATE")
    assert importlib.reload(C).BET_STATE == "ME"
