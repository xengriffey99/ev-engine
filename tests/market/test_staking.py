import pytest

from market import config as C
from market.odds import american_to_decimal
from market.staking import cap_groups, kelly_fraction, round_units, stake_units


def test_spec_kelly_vector():
    # NBA_V2_PLAN Phase 2: p = 0.55 at -110 -> f* = 5.5%, 1/16 -> 0.34% of bankroll.
    d = american_to_decimal(-110)
    assert kelly_fraction(0.55, d) == pytest.approx(0.055, abs=1e-3)
    assert stake_units(0.55, d) == pytest.approx(0.34, abs=0.01)
    assert C.KELLY_FRACTION == pytest.approx(1 / 16)


def test_no_edge_no_stake():
    d = american_to_decimal(-110)
    assert kelly_fraction(0.50, d) == 0.0
    assert stake_units(0.52, d) == 0.0  # 0.52 x 1.909 < 1
    assert kelly_fraction(0.9, 1.0) == 0.0


def test_shared_thesis_cap_scales_only_groups_over_the_cap():
    stakes = [0.8, 0.7, 0.9, 0.5]
    groups = ["home", "home", "home", "over"]
    out = cap_groups(stakes, groups, 2.0)
    assert sum(out[:3]) == pytest.approx(2.0)
    assert out[0] / out[1] == pytest.approx(0.8 / 0.7)  # pro rata
    assert out[3] == 0.5  # under the cap: untouched


def test_round_down_never_breaks_a_cap():
    assert round_units(0.6666) == 0.66
    assert sum(round_units(x) for x in cap_groups([1.0, 1.0, 1.0], [1, 1, 1], 2.0)) <= 2.0
