import pytest

from market.odds import (
    american_to_decimal,
    decimal_to_american,
    expected_value,
    hold,
    implied_prob_american,
    overround,
)


@pytest.mark.parametrize(
    "american, decimal",
    [(-110, 1.9090909090909092), (100, 2.0), (-100, 2.0), (150, 2.5), (-250, 1.4)],
)
def test_american_decimal_round_trip(american, decimal):
    assert american_to_decimal(american) == pytest.approx(decimal, abs=1e-12)
    back = decimal_to_american(decimal)
    if american in (100, -100):
        assert back == pytest.approx(100.0)
    else:
        assert back == pytest.approx(american, abs=1e-9)


@pytest.mark.parametrize("bad", [0, 50, -99.9])
def test_invalid_american_rejected(bad):
    with pytest.raises(ValueError):
        american_to_decimal(bad)


@pytest.mark.parametrize(
    "price, breakeven, over_round, hold_pct",
    # Research doc, "Hold and breakeven" table.
    [(-110, 0.5238, 0.0476, 0.0455), (-115, 0.5349, 0.0698, 0.0652), (-120, 0.5455, 0.0909, 0.0833)],
)
def test_hold_and_breakeven_match_research_doc(price, breakeven, over_round, hold_pct):
    pis = [implied_prob_american(price)] * 2
    assert pis[0] == pytest.approx(breakeven, abs=5e-5)
    assert overround(pis) == pytest.approx(over_round, abs=5e-5)
    assert hold(pis) == pytest.approx(hold_pct, abs=5e-5)


def test_expected_value():
    assert expected_value(0.55, american_to_decimal(-110)) == pytest.approx(0.05, abs=1e-12)
