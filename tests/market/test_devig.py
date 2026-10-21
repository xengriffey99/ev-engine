import random

import pytest

from market.devig import (
    METHODS,
    DevigError,
    OneWayMarketError,
    additive,
    devig,
    devig_american,
    devig_two_way,
    multiplicative,
    power,
    shin,
)
from market.odds import implied_prob_american

# Reference values computed independently with scipy.optimize.brentq
# (not with this module), rounded to 9 dp.
FIXTURES = {
    (-150, 130): {
        "multiplicative": (0.579831933, 0.420168067),
        "power": (0.583982635, 0.416017365),
        "shin": (0.582608696, 0.417391304),
        "additive": (0.582608696, 0.417391304),
    },
    (-700, 450): {
        "multiplicative": (0.827956989, 0.172043011),
        "power": (0.858150462, 0.141849538),
        "shin": (0.846590909, 0.153409091),
        "additive": (0.846590909, 0.153409091),
    },
    (-125, -105): {
        "multiplicative": (0.520304569, 0.479695431),
        "power": (0.52242582, 0.47757418),
        "shin": (0.521680217, 0.478319783),
        "additive": (0.521680217, 0.478319783),
    },
    (150, 240, 190): {
        "multiplicative": (0.385005857, 0.283092542, 0.331901601),
        "power": (0.386911967, 0.281335746, 0.331752287),
        "shin": (0.386501485, 0.28164061, 0.331857906),
        "additive": (0.387018256, 0.281135903, 0.331845842),
    },
}


@pytest.mark.parametrize("prices", list(FIXTURES))
@pytest.mark.parametrize("method", sorted(METHODS))
def test_matches_independent_reference_to_1e6(prices, method):
    result = devig_american(prices, method)
    assert result.method == method
    for got, want in zip(result.probs, FIXTURES[prices][method]):
        assert got == pytest.approx(want, abs=1e-6)


@pytest.mark.parametrize("method", sorted(METHODS))
@pytest.mark.parametrize("price", [-110, -115, -120])
def test_symmetric_market_is_fifty_fifty(method, price):
    result = devig_american([price, price], method)
    assert result.probs == pytest.approx((0.5, 0.5), abs=1e-12)


@pytest.mark.parametrize("method", sorted(METHODS))
def test_probabilities_sum_to_one(method):
    rng = random.Random(7)
    checked = 0
    for _ in range(400):
        n = rng.choice([2, 2, 2, 3, 4])
        # raw implied probs with overround between 0% and 15%
        fair = [rng.uniform(0.05, 1.0) for _ in range(n)]
        s = sum(fair)
        margin = rng.uniform(1.0, 1.15)
        pis = [min(0.97, f / s * margin) for f in fair]
        if sum(pis) < 1.0:
            continue
        try:
            result = devig(pis, method)
        except DevigError:
            assert method == "additive"  # can go negative on extreme longshots
            continue
        assert sum(result.probs) == pytest.approx(1.0, abs=1e-12)
        assert all(0.0 < p < 1.0 for p in result.probs)
        checked += 1
    assert checked > 300


def test_shin_equals_additive_on_two_way_markets():
    # Research doc (implied package): identical on two-outcome markets.
    rng = random.Random(11)
    for _ in range(500):
        fav = rng.uniform(-1500, -101)
        dog = rng.uniform(100, 1200)
        pis = [implied_prob_american(fav), implied_prob_american(dog)]
        if sum(pis) < 1.0:
            continue
        s, a = shin(pis).probs, additive(pis).probs
        assert s == pytest.approx(a, abs=1e-9)


def test_power_strips_more_from_the_longshot_than_multiplicative():
    pw = devig_american([-700, 450], "power").probs
    mu = devig_american([-700, 450], "multiplicative").probs
    assert pw[1] < mu[1]
    assert pw[0] > mu[0]


def test_one_way_market_is_rejected():
    with pytest.raises(OneWayMarketError):
        devig([implied_prob_american(-110)], "power")
    with pytest.raises(OneWayMarketError):
        devig_two_way(-110, None)
    with pytest.raises(OneWayMarketError):
        devig_two_way(None, 120)


def test_shin_rejects_negative_overround_but_power_handles_it():
    pis = [0.49, 0.49]  # exchange-style book under 100%
    with pytest.raises(DevigError):
        shin(pis)
    result = power(pis)
    assert result.param < 1.0
    assert result.probs == pytest.approx((0.5, 0.5))
    assert sum(multiplicative(pis).probs) == pytest.approx(1.0)


def test_additive_rejects_negative_probability():
    with pytest.raises(DevigError):
        additive([0.90, 0.30, 0.02])  # shift (0.22/3 = 0.073) exceeds the 0.02 longshot


def test_unknown_method_rejected():
    with pytest.raises(ValueError):
        devig([0.5, 0.55], "logit")


def test_invalid_probability_rejected():
    with pytest.raises(DevigError):
        devig([0.0, 1.0], "power")
