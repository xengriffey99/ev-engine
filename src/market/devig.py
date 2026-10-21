"""Two-sided de-vig (NN4): multiplicative, power, Shin, additive.

Inputs are raw implied probabilities $\\pi_i$ for ONE book's complete market
(every outcome, same line, same snapshot). A one-sided market cannot be
de-vigged and raises OneWayMarketError.

- Multiplicative: $p_i = \\pi_i / \\sum_j \\pi_j$
- Power: $p_i = \\pi_i^{k}$ with $k$ solving $\\sum_j \\pi_j^{k} = 1$ (default)
- Additive: $p_i = \\pi_i - (\\sum_j \\pi_j - 1)/n$
- Shin: $p_i = \\frac{\\sqrt{z^2 + 4(1-z)\\pi_i^2/S} - z}{2(1-z)}$, $z$ solving $\\sum p_i = 1$

On two-outcome markets Shin and additive are identical (checked in tests).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from math import sqrt

from .odds import american_to_decimal, implied_prob, overround


class DevigError(ValueError):
    """The market cannot be de-vigged with the requested method."""


class OneWayMarketError(DevigError):
    """Only one side is posted. Edge is never computed off a one-way market."""


@dataclass(frozen=True)
class DevigResult:
    probs: tuple[float, ...]
    method: str
    overround: float
    param: float | None  # k for power, z for Shin, None otherwise


_TOL = 1e-15
_ITERS = 300


def _validate(implied: Sequence[float]) -> tuple[float, ...]:
    pis = tuple(float(x) for x in implied)
    if len(pis) < 2:
        raise OneWayMarketError(
            f"De-vig needs every side of the market; got {len(pis)} side(s)"
        )
    for x in pis:
        if not 0.0 < x < 1.0:
            raise DevigError(f"Implied probability {x!r} outside (0, 1)")
    return pis


def multiplicative(implied: Sequence[float]) -> DevigResult:
    pis = _validate(implied)
    s = sum(pis)
    return DevigResult(tuple(x / s for x in pis), "multiplicative", s - 1.0, None)


def additive(implied: Sequence[float]) -> DevigResult:
    pis = _validate(implied)
    s = sum(pis)
    shift = (s - 1.0) / len(pis)
    probs = tuple(x - shift for x in pis)
    if any(p <= 0.0 for p in probs):
        raise DevigError("Additive de-vig produced a non-positive probability")
    return DevigResult(probs, "additive", s - 1.0, None)


def power(implied: Sequence[float]) -> DevigResult:
    pis = _validate(implied)

    def excess(k: float) -> float:
        return sum(x**k for x in pis) - 1.0  # strictly decreasing in k

    lo, hi = 1e-9, 1.0
    while excess(hi) > 0.0:  # sum > 1 at k=1 means k > 1
        hi *= 2.0
        if hi > 1e6:
            raise DevigError("Power de-vig failed to bracket k")
    for _ in range(_ITERS):
        mid = 0.5 * (lo + hi)
        if excess(mid) > 0.0:
            lo = mid
        else:
            hi = mid
        if hi - lo < _TOL:
            break
    k = 0.5 * (lo + hi)
    probs = tuple(x**k for x in pis)
    s = sum(probs)
    probs = tuple(p / s for p in probs)  # remove residual solver error
    return DevigResult(probs, "power", sum(pis) - 1.0, k)


def shin(implied: Sequence[float]) -> DevigResult:
    pis = _validate(implied)
    s = sum(pis)
    if s < 1.0:
        raise DevigError("Shin de-vig is undefined for a negative overround")

    def probs_at(z: float) -> tuple[float, ...]:
        return tuple(
            (sqrt(z * z + 4.0 * (1.0 - z) * x * x / s) - z) / (2.0 * (1.0 - z))
            for x in pis
        )

    lo, hi = 0.0, 1.0 - 1e-12  # sum(p) >= 1 at z=0, < 1 as z -> 1
    for _ in range(_ITERS):
        mid = 0.5 * (lo + hi)
        if sum(probs_at(mid)) > 1.0:
            lo = mid
        else:
            hi = mid
        if hi - lo < _TOL:
            break
    z = 0.5 * (lo + hi)
    probs = probs_at(z)
    total = sum(probs)
    probs = tuple(p / total for p in probs)
    return DevigResult(probs, "shin", s - 1.0, z)


METHODS: dict[str, Callable[[Sequence[float]], DevigResult]] = {
    "multiplicative": multiplicative,
    "power": power,
    "shin": shin,
    "additive": additive,
}


def devig(implied: Sequence[float], method: str = "power") -> DevigResult:
    try:
        fn = METHODS[method]
    except KeyError as exc:
        raise ValueError(f"Unknown de-vig method {method!r}; use one of {sorted(METHODS)}") from exc
    return fn(implied)


def devig_decimal(prices: Sequence[float], method: str = "power") -> DevigResult:
    return devig([implied_prob(d) for d in prices], method)


def devig_american(prices: Sequence[float], method: str = "power") -> DevigResult:
    return devig_decimal([american_to_decimal(a) for a in prices], method)


def devig_two_way(
    price_a: float | None,
    price_b: float | None,
    method: str = "power",
    *,
    american: bool = True,
) -> DevigResult:
    """De-vig one book's two-way market. Both sides are required."""
    if price_a is None or price_b is None:
        raise OneWayMarketError("Two-way de-vig needs both sides from the same book")
    if american:
        return devig_american([price_a, price_b], method)
    return devig_decimal([price_a, price_b], method)


__all__ = [
    "DevigError",
    "OneWayMarketError",
    "DevigResult",
    "METHODS",
    "multiplicative",
    "power",
    "shin",
    "additive",
    "devig",
    "devig_decimal",
    "devig_american",
    "devig_two_way",
    "overround",
]
