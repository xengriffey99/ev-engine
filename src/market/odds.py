"""Price conversions, implied probability, overround and hold.

$p_{be} = A/(A+100)$ for American odds $-A$; $\\text{hold} = 1 - 1/\\sum_j \\pi_j$.
"""

from __future__ import annotations

from collections.abc import Sequence


def american_to_decimal(american: float) -> float:
    a = float(american)
    if abs(a) < 100:
        raise ValueError(f"Invalid American odds {american!r}: |odds| must be >= 100")
    return 1.0 + (a / 100.0 if a > 0 else 100.0 / -a)


def decimal_to_american(decimal: float) -> float:
    d = float(decimal)
    if d <= 1.0:
        raise ValueError(f"Invalid decimal odds {decimal!r}: must be > 1")
    if d >= 2.0:
        return 100.0 * (d - 1.0)
    return -100.0 / (d - 1.0)


def implied_prob(decimal: float) -> float:
    d = float(decimal)
    if d <= 1.0:
        raise ValueError(f"Invalid decimal odds {decimal!r}: must be > 1")
    return 1.0 / d


def implied_prob_american(american: float) -> float:
    return implied_prob(american_to_decimal(american))


def overround(implied: Sequence[float]) -> float:
    return float(sum(implied)) - 1.0


def hold(implied: Sequence[float]) -> float:
    return 1.0 - 1.0 / float(sum(implied))


def expected_value(p: float, decimal: float) -> float:
    """EV per unit staked: $p \\cdot d - 1$."""
    return p * decimal - 1.0
