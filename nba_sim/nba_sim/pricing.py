"""Odds conversion, no-vig pricing, EV and fractional Kelly."""
from __future__ import annotations

import math
from scipy.optimize import brentq

from .config import BANKROLL


def american_to_decimal(a: float) -> float:
    return 1 + (a / 100 if a > 0 else 100 / abs(a))


def decimal_to_american(d: float) -> float:
    if d <= 1:
        raise ValueError("decimal odds must exceed 1.0")
    return round((d - 1) * 100) if d >= 2 else round(-100 / (d - 1))


def implied_prob(a: float) -> float:
    return 1 / american_to_decimal(a)


def prob_to_american(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return decimal_to_american(1 / p)


def devig_multiplicative(a_side: float, a_other: float) -> tuple[float, float]:
    pa, pb = implied_prob(a_side), implied_prob(a_other)
    s = pa + pb
    return pa / s, pb / s


def devig_power(a_side: float, a_other: float) -> tuple[float, float]:
    """Power method: find k with pa^k + pb^k = 1. Better for lopsided prices."""
    pa, pb = implied_prob(a_side), implied_prob(a_other)
    k = brentq(lambda k: pa ** k + pb ** k - 1, 0.5, 2.5)
    return pa ** k, pb ** k


def devig(a_side: float, a_other: float) -> tuple[float, float]:
    if min(a_side, a_other) <= -200:
        return devig_power(a_side, a_other)
    return devig_multiplicative(a_side, a_other)


def hold(a_side: float, a_other: float) -> float:
    return implied_prob(a_side) + implied_prob(a_other) - 1


def ev_pct(p_win: float, american: float, p_push: float = 0.0) -> float:
    """EV per unit staked. With pushes: p_win*(d-1) - p_loss."""
    d = american_to_decimal(american)
    p_loss = 1 - p_win - p_push
    return p_win * (d - 1) - p_loss


def kelly_units(p_win: float, american: float, p_push: float = 0.0,
                fraction: float | None = None, cap: float | None = None) -> float:
    """Fractional Kelly in units (1u = 1% bankroll), capped. Push-adjusted."""
    fraction = BANKROLL.kelly_fraction if fraction is None else fraction
    cap = BANKROLL.max_units if cap is None else cap
    b = american_to_decimal(american) - 1
    p_loss = 1 - p_win - p_push
    f_full = (b * p_win - p_loss) / b
    if f_full <= 0:
        return 0.0
    return round(min(fraction * f_full * 100, cap), 2)


def fair_american(p_win: float, p_push: float = 0.0) -> float:
    """Fair price ignoring pushes (stake returned)."""
    denom = 1 - p_push
    return prob_to_american(p_win / denom if denom > 0 else p_win)


def ev_threshold(market: str) -> float:
    return {"spread": BANKROLL.ev_min_sides_totals, "total": BANKROLL.ev_min_sides_totals,
            "moneyline": BANKROLL.ev_min_moneyline, "prop": BANKROLL.ev_min_props}[market]


def breakeven_prob(american: float, ev_req: float = 0.0) -> float:
    return (1 + ev_req) / american_to_decimal(american)


def round_or_none(x, nd=4):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(x, nd)
