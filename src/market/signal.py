"""Daily adverse-selection tracker (added 2026-10-22).

Day 1 finding: when the best bettable price beat the reference fair price, the reference then
moved AGAINST that side (r = -0.42, t = -3.0, n = 44, one slate). This module re-measures it
every day on game markets so the lead is confirmed or killed out of sample.

  adv  = p_market_novig * decimal(price_american) - 1     (price advantage vs the logged reference; fees ignored)
  move = p_close_novig - p_market_novig                   (reference move toward the side, probability points)
"""

from __future__ import annotations

from math import sqrt

from . import config as C
from .odds import american_to_decimal

GAME_MARKETS = C.SPREAD_LIKE_MARKETS | C.TOTAL_LIKE_MARKETS


def signal_stats(rows: list[dict], near_fair: float = -0.005) -> dict:
    xs, ys = [], []
    for r in rows:
        if r["market"] not in GAME_MARKETS or not r.get("p_close_novig") or not r.get("p_market_novig"):
            continue
        d = american_to_decimal(float(r["price_american"]))
        xs.append(float(r["p_market_novig"]) * d - 1.0)
        ys.append(float(r["p_close_novig"]) - float(r["p_market_novig"]))
    n = len(xs)
    out = {"n": n, "r": None, "t": None, "slope": None, "near_fair_n": 0, "near_fair_move": None,
           "other_move": None}
    if n < 3:
        return out
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    if sxx > 0 and syy > 0:
        r = sxy / sqrt(sxx * syy)
        out.update(r=r, slope=sxy / sxx, t=r * sqrt((n - 2) / max(1e-12, 1 - r * r)))
    near = [y for x, y in zip(xs, ys) if x >= near_fair]
    rest = [y for x, y in zip(xs, ys) if x < near_fair]
    out["near_fair_n"] = len(near)
    out["near_fair_move"] = sum(near) / len(near) if near else None
    out["other_move"] = sum(rest) / len(rest) if rest else None
    return out
