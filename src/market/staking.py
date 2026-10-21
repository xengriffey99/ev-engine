"""Staking (NN7): 1/16 Kelly on the blended probability, 1u = 1% of bankroll.

For a bet at decimal odds d (b = d - 1) won with probability p (no-push basis):
$f^* = \\frac{b p - (1 - p)}{b}$, stake = KELLY_FRACTION x f* x 100 units.
Spec vector (NBA_V2_PLAN Phase 2): p = 0.55 at -110 -> f* = 5.5%, 1/16 -> 0.34u.

Caps:
- per game: 2u across every leg on one game (enforced again by the opinion log);
- shared thesis (approved by Chris 2026-10-21): gate passes from one
  model version, one lane and one direction on one slate (e.g. every home side) ride a
  common model factor, so they are one position (NBA_CONTEXT.md section 12, "collapse
  correlated +EV legs into one position on the shared thesis"). Their stakes are scaled
  pro rata so the group total is at most SHARED_THESIS_CAP_U.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Hashable, Sequence

from . import config as C


def kelly_fraction(p: float, decimal: float) -> float:
    b = decimal - 1.0
    if b <= 0:
        return 0.0
    return max(0.0, (b * p - (1.0 - p)) / b)


def stake_units(p: float, decimal: float, fraction: float = C.KELLY_FRACTION) -> float:
    """Fractional Kelly in units (1u = 1% of bankroll), before caps."""
    return 100.0 * fraction * kelly_fraction(p, decimal)


def cap_groups(stakes: Sequence[float], groups: Sequence[Hashable], cap: float) -> list[float]:
    """Scale stakes pro rata so each group's total is <= cap. Order is preserved."""
    total: dict[Hashable, float] = defaultdict(float)
    for s, g in zip(stakes, groups):
        total[g] += s
    return [s * min(1.0, cap / total[g]) if total[g] > 0 else 0.0 for s, g in zip(stakes, groups)]


def round_units(u: float) -> float:
    """Round down to 0.01u so a cap is never exceeded by rounding."""
    return int(u * 100 + 1e-9) / 100.0
