"""Module 3: coaching philosophy and tactical adjustments.

Inputs are optional; each function returns 0.0 (neutral) when its feed is not loaded so the
engine stays honest about what it knows.
"""
from __future__ import annotations

# League points per shot by zone, APPROX 2025-26; replace with tracking-data values.
LEAGUE_PPS = {"rim": 1.30, "short_mid": 0.85, "long_mid": 0.82, "corner3": 1.17, "above_break3": 1.06, "ft_trip": 1.55}

# Scheme vs ball-handler matrix: points per 100 for the OFFENSE, APPROX starting grid.
SCHEME_MATRIX = {
    ("drop", "pullup_shooter"): +1.2, ("drop", "rim_driver"): -0.6, ("drop", "pnr_passer"): +0.3,
    ("switch", "pullup_shooter"): -0.4, ("switch", "rim_driver"): +0.2, ("switch", "pnr_passer"): -0.3,
    ("blitz", "pullup_shooter"): -0.8, ("blitz", "rim_driver"): -0.2, ("blitz", "pnr_passer"): +0.9,
}


def xpps(shot_mix: dict | None, pps: dict = LEAGUE_PPS) -> float | None:
    """Expected points per shot from a scheme's shot-zone frequencies (sums to 1)."""
    if not shot_mix:
        return None
    return sum(freq * pps[z] for z, freq in shot_mix.items())


def shot_profile_adj(off_mix: dict | None, def_allowed_mix: dict | None, shots_per_100: float = 88.0) -> float:
    """Points per 100: offense's preferred mix blended 50/50 with what the defense concedes, vs league."""
    if not (off_mix and def_allowed_mix):
        return 0.0
    blended = {z: 0.5 * off_mix.get(z, 0) + 0.5 * def_allowed_mix.get(z, 0) for z in LEAGUE_PPS}
    league_mix = {"rim": 0.32, "short_mid": 0.12, "long_mid": 0.06, "corner3": 0.09, "above_break3": 0.29, "ft_trip": 0.12}
    return (xpps(blended) - xpps(league_mix)) * shots_per_100


def scheme_adj(defense_scheme: str | None, handler_type: str | None, usage_share: float = 0.35) -> float:
    if not (defense_scheme and handler_type):
        return 0.0
    return SCHEME_MATRIX.get((defense_scheme, handler_type), 0.0) * usage_share / 0.35


def minutes_cap(base_min: float, rest_days: int, load_managed: bool, age: int | None = None) -> float:
    """Rotation management: cap a player's projected minutes on B2B or load management."""
    m = base_min
    if rest_days == 0:
        m *= 0.92
    if load_managed:
        m = min(m, 28.0)
    if age and age >= 36:
        m = min(m, 32.0)
    return round(m, 1)
