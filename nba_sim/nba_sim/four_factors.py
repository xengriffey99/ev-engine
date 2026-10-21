"""Module 2: Four Factors matchup matrix and opponent-adjusted efficiency."""
from __future__ import annotations

from .config import PRIORS

FACTORS = ("efg", "tov", "orb", "ftr")
# sign: +1 when a higher offensive value is good for the offense
SIGN = {"efg": 1, "tov": -1, "orb": 1, "ftr": 1}
# league SD of team-season factor values, APPROX; replace with fitted values
FACTOR_SD = {"efg": 0.018, "tov": 0.011, "orb": 0.022, "ftr": 0.025}
# points per 100 possessions per 1 SD of combined factor edge, APPROX
PTS_PER_SD = 1.6


def four_factors(fgm, fg3m, fga, tov, fta, ftm, oreb, opp_dreb) -> dict:
    poss_like = fga + 0.44 * fta + tov
    return {
        "efg": (fgm + 0.5 * fg3m) / fga,
        "tov": tov / poss_like,
        "orb": oreb / (oreb + opp_dreb),
        "ftr": ftm / fga,
    }


def factor_edge(off: dict | None, opp_def: dict | None, lg: dict | None) -> tuple[float, dict]:
    """Return (points per 100 adjustment, per-factor z contributions).

    off: offense's factors; opp_def: factors the opposing defense allows; lg: league means.
    Neutral (0.0) when the factor feed is not loaded.
    """
    if not (off and opp_def and lg):
        return 0.0, {}
    contrib, total = {}, 0.0
    w = PRIORS.four_factor_weights
    for f in FACTORS:
        z = SIGN[f] * ((off[f] - lg[f]) + (opp_def[f] - lg[f])) / FACTOR_SD[f]
        contrib[f] = round(w[f] * z, 3)
        total += w[f] * z
    return PRIORS.four_factor_gamma * PTS_PER_SD * total, contrib


def expected_ortg(off_rating: float, opp_def_rating: float, lg: float = PRIORS.league_ortg,
                  factor_adj: float = 0.0) -> float:
    """Additive opponent adjustment plus factor-level mismatch."""
    return off_rating + (opp_def_rating - lg) + factor_adj


def regress_rating(r: float, lg: float = PRIORS.league_ortg, rho: float = PRIORS.net_carryover_rho) -> float:
    return lg + rho * (r - lg)
