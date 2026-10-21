"""Module 1: possession and pace engine (team and lineup adjusted)."""
from __future__ import annotations

from .config import PRIORS


def possessions(fga: float, fta: float, oreb: float, tov: float) -> float:
    return fga + 0.44 * fta - oreb + tov


def game_possessions(home: dict, away: dict) -> float:
    """Average both teams' box-score possession estimates for one game."""
    return 0.5 * (possessions(**home) + possessions(**away))


def regress_pace(pace: float, lg: float = PRIORS.league_pace, rho: float = PRIORS.pace_carryover) -> float:
    return lg + rho * (pace - lg)


def matchup_pace(pace_h: float, pace_a: float, lg: float = PRIORS.league_pace) -> float:
    return pace_h * pace_a / lg


def shrunk_lineup_pace(lineup_pace: float, lineup_poss: float, team_pace: float, k: float = 300) -> float:
    return (lineup_poss * lineup_pace + k * team_pace) / (lineup_poss + k)


def lineup_weighted_pace(lineups: list[dict], team_pace: float, k: float = 300) -> float:
    """lineups: [{'pace': float, 'poss': float, 'proj_min': float}] for expected rotation.

    Projected minutes come from the rotation model (RAPM-weighted substitution pattern).
    Falls back to team pace when no lineup data are loaded.
    """
    if not lineups:
        return team_pace
    num = sum(l["proj_min"] * shrunk_lineup_pace(l["pace"], l["poss"], team_pace, k) for l in lineups)
    den = sum(l["proj_min"] for l in lineups)
    return num / den if den else team_pace
