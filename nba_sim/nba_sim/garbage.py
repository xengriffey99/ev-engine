"""Garbage-time filter: drop Q4 possessions where the leader's live win probability > 98%."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

GAME_SECONDS = 2880
MARGIN_SD = 12.8


def live_win_prob(lead: np.ndarray, seconds_left: np.ndarray, pregame_spread_home: float = 0.0) -> np.ndarray:
    """Home win probability from current lead and time left (Brownian-motion approximation)."""
    frac = np.clip(seconds_left / GAME_SECONDS, 1e-4, 1)
    drift = -pregame_spread_home * frac  # expected remaining margin from pregame line
    return norm.cdf((lead + drift) / (MARGIN_SD * np.sqrt(frac)))


def filter_garbage(pbp: pd.DataFrame, threshold: float = 0.98, spread_col: str | None = None) -> pd.DataFrame:
    """pbp columns: period, seconds_left_game, home_lead (+ optional pregame home spread column)."""
    spread = pbp[spread_col].to_numpy() if spread_col else 0.0
    wp = live_win_prob(pbp["home_lead"].to_numpy(), pbp["seconds_left_game"].to_numpy(), spread)
    leader_wp = np.maximum(wp, 1 - wp)
    garbage = (pbp["period"] >= 4) & (leader_wp > threshold)
    out = pbp.loc[~garbage].copy()
    out.attrs["garbage_rows_dropped"] = int(garbage.sum())
    return out
