"""Module 7: Monte Carlo game-script engine (bivariate normal, integer scores, overtime)."""
from __future__ import annotations

import numpy as np

from .config import SIM


def simulate(mean_margin: float, mean_total: float, n: int = SIM.n_sims, seed: int = SIM.seed,
             team_sd: float = SIM.team_sd, rho: float = SIM.rho, calibrate_ot: bool = True) -> dict:
    """Market totals include overtime, so the regulation mean is lowered by the expected OT points
    (estimated with a pilot run on the same seed) so the full-game mean matches mean_total."""
    if calibrate_ot:
        pilot = _simulate(mean_margin, mean_total, n, seed, team_sd, rho)
        mean_total -= float(pilot["total"].mean()) - mean_total
    return _simulate(mean_margin, mean_total, n, seed, team_sd, rho)


def _simulate(mean_margin, mean_total, n, seed, team_sd, rho) -> dict:
    rng = np.random.default_rng(seed)
    mu = np.array([(mean_total + mean_margin) / 2, (mean_total - mean_margin) / 2])
    cov = team_sd ** 2 * np.array([[1, rho], [rho, 1]])
    s = np.rint(rng.multivariate_normal(mu, cov, size=n)).astype(int)
    home, away = s[:, 0], s[:, 1]
    tied = home == away
    ot_periods = np.zeros(n, dtype=int)
    while tied.any():  # resolve ties with overtime periods
        k = int(tied.sum())
        home[tied] += np.rint(rng.normal(SIM.ot_mean_pts, SIM.ot_sd_pts, k)).astype(int)
        away[tied] += np.rint(rng.normal(SIM.ot_mean_pts, SIM.ot_sd_pts, k)).astype(int)
        ot_periods[tied] += 1
        tied = home == away
    return {"home": home, "away": away, "margin": home - away, "total": home + away, "ot": ot_periods}


def summarize(sim: dict) -> dict:
    m, t = sim["margin"], sim["total"]
    return {
        "home_win_prob": float((m > 0).mean()),
        "exp_margin": float(m.mean()),
        "sd_margin": float(m.std()),
        "exp_total": float(t.mean()),
        "sd_total": float(t.std()),
        "ot_rate": float((sim["ot"] > 0).mean()),
        "exp_home_pts": float(sim["home"].mean()),
        "exp_away_pts": float(sim["away"].mean()),
    }


def cover_probs(sim: dict, home_spread: float) -> dict:
    """Home covers when margin + home_spread > 0."""
    adj = sim["margin"] + home_spread
    return {"home": float((adj > 0).mean()), "push": float((adj == 0).mean()), "away": float((adj < 0).mean())}


def total_probs(sim: dict, line: float) -> dict:
    t = sim["total"]
    return {"over": float((t > line).mean()), "push": float((t == line).mean()), "under": float((t < line).mean())}


def histogram(values: np.ndarray, lo: int, hi: int, step: int = 2) -> list[dict]:
    edges = np.arange(lo, hi + step, step)
    counts, _ = np.histogram(values, bins=edges)
    return [{"bin": int(e), "pct": round(float(c) / len(values), 5)} for e, c in zip(edges[:-1], counts)]
