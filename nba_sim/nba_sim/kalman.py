"""Team-strength state space and market-as-sensor fusion (shrink toward market before Kelly)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import FUSION, PRIORS


@dataclass
class TeamState:
    theta: float      # net rating, points per 100
    var: float        # state variance


def offseason_prior(last_net: float, roster_delta: float, sd_carry: float = 2.0, sd_roster: float = 2.5,
                    rho: float = PRIORS.net_carryover_rho) -> TeamState:
    return TeamState(theta=rho * last_net + roster_delta, var=sd_carry ** 2 + sd_roster ** 2)


class LeagueKalman:
    """Scalar-per-team Kalman filter on game margins. y = th_H - th_A + HCA + v."""

    def __init__(self, states: dict[str, TeamState], process_var: float = 0.15, obs_sd: float = 12.8,
                 hca: float = PRIORS.hca_points):
        self.s, self.q, self.r, self.hca = states, process_var, obs_sd ** 2, hca

    def predict(self):
        for st in self.s.values():
            st.var += self.q

    def update(self, home: str, away: str, margin_per100: float):
        h, a = self.s[home], self.s[away]
        H = np.array([1.0, -1.0])
        P = np.diag([h.var, a.var])
        x = np.array([h.theta, a.theta])
        y = margin_per100 - (x @ H + self.hca)
        S = H @ P @ H + self.r
        K = P @ H / S
        x = x + K * y
        P = P - np.outer(K, H) @ P
        h.theta, a.theta = float(x[0]), float(x[1])
        h.var, a.var = float(P[0, 0]), float(P[1, 1])


def fuse(mu_model: float, mu_market: float, sd_model: float, sd_market: float) -> dict:
    """Precision-weighted fusion. K is the weight on the market."""
    K = sd_model ** 2 / (sd_model ** 2 + sd_market ** 2)
    mu = mu_model + K * (mu_market - mu_model)
    sd = (sd_model ** -2 + sd_market ** -2) ** -0.5
    return {"mu": mu, "sd": sd, "w_market": K, "w_model": 1 - K}


def fuse_side(mu_model, mu_market, sd_model=FUSION.sd_model_side, sd_market=FUSION.sd_market_side):
    return fuse(mu_model, mu_market, sd_model, sd_market)


def fuse_total(mu_model, mu_market, sd_model=FUSION.sd_model_total, sd_market=FUSION.sd_market_total):
    return fuse(mu_model, mu_market, sd_model, sd_market)
