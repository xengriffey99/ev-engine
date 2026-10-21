"""Prop engine v1 (NN5): minutes x per-minute rate x pace x matchup, priced as a simulated
distribution against the posted line (median reported, never a fixed-sigma normal).

One joint simulation per game, shared by every player in it:

1. Game environment (NBA_CONTEXT.md section 14): home margin M and total T drawn around the
   de-vigged MARKET spread and total, not around the game sim. Team points: home (T + M) / 2,
   away (T - M) / 2. Pace and matchup enter here, through the market's team totals.
2. Availability: each player plays with his status probability (out 0, doubtful 0.15,
   questionable 0.50, probable 0.85); one draw per simulated game.
3. Minutes engine: the healthy rotation takes the top 240 projected minutes in MPG order. Minutes
   vacated by absences go BETA to the remaining rotation (pro rata, capped) and the rest down the
   depth chart. Blowouts (|M| beyond a threshold) move starter minutes to the bench. Player-level
   minutes noise, renormalised to 240 per team (+25 per overtime period).
4. Rate engine with usage cascade: per-minute scoring weight r_i = pts/100 x poss/48 / 48 (career
   projections, already shrunk). Expected points lambda_i = team points x r_i m_i / sum_j r_j m_j,
   so an absent scorer's share flows to the players who take his minutes and shots.
5. Outcome: made twos, threes and free throws, each negative binomial (gamma-Poisson) around
   lambda_i x shot-profile share; points = 2 X2 + 3 X3 + X1. Skewed and overdispersed by
   construction.

Every number in EngineParams is an ESTIMATE until it is fit walk-forward (Phases 3-4); the run
records them in its payload.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np

PLAY_PROB = {"out": 0.0, "doubtful": 0.15, "questionable": 0.50, "probable": 0.85, "available": 1.0}


@dataclass(frozen=True)
class PlayerIn:
    name: str
    team: str
    mpg: float  # projected minutes per game when healthy
    pct_start: float
    pts100: float  # points per 100 possessions
    poss48: float  # team possessions per 48 minutes with him on court
    fg3a100: float
    fg3pct: float
    fta100: float
    ftpct: float
    status: str = "available"
    two_way: bool = False  # two-way / 10-day contract: banned prop type (section 11)

    @property
    def p_play(self) -> float:
        return PLAY_PROB[self.status]


@dataclass(frozen=True)
class MarketEnv:
    home: str
    away: str
    mu_margin: float  # home margin implied by the de-vigged market spread
    mu_total: float  # implied by the de-vigged market total
    sd_margin: float = 12.0  # PLACEHOLDER (config CLV_SHIFT_SIGMA); fit walk-forward
    sd_total: float = 18.0  # PLACEHOLDER


@dataclass(frozen=True)
class EngineParams:
    n_sims: int = 20_000
    seed: int = 20261021
    team_minutes: float = 240.0
    vacated_to_rotation: float = 0.40  # BETA: share of vacated minutes kept by the rotation
    max_minutes: float = 42.0
    max_bump: float = 1.15  # a rotation player rises at most 15% over his healthy minutes
    blowout_start: float = 12.0  # |final margin| where starters begin to sit late
    blowout_slope: float = 0.5  # starter minutes lost per point beyond the start
    blowout_max: float = 8.0
    minutes_noise_cv: float = 0.10
    ot_rate: float = 0.06
    k2: float = 40.0  # NB dispersion of made twos (league-typical SD/mean, not fit)
    k3: float = 20.0  # made threes
    kft: float = 15.0  # made free throws


def healthy_allocation(mpg: np.ndarray, team_minutes: float) -> np.ndarray:
    """Top-`team_minutes` projected minutes in MPG order (deep bench gets what is left)."""
    out = np.zeros_like(mpg)
    left = team_minutes
    for i in np.argsort(-mpg, kind="stable"):
        take = min(mpg[i], left)
        out[i] = max(0.0, take)
        left -= out[i]
    return out


def allocate_minutes(mpg: np.ndarray, active: np.ndarray, p: EngineParams) -> np.ndarray:
    """Regulation minutes for one availability state (active: bool array)."""
    healthy = healthy_allocation(mpg, p.team_minutes)
    if active.all():
        return healthy
    vacated = healthy[~active].sum()
    m = np.where(active, healthy, 0.0)
    # 1) BETA of the vacated minutes to the active rotation, pro rata, capped.
    rot = active & (healthy > 0)
    give = p.vacated_to_rotation * vacated
    for _ in range(5):  # redistribute what the caps leave over
        if give <= 1e-9 or not rot.any():
            break
        cap = np.minimum(p.max_minutes, p.max_bump * healthy) - m
        room = np.where(rot, np.maximum(cap, 0.0), 0.0)
        if room.sum() <= 1e-9:
            break
        w = np.where(room > 0, healthy, 0.0)
        add = np.minimum(room, give * w / w.sum())
        m += add
        give -= add.sum()
    # 2) the rest down the depth chart (next men up by MPG), each up to his own MPG.
    left = p.team_minutes - m.sum()
    for i in np.argsort(-mpg, kind="stable"):
        if left <= 1e-9:
            break
        if not active[i]:
            continue
        cap = min(mpg[i], p.max_minutes) - m[i]
        if cap > 0:
            add = min(cap, left)
            m[i] += add
            left -= add
    if left > 1e-6:  # short bench: spread the remainder pro rata over active players
        m += left * np.where(active, m, 0.0) / np.where(active, m, 0.0).sum()
    return m


def _states(players: Sequence[PlayerIn], rng: np.random.Generator, n: int) -> np.ndarray:
    pp = np.array([pl.p_play for pl in players])
    return rng.random((n, len(players))) < pp


def simulate_team(players: Sequence[PlayerIn], team_pts: np.ndarray, team_margin: np.ndarray,
                  ot: np.ndarray, p: EngineParams, rng: np.random.Generator) -> dict:
    """Simulate one team's players across n games. team_margin is from this team's side."""
    n = team_pts.shape[0]
    mpg = np.array([pl.mpg for pl in players], dtype=float)
    active = _states(players, rng, n)
    # Minutes per distinct availability state (few states: cache them).
    keys, inverse = np.unique(active, axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    base = np.stack([allocate_minutes(mpg, k.astype(bool), p) for k in keys])[inverse]
    # Blowouts: starters (top five by healthy minutes among the active) sit late.
    order = np.argsort(-base, axis=1)
    starters = np.zeros_like(base, dtype=bool)
    np.put_along_axis(starters, order[:, :5], True, axis=1)
    loss = np.clip(p.blowout_slope * (np.abs(team_margin) - p.blowout_start), 0.0, p.blowout_max)
    take = np.minimum(base, loss[:, None]) * starters
    bench_w = np.where(active & ~starters, mpg[None, :], 0.0)
    bench_w = bench_w / np.maximum(bench_w.sum(axis=1, keepdims=True), 1e-9)
    m = base - take + take.sum(axis=1, keepdims=True) * bench_w
    # Minutes noise, then renormalise to 240 (+25 per OT period).
    noise = rng.lognormal(-0.5 * p.minutes_noise_cv ** 2, p.minutes_noise_cv, m.shape)
    m = m * noise * active
    total = p.team_minutes + 25.0 * ot
    m = m * (total / np.maximum(m.sum(axis=1), 1e-9))[:, None]
    m = np.minimum(m, 53.0)
    # Usage cascade: team points split by rate x minutes.
    r = np.array([pl.pts100 / 100.0 * pl.poss48 / 48.0 for pl in players])
    w = r[None, :] * m
    lam = team_pts[:, None] * w / np.maximum(w.sum(axis=1, keepdims=True), 1e-9)
    # Shot profile shares of points.
    s3 = np.array([3 * pl.fg3a100 * pl.fg3pct / pl.pts100 for pl in players])
    sf = np.array([pl.fta100 * pl.ftpct / pl.pts100 for pl in players])
    s3, sf = np.clip(s3, 0, 0.8), np.clip(sf, 0, 0.5)
    s2 = np.clip(1.0 - s3 - sf, 0.05, None)
    tot = s2 + s3 + sf
    s2, s3, sf = s2 / tot, s3 / tot, sf / tot

    def nb(mean, k):
        g = rng.gamma(k, 1.0 / k, mean.shape)
        return rng.poisson(mean * g)

    pts = 2 * nb(lam * s2 / 2, p.k2) + 3 * nb(lam * s3 / 3, p.k3) + nb(lam * sf, p.kft)
    return {"minutes": m, "lambda": lam, "points": pts, "active": active}


def simulate_game(env: MarketEnv, home: Sequence[PlayerIn], away: Sequence[PlayerIn],
                  p: EngineParams = EngineParams()) -> dict:
    """Joint simulation of every player in one game. Returns per-player arrays + env draws."""
    rng = np.random.default_rng(p.seed)
    n = p.n_sims
    M = rng.normal(env.mu_margin, env.sd_margin, n)
    T = rng.normal(env.mu_total, env.sd_total, n)
    ot = (rng.random(n) < p.ot_rate).astype(float)
    hp, ap = (T + M) / 2.0, (T - M) / 2.0
    h = simulate_team(home, hp, M, ot, p, rng)
    a = simulate_team(away, ap, -M, ot, p, rng)
    out = {}
    for team, players, res in ((env.home, home, h), (env.away, away, a)):
        for j, pl in enumerate(players):
            out[pl.name] = {
                "team": team, "player": pl,
                "minutes": res["minutes"][:, j], "points": res["points"][:, j],
                "active": res["active"][:, j],
            }
    return {"players": out, "margin": M, "total": T, "home_pts": hp, "away_pts": ap,
            "params": asdict(p), "env": asdict(env)}


def summarize(sim_player: dict) -> dict:
    pts, mins, act = sim_player["points"], sim_player["minutes"], sim_player["active"]
    return {
        "p_play": float(act.mean()),
        "exp_minutes": float(mins.mean()),
        "minutes_if_playing_p50": float(np.median(mins[act])) if act.any() else 0.0,
        "mean": float(pts.mean()), "median": float(np.median(pts)), "sd": float(pts.std()),
        "p10": float(np.percentile(pts, 10)), "p90": float(np.percentile(pts, 90)),
    }


def prob_over(points: np.ndarray, line: float) -> float:
    """P(points > line) on the no-push basis used by the de-vigged market."""
    over = float((points > line).mean())
    push = float((points == line).mean())
    return over / (1.0 - push) if push < 1.0 else 0.5


# --- 2026-27 restrictions (NBA_CONTEXT.md section 11) ---------------------------------------
MIN_UNDER_MINUTES = 12.0
BANNED_MARKETS = frozenset({"player_turnovers", "player_fouls", "player_first_basket",
                            "player_first_stat", "opening_tip", "player_missed_free_throws"})


def restriction(market: str, side: str, exp_minutes: float, two_way: bool,
                is_main_line: bool = True) -> str | None:
    """Reason the opinion is off the board, or None. Partner books removed these in 2026-27."""
    if market in BANNED_MARKETS:
        return "banned market type"
    if two_way:
        return "two-way / 10-day player"
    if side == "Under" and exp_minutes < MIN_UNDER_MINUTES:
        return f"under on a player projected {exp_minutes:.1f} < {MIN_UNDER_MINUTES:g} minutes"
    if side == "Under" and not is_main_line:
        return "alternate under line"
    return None
