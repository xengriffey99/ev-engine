"""Model constants. Bankroll, risk and situational values come from the NBA Project "Context" doc.

Values marked APPROX are analyst priors that must be replaced by calibrated estimates
from the backtest harness (nba_sim.backtest) once in-season data accumulate.
"""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Bankroll:
    starting: float = 10_000.0
    unit_pct: float = 0.01            # 1.0u = 1% of current bankroll
    max_units: float = 2.5            # hard cap per wager (2.5% of bankroll)
    kelly_fraction: float = 0.25      # quarter Kelly
    ev_min_sides_totals: float = 0.025
    ev_min_props: float = 0.040
    ev_min_moneyline: float = 0.025   # not set in Context doc; treated like sides (assumption)


@dataclass(frozen=True)
class Situational:
    denver_altitude: float = 1.8      # home margin, opponent on 0-1 days rest
    utah_altitude: float = 1.4
    road_fatigue_ortg: float = -2.1   # visitor ORtg on 3-in-4 / 4-in-5
    matinee_total: float = -1.5       # tip before 3:30 pm local
    matinee_weekend_only: bool = True # Context doc rule reads "Early Weekend Matinee"
    eastbound_tz_penalty: float = -0.6  # APPROX, uncalibrated


@dataclass(frozen=True)
class Simulation:
    n_sims: int = 10_000
    seed: int = 20261020
    team_sd: float = 11.25            # gives margin SD 12.8, total SD 18.5 with rho 0.35
    rho: float = 0.35
    ot_mean_pts: float = 11.0         # per team per 5-min OT, APPROX
    ot_sd_pts: float = 3.5


@dataclass(frozen=True)
class Priors:
    league_ortg: float = 115.5        # APPROX 2025-26 league average
    league_pace: float = 99.6         # APPROX
    hca_points: float = 2.4           # APPROX recent-season home-court value
    net_carryover_rho: float = 0.70   # year-over-year regression of ratings
    pace_carryover: float = 0.80
    replacement_epm: float = -2.0
    pos_synergy_lambda: float = 0.35  # convex penalty, points per 100
    q_play_prob: float = 0.50         # Questionable
    p_play_prob: float = 0.85         # Probable
    four_factor_gamma: float = 1.0
    four_factor_weights: dict = field(default_factory=lambda: {
        "efg": 0.40, "tov": 0.25, "orb": 0.20, "ftr": 0.15})


@dataclass(frozen=True)
class Fusion:
    """Kalman fusion of structural projection with the market line (shrink before Kelly)."""
    sd_model_side: float = 4.0        # opening night; shrinks as games are observed
    sd_market_side: float = 1.5
    sd_model_total: float = 6.0
    sd_market_total: float = 2.0
    rlm_shift_points: float = 0.5     # posterior nudge toward RLM side, uncalibrated
    sharp_shift_points: float = 0.5


@dataclass(frozen=True)
class MarketRules:
    rlm_public_threshold: float = 0.65
    sharp_money_min: float = 0.60
    sharp_ticket_max: float = 0.40
    key_numbers: tuple = (1.5, 2.5, 3.5, 4.5, 5.5, 7.5, 9.5)
    steam_points: float = 1.0
    steam_books: int = 3
    steam_window_min: int = 10


BANKROLL = Bankroll()
SITUATIONAL = Situational()
SIM = Simulation()
PRIORS = Priors()
FUSION = Fusion()
MARKET = MarketRules()
