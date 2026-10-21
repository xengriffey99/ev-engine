"""Phase 0 configuration.

Every number here traces to NBA_CONTEXT.md section 4 (parameters), section 5
(reference prices), section 6 (timing) or the section 14 amendments approved
on 2026-10-21. Values marked PLACEHOLDER are estimates, not sourced; they are
fit later (see the comment on each).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import time

# --- The Odds API -----------------------------------------------------------
API_HOST = "https://api.the-odds-api.com"
SPORT_KEY = "basketball_nba"
API_KEY_ENV = "ODDS_API_KEY"
MAX_BOOKS_PER_REGION = 10  # every group of 10 bookmakers is billed as 1 region

# --- Market classes -----------------------------------------------------------
SIDES_TOTALS_MARKETS = frozenset(
    {"h2h", "spreads", "totals", "alternate_spreads", "alternate_totals"}
)
SPREAD_LIKE_MARKETS = frozenset({"spreads", "alternate_spreads"})
TOTAL_LIKE_MARKETS = frozenset({"totals", "alternate_totals"})
LINELESS_MARKETS = frozenset({"h2h"})

SIDES_TOTALS = "sides_totals"
PROPS = "props"


def market_class(market: str) -> str:
    """Return the lane a market belongs to: sides/totals or props."""
    if market in SIDES_TOTALS_MARKETS:
        return SIDES_TOTALS
    if market.startswith("player_"):
        return PROPS
    raise ValueError(f"Unknown market class for {market!r}")


def stat_for_market(market: str) -> str | None:
    """'player_points' -> 'points'; game markets have no stat."""
    if market.startswith("player_"):
        return market[len("player_"):]
    return None


# --- Blend, gates, staking (section 4) -----------------------------------------
BLEND_W = {SIDES_TOTALS: 0.40, PROPS: 0.50}
EDGE_GATE = {SIDES_TOTALS: 0.025, PROPS: 0.035}
TIER2_CAP = 0.02  # max probability move from judgment inputs
PER_GAME_CAP_U = 2.0
KELLY_FRACTION = 1.0 / 16.0
UNIT_PCT_BANKROLL = 0.01
# Approved 2026-10-21: ceiling set from walk-forward where edge-bucket
# reliability breaks down. Disabled (None) until fit.
OUTSIZED_EDGE_CEILING: dict[str, float | None] = {SIDES_TOTALS: None, PROPS: None}

# --- Books (section 5 + section 14) ------------------------------------------
# <=10 books per call keeps each call at 1 region of cost. Each list mixes the
# reference books with every book you can bet at in MA or ME that the free tier serves.
SIDES_TOTALS_BOOKS = (
    "pinnacle",  # sharp reference (public-site feed, can lag)
    "lowvig",  # extra reference opinion (not bettable)
    "draftkings",  # bettable ME + MA
    "fanduel",  # bettable MA
    "betmgm",  # bettable MA
    "espnbet",  # theScore Bet (formerly ESPN Bet), bettable MA
    "ballybet",  # bettable MA
    "kalshi",  # exchange; sports contracts blocked in MA, available in ME
    "novig",  # exchange
    "prophetx",  # exchange
)
PROPS_BOOKS = (
    "fanduel",  # originates its own props: the key prop reference
    "draftkings",
    "betmgm",
    "espnbet",
    "ballybet",
    "betrivers",  # reference opinion only (not live in MA or ME)
    "hardrockbet",  # reference opinion only (not live in MA or ME)
    "kalshi",
    "novig",
    "prophetx",
)
SHARP_BOOKS = frozenset({"pinnacle"})  # Circa is not on The Odds API
EXCHANGE_BOOKS = frozenset({"kalshi", "novig", "prophetx", "polymarket_us"})

# Books you can actually place a paper bet at, by the state you are physically in
# (sportsbooks geofence on location, not residency). Checked 2026-10-21:
#   MA: DraftKings, FanDuel, BetMGM, Caesars, Fanatics, theScore Bet, Bally Bet live;
#       Kalshi sports contracts blocked by a court injunction.
#   ME: DraftKings and Caesars only (tribal licenses); prediction markets listed available.
#   Novig: all states except AZ, MI, NV. ProphetX: MA and ME not on its restricted list.
# williamhill_us (Caesars) and fanatics are paid-tier only on The Odds API, so the
# free tier never prices them; they stay listed for when the plan is upgraded.
BETTABLE_BY_STATE = {
    "ME": frozenset({"draftkings", "williamhill_us", "kalshi", "novig", "prophetx"}),
    "MA": frozenset(
        {"draftkings", "fanduel", "betmgm", "williamhill_us", "fanatics", "espnbet",
         "ballybet", "novig", "prophetx"}
    ),
}
# Default ME. Set NBA_EV_BET_STATE=MA to price at Massachusetts books instead.
BET_STATE = os.environ.get("NBA_EV_BET_STATE", "ME").upper()
if BET_STATE not in BETTABLE_BY_STATE:
    raise ValueError(f"NBA_EV_BET_STATE must be one of {sorted(BETTABLE_BY_STATE)}")
BETTABLE_BOOKS = BETTABLE_BY_STATE[BET_STATE]
# Books that share a feed count as ONE opinion in a consensus.
# dk_feed: spec section 5 (sourced to NFL props; verified on NBA data by feed_check).
# bol_group: ASSUMPTION that LowVig and BetOnline share pricing; verify with feed_check.
FEED_GROUPS = {
    "draftkings": "dk_feed",
    "betmgm": "dk_feed",
    "williamhill_us": "dk_feed",
    "lowvig": "bol_group",
    "betonlineag": "bol_group",
}


def opinion_group(book: str) -> str:
    return FEED_GROUPS.get(book, book)


# Reference quality, best to worst. Retail-only references are lower quality (NN4).
REF_QUALITY_ORDER = ("sharp", "exchange", "consensus_retail", "single_retail")
DEFAULT_DEVIG_METHOD = "power"

# --- CLV ------------------------------------------------------------------------
# PLACEHOLDER estimates of the final-margin and final-total spread around the
# closing number, used ONLY to shift a closing probability to the bet line when
# the line moved. Rows that use them are labeled clv_method="line_shift_normal_est".
# Fit walk-forward in Phase 1b from the historical close backfill.
CLV_SHIFT_SIGMA = {"spreads": 12.0, "totals": 18.0}
# A reference price whose market last_update is older than this at capture is stale.
MAX_REF_STALENESS_MIN = 10


# ESTIMATE (live slate 2026-10-21: 11 games, 9 distinct tips): average night ~6 close
# calls x 2 markets + 2 for the decision = 14 credits for full sides/totals coverage.
# Used to protect the rest of the month's closes; refine from the credit ledger.
BASELINE_SIDES_CREDITS_PER_DAY = 14.0

# --- Free-tier snapshot schedule (Option A, approved 2026-10-21) ----------------
@dataclass(frozen=True)
class FreeTierSchedule:
    sides_markets: tuple[str, ...] = ("spreads", "totals")
    props_markets: tuple[str, ...] = ("player_points",)
    sides_books: tuple[str, ...] = SIDES_TOTALS_BOOKS
    props_books: tuple[str, ...] = PROPS_BOOKS
    props_games_per_day: int = 2
    decision_time_et: time = time(12, 0)
    early_tip_cutoff_et: time = time(13, 0)  # tips before this get a T-60 decision
    early_decision_lead_min: int = 60
    decision_window_min: int = 30  # a decision job may run up to 30 min late
    close_lead_min: int = 5  # close = T-5
    cluster_window_min: int = 5  # tips within 5 min share one close call (keeps T-10..T-0)
    monthly_credits: int = 500
    reserve_credits: int = 60  # kept back for /markets checks and retries
    draw_salt: str = "nba-ev-v2-props-archive"


FREE_TIER = FreeTierSchedule()
