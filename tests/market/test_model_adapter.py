from datetime import date, datetime, timedelta, timezone
from math import erf, sqrt

import pytest
from factories import AWAY, FETCHED, HOME, game

from market import config as C
from market.model_adapter import (
    GameProjection,
    build_opinions,
    canonical_team,
    from_nba_sim_payload,
    match_events,
    p_model,
)

DAY = date(2026, 10, 22)
MA = C.BETTABLE_BY_STATE["MA"]


def Phi(x):
    return 0.5 * (1 + erf(x / sqrt(2)))


def proj(mu=3.0, total=224.0, as_of=FETCHED - timedelta(hours=2), **kw):
    base = dict(model_game_id="2026-10-22-NYK-BOS", date_et=DAY, home=HOME, away=AWAY,
                mu_home_margin=mu, sd_margin=12.0, mu_total=total, sd_total=18.0,
                model_version="test-model", as_of=as_of, source="test")
    base.update(kw)
    return GameProjection(**base)


def nba_sim_payload():
    """Shape of nba_sim 0.1.0 payload_<date>.json (Layer 3 schema, 2026-10-20 chat)."""
    return {
        "schema_version": "1.0",
        "model_version": "nba-sim 0.1.0 (opening-night priors)",
        "generated_at": "2026-10-22T14:30:00+00:00",
        "slate_date": "2026-10-22",
        "games": [{
            "game_id": "2026-10-22-NYK-BOS", "away": "NYK", "home": "BOS",
            "projection": {
                "structural": {"home_margin": 6.2, "total": 219.4, "pace": 97.1},
                "market": {"home_margin": 4.5, "total": 221.5},
                "posterior": {"home_margin": 4.71, "total": 221.2, "w_model_side": 0.12},
            },
            "simulation": {"n": 10000, "home_win_prob": 0.64, "sd_margin": 13.04, "exp_total": 221.3,
                           "sd_total": 18.81},
            "markets": [], "recommendation": "PASS",
        }],
    }


# --- teams --------------------------------------------------------------------------
@pytest.mark.parametrize("name, full", [
    ("BOS", "Boston Celtics"), ("bkn", "Brooklyn Nets"), ("BRK", "Brooklyn Nets"),
    ("PHO", "Phoenix Suns"), ("LA Clippers", "Los Angeles Clippers"), ("Utah Jazz", "Utah Jazz"),
    ("GS", "Golden State Warriors"),
])
def test_canonical_team(name, full):
    assert canonical_team(name) == full


def test_unknown_team_raises():
    with pytest.raises(KeyError):
        canonical_team("Seattle SuperSonics")


# --- nba_sim payload -----------------------------------------------------------------
def test_nba_sim_uses_structural_not_market_fused_posterior():
    (p,) = from_nba_sim_payload(nba_sim_payload())
    assert p.mu_home_margin == 6.2 and p.mu_total == 219.4  # structural
    assert p.sd_margin == 13.04 and p.sd_total == 18.81
    assert p.home == "Boston Celtics" and p.away == "New York Knicks"
    assert p.as_of == datetime(2026, 10, 22, 14, 30, tzinfo=timezone.utc)


# --- p_model ------------------------------------------------------------------------
def test_p_model_spread_closed_form():
    p = proj(mu=4.5)
    assert p_model(p, "spreads", HOME, -4.5) == pytest.approx(0.5, abs=1e-12)
    assert p_model(p, "spreads", HOME, -2.5) == pytest.approx(Phi(2 / 12), abs=1e-12)
    assert p_model(p, "spreads", AWAY, 2.5) == pytest.approx(1 - Phi(2 / 12), abs=1e-12)


@pytest.mark.parametrize("line", [-4.0, -4.5, 3.0])
def test_p_model_sides_complement_on_no_push_basis(line):
    p = proj(mu=3.3)
    assert p_model(p, "spreads", HOME, line) + p_model(p, "spreads", AWAY, -line) == pytest.approx(1.0)


@pytest.mark.parametrize("line", [221.0, 221.5])
def test_p_model_totals_complement(line):
    p = proj(total=224.0)
    over, under = p_model(p, "totals", "Over", line), p_model(p, "totals", "Under", line)
    assert over + under == pytest.approx(1.0) and over > 0.5


def test_p_model_rejects_wrong_side():
    with pytest.raises(ValueError):
        p_model(proj(), "spreads", "Miami Heat", -4.5)
    with pytest.raises(ValueError):
        p_model(proj(), "player_points", "Over", 24.5)


# --- build_opinions ----------------------------------------------------------------
def board(pin_home=-110, pin_away=-110, extra=()):
    rows = [
        game("pinnacle", HOME, -4.5, pin_home), game("pinnacle", AWAY, 4.5, pin_away),
        game("draftkings", HOME, -4.5, -112), game("draftkings", AWAY, 4.5, -108),
        game("fanduel", HOME, -4.5, -105), game("fanduel", AWAY, 4.5, -115),
        game("draftkings", HOME, -5.5, -102), game("draftkings", AWAY, 5.5, -118),  # retail-only alt line
        game("pinnacle", "Over", 221.5, -110, market="totals"),
        game("pinnacle", "Under", 221.5, -110, market="totals"),
        game("draftkings", "Over", 221.5, -110, market="totals"),
        game("draftkings", "Under", 221.5, -110, market="totals"),
    ]
    return rows + list(extra)


def test_builds_one_opinion_per_market_at_the_sharp_line():
    ops, skipped = build_opinions([proj(mu=8.0, total=224.0)], board(), books=MA)
    assert skipped == []
    spread = next(o for o in ops if o.market == "spreads")
    total = next(o for o in ops if o.market == "totals")
    assert spread.side == HOME and spread.line == -4.5  # Pinnacle's line, not the retail-only -5.5
    assert spread.book == "fanduel" and spread.price_american == pytest.approx(-105)
    assert spread.ref_quality == "sharp" and spread.p_market_novig == pytest.approx(0.5)
    assert spread.p_model == pytest.approx(Phi(3.5 / 12), abs=1e-9)
    assert total.side == "Over" and total.line == 221.5
    assert spread.timestamp == FETCHED and spread.game_id == "evt1"
    assert "model_game_id=2026-10-22-NYK-BOS" in spread.notes


def test_bettable_profile_changes_the_price_taken():
    ops, _ = build_opinions([proj(mu=8.0)], board(), books=C.BETTABLE_BY_STATE["ME"])
    spread = next(o for o in ops if o.market == "spreads")
    assert spread.book == "draftkings" and spread.price_american == pytest.approx(-112)


def test_side_choice_follows_ev_not_model_direction_alone():
    # Model leans home by a hair, but the away price is far better: EV picks away.
    rows = board(extra=[]) + []
    rows = [r for r in rows if not (r.book == "fanduel")]
    rows += [game("fanduel", HOME, -4.5, -140), game("fanduel", AWAY, 4.5, +125)]
    ops, _ = build_opinions([proj(mu=4.8)], rows, books=MA)
    spread = next(o for o in ops if o.market == "spreads")
    assert spread.side == AWAY and spread.book == "fanduel"


def test_projection_after_snapshot_is_refused():
    ops, skipped = build_opinions([proj(as_of=FETCHED + timedelta(minutes=1))], board(), books=MA)
    assert ops == [] and "as-of rule" in skipped[0]


def test_unmatched_and_unbettable_are_reported():
    other = proj(model_game_id="2026-10-22-MIA-ORL", home="Orlando Magic", away="Miami Heat")
    ops, skipped = build_opinions([other], board(), books=MA)
    assert ops == [] and "no matching event" in skipped[0]
    pin_only = [r for r in board() if r.book == "pinnacle"]
    ops, skipped = build_opinions([proj()], pin_only, books=MA)
    assert ops == [] and all("no two-way reference with a bettable price" in s for s in skipped)


def test_match_events_uses_et_date():
    assert match_events([proj()], board()) == {"2026-10-22-NYK-BOS": "evt1"}
    assert match_events([proj(date_et=DAY + timedelta(days=1))], board()) == {}
