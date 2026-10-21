"""One simulated game day through the whole Phase 0 chain:
plan -> decision snapshots -> opinions -> close snapshots -> CLV -> export."""

import csv
from datetime import date, datetime, timedelta, timezone

import pytest
from fake_api import FakeOddsApi, iso

from market import config as C
from market.opinion_log import EXPORT_COLUMNS, NotCloseCapturableError, Opinion, OpinionLog
from market.odds_api import OddsApiClient
from market.model_adapter import from_nba_sim_payload
from market.run_snapshots import close_out_opinions, close_plan_from, plan, record_opinions, run_due
from market.scheduler import ET
from market.selection import best_price, reference_prob
from market.store import DONE, MISSED, MarketStore

DAY = date(2026, 10, 22)


def at(hh, mm=0):
    return datetime(2026, 10, 22, hh, mm, tzinfo=ET).astimezone(timezone.utc)


EVENTS = [
    {"id": "g1", "commence_time": iso(at(19)), "home_team": "Boston Celtics", "away_team": "New York Knicks"},
    {"id": "g2", "commence_time": iso(at(19)), "home_team": "Miami Heat", "away_team": "Orlando Magic"},
    {"id": "g3", "commence_time": iso(at(22)), "home_team": "Denver Nuggets", "away_team": "Utah Jazz"},
]


def board(ev, markets, books, now):
    """Noon: HOME -4.5 at -110/-110. Close: money on HOME, -4.5 now -125/+105."""
    closing = now >= at(18, 30)
    out = []
    for b in books:
        mk = []
        if "spreads" in markets:
            home_px, away_px = (1.80, 2.05) if closing else (1.91, 1.91)
            if b == "pinnacle":
                home_px, away_px = (1.82, 2.08) if closing else (1.95, 1.95)
            mk.append({"key": "spreads", "last_update": iso(now), "outcomes": [
                {"name": ev["home_team"], "price": home_px, "point": -4.5},
                {"name": ev["away_team"], "price": away_px, "point": 4.5}]})
        if "totals" in markets:
            mk.append({"key": "totals", "last_update": iso(now), "outcomes": [
                {"name": "Over", "price": 1.91, "point": 221.5},
                {"name": "Under", "price": 1.91, "point": 221.5}]})
        if "player_points" in markets and b in ("fanduel", "draftkings", "betmgm"):
            mk.append({"key": "player_points", "last_update": iso(now), "outcomes": [
                {"name": "Over", "description": "Star", "price": 2.20 if b == "fanduel" else 1.95, "point": 24.5},
                {"name": "Under", "description": "Star", "price": 1.70 if b == "fanduel" else 1.87, "point": 24.5}]})
        if mk:
            out.append({"key": b, "title": b, "markets": mk})
    return out


@pytest.fixture
def world(tmp_path):
    clock = {"now": at(9)}
    api = FakeOddsApi(EVENTS, board, remaining=500, clock=lambda: clock["now"])
    store = MarketStore(tmp_path / "market.db")
    client = OddsApiClient("k", transport=api, ledger=store, clock=lambda: clock["now"], sleep=lambda s: None)
    log = OpinionLog(tmp_path / "market.db", close_plan=close_plan_from(store))
    yield clock, api, store, client, log
    log.close()
    store.close()


def test_full_day(world, tmp_path):
    clock, api, store, client, log = world

    n_jobs, warnings = plan(store, client, DAY)
    assert warnings == []
    kinds = sorted(j.kind for j in store.jobs())
    assert kinds.count("sides_close") == 2 and kinds.count("sides_decision") == 1
    assert kinds.count("props_decision") == 2 and kinds.count("props_close") == 2
    assert api.remaining == 500  # /events is free

    clock["now"] = at(12, 1)
    msgs = run_due(store, client, clock["now"])
    assert sum(m.startswith("DONE") for m in msgs) == 3  # sides decision + 2 props decisions
    dec_job = next(j for j in store.jobs(status=DONE) if j.kind == "sides_decision")
    noon = store.load_quotes(dec_job.snapshot_id, "g1")

    # Opinion on the AWAY dog +4.5 off the noon snapshot (v1 model says 56%).
    kw = dict(event_id="g1", market="spreads", side="New York Knicks", line=4.5)
    pick = best_price(noon, books=C.BETTABLE_BOOKS, **kw)
    ref = reference_prob(noon, **kw)
    assert ref.ref_quality == "sharp" and ref.p == pytest.approx(0.5)
    oid = log.record(Opinion(
        game_id="g1", market="spreads", side="New York Knicks", line=4.5, book=pick.book,
        price_american=pick.price_american, p_model=0.56, p_market_novig=ref.p,
        model_version="v1-kalman", ref_quality=ref.ref_quality, timestamp=clock["now"],
    ))

    # A market without a scheduled close cannot be logged.
    with pytest.raises(NotCloseCapturableError):
        log.record(Opinion(
            game_id="g1", market="h2h", side="New York Knicks", line=None, book="fanduel",
            price_american=150, p_model=0.4, p_market_novig=0.38, model_version="v1-kalman",
            ref_quality="sharp", timestamp=clock["now"],
        ))

    clock["now"] = at(18, 55)
    msgs = run_due(store, client, clock["now"])
    assert sum(m.startswith("DONE") for m in msgs) >= 2  # 19:00 cluster close + prop closes

    # The runner misses the 21:55 close for g3 (machine asleep) -> marked missed.
    clock["now"] = at(22, 1)
    msgs = run_due(store, client, clock["now"])
    assert any(m.startswith("MISSED sides_close") for m in msgs)
    assert any(j.status == MISSED and "g3" in j.event_ids for j in store.jobs())

    closed = close_out_opinions(store, log)
    row = log.rows()[0]
    assert row["opinion_id"] == oid and row["clv_method"] == "exact_line"
    assert row["clv_pct"] < 0  # money came on the home side; the dog bet lost value
    assert row["close_price"] == pytest.approx(108)  # Pinnacle's closing dog price
    assert row["close_ts_et"].startswith("2026-10-22T18:55")

    # Credit accounting: billed credits equal the planned cost of the jobs that ran.
    done = store.jobs(status=DONE)
    assert 500 - api.remaining == sum(j.est_cost for j in done)
    assert store.remaining == api.remaining
    missed_props = [j for j in store.jobs(status=MISSED) if j.kind == "props_close"]
    assert len(done) + len(store.jobs(status=MISSED)) == len(store.jobs())
    assert all("g3" in j.event_ids for j in missed_props)  # only the 22:00 game was missed

    path = log.export_csv(tmp_path / "opinion_log.csv")
    with path.open() as fh:
        reader = csv.DictReader(fh)
        assert tuple(reader.fieldnames) == EXPORT_COLUMNS
        assert len(list(reader)) == 1


def test_nba_sim_payload_to_logged_opinions(world):
    clock, api, store, client, log = world
    plan(store, client, DAY)
    payload = {
        "model_version": "nba-sim 0.1.0", "generated_at": "2026-10-22T14:30:00+00:00",
        "slate_date": "2026-10-22",
        "games": [
            {"game_id": "2026-10-22-NYK-BOS", "away": "NYK", "home": "BOS",
             "projection": {"structural": {"home_margin": 9.0, "total": 216.0}},
             "simulation": {"sd_margin": 13.0, "sd_total": 18.8}},
            {"game_id": "2026-10-22-UTA-DEN", "away": "UTA", "home": "DEN",
             "projection": {"structural": {"home_margin": 4.4, "total": 221.6}},
             "simulation": {"sd_margin": 13.0, "sd_total": 18.8}},
            {"game_id": "2026-10-22-LAL-GSW", "away": "LAL", "home": "GSW",  # not on tonight's board
             "projection": {"structural": {"home_margin": 1.0, "total": 230.0}},
             "simulation": {"sd_margin": 13.0, "sd_total": 18.8}},
        ],
    }
    projections = from_nba_sim_payload(payload)
    msgs = record_opinions(store, log, projections, DAY)
    assert msgs[0].startswith("No completed sides_decision")

    clock["now"] = at(12, 1)
    run_due(store, client, clock["now"])
    msgs = record_opinions(store, log, projections, DAY)
    rows = log.rows()
    assert len(rows) == 4  # 2 games x (spread, total); passes included
    assert any("LAL-GSW" in m and "not in any decision snapshot" in m for m in msgs)
    bos_spread = next(r for r in rows if r["game_id"] == "g1" and r["market"] == "spreads")
    assert bos_spread["side"] == "Boston Celtics" and bos_spread["gate_pass"] is True
    assert bos_spread["timestamp_et"].startswith("2026-10-22T12:01")
    assert all(r["model_version"] == "nba-sim 0.1.0" for r in rows)
    assert record_opinions(store, log, projections, DAY)[0].startswith("already logged")
    assert len(log.rows()) == 4

    clock["now"] = at(18, 55)
    run_due(store, client, clock["now"])
    clock["now"] = at(21, 55)
    run_due(store, client, clock["now"])
    close_out_opinions(store, log)
    assert all(r["clv_method"] == "exact_line" for r in log.rows())


def test_transient_error_keeps_job_pending_until_it_succeeds(world):
    clock, api, store, client, log = world
    plan(store, client, DAY)
    clock["now"] = at(12, 1)
    real = api.__call__

    outage = {"left": 3}  # three 502s exhaust the client's in-call retries once

    def flaky(url, params):
        if "/odds" in url and outage["left"] > 0:
            outage["left"] -= 1
            api.calls.append((url, dict(params)))
            return 502, {"x-requests-last": "0"}, b"Internal Server Error"
        return real(url, params)

    client.transport = flaky
    msgs = run_due(store, client, clock["now"])
    assert any(m.startswith("RETRY sides_decision") for m in msgs)
    dec = next(j for j in store.jobs() if j.kind == "sides_decision")
    assert dec.status == "pending"
    clock["now"] = at(12, 2)
    msgs = run_due(store, client, clock["now"])
    dec = next(j for j in store.jobs() if j.kind == "sides_decision")
    assert dec.status == "done"


def test_late_runner_marks_decision_missed(world):
    clock, api, store, client, log = world
    plan(store, client, DAY)
    clock["now"] = at(12, 45)  # past the 30-minute decision window
    msgs = run_due(store, client, clock["now"])
    assert any(m.startswith("MISSED sides_decision") for m in msgs)
    assert api.remaining == 500


def test_opinion_with_missed_close_is_kept_and_flagged(world):
    clock, api, store, client, log = world
    plan(store, client, DAY)
    clock["now"] = at(12, 1)
    run_due(store, client, clock["now"])
    dec = next(j for j in store.jobs(status=DONE) if j.kind == "sides_decision")
    quotes = store.load_quotes(dec.snapshot_id, "g3")
    ref = reference_prob(quotes, event_id="g3", market="spreads", side="Utah Jazz", line=4.5)
    log.record(Opinion(
        game_id="g3", market="spreads", side="Utah Jazz", line=4.5, book="draftkings",
        price_american=-110, p_model=0.55, p_market_novig=ref.p, model_version="v1-kalman",
        ref_quality=ref.ref_quality, timestamp=clock["now"],
    ))
    clock["now"] = at(22, 5)
    run_due(store, client, clock["now"])  # the 21:55 close is past its deadline
    close_out_opinions(store, log)
    row = log.rows()[0]
    assert row["clv_method"] == "close_missing" and row["clv_pct"] is None


def _late_payload(generated_at):
    return {
        "model_version": "nba-sim 0.2.0", "generated_at": generated_at, "slate_date": "2026-10-22",
        "games": [
            {"game_id": "2026-10-22-NYK-BOS", "away": "NYK", "home": "BOS",
             "projection": {"structural": {"home_margin": 9.0, "total": 210.0}},
             "simulation": {"sd_margin": 13.0, "sd_total": 18.8}},
            {"game_id": "2026-10-22-ORL-MIA", "away": "ORL", "home": "MIA",
             "projection": {"structural": {"home_margin": 8.5, "total": 230.0}},
             "simulation": {"sd_margin": 13.0, "sd_total": 18.8}},
        ],
    }


def test_late_projection_needs_a_later_snapshot_then_holds_and_stakes(world):
    from market.run_snapshots import decide_now

    clock, api, store, client, log = world
    plan(store, client, DAY)
    clock["now"] = at(12, 1)
    run_due(store, client, clock["now"])
    # Projection made at 13:30 ET, after the noon snapshot: refused against it (as-of rule).
    projections = from_nba_sim_payload(_late_payload(at(13, 30).isoformat()))
    msgs = record_opinions(store, log, projections, DAY)
    assert log.rows() == [] and all("take one with `decide`" in m for m in msgs)

    clock["now"] = at(13, 35)
    before = api.remaining
    msgs = decide_now(store, client, DAY, projections, clock["now"])
    assert any(m.startswith("DONE sides_decision") for m in msgs)
    assert before - api.remaining == 2  # spreads + totals, one region

    msgs = record_opinions(store, log, projections, DAY, hold_markets=frozenset({"totals"}))
    rows = log.rows()
    assert len(rows) == 4
    assert all(r["timestamp_et"].startswith("2026-10-22T13:35") for r in rows)
    totals = [r for r in rows if r["market"] == "totals"]
    assert all(r["pass_reason"] == "model_hold" and r["stake_u"] == 0 for r in totals)
    sides = [r for r in rows if r["market"] == "spreads"]
    assert all(r["gate_pass"] and r["stake_u"] > 0 for r in sides)  # both home sides pass
    # Both home sides share one thesis: combined stake <= the shared-thesis cap.
    assert sum(r["stake_u"] for r in sides) <= C.SHARED_THESIS_CAP_U + 1e-9
    assert all(r["stake_u"] <= C.PER_GAME_CAP_U for r in rows)
