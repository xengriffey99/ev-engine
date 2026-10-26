"""Approved one-off snapshot requests: schedule, decide, log record-only, close, CLV."""

import json
from datetime import date, datetime, timezone

import pytest
from fake_api import FakeOddsApi, iso

from market import extra
from market.odds_api import OddsApiClient
from market.opinion_log import OpinionLog
from market.run_snapshots import close_out_opinions, close_plan_from, plan, run_due
from market.scheduler import ET
from market.store import DONE, MarketStore

DAY = date(2026, 10, 24)


def at(hh, mm=0):
    return datetime(2026, 10, 24, hh, mm, tzinfo=ET).astimezone(timezone.utc)


EVENTS = [{"id": "wc", "commence_time": iso(at(20)), "home_team": "Chicago Bulls",
           "away_team": "Washington Wizards"}]


def board(ev, markets, books, now):
    closing = now >= at(19, 30)
    out = []
    for b in books:
        mk = []
        if "player_points" in markets and b in ("fanduel", "draftkings"):
            mk.append({"key": "player_points", "last_update": iso(now), "outcomes": [
                {"name": "Over", "description": "Josh Giddey", "price": 1.91, "point": 16.5},
                {"name": "Under", "description": "Josh Giddey", "price": 1.91, "point": 16.5}]})
        if "player_rebounds" in markets and b in ("hardrockbet", "draftkings"):
            over, under = (1.70, 2.20) if closing else (1.80, 2.00)
            if b == "draftkings":
                over, under = (1.72, 2.15) if closing else (1.79, 2.02)
            mk.append({"key": "player_rebounds", "last_update": iso(now), "outcomes": [
                {"name": "Over", "description": "Josh Giddey", "price": over, "point": 6.5},
                {"name": "Under", "description": "Josh Giddey", "price": under, "point": 6.5},
                {"name": "Over", "description": "Kyshawn George", "price": 2.30, "point": 4.5}]})
        if mk:
            out.append({"key": b, "title": b, "markets": mk})
    return out


@pytest.fixture
def world(tmp_path):
    clock = {"now": at(9)}
    api = FakeOddsApi(EVENTS, board, remaining=400, clock=lambda: clock["now"])
    store = MarketStore(tmp_path / "market.db")
    client = OddsApiClient("k", transport=api, ledger=store, clock=lambda: clock["now"], sleep=lambda s: None)
    log = OpinionLog(tmp_path / "market.db", close_plan=close_plan_from(store))
    req = {"requests": [{
        "id": "t-was-chi-reb", "date_et": "2026-10-24",
        "home_team": "Chicago Bulls", "away_team": "Washington Wizards",
        "markets": ["player_rebounds"],
        "decision_at_et": "2026-10-24T12:00:00-04:00",
        "decision_deadline_et": "2026-10-24T19:30:00-04:00",
        "approved": "test", "model_version": "sidequest-reb-0",
        "projection_as_of_et": "2026-10-24T11:10:00-04:00",
        "opinions": [
            {"player": "Josh Giddey", "market": "player_rebounds", "line": 6.5, "p_over_model": 0.615},
            {"player": "Kyshawn George", "market": "player_rebounds", "line": 4.5, "p_over_model": 0.347},
        ]}]}
    path = tmp_path / "extra.json"
    path.write_text(json.dumps(req))
    yield clock, api, store, client, log, path
    log.close()
    store.close()


def test_extra_request_end_to_end(world):
    clock, api, store, client, log, path = world
    plan(store, client, DAY)  # the normal plan also archives points on this game (1-game slate)
    clock["now"] = at(11, 20)
    msgs = extra.schedule_extra(store, client, DAY, clock["now"], path)
    assert any("2 job(s)" in m for m in msgs)
    assert extra.schedule_extra(store, client, DAY, clock["now"], path) == []  # idempotent

    clock["now"] = at(12, 1)
    run_due(store, client, clock["now"])
    dec = [j for j in store.jobs(date_et="2026-10-24") if j.kind == extra.DECISION]
    assert dec[0].status == DONE and dec[0].markets == ("player_rebounds",)

    msgs = extra.record_extra_opinions(store, log, DAY, path)
    assert any("Josh Giddey" in m and "model_hold" in m for m in msgs)
    # George is over-only in the fake board: one-way, so no row (NN4)
    assert any("Kyshawn George" in m and "not logged" in m for m in msgs)
    rows = log.rows()
    assert len(rows) == 1 and rows[0]["pass_reason"] == "model_hold" and rows[0]["stake_u"] == 0
    assert extra.record_extra_opinions(store, log, DAY, path) == []  # idempotent

    clock["now"] = at(19, 55)
    run_due(store, client, clock["now"])
    msgs = close_out_opinions(store, log)
    row = log.rows()[0]
    assert row["clv_method"] == "exact_line" and row["p_close_novig"] is not None
    # the points close on the same game must not be picked for a rebounds row
    assert store.close_job_for_market("wc", ("props_close", extra.CLOSE), "player_rebounds").kind == extra.CLOSE
    assert store.close_job_for_market("wc", ("props_close", extra.CLOSE), "player_points").kind == "props_close"


def test_projection_newer_than_snapshot_is_not_priced(world):
    clock, api, store, client, log, path = world
    req = json.loads(path.read_text())
    req["requests"][0]["projection_as_of_et"] = "2026-10-24T13:00:00-04:00"
    path.write_text(json.dumps(req))
    clock["now"] = at(11, 20)
    extra.schedule_extra(store, client, DAY, clock["now"], path)
    clock["now"] = at(12, 1)
    run_due(store, client, clock["now"])
    assert extra.record_extra_opinions(store, log, DAY, path) == []
    assert log.rows() == []


def test_alt_line_used_when_request_line_moved(world):
    clock, api, store, client, log, path = world
    req = json.loads(path.read_text())
    giddey = req["requests"][0]["opinions"][0]
    giddey["line"], giddey["p_over_model"] = 7.5, 0.48          # line the board no longer shows
    giddey["alt_lines"] = [{"line": 8.5, "p_over_model": 0.35},  # not on the board either
                           {"line": 6.5, "p_over_model": 0.615}]  # the posted line
    path.write_text(json.dumps(req))
    clock["now"] = at(11, 20)
    extra.schedule_extra(store, client, DAY, clock["now"], path)
    clock["now"] = at(12, 1)
    run_due(store, client, clock["now"])
    msgs = extra.record_extra_opinions(store, log, DAY, path)
    rows = [r for r in log.rows() if r["player"] == "Josh Giddey"]
    assert len(rows) == 1 and rows[0]["line"] == 6.5 and abs(rows[0]["p_model"] - 0.615) < 1e-9
    assert "alt line (request line 7.5)" in rows[0]["notes"]
    assert any("Josh Giddey" in m and " 6.5 " in m for m in msgs)
    assert extra.record_extra_opinions(store, log, DAY, path) == []  # one row per player-market


def test_request_for_missing_game_warns(world, tmp_path):
    clock, api, store, client, log, path = world
    req = json.loads(path.read_text())
    req["requests"][0]["home_team"] = "Boston Celtics"
    path.write_text(json.dumps(req))
    msgs = extra.schedule_extra(store, client, DAY, at(11, 20), path)
    assert msgs and msgs[0].startswith("WARNING")
