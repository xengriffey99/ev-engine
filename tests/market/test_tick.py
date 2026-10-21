"""The Actions loop's unit of work: tick(), the baseline logger and the active window."""

from datetime import date, datetime, timedelta, timezone

import pytest
from fake_api import FakeOddsApi, iso
from test_runner_e2e import EVENTS, at, board

from market.model_adapter import BASELINE_VERSION, build_baseline_opinions
from market.odds_api import OddsApiClient
from market.opinion_log import OpinionLog
from market.run_snapshots import (
    close_plan_from,
    in_active_window,
    slate_day,
    summary_markdown,
    tick,
)
from market.scheduler import ET
from market.store import MarketStore


@pytest.fixture
def world(tmp_path):
    clock = {"now": at(8, 50)}
    api = FakeOddsApi(EVENTS, board, remaining=500, clock=lambda: clock["now"])
    store = MarketStore(tmp_path / "m.db")
    client = OddsApiClient("k", transport=api, ledger=store, clock=lambda: clock["now"], sleep=lambda s: None)
    log = OpinionLog(tmp_path / "m.db", close_plan=close_plan_from(store))
    yield clock, api, store, client, log, tmp_path
    log.close()
    store.close()


def run_day(clock, store, client, log, tmp_path, start=(8, 50), end=(23, 0)):
    out = []
    t = at(*start)
    while t <= at(*end):
        clock["now"] = t
        out += tick(store, client, log, t, payload_dir=tmp_path)
        t += timedelta(minutes=1)
    return out


def test_active_window_and_slate_day():
    assert not in_active_window(at(8, 30)) and in_active_window(at(8, 45))
    assert in_active_window(at(23, 59))
    late = datetime(2026, 10, 23, 1, 30, tzinfo=ET)
    assert in_active_window(late) and slate_day(late) == date(2026, 10, 22)
    assert not in_active_window(datetime(2026, 10, 23, 2, 0, tzinfo=ET))


def test_minute_loop_runs_a_full_day(world):
    clock, api, store, client, log, tmp_path = world
    msgs = run_day(clock, store, client, log, tmp_path)
    assert sum(m.startswith("planned 2026-10-22") for m in msgs) == 1  # plans once
    rows = log.rows()
    assert {r["model_version"] for r in rows} == {BASELINE_VERSION}
    assert len(rows) == 6  # 3 games x (spreads, totals)
    assert all(r["clv_method"] == "exact_line" for r in rows)
    assert all(r["p_model"] == r["p_market_novig"] for r in rows)
    jobs = store.jobs()
    assert all(j.status == "done" for j in jobs)
    # Closes ran inside T-5..T-4 (minute loop), never after tip
    tips = {e["id"]: datetime.fromisoformat(e["commence_time"].replace("Z", "+00:00")) for e in EVENTS}
    for j in jobs:
        if j.kind.endswith("close"):
            snap = store.conn.execute("SELECT fetched_at_utc FROM snapshots WHERE snapshot_id=?",
                                      (j.snapshot_id,)).fetchone()[0]
            taken = datetime.fromisoformat(snap)
            for eid in j.event_ids:
                assert timedelta(minutes=4) <= tips[eid] - taken <= timedelta(minutes=5)
    # A second pass the same day changes nothing and spends nothing
    before = api.remaining
    assert run_day(clock, store, client, log, tmp_path, (23, 1), (23, 10)) == []
    assert api.remaining == before


def test_payload_in_folder_is_logged_alongside_baseline(world):
    clock, api, store, client, log, tmp_path = world
    (tmp_path / "payload_2026-10-22.json").write_text("""{
      "model_version": "nba-sim 0.1.0", "generated_at": "2026-10-22T14:30:00+00:00",
      "slate_date": "2026-10-22",
      "games": [{"game_id": "x", "away": "NYK", "home": "BOS",
                 "projection": {"structural": {"home_margin": 9.0, "total": 216.0}},
                 "simulation": {"sd_margin": 13.0, "sd_total": 18.8}}]}""")
    run_day(clock, store, client, log, tmp_path, end=(12, 5))
    versions = [r["model_version"] for r in log.rows()]
    assert versions.count("nba-sim 0.1.0") == 2 and versions.count(BASELINE_VERSION) == 6


def test_baseline_picks_best_net_price_side():
    from factories import AWAY, HOME, game
    quotes = [
        game("pinnacle", HOME, -4.5, -110), game("pinnacle", AWAY, 4.5, -110),
        game("draftkings", HOME, -4.5, -120), game("draftkings", AWAY, 4.5, +100),  # soft AWAY
    ]
    ops, skipped = build_baseline_opinions(quotes, books=frozenset({"draftkings"}), markets=("spreads",))
    (op,) = ops
    assert op.side == AWAY and op.price_american == pytest.approx(100)
    assert op.p_model == op.p_market_novig == pytest.approx(0.5)
    assert "excluded from promotion" in op.notes


def test_summary_has_no_prices(world):
    clock, api, store, client, log, tmp_path = world
    run_day(clock, store, client, log, tmp_path)
    text = summary_markdown(store, log)
    assert "Close captured | 6 (100.0%)" in text
    for leak in ("-110", "+100", "1.91", "221.5", "price"):
        assert leak not in text
