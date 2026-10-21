from datetime import date, datetime, timedelta, timezone

import pytest

from market import config as C
from market.scheduler import (
    ET,
    Event,
    daily_allowance,
    days_left_in_month,
    draw_prop_games,
    plan_day,
    projected_monthly_credits,
    tip_clusters,
)

DAY = date(2026, 10, 22)


def ev(eid, hh, mm=0, day=DAY):
    tip = datetime(day.year, day.month, day.day, hh, mm, tzinfo=ET).astimezone(timezone.utc)
    return Event(eid, tip, f"H{eid}", f"A{eid}")


SLATE = [ev("g1", 19), ev("g2", 19), ev("g3", 19, 30), ev("g4", 22), ev("g5", 22, 30)]


def et(job_time):
    return job_time.astimezone(ET).strftime("%H:%M")


def test_tip_clusters_respect_five_minute_window():
    games = [ev("a", 19), ev("b", 19, 5), ev("c", 19, 6), ev("d", 22)]
    clusters = tip_clusters(games, 5)
    assert [[e.id for e in c] for c in clusters] == [["a", "b"], ["c"], ["d"]]


def test_plan_day_jobs_and_times():
    jobs, warnings = plan_day(SLATE, DAY)
    assert warnings == []
    sides_dec = [j for j in jobs if j.kind == "sides_decision"]
    closes = [j for j in jobs if j.kind == "sides_close"]
    assert len(sides_dec) == 1 and et(sides_dec[0].run_at) == "12:00"
    assert set(sides_dec[0].event_ids) == {"g1", "g2", "g3", "g4", "g5"}
    assert [et(j.run_at) for j in closes] == ["18:55", "19:25", "21:55", "22:25"]
    assert [et(j.deadline) for j in closes] == ["19:00", "19:30", "22:00", "22:30"]
    assert closes[0].event_ids == ("g1", "g2")
    assert all(j.markets == ("spreads", "totals") and j.endpoint == "featured" for j in closes)
    props = [j for j in jobs if j.kind.startswith("props")]
    assert len(props) == 4 and all(j.markets == ("player_points",) and j.endpoint == "event" for j in props)
    # Cost: 2 x (1 decision + 4 closes) + 2 games x 2 snapshots x 1 = 14 credits
    assert sum(j.est_cost for j in jobs) == 14


def test_every_close_lands_inside_t10_to_t0():
    games = [ev("a", 19), ev("b", 19, 5), ev("c", 19, 30)]
    jobs, _ = plan_day(games, DAY)
    tips = {g.id: g.commence_time for g in games}
    for j in (j for j in jobs if j.kind.endswith("close")):
        for eid in j.event_ids:
            lead = tips[eid] - j.run_at
            assert timedelta(minutes=0) < lead <= timedelta(minutes=10)


def test_early_tip_gets_t60_decision():
    games = [ev("early", 12, 30), ev("late", 19)]
    jobs, _ = plan_day(games, DAY)
    dec = sorted((j for j in jobs if j.kind == "sides_decision"), key=lambda j: j.run_at)
    assert [et(j.run_at) for j in dec] == ["11:30", "12:00"]
    assert dec[0].event_ids == ("early",)


def test_prop_draw_is_seeded_and_order_independent():
    a = draw_prop_games(SLATE, DAY, 2, "salt")
    b = draw_prop_games(list(reversed(SLATE)), DAY, 2, "salt")
    assert a == b and len(a) == 2
    other_days = {tuple(draw_prop_games(SLATE, DAY + timedelta(days=i), 2, "salt")) for i in range(30)}
    assert len(other_days) > 3  # the draw actually varies by date


def test_only_games_on_the_date_are_planned():
    jobs, _ = plan_day(SLATE + [ev("tomorrow", 19, day=DAY + timedelta(days=1))], DAY)
    assert all("tomorrow" not in j.event_ids for j in jobs)


def test_allowance_drops_props_archive_as_a_unit():
    jobs, warnings = plan_day(SLATE, DAY, allowance=12)  # 14 planned -> props (4) go, 10 left
    assert {j.kind for j in jobs} == {"sides_decision", "sides_close"}
    assert all(j.markets == ("spreads", "totals") for j in jobs)
    assert any("props archive dropped" in w for w in warnings)


def test_allowance_goes_spreads_only_on_decision_and_closes_together():
    jobs, warnings = plan_day(SLATE, DAY, allowance=6)  # 10 sides credits -> spreads-only = 5
    assert sum(1 for j in jobs if j.kind == "sides_decision") == 1  # decision is never dropped
    assert sum(1 for j in jobs if j.kind == "sides_close") == 4
    assert all(j.markets == ("spreads",) and j.est_cost == 1 for j in jobs)
    assert any("spreads-only" in w for w in warnings)
    assert not any("Upgrade trigger 1" in w for w in warnings)


def test_allowance_never_strands_a_decision_without_closes():
    jobs, warnings = plan_day(SLATE, DAY, allowance=3)
    kinds = [j.kind for j in jobs]
    assert kinds.count("sides_decision") == 1 and kinds.count("sides_close") == 4
    assert any("Upgrade trigger 1" in w for w in warnings)


def test_live_slate_2026_10_21_cost():
    """Tip times from the live /events call on 2026-10-21 (11 games, 9 distinct tips)."""
    tips = [(19, 0), (19, 10), (19, 30), (19, 40), (19, 40), (20, 10), (20, 10), (20, 40),
            (22, 0), (22, 10), (22, 40)]
    day = date(2026, 10, 21)
    slate = [ev(f"g{i}", h, m, day=day) for i, (h, m) in enumerate(tips)]
    jobs, _ = plan_day(slate, day)
    assert sum(1 for j in jobs if j.kind == "sides_close") == 9
    assert sum(j.est_cost for j in jobs) == 24  # 2 + 9 x 2 + 4


def test_daily_allowance_protects_a_baseline_for_remaining_days():
    assert days_left_in_month(date(2026, 10, 22)) == 10
    # 500 - 60 reserve - 14 x 9 future days = 314 available today
    assert daily_allowance(500, 60, date(2026, 10, 22)) == pytest.approx(314.0)
    # Nov 1 on a fresh 500: 500 - 60 - 14 x 29 = 34 -> a 24-credit full slate still fits
    assert daily_allowance(500, 60, date(2026, 11, 1)) == pytest.approx(34.0)
    # Late month, budget spent: today's allowance collapses, closes then go spreads-only
    assert daily_allowance(120, 60, date(2026, 11, 25)) == pytest.approx(0.0)
    assert daily_allowance(50, 60, date(2026, 10, 31)) == 0.0


def test_month_simulation_on_free_tier():
    """30 straight 18-credit nights (6 closes + props) on 500 credits: what degrades, and when."""
    remaining, kinds_by_day = 500, []
    tips = [(19, 0), (19, 30), (20, 0), (20, 30), (22, 0), (22, 30)]
    for d in range(1, 31):
        day = date(2026, 11, d)
        slate = [ev(f"g{d}_{i}", h, m, day=day) for i, (h, m) in enumerate(tips)]
        jobs, _ = plan_day(slate, day, allowance=daily_allowance(remaining, 60, day))
        remaining -= sum(j.est_cost for j in jobs)
        kinds_by_day.append({(j.kind, j.markets) for j in jobs})
    assert remaining >= 0  # never overdrawn
    full_sides = sum(any(k == ("sides_close", ("spreads", "totals")) for k in s) for s in kinds_by_day)
    with_props = sum(any(k[0] == "props_close" for k in s) for s in kinds_by_day)
    assert full_sides == 30  # sides/totals keep full coverage all month
    assert with_props < 30  # the props archive is what gives way (upgrade trigger 2)


def test_monthly_projection_matches_option_a_budget():
    # 28 game days, ~210 games, ~4 tip clusters/day -> 280 sides + 112 props = 392
    assert projected_monthly_credits(28, 210, 4) == pytest.approx(392)
    assert projected_monthly_credits(28, 210, 4) <= C.FREE_TIER.monthly_credits - C.FREE_TIER.reserve_credits
