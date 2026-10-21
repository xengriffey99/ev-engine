"""Free-tier snapshot plan (Option A, approved 2026-10-21).

Per game day:
  sides_decision  spreads+totals, 12:00 ET (T-60 for tips before 13:00 ET), one call for all games
  sides_close     spreads+totals, T-5 per tip cluster (tips within 5 min share a call)
  props_decision  player_points, decision time, 2 games drawn by a seeded random rule
  props_close     player_points, T-5, same 2 games
Estimated cost per day: $2(1 + K) + 2 \\cdot 2$ credits for K tip clusters.

When the daily credit allowance is short: props archive first, then the
sides lane goes spreads-only (decision AND closes together), then upgrade
trigger 1 is flagged. A decision snapshot is never dropped while its closes
stay, and an opinion is never logged for a market whose close was cut.

Live check 2026-10-21: an 11-game slate had 9 distinct tip times (tips sit at
:00/:10/:30/:40), so K is closer to 0.8 x games than the 4/day first assumed.
"""

from __future__ import annotations

import calendar
import hashlib
import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from . import config as C
from .odds_api import estimate_cost
from .quotes import parse_ts

ET = ZoneInfo("America/New_York")

PRIORITY = {"sides_close": 1, "sides_decision": 2, "props_close": 3, "props_decision": 4}


@dataclass(frozen=True)
class Event:
    id: str
    commence_time: datetime  # UTC
    home_team: str
    away_team: str

    @property
    def tip_et(self) -> datetime:
        return self.commence_time.astimezone(ET)


@dataclass(frozen=True)
class SnapshotJob:
    job_id: str
    date_et: str
    kind: str
    run_at: datetime  # UTC
    deadline: datetime  # UTC; a job not run by now is missed
    event_ids: tuple[str, ...]
    markets: tuple[str, ...]
    books: tuple[str, ...]
    endpoint: str  # "featured" or "event"
    est_cost: int
    priority: int
    note: str = ""


def events_from_payload(payload: Iterable[dict]) -> list[Event]:
    return [
        Event(e["id"], parse_ts(e["commence_time"]), e["home_team"], e["away_team"]) for e in payload
    ]


def _job_id(kind: str, run_at: datetime, event_ids: Sequence[str]) -> str:
    raw = f"{kind}|{run_at.astimezone(timezone.utc).isoformat()}|{','.join(sorted(event_ids))}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def tip_clusters(events: Sequence[Event], window_min: int) -> list[list[Event]]:
    clusters: list[list[Event]] = []
    for ev in sorted(events, key=lambda e: (e.commence_time, e.id)):
        if clusters and ev.commence_time - clusters[-1][0].commence_time <= timedelta(minutes=window_min):
            clusters[-1].append(ev)
        else:
            clusters.append([ev])
    return clusters


def draw_prop_games(events: Sequence[Event], date_et: date, n: int, salt: str) -> list[str]:
    """Pre-registered random draw: seeded by the date, not chosen after seeing lines."""
    ids = sorted(e.id for e in events)
    seed = int(hashlib.sha256(f"{salt}|{date_et.isoformat()}".encode()).hexdigest(), 16)
    return sorted(random.Random(seed).sample(ids, min(n, len(ids))))


def decision_time(ev: Event, cfg: C.FreeTierSchedule) -> datetime:
    tip = ev.tip_et
    if tip.time() < cfg.early_tip_cutoff_et:
        return (tip - timedelta(minutes=cfg.early_decision_lead_min)).astimezone(timezone.utc)
    noon = datetime.combine(tip.date(), cfg.decision_time_et, tzinfo=ET)
    return noon.astimezone(timezone.utc)


def days_left_in_month(today: date) -> int:
    return calendar.monthrange(today.year, today.month)[1] - today.day + 1


def daily_allowance(remaining: int, reserve: int, today: date,
                    baseline: float = C.BASELINE_SIDES_CREDITS_PER_DAY) -> float:
    """Credits today may use while still protecting a sides-lane baseline for every
    remaining day of the month: quiet nights fund big ones, never the reverse."""
    return max(0.0, remaining - reserve - baseline * (days_left_in_month(today) - 1))


def plan_day(
    events: Iterable[Event],
    date_et: date,
    cfg: C.FreeTierSchedule = C.FREE_TIER,
    allowance: float | None = None,
) -> tuple[list[SnapshotJob], list[str]]:
    day = [e for e in events if e.tip_et.date() == date_et]
    warnings: list[str] = []
    jobs: list[SnapshotJob] = []
    if not day:
        return jobs, warnings
    ds = date_et.isoformat()
    sides_cost = estimate_cost(cfg.sides_markets, cfg.sides_books)
    props_cost = estimate_cost(cfg.props_markets, cfg.props_books)

    def add(kind, run_at, deadline, ids, markets, books, endpoint, cost, note=""):
        jobs.append(SnapshotJob(_job_id(kind, run_at, ids), ds, kind, run_at, deadline, tuple(ids),
                                tuple(markets), tuple(books), endpoint, cost, PRIORITY[kind], note))

    # Sides/totals decision: one featured call per distinct decision time.
    by_time: dict[datetime, list[str]] = {}
    for ev in day:
        by_time.setdefault(decision_time(ev, cfg), []).append(ev.id)
    for run_at, ids in sorted(by_time.items()):
        add("sides_decision", run_at, run_at + timedelta(minutes=cfg.decision_window_min),
            sorted(ids), cfg.sides_markets, cfg.sides_books, "featured", sides_cost)

    # Sides/totals close: T-5 per tip cluster.
    for cl in tip_clusters(day, cfg.cluster_window_min):
        first = cl[0].commence_time
        add("sides_close", first - timedelta(minutes=cfg.close_lead_min), first,
            sorted(e.id for e in cl), cfg.sides_markets, cfg.sides_books, "featured", sides_cost)

    # Props archive: seeded draw, decision + close per game.
    picked = draw_prop_games(day, date_et, cfg.props_games_per_day, cfg.draw_salt)
    note = f"seeded draw {cfg.props_games_per_day} of {len(day)}"
    for ev in (e for e in day if e.id in picked):
        run_at = decision_time(ev, cfg)
        add("props_decision", run_at, min(run_at + timedelta(minutes=cfg.decision_window_min),
                                          ev.commence_time),
            [ev.id], cfg.props_markets, cfg.props_books, "event", props_cost, note)
        add("props_close", ev.commence_time - timedelta(minutes=cfg.close_lead_min), ev.commence_time,
            [ev.id], cfg.props_markets, cfg.props_books, "event", props_cost, note)

    if allowance is not None:
        jobs, more = _fit_allowance(jobs, allowance, ds, cfg)
        warnings += more
    return sorted(jobs, key=lambda j: (j.run_at, j.priority)), warnings


def _fit_allowance(jobs: list[SnapshotJob], allowance: float, ds: str,
                   cfg: C.FreeTierSchedule) -> tuple[list[SnapshotJob], list[str]]:
    """Cut coverage in a way that never strands an opinion without its close.

    1. Drop the props archive (decision and close together).
    2. Sides lane goes spreads-only: totals leave BOTH the decision and every close,
       so no totals opinion is logged that could not be closed.
    3. If spreads-only still does not fit, keep the plan and flag upgrade trigger 1:
       the runtime guard protects closes, and a skipped close is better recorded
       than silently never attempted.
    """
    warnings: list[str] = []

    def total(js):
        return sum(j.est_cost for j in js)

    if total(jobs) > allowance:
        props = [j for j in jobs if j.kind.startswith("props")]
        if props:
            jobs = [j for j in jobs if not j.kind.startswith("props")]
            warnings.append(f"Quota limits coverage on {ds}: props archive dropped "
                            f"({len(props)} jobs) to fit a {allowance:.0f}-credit daily allowance")
    if total(jobs) > allowance and any("totals" in j.markets for j in jobs):
        cost_spreads = estimate_cost(["spreads"], cfg.sides_books)
        jobs = [
            SnapshotJob(
                _job_id(j.kind, j.run_at, j.event_ids) + "s", j.date_et, j.kind, j.run_at, j.deadline,
                j.event_ids, tuple(m for m in j.markets if m != "totals"), j.books, j.endpoint,
                cost_spreads, j.priority, (j.note + " spreads-only").strip(),
            ) if j.kind.startswith("sides") else j
            for j in jobs
        ]
        warnings.append(f"Quota limits coverage on {ds}: sides lane spreads-only (totals not logged today)")
    if total(jobs) > allowance:
        warnings.append(f"Sides closes alone need {total(jobs)} credits on {ds} vs a {allowance:.0f} "
                        "allowance. Upgrade trigger 1: move to the 20K plan.")
    return jobs, warnings


def projected_monthly_credits(
    game_days: int, games: int, clusters_per_day: float, cfg: C.FreeTierSchedule = C.FREE_TIER
) -> float:
    """Planning estimate: $C = D \\cdot M_{st}(1 + K) + G_p \\cdot M_p \\cdot 2$."""
    sides = estimate_cost(cfg.sides_markets, cfg.sides_books)
    props = estimate_cost(cfg.props_markets, cfg.props_books)
    prop_games = min(games, game_days * cfg.props_games_per_day)
    return game_days * sides * (1 + clusters_per_day) + prop_games * props * 2
