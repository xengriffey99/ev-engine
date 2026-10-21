"""Snapshot runner. Run `run-due` every 5 minutes (cron / Task Scheduler).

  python -m market.run_snapshots plan --date 2026-10-22      # free /events call, stores jobs
  python -m market.run_snapshots run-due                     # executes jobs that are due
  python -m market.run_snapshots opinions --payload data/payload_2026-10-22.json
                                                             # model output -> opinion rows (after 12:00 ET)
  python -m market.run_snapshots close-out                   # fills closes + CLV on open opinions
  python -m market.run_snapshots export --out data/opinion_log.csv
  python -m market.run_snapshots status

A job not executed before its deadline is marked missed: a late close is not
a close. Opinions whose close was missed stay in the log as `close_missing`.
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from . import config as C
from .clv import compute_clv
from .model_adapter import build_baseline_opinions, build_opinions, from_nba_sim_payload, match_events
from .odds_api import BudgetExceeded, OddsApiClient, OddsApiError, estimate_cost
from .opinion_log import OpinionLog, OpinionLogError
from .quotes import parse_odds_payload
from .scheduler import ET, daily_allowance, events_from_payload, plan_day
from .store import DONE, ERROR, MISSED, PENDING, SKIPPED_BUDGET, MarketStore

DEFAULT_DB = Path("data/market.db")


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def plan(store: MarketStore, client: OddsApiClient, day: date,
         cfg: C.FreeTierSchedule = C.FREE_TIER) -> tuple[int, list[str]]:
    start = datetime.combine(day, datetime.min.time(), tzinfo=ET)
    resp = client.events(start, start + timedelta(days=1))  # free
    events = events_from_payload(resp.body or [])
    remaining = store.remaining
    allowance = None if remaining is None else daily_allowance(remaining, cfg.reserve_credits, day)
    jobs, warnings = plan_day(events, day, cfg, allowance)
    return store.upsert_jobs(jobs), warnings


def run_due(store: MarketStore, client: OddsApiClient, now: datetime | None = None) -> list[str]:
    now = now or now_utc()
    messages: list[str] = []
    for job in store.jobs(status=PENDING):
        if now < job.run_at:
            continue
        if now >= job.deadline:
            store.mark_job(job.job_id, MISSED, note=f"runner first saw it at {now.isoformat()}")
            messages.append(f"MISSED {job.kind} {job.job_id} (deadline {job.deadline.isoformat()})")
            continue
        protected = job.kind == "sides_close"
        snapshot_id = uuid.uuid4().hex
        try:
            if job.endpoint == "featured":
                resps = [client.featured_odds(job.markets, job.books, event_ids=job.event_ids,
                                              protected=protected)]
            else:
                resps = [client.event_odds(eid, job.markets, job.books, protected=protected)
                         for eid in job.event_ids]
        except BudgetExceeded as err:
            store.mark_job(job.job_id, SKIPPED_BUDGET, note=str(err))
            tag = " -- upgrade trigger 1" if protected else ""
            messages.append(f"SKIPPED {job.kind} {job.job_id}: {err}{tag}")
            continue
        except OddsApiError as err:
            if err.retryable:
                # Transient (5xx, 429, network): stays pending and is retried on the next
                # tick until its deadline, after which it is marked missed as usual.
                store.mark_job(job.job_id, PENDING, note=f"retrying: {err}")
                messages.append(f"RETRY {job.kind} {job.job_id}: {err}")
            else:
                store.mark_job(job.job_id, ERROR, note=str(err))
                messages.append(f"ERROR {job.kind} {job.job_id}: {err}")
            continue
        quotes = []
        for r in resps:
            if r.body:
                quotes.extend(parse_odds_payload(r.body, snapshot_id, r.fetched_at))
        last = resps[-1]
        n = store.save_snapshot(
            snapshot_id=snapshot_id, job_id=job.job_id, kind=job.kind, fetched_at=last.fetched_at,
            endpoint=job.endpoint, markets=job.markets, books=job.books, event_ids=job.event_ids,
            cost=sum(r.cost_last or 0 for r in resps), used=last.used, remaining=last.remaining,
            quotes=quotes,
        )
        store.mark_job(job.job_id, DONE, snapshot_id=snapshot_id)
        messages.append(f"DONE {job.kind} {job.job_id}: {n} quotes, {last.remaining} credits left")
    return messages


def close_out_opinions(store: MarketStore, log: OpinionLog) -> list[str]:
    messages: list[str] = []
    for row in log.open_rows():
        kind = "sides_close" if C.market_class(row["market"]) == C.SIDES_TOTALS else "props_close"
        job = store.close_job_for(row["game_id"], kind)
        if job is None or job.status in (MISSED, SKIPPED_BUDGET, ERROR):
            log.mark_close_missing(row["opinion_id"])
            messages.append(f"close_missing {row['opinion_id']} ({job.status if job else 'no close job'})")
            continue
        if job.status != DONE:
            continue  # close not taken yet
        quotes = store.load_quotes(job.snapshot_id, row["game_id"])
        clv = compute_clv(
            quotes, event_id=row["game_id"], market=row["market"], side=row["side"],
            bet_line=row["line"], bet_price_american=row["price_american"], player=row["player"],
            method=row["devig_method"],
        )
        log.close_out(row["opinion_id"], clv)
        messages.append(f"closed {row['opinion_id']}: {clv.clv_method} clv={clv.clv_pct}")
    return messages


def record_opinions(store: MarketStore, log: OpinionLog, projections, day: date, *,
                    hold_markets: frozenset[str] = frozenset(), stake: bool = True) -> list[str]:
    """Build and log sides/totals opinions from the day's decision snapshot(s).

    Each projection is priced against the EARLIEST completed decision snapshot taken at or
    after it was made (as-of rule, NN9). Markets in hold_markets are logged as passes
    (pass_reason model_hold) so their closes and CLV are still captured. Gate passes are
    staked at 1/16 Kelly with the per-game and shared-thesis caps (staking.py).
    Re-running is safe: a (game, market, model_version) already logged is skipped.
    """
    from .odds import american_to_decimal
    from .staking import cap_groups, round_units, stake_units

    messages: list[str] = []
    done = [j for j in store.jobs(status=DONE, date_et=day.isoformat()) if j.kind == "sides_decision"]
    if not done:
        return [f"No completed sides_decision snapshot for {day}; run `run-due` after the decision time"]
    snaps = []
    for job in done:
        quotes = store.load_quotes(job.snapshot_id)
        if quotes:
            snaps.append((max(q.fetched_at for q in quotes), quotes))
    snaps.sort(key=lambda s: s[0])
    existing = {(r["game_id"], r["market"], r["model_version"], r["player"]) for r in log.rows()}
    pending: list[tuple] = []  # (opinion, projection)
    unmatched: list[str] = []
    for p in projections:
        chosen = None
        for snap_time, quotes in snaps:
            if snap_time < p.as_of:
                continue
            if match_events([p], quotes):
                chosen = quotes
                break
        if chosen is None:
            later = any(t >= p.as_of for t, _ in snaps)
            why = ("not in any decision snapshot taken after the projection" if later else
                   f"no decision snapshot at or after the projection ({p.as_of.isoformat()}); "
                   "take one with `decide`")
            unmatched.append(f"skipped {p.model_game_id}: {why}")
            continue
        opinions, skipped = build_opinions([p], chosen)
        messages += [f"skipped {s}" for s in skipped]
        pending += [(op, p) for op in opinions]

    # Gate, hold and stake.
    raw, groups = [], []
    for op, p in pending:
        cls = C.market_class(op.market)
        w = C.BLEND_W[cls]
        p_bet = w * op.p_model + (1.0 - w) * op.p_market_novig
        dec = american_to_decimal(op.price_american)
        ev = p_bet * dec - 1.0
        if op.market in hold_markets:
            op.pass_reason = "model_hold"
        passes = op.pass_reason is None and ev >= C.EDGE_GATE[cls]
        raw.append(stake_units(p_bet, dec) if (passes and stake) else 0.0)
        if op.market in C.SPREAD_LIKE_MARKETS:
            direction = "home" if op.side == p.home else "away"
        else:
            direction = op.side.lower()
        groups.append((op.model_version, cls, direction))
    capped = cap_groups(raw, groups, C.SHARED_THESIS_CAP_U)
    per_game: dict[str, float] = {}
    for (op, p), s_raw, s_cap, g in zip(pending, raw, capped, groups):
        s = round_units(min(s_cap, C.PER_GAME_CAP_U - per_game.get(op.game_id, 0.0)))
        if s_raw > 0:
            op.stake_u = s
            per_game[op.game_id] = per_game.get(op.game_id, 0.0) + s
            op.notes += (f"; stake 1/16 Kelly {s_raw:.2f}u -> {s:.2f}u after shared-thesis cap "
                         f"{C.SHARED_THESIS_CAP_U}u on {g[1]}/{g[2]}")
    messages += _log_all(log, [op for op, _ in pending], existing)
    return messages + unmatched


def decide_now(store: MarketStore, client: OddsApiClient, day: date, projections,
               now: datetime | None = None, cfg: C.FreeTierSchedule = C.FREE_TIER) -> list[str]:
    """On-demand decision snapshot for a projection made after the scheduled decision time.

    Adds one sides_decision job (spreads + totals, the sides book list) for the events still
    before tip and runs it at once. Not protected: the budget guard keeps the reserve.
    """
    now = now or now_utc()
    events = {}
    for job in store.jobs(date_et=day.isoformat()):
        if job.kind == "sides_close" and job.deadline > now:
            for eid in job.event_ids:
                events[eid] = job.deadline  # deadline = tip of the cluster
    if not events:
        return [f"No games left before tip on {day}"]
    from .scheduler import SnapshotJob, PRIORITY, _job_id
    ids = sorted(events)
    deadline = min(min(events.values()), now + timedelta(minutes=cfg.decision_window_min))
    job = SnapshotJob(_job_id("sides_decision", now, ids), day.isoformat(), "sides_decision", now, deadline,
                      tuple(ids), tuple(cfg.sides_markets), tuple(cfg.sides_books), "featured",
                      estimate_cost(cfg.sides_markets, cfg.sides_books), PRIORITY["sides_decision"],
                      "model decision: projection made after the scheduled decision snapshot")
    store.upsert_jobs([job])
    return run_due(store, client, now)


def _log_all(log: OpinionLog, opinions, existing: set, quiet_dupes: bool = False) -> list[str]:
    messages: list[str] = []
    for op in opinions:
        key = (op.game_id, op.market, op.model_version, op.player)
        if key in existing:
            if not quiet_dupes:
                messages.append(f"already logged {op.game_id} {op.market}" + (f" {op.player}" if op.player else ""))
            continue
        try:
            oid = log.record(op)
        except OpinionLogError as err:
            messages.append(f"rejected {op.game_id} {op.market}: {err}")
            continue
        existing.add(key)
        row = next(r for r in log.rows() if r["opinion_id"] == oid)
        verdict = "BET-QUALIFIED" if row["gate_pass"] else f"pass ({row['pass_reason']})"
        line = f"{op.line:+g}" if op.market in C.SPREAD_LIKE_MARKETS else f"{op.line:g}"
        who = f"{op.player} " if op.player else ""
        messages.append(
            f"logged [{op.model_version}] {op.market} {who}{op.side} {line} {row['price_american']:+.0f} "
            f"@{op.book}: EV {row['ev_pct'] * 100:+.2f}% -> {verdict}"
        )
    return messages


def record_baseline_opinions(store: MarketStore, log: OpinionLog, day: date) -> list[str]:
    """Market-baseline rows for every game in each completed decision snapshot (idempotent)."""
    existing = {(r["game_id"], r["market"], r["model_version"], r["player"]) for r in log.rows()}
    messages: list[str] = []
    for job in store.jobs(status=DONE, date_et=day.isoformat()):
        if job.kind != "sides_decision":
            continue
        opinions, skipped = build_baseline_opinions(store.load_quotes(job.snapshot_id))
        messages += _log_all(log, opinions, existing, quiet_dupes=True)
        if any((o.game_id, o.market, o.model_version, o.player) not in existing for o in opinions):
            messages += [f"skipped {s}" for s in skipped]
    return messages


ACTIVE_START = (8, 45)  # ET; nothing is scheduled before the 09:00 plan
ACTIVE_END = (2, 0)  # ET, next day; last closes are ~22:35-23:05
OUTSIDE_WINDOW = 10  # exit code for `tick` outside the active window


def in_active_window(now: datetime) -> bool:
    t = now.astimezone(ET)
    hm = (t.hour, t.minute)
    return hm >= ACTIVE_START or hm < ACTIVE_END


def slate_day(now: datetime) -> date:
    """Before 02:00 ET still belongs to the previous slate day."""
    t = now.astimezone(ET)
    return (t - timedelta(days=1)).date() if (t.hour, t.minute) < ACTIVE_END else t.date()


def tick(store: MarketStore, client: OddsApiClient, log: OpinionLog, now: datetime | None = None,
         payload_dir: Path = Path("data/payloads")) -> list[str]:
    """One pass of everything: plan once per day, take due snapshots, log, close out."""
    now = now or now_utc()
    day = slate_day(now)
    messages: list[str] = []
    if not store.is_planned(day.isoformat()) and now.astimezone(ET).date() == day:
        n, warnings = plan(store, client, day)
        store.mark_planned(day.isoformat())
        messages.append(f"planned {day}: {n} job(s), credits remaining {store.remaining}")
        messages += [f"WARNING: {w}" for w in warnings]
    messages += run_due(store, client, now)
    messages += record_baseline_opinions(store, log, day)
    payload = payload_dir / f"payload_{day.isoformat()}.json"
    if payload.exists():
        projections = from_nba_sim_payload(json.loads(payload.read_text()))
        msgs = record_opinions(store, log, projections, day)
        messages += [m for m in msgs if not m.startswith(("already logged", "No completed"))]
    messages += close_out_opinions(store, log)
    from . import pilot
    messages += pilot.record_pilot(log.db_path, log.rows(), day)
    messages += grade_results(store, log, now)
    return messages


RESULT_FETCH_EVERY = timedelta(minutes=15)
RESULT_DELAY = timedelta(hours=2, minutes=45)  # tip + this before a game is expected final


def event_teams(store: MarketStore) -> dict[str, tuple[str, str]]:
    rows = store.conn.execute("SELECT DISTINCT event_id, home_team, away_team FROM quotes").fetchall()
    return {r["event_id"]: (r["home_team"], r["away_team"]) for r in rows}


def grade_results(store: MarketStore, log: OpinionLog, now: datetime, fetch=None) -> list[str]:
    """Fill `result` on game-market rows once their games are final (free ESPN scoreboard)."""
    from .quotes import parse_ts
    from .results import fetch_finals, grade_rows

    open_rows = [r for r in log.rows() if not r["result"] and r["market"] in (C.SPREAD_LIKE_MARKETS | C.TOTAL_LIKE_MARKETS)
                 and r["clv_method"] is not None]
    if not open_rows:
        return []
    tips = {r["event_id"]: parse_ts(r["commence_utc"]) for r in
            store.conn.execute("SELECT DISTINCT event_id, commence_utc FROM quotes").fetchall()}
    due = [r for r in open_rows if r["game_id"] in tips and now >= tips[r["game_id"]] + RESULT_DELAY]
    if not due:
        return []
    last = store.conn.execute("SELECT value FROM meta WHERE key = 'results_last_fetch'").fetchone()
    if last and now - parse_ts(last[0]) < RESULT_FETCH_EVERY:
        return []
    store.conn.execute("INSERT OR REPLACE INTO meta VALUES ('results_last_fetch', ?)", (now.isoformat(),))
    store.conn.commit()
    fetch = fetch or fetch_finals
    days = sorted({tips[r["game_id"]].astimezone(ET).date() for r in due})
    finals = []
    for d in days:
        try:
            finals += fetch(d)
        except Exception as err:  # network: retry on a later tick
            return [f"results fetch failed for {d}: {err}"]
    graded = grade_rows(due, event_teams(store), finals)
    for oid, res in graded.items():
        log.set_result(oid, res)
    return [f"graded {len(graded)} row(s)"] if graded else []


def close_plan_from(store: MarketStore):
    """Opinion-log gate: only markets with a scheduled close snapshot can be logged."""
    def plan_fn(game_id: str, market: str) -> bool:
        kind = "sides_close" if C.market_class(market) == C.SIDES_TOTALS else "props_close"
        job = store.close_job_for(game_id, kind)
        return job is not None and market in job.markets and job.status in (PENDING, DONE)
    return plan_fn


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="market.run_snapshots")
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_plan = sub.add_parser("plan")
    p_plan.add_argument("--date", type=date.fromisoformat, default=datetime.now(ET).date())
    sub.add_parser("run-due")
    p_op = sub.add_parser("opinions")
    p_op.add_argument("--payload", type=Path, required=True, help="nba_sim payload_<date>.json")
    p_op.add_argument("--decide-now", action="store_true",
                      help="take an on-demand decision snapshot first (projection made after noon)")
    p_op.add_argument("--hold", default="", help="comma-separated markets logged as model_hold passes")
    sub.add_parser("close-out")
    p_tick = sub.add_parser("tick", help="plan-if-needed + run-due + log + close-out (prints only changes)")
    p_tick.add_argument("--payload-dir", type=Path, default=Path("data/payloads"))
    p_exp = sub.add_parser("export")
    p_exp.add_argument("--out", type=Path, default=Path("data/opinion_log.csv"))
    sub.add_parser("status")
    p_pil = sub.add_parser("pilot", help="pilot tier: status | picks --date | placed --id --book --price")
    p_pil.add_argument("action", choices=["status", "picks", "placed"])
    p_pil.add_argument("--date", type=date.fromisoformat, default=datetime.now(ET).date())
    p_pil.add_argument("--id")
    p_pil.add_argument("--book")
    p_pil.add_argument("--price", type=float)
    p_sum = sub.add_parser("summary", help="aggregate status with no prices (safe to publish)")
    p_sum.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    store = MarketStore(args.db)
    client = OddsApiClient(ledger=store)
    if args.cmd == "plan":
        n, warnings = plan(store, client, args.date)
        print(f"{n} new job(s) for {args.date}")
        for w in warnings:
            print(f"WARNING: {w}")
    elif args.cmd == "run-due":
        for m in run_due(store, client):
            print(m)
    elif args.cmd == "opinions":
        payload = json.loads(args.payload.read_text())
        projections = from_nba_sim_payload(payload)
        log = OpinionLog(args.db, close_plan=close_plan_from(store))
        day = date.fromisoformat(payload["slate_date"])
        if args.decide_now:
            for m in decide_now(store, client, day, projections):
                print(m)
        holds = frozenset(x.strip() for x in args.hold.split(",") if x.strip())
        for m in record_opinions(store, log, projections, day, hold_markets=holds):
            print(m)
    elif args.cmd in ("close-out", "export"):
        log = OpinionLog(args.db, close_plan=close_plan_from(store))
        if args.cmd == "close-out":
            for m in close_out_opinions(store, log):
                print(m)
        else:
            print(log.export_csv(args.out))
    elif args.cmd == "tick":
        now = now_utc()
        if not in_active_window(now):
            return OUTSIDE_WINDOW
        log = OpinionLog(args.db, close_plan=close_plan_from(store))
        try:
            msgs = tick(store, client, log, now, args.payload_dir)
        except OddsApiError as err:
            msgs = [f"ERROR: {err}"]
        for m in msgs:
            print(m)
    elif args.cmd == "status":
        print(f"credits remaining (last seen): {store.remaining}")
        for j in store.jobs():
            print(f"{j.date_et} {j.kind:15s} {j.run_at.astimezone(ET):%H:%M} ET  {j.status:14s} cost~{j.est_cost}")
    elif args.cmd == "pilot":
        from . import pilot
        log = OpinionLog(args.db, close_plan=close_plan_from(store))
        if args.action == "status":
            print(json.dumps(pilot.status(args.db, log.rows(), args.date), indent=1, default=str))
        elif args.action == "picks":
            for b in pilot.picks(args.db, args.date):
                print(f"{b['status']:11s} {b['side']} {b['line']:+g} @{b['book']} {b['price_american']:+.0f} "
                      f"pilot {b['pilot_stake_u']:.2f}u (paper {b['paper_stake_u']:.2f}u) {b['note'] or ''}")
        else:
            pilot.mark_placed(args.db, args.id, args.book, args.price, now_utc())
            print(f"marked {args.id} placed at {args.price:+.0f} @{args.book}")
    elif args.cmd == "summary":
        log = OpinionLog(args.db, close_plan=close_plan_from(store))
        text = summary_markdown(store, log)
        if args.out:
            args.out.write_text(text)
        print(text)
    return 0


def summary_markdown(store: MarketStore, log: OpinionLog) -> str:
    """Counts only: no prices, lines or probabilities (odds data stays encrypted)."""
    from collections import Counter

    jobs = store.jobs()
    rows = log.rows()
    lines = [
        "# NBA EV runner status",
        f"Updated {now_utc().astimezone(ET):%Y-%m-%d %H:%M} ET | credits remaining (last seen): {store.remaining}",
        "",
        "| Day | Planned | Done | Missed | Skipped (budget) | Errors |",
        "|---|---|---|---|---|---|",
    ]
    by_day: dict[str, Counter] = {}
    for j in jobs:
        by_day.setdefault(j.date_et, Counter())[j.status] += 1
    for d in sorted(by_day)[-14:]:
        c = by_day[d]
        lines.append(f"| {d} | {sum(c.values())} | {c['done']} | {c['missed']} | {c['skipped_budget']} | {c['error']} |")
    closed = [r for r in rows if r["clv_method"] is not None]
    captured = [r for r in closed if r["clv_method"] != "close_missing"]
    lines += [
        "",
        "| Opinion rows | Value |",
        "|---|---|",
        f"| Logged | {len(rows)} |",
        f"| Closed out | {len(closed)} |",
        f"| Close captured | {len(captured)} ({len(captured) / len(closed) * 100:.1f}%) |" if closed
        else "| Close captured | n/a |",
        f"| Gate passes | {sum(1 for r in rows if r['gate_pass'])} |",
    ]
    from . import pilot
    try:
        st = pilot.status(log.db_path, rows, now_utc().astimezone(ET).date())
        checks = ", ".join(f"{k} {'ok' if c['passed'] else 'pending'}" for k, c in st["checks"].items())
        lines += ["", f"Pilot tier: {'ACTIVE' if st['active'] else 'inactive'} ({checks}; "
                      f"kill switch {'STOP' if st['kill_switch']['stop'] else 'ok'} at n={st['kill_switch']['n']})"]
    except Exception as err:  # status must never break the summary
        lines += ["", f"Pilot tier: status unavailable ({type(err).__name__})"]
    by_model = Counter(r["model_version"] for r in rows)
    lines += ["", "| Model version | Rows |", "|---|---|"] + [f"| {m} | {n} |" for m, n in by_model.items()]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    try:
        sys.exit(main())
    except OddsApiError as err:
        print(f"ERROR: {err}", file=sys.stderr)
        sys.exit(2)
