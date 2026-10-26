"""Approved one-off snapshot requests (on top of the Option A plan).

A request file (default `requests/extra_snapshots.json`) lists markets Chris approved for a
specific game outside the free-tier plan. For each request on today's slate the runner adds:

  extra_decision  the requested markets at `decision_at_et` (or as soon as the runner sees it,
                  up to `decision_deadline_et`, never after tip)
  extra_close     the same markets at T-5, deadline at tip

Each job is one event-odds call: markets x regions credits (1 market, <=10 books = 1 credit).
Opinions listed in the request are logged after the decision snapshot, record-only
(pass_reason = model_hold), so the close can grade them. The close is scheduled before any
opinion is logged, so every logged row is close-capturable (spec section 14).

Request format:
{
  "requests": [{
    "id": "2026-10-24-was-chi-reb",
    "date_et": "2026-10-24",
    "home_team": "Chicago Bulls", "away_team": "Washington Wizards",
    "markets": ["player_rebounds"],
    "decision_at_et": "2026-10-24T12:00:00-04:00",
    "decision_deadline_et": "2026-10-24T19:30:00-04:00",
    "approved": "Chris 2026-10-24 11:15 ET",
    "model_version": "sidequest-reb-0",
    "projection_as_of_et": "2026-10-24T11:10:00-04:00",
    "opinions": [{"player": "Josh Giddey", "market": "player_rebounds", "line": 6.5,
                  "p_over_model": 0.615, "notes": "...",
                  "alt_lines": [{"line": 5.5, "p_over_model": 0.75}]}]
  }]
}

`alt_lines` (optional) carries the same projection priced at nearby lines, for when the posted
line moves between the projection and the decision snapshot. The request line is tried first;
one row per player-market is logged, at the first line with a two-way reference (no ladders).
"""

from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from . import config as C
from .devig import OneWayMarketError
from .fees import effective_price, fee_note
from .odds import decimal_to_american
from .odds_api import estimate_cost
from .opinion_log import Opinion, OpinionLog
from .quotes import parse_ts
from .scheduler import ET, SnapshotJob, events_from_payload
from .selection import NoQuoteError, best_price, reference_prob
from .store import DONE, MarketStore

DEFAULT_PATH = Path("requests/extra_snapshots.json")
DECISION, CLOSE = "extra_decision", "extra_close"
PRIORITY = {CLOSE: 3, DECISION: 4}


def load_requests(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return list(json.loads(path.read_text()).get("requests", []))


def _job_id(req_id: str, kind: str) -> str:
    return hashlib.sha1(f"extra|{req_id}|{kind}".encode()).hexdigest()[:16]


def _meta_key(req_id: str) -> str:
    return f"extra:{req_id}"


def _has_meta(store: MarketStore, key: str) -> bool:
    return store.conn.execute("SELECT 1 FROM meta WHERE key = ?", (key,)).fetchone() is not None


def _set_meta(store: MarketStore, key: str, value: str) -> None:
    store.conn.execute("INSERT OR REPLACE INTO meta VALUES (?, ?)", (key, value))
    store.conn.commit()


def schedule_extra(store: MarketStore, client, day: date, now: datetime,
                   path: Path = DEFAULT_PATH, cfg: C.FreeTierSchedule = C.FREE_TIER) -> list[str]:
    """Add the decision and close jobs for today's approved requests (idempotent, free /events call)."""
    todays = [r for r in load_requests(path)
              if r.get("date_et") == day.isoformat() and not _has_meta(store, _meta_key(r["id"]))]
    if not todays:
        return []
    start = datetime.combine(day, datetime.min.time(), tzinfo=ET)
    events = events_from_payload(client.events(start, start + timedelta(days=1)).body or [])
    messages: list[str] = []
    for req in todays:
        ev = next((e for e in events if e.home_team == req["home_team"]
                   and e.away_team == req["away_team"]), None)
        if ev is None:
            messages.append(f"WARNING: extra request {req['id']}: no event "
                            f"{req['away_team']} @ {req['home_team']} on {day}")
            continue
        markets = tuple(req["markets"])
        books = tuple(req.get("books") or cfg.props_books)
        cost = estimate_cost(markets, books)
        tip = ev.commence_time
        dec_at = parse_ts(req["decision_at_et"]).astimezone(timezone.utc)
        dec_deadline = min(parse_ts(req["decision_deadline_et"]).astimezone(timezone.utc),
                           tip - timedelta(minutes=cfg.close_lead_min))
        note = f"extra request {req['id']} approved {req.get('approved', '?')}"
        jobs = [
            SnapshotJob(_job_id(req["id"], DECISION), day.isoformat(), DECISION, dec_at, dec_deadline,
                        (ev.id,), markets, books, "event", cost, PRIORITY[DECISION], note),
            SnapshotJob(_job_id(req["id"], CLOSE), day.isoformat(), CLOSE,
                        tip - timedelta(minutes=cfg.close_lead_min), tip,
                        (ev.id,), markets, books, "event", cost, PRIORITY[CLOSE], note),
        ]
        n = store.upsert_jobs(jobs)
        _set_meta(store, _meta_key(req["id"]), json.dumps({"event_id": ev.id, "scheduled": now.isoformat()}))
        messages.append(f"extra {req['id']}: {n} job(s) for {ev.away_team} @ {ev.home_team} "
                        f"({','.join(markets)}, ~{2 * cost} credits)")
    return messages


def _event_for(store: MarketStore, req: dict) -> str | None:
    row = store.conn.execute("SELECT value FROM meta WHERE key = ?", (_meta_key(req["id"]),)).fetchone()
    return json.loads(row[0])["event_id"] if row else None


def record_extra_opinions(store: MarketStore, log: OpinionLog, day: date,
                          path: Path = DEFAULT_PATH) -> list[str]:
    """Log the request's opinions against its decision snapshot, record-only (idempotent)."""
    messages: list[str] = []
    existing = {(r["game_id"], r["market"], r["model_version"], r["player"], r["line"]) for r in log.rows()}
    for req in load_requests(path):
        if req.get("date_et") != day.isoformat() or not req.get("opinions"):
            continue
        event_id = _event_for(store, req)
        if not event_id:
            continue
        job = next((j for j in store.jobs(status=DONE, date_et=day.isoformat())
                    if j.job_id == _job_id(req["id"], DECISION)), None)
        if job is None:
            continue
        quotes = store.load_quotes(job.snapshot_id, event_id)
        snap_time = max((q.fetched_at for q in quotes), default=None)
        as_of = parse_ts(req["projection_as_of_et"])
        if snap_time is None or snap_time < as_of:
            continue  # as-of rule: price only against a snapshot taken after the projection
        for spec in req["opinions"]:
            # The request line first, then any alt_lines (same projection, priced at a moved line).
            # Exactly one row per player-market is logged: the first line with a two-way reference.
            options = [(spec["line"], spec["p_over_model"])] + [
                (a["line"], a["p_over_model"]) for a in spec.get("alt_lines", [])]
            keys = {(event_id, spec["market"], req["model_version"], spec["player"], ln) for ln, _ in options}
            skip_key = f"{_meta_key(req['id'])}:skip:{spec['player']}:{spec['market']}:{spec['line']}"
            if keys & existing or _has_meta(store, skip_key):
                continue
            chosen = None
            for line, p_over in options:
                cands = {}
                for side, pm in (("Over", p_over), ("Under", 1 - p_over)):
                    try:
                        ref = reference_prob(quotes, event_id=event_id, market=spec["market"], side=side,
                                             line=line, player=spec["player"])
                        pick = best_price(quotes, event_id=event_id, market=spec["market"], side=side,
                                          line=line, player=spec["player"], books=C.BETTABLE_BOOKS)
                    except (NoQuoteError, OneWayMarketError):
                        continue
                    w = C.BLEND_W[C.PROPS]
                    cands[side] = ((w * pm + (1 - w) * ref.p) * effective_price(pick) - 1, pm, ref, pick)
                if cands:
                    chosen = (line, cands)
                    break
            if chosen is None:
                tried = ",".join(f"{ln:g}" for ln, _ in options)
                messages.append(f"extra {req['id']}: {spec['player']} {spec['market']} (lines {tried}): "
                                "no two-way reference with a bettable price at decision; not logged")
                _set_meta(store, skip_key, "no two-way reference")
                continue
            line, cands = chosen
            side = max(cands, key=lambda s: cands[s][0])
            _, pm, ref, pick = cands[side]
            alt_note = f"; alt line (request line {spec['line']:g})" if line != spec["line"] else ""
            fee = fee_note(pick)
            op = Opinion(
                game_id=event_id, market=spec["market"], side=side, line=line, book=pick.book,
                price_american=decimal_to_american(effective_price(pick)),
                p_model=min(max(pm, 1e-4), 1 - 1e-4), p_market_novig=ref.p,
                model_version=req["model_version"], ref_quality=ref.ref_quality,
                player=spec["player"], stat=C.stat_for_market(spec["market"]), pass_reason="model_hold",
                timestamp=snap_time,
                notes=(f"{spec.get('notes', '')}; extra request {req['id']}{alt_note}; "
                       f"ref_books={','.join(ref.books)} (retail reference); bet_state={C.BET_STATE}"
                       + (f"; {fee}" if fee else "")),
            )
            try:
                oid = log.record(op)
            except Exception as err:  # noqa: BLE001 - report and continue; a bad row must not stop the tick
                messages.append(f"rejected extra {spec['player']} {spec['market']}: {err}")
                _set_meta(store, skip_key, f"rejected: {err}")
                continue
            existing.add((event_id, spec["market"], req["model_version"], spec["player"], line))
            row = next(r for r in log.rows() if r["opinion_id"] == oid)
            messages.append(f"logged [{req['model_version']}] {spec['market']} {spec['player']} {side} "
                            f"{line:g} {row['price_american']:+.0f} @{pick.book}: "
                            f"EV {row['ev_pct'] * 100:+.2f}% -> pass (model_hold)")
    return messages
