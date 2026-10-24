"""SQLite storage for quotes, snapshots, scheduled jobs and the credit ledger.

Every stored price carries two timestamps: when we fetched it and the API's
market-level last_update (data discipline: every input is time-stamped).
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .quotes import Quote, parse_ts

SCHEMA = """
CREATE TABLE IF NOT EXISTS quotes (
    snapshot_id TEXT, fetched_at_utc TEXT, event_id TEXT, commence_utc TEXT,
    home_team TEXT, away_team TEXT, book TEXT, market TEXT, side TEXT, line REAL,
    price_decimal REAL, player TEXT, market_last_update_utc TEXT
);
CREATE INDEX IF NOT EXISTS ix_quotes_snapshot ON quotes(snapshot_id);
CREATE INDEX IF NOT EXISTS ix_quotes_event ON quotes(event_id, market);
CREATE TABLE IF NOT EXISTS snapshots (
    snapshot_id TEXT PRIMARY KEY, job_id TEXT, kind TEXT, fetched_at_utc TEXT,
    endpoint TEXT, markets TEXT, books TEXT, event_ids TEXT,
    cost INTEGER, used INTEGER, remaining INTEGER, n_quotes INTEGER
);
CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY, date_et TEXT, kind TEXT, run_at_utc TEXT, deadline_utc TEXT,
    event_ids TEXT, markets TEXT, books TEXT, endpoint TEXT, est_cost INTEGER,
    priority INTEGER, status TEXT, snapshot_id TEXT, note TEXT
);
CREATE TABLE IF NOT EXISTS credit_ledger (
    ts_utc TEXT, path TEXT, est_cost INTEGER, cost_last INTEGER, used INTEGER, remaining INTEGER
);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
"""

PENDING, DONE, MISSED, SKIPPED_BUDGET, ERROR = "pending", "done", "missed", "skipped_budget", "error"


def _iso(ts: datetime | None) -> str | None:
    return ts.astimezone(timezone.utc).isoformat() if ts else None


@dataclass(frozen=True)
class JobRow:
    job_id: str
    date_et: str
    kind: str
    run_at: datetime
    deadline: datetime
    event_ids: tuple[str, ...]
    markets: tuple[str, ...]
    books: tuple[str, ...]
    endpoint: str
    est_cost: int
    priority: int
    status: str
    snapshot_id: str | None
    note: str | None


class MarketStore:
    def __init__(self, db_path: str | Path):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path))
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    # --- credit ledger (satisfies odds_api.Ledger) -------------------------------
    @property
    def remaining(self) -> int | None:
        row = self.conn.execute(
            "SELECT remaining FROM credit_ledger WHERE remaining IS NOT NULL ORDER BY rowid DESC LIMIT 1"
        ).fetchone()
        return None if row is None else int(row[0])

    def record(self, *, ts: datetime, path: str, est_cost: int, cost_last: int | None,
               used: int | None, remaining: int | None) -> None:
        self.conn.execute(
            "INSERT INTO credit_ledger VALUES (?, ?, ?, ?, ?, ?)",
            (_iso(ts), path, est_cost, cost_last, used, remaining),
        )
        self.conn.commit()

    def credits_spent_since(self, since: datetime) -> int:
        row = self.conn.execute(
            "SELECT COALESCE(SUM(cost_last), 0) FROM credit_ledger WHERE ts_utc >= ?", (_iso(since),)
        ).fetchone()
        return int(row[0])

    # --- quotes and snapshots ------------------------------------------------------
    def save_snapshot(self, *, snapshot_id: str, job_id: str | None, kind: str, fetched_at: datetime,
                      endpoint: str, markets: Sequence[str], books: Sequence[str],
                      event_ids: Sequence[str], cost: int | None, used: int | None,
                      remaining: int | None, quotes: Iterable[Quote]) -> int:
        quotes = list(quotes)
        self.conn.executemany(
            "INSERT INTO quotes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (q.snapshot_id, _iso(q.fetched_at), q.event_id, _iso(q.commence_time), q.home_team,
                 q.away_team, q.book, q.market, q.side, q.line, q.price_decimal, q.player,
                 _iso(q.market_last_update))
                for q in quotes
            ],
        )
        self.conn.execute(
            "INSERT INTO snapshots VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (snapshot_id, job_id, kind, _iso(fetched_at), endpoint, json.dumps(list(markets)),
             json.dumps(list(books)), json.dumps(list(event_ids)), cost, used, remaining, len(quotes)),
        )
        self.conn.commit()
        return len(quotes)

    def load_quotes(self, snapshot_id: str, event_id: str | None = None) -> list[Quote]:
        sql, args = "SELECT * FROM quotes WHERE snapshot_id = ?", [snapshot_id]
        if event_id is not None:
            sql += " AND event_id = ?"
            args.append(event_id)
        return [
            Quote(
                snapshot_id=r["snapshot_id"], fetched_at=parse_ts(r["fetched_at_utc"]),
                event_id=r["event_id"], commence_time=parse_ts(r["commence_utc"]),
                home_team=r["home_team"], away_team=r["away_team"], book=r["book"],
                market=r["market"], side=r["side"], line=r["line"],
                price_decimal=r["price_decimal"], player=r["player"],
                market_last_update=parse_ts(r["market_last_update_utc"]),
            )
            for r in self.conn.execute(sql, args).fetchall()
        ]

    # --- jobs ------------------------------------------------------------------------
    def upsert_jobs(self, jobs: Iterable) -> int:
        n = 0
        for j in jobs:
            cur = self.conn.execute(
                "INSERT OR IGNORE INTO jobs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (j.job_id, j.date_et, j.kind, _iso(j.run_at), _iso(j.deadline),
                 json.dumps(list(j.event_ids)), json.dumps(list(j.markets)), json.dumps(list(j.books)),
                 j.endpoint, j.est_cost, j.priority, PENDING, None, j.note),
            )
            n += cur.rowcount
        self.conn.commit()
        return n

    def _job(self, r: sqlite3.Row) -> JobRow:
        return JobRow(
            job_id=r["job_id"], date_et=r["date_et"], kind=r["kind"],
            run_at=parse_ts(r["run_at_utc"]), deadline=parse_ts(r["deadline_utc"]),
            event_ids=tuple(json.loads(r["event_ids"])), markets=tuple(json.loads(r["markets"])),
            books=tuple(json.loads(r["books"])), endpoint=r["endpoint"], est_cost=r["est_cost"],
            priority=r["priority"], status=r["status"], snapshot_id=r["snapshot_id"], note=r["note"],
        )

    def jobs(self, status: str | None = None, date_et: str | None = None) -> list[JobRow]:
        sql, args = "SELECT * FROM jobs WHERE 1=1", []
        if status:
            sql += " AND status = ?"
            args.append(status)
        if date_et:
            sql += " AND date_et = ?"
            args.append(date_et)
        sql += " ORDER BY run_at_utc, priority"
        return [self._job(r) for r in self.conn.execute(sql, args).fetchall()]

    def mark_job(self, job_id: str, status: str, *, snapshot_id: str | None = None,
                 note: str | None = None) -> None:
        self.conn.execute(
            "UPDATE jobs SET status = ?, snapshot_id = COALESCE(?, snapshot_id), "
            "note = COALESCE(?, note) WHERE job_id = ?",
            (status, snapshot_id, note, job_id),
        )
        self.conn.commit()

    def close_job_for(self, event_id: str, kind: str) -> JobRow | None:
        for j in self.jobs():
            if j.kind == kind and event_id in j.event_ids:
                return j
        return None

    def close_job_for_market(self, event_id: str, kinds: tuple[str, ...], market: str) -> JobRow | None:
        """The close job that actually covers this market (a game can have a points close and an
        approved extra rebounds close; the first job of the kind is not necessarily the right one)."""
        for j in self.jobs():
            if j.kind in kinds and event_id in j.event_ids and market in j.markets:
                return j
        return None

    def is_planned(self, date_et: str) -> bool:
        return self.conn.execute("SELECT 1 FROM meta WHERE key = ?", (f"planned:{date_et}",)).fetchone() is not None

    def mark_planned(self, date_et: str) -> None:
        self.conn.execute("INSERT OR REPLACE INTO meta VALUES (?, ?)",
                          (f"planned:{date_et}", datetime.now(timezone.utc).isoformat()))
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()
