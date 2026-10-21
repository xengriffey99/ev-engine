"""Real-money pilot tier (approved by Chris 2026-10-21; NBA_CONTEXT.md section 15).

Scope: spreads from an nba-sim model only (totals and props stay held). A logged gate pass becomes
a pilot pick at half its paper stake (1/32 Kelly), capped at 0.25u per bet and 1u per day, only
on days when every entry check has passed:

  close_capture  previous slate closed >= 95% of its opinion rows (automatic, from the log)
  novig_fee      Novig's pregame fee confirmed in the app (manual: pilot_checks.json)
  home_lean      the sides home lean diagnosed, and fixed or kept under the shared-thesis cap
                 (manual: pilot_checks.json)

Stops (checked before any new pick):
  drawdown pause  cumulative pilot P/L <= -5u (risk control; results never grade the model)
  CLV kill switch at n >= 50 closed pilot opinions: mean CLV < 0, or the model trails the
                  line-shopping baseline on the same game and market at t < -1
Full promotion (1/16 Kelly) still needs the section 4 CLV test and the calibration regression.
The pilot ledger lives in the encrypted state DB; pilot_checks.json holds no prices.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime
from math import sqrt
from pathlib import Path
from statistics import mean, stdev

from . import config as C
from .odds import american_to_decimal
from .staking import cap_groups, round_units

SCHEMA = """
CREATE TABLE IF NOT EXISTS pilot_bets (
  opinion_id TEXT PRIMARY KEY, date_et TEXT, game_id TEXT, market TEXT, side TEXT, line REAL,
  book TEXT, price_american REAL, paper_stake_u REAL, pilot_stake_u REAL,
  status TEXT DEFAULT 'recommended', placed_book TEXT, placed_price REAL, placed_ts TEXT, note TEXT
);
"""
CHECKS_FILE = Path("pilot_checks.json")


@dataclass(frozen=True)
class PilotTier:
    markets: tuple[str, ...] = ("spreads",)
    model_prefix: str = "nba-sim"
    kelly_multiplier: float = 0.5  # half the paper stake: 1/32 Kelly
    per_bet_cap_u: float = 0.25
    daily_cap_u: float = 1.0
    drawdown_pause_u: float = -5.0
    kill_min_n: int = 50
    kill_t: float = -1.0
    min_close_capture: float = 0.95


PILOT = PilotTier()


def _conn(db_path) -> sqlite3.Connection:
    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)
    return con


def load_manual_checks(path: Path = CHECKS_FILE) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text()).get("checks", {})


def close_capture(rows: list[dict], before: date) -> tuple[float | None, str | None]:
    """Capture rate of the most recent slate day before `before` that has closed rows."""
    days: dict[str, list[dict]] = {}
    for r in rows:
        if r["clv_method"] is None:
            continue
        d = r["timestamp_et"][:10]
        if d < before.isoformat():
            days.setdefault(d, []).append(r)
    if not days:
        return None, None
    d = max(days)
    rs = days[d]
    return sum(1 for r in rs if r["clv_method"] != "close_missing") / len(rs), d


def entry_checks(rows: list[dict], day: date, manual: dict, tier: PilotTier = PILOT) -> dict[str, dict]:
    rate, d = close_capture(rows, day)
    checks = {
        "close_capture": {
            "passed": rate is not None and rate >= tier.min_close_capture,
            "evidence": f"{d}: {rate:.1%} of rows closed" if rate is not None else "no closed slate yet",
        }
    }
    for name in ("novig_fee", "home_lean"):
        m = manual.get(name, {})
        checks[name] = {"passed": bool(m.get("passed")), "evidence": m.get("evidence", "not confirmed")}
    return checks


def pilot_pnl(con: sqlite3.Connection, rows_by_id: dict[str, dict]) -> float:
    pnl = 0.0
    for b in con.execute("SELECT * FROM pilot_bets WHERE status != 'skipped'"):
        r = rows_by_id.get(b["opinion_id"])
        if not r or not r["result"]:
            continue
        price = b["placed_price"] if b["placed_price"] is not None else b["price_american"]
        if r["result"] == "win":
            pnl += b["pilot_stake_u"] * (american_to_decimal(price) - 1.0)
        elif r["result"] == "loss":
            pnl -= b["pilot_stake_u"]
    return pnl


def kill_switch(con: sqlite3.Connection, rows: list[dict], tier: PilotTier = PILOT) -> dict:
    by_id = {r["opinion_id"]: r for r in rows}
    base = {(r["game_id"], r["market"]): r for r in rows
            if r["model_version"].startswith("market-baseline") and r["clv_pct"] is not None}
    clv, diff = [], []
    for b in con.execute("SELECT opinion_id FROM pilot_bets WHERE status != 'skipped'"):
        r = by_id.get(b["opinion_id"])
        if not r or r["clv_pct"] is None:
            continue
        clv.append(r["clv_pct"])
        br = base.get((r["game_id"], r["market"]))
        if br is not None:
            diff.append(r["clv_pct"] - br["clv_pct"])
    n = len(clv)
    t = (mean(diff) / (stdev(diff) / sqrt(len(diff)))) if len(diff) >= 3 and stdev(diff) > 0 else None
    stop = n >= tier.kill_min_n and ((mean(clv) < 0) or (t is not None and t < tier.kill_t))
    return {"n": n, "mean_clv": mean(clv) if clv else None, "t_vs_baseline": t, "stop": stop}


def status(db_path, rows: list[dict], day: date, manual: dict | None = None, tier: PilotTier = PILOT) -> dict:
    con = _conn(db_path)
    manual = load_manual_checks() if manual is None else manual
    checks = entry_checks(rows, day, manual, tier)
    ks = kill_switch(con, rows, tier)
    pnl = pilot_pnl(con, {r["opinion_id"]: r for r in rows})
    paused = pnl <= tier.drawdown_pause_u
    active = all(c["passed"] for c in checks.values()) and not ks["stop"] and not paused
    return {"active": active, "checks": checks, "kill_switch": ks, "pnl_u": pnl, "drawdown_paused": paused}


def record_pilot(db_path, rows: list[dict], day: date, manual: dict | None = None,
                 tier: PilotTier = PILOT) -> list[str]:
    """Turn today's qualifying gate passes into pilot picks (idempotent). Returns messages.

    A candidate on a day the pilot is inactive is written once as status 'skipped' with the
    reason, so the audit trail shows every pick the pilot declined and why.
    """
    con = _conn(db_path)
    have = {r["opinion_id"] for r in con.execute("SELECT opinion_id FROM pilot_bets")}
    new = [r for r in rows if r["timestamp_et"][:10] == day.isoformat() and r["gate_pass"]
           and r["market"] in tier.markets and r["model_version"].startswith(tier.model_prefix)
           and (r["stake_u"] or 0) > 0 and r["opinion_id"] not in have]
    if not new:
        return []
    st = status(db_path, rows, day, manual, tier)
    ins = ("INSERT INTO pilot_bets (opinion_id, date_et, game_id, market, side, line, book, price_american, "
           "paper_stake_u, pilot_stake_u, status, note) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)")
    if not st["active"]:
        why = [k for k, c in st["checks"].items() if not c["passed"]]
        if st["kill_switch"]["stop"]:
            why.append("CLV kill switch")
        if st["drawdown_paused"]:
            why.append(f"drawdown {st['pnl_u']:.2f}u")
        reason = "pilot inactive: " + ", ".join(why)
        for r in new:
            con.execute(ins, (r["opinion_id"], day.isoformat(), r["game_id"], r["market"], r["side"], r["line"],
                              r["book"], r["price_american"], r["stake_u"], 0.0, "skipped", reason))
        con.commit()
        return [f"pilot skipped {len(new)} candidate(s) for {day} ({reason})"]
    used = con.execute("SELECT COALESCE(SUM(pilot_stake_u), 0) FROM pilot_bets WHERE date_et = ? "
                       "AND status != 'skipped'", (day.isoformat(),)).fetchone()[0]
    raw = [min(tier.per_bet_cap_u, tier.kelly_multiplier * r["stake_u"]) for r in new]
    sized = cap_groups(raw, [0] * len(raw), max(0.0, tier.daily_cap_u - used))
    msgs = []
    for r, s in zip(new, sized):
        s = round_units(s)
        con.execute(ins, (r["opinion_id"], day.isoformat(), r["game_id"], r["market"], r["side"], r["line"],
                          r["book"], r["price_american"], r["stake_u"], s,
                          "recommended" if s > 0 else "skipped", "" if s > 0 else "daily cap reached"))
        if s > 0:
            msgs.append(f"PILOT PICK {r['side']} {r['line']:+g} @{r['book']} {r['price_american']:+.0f}: {s:.2f}u")
    con.commit()
    return msgs


def picks(db_path, day: date) -> list[dict]:
    con = _conn(db_path)
    return [dict(r) for r in con.execute("SELECT * FROM pilot_bets WHERE date_et = ? ORDER BY pilot_stake_u DESC",
                                         (day.isoformat(),))]


def mark_placed(db_path, opinion_id: str, book: str, price: float, ts: datetime, note: str = "") -> None:
    con = _conn(db_path)
    con.execute("UPDATE pilot_bets SET status='placed', placed_book=?, placed_price=?, placed_ts=?, note=? "
                "WHERE opinion_id=?", (book, price, ts.isoformat(), note, opinion_id))
    con.commit()
