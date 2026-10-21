"""Opinion log (NN1, NN8): one row per opinion, passes included.

Storage is SQLite (rows are opened at decision time and closed out at tip);
export_csv() writes the Excel-ready file. Column order: `opinion_id`, then the
NBA_CONTEXT.md section 9 columns in spec order, then the six columns approved
on 2026-10-21.

Probability chain written on every row:
  p_bet   = w * p_model + (1 - w) * p_market_novig        (NN3)
  p_final = p_bet + tier2_adj,  |tier2_adj| <= Tier-2 cap  (Tier-2 overlay)
  ev_pct  = p_final * decimal(price_american) - 1
Rules enforced at write time:
  - two-way reference only (NN4); one-way markets never become rows
  - log only what you can close-capture: the market must be on the close schedule
  - gate per lane (NN8); below the gate is a pass with stake 0 and a pass_reason
  - per-game cap 2u (NN7)
Missed closes stay in the log, flagged `close_missing`; rows are never deleted.
"""

from __future__ import annotations

import csv
import sqlite3
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from . import config as C
from .clv import MISSING, ClvResult
from .devig import METHODS
from .odds import american_to_decimal
from .selection import worse_quality

ET = ZoneInfo("America/New_York")

SPEC_COLUMNS = (
    "timestamp_et", "game_id", "market", "player", "stat", "line", "side", "book",
    "price_american", "two_way", "p_model", "p_market_novig", "w", "p_bet",
    "tier2_adj", "ev_pct", "gate_pass", "stake_u", "close_line", "close_price",
    "p_close_novig", "clv_pct", "result", "notes",
)
LEADING_COLUMNS = ("opinion_id",)
TRAILING_COLUMNS = (
    "model_version", "ref_quality", "devig_method", "close_ts_et", "clv_method", "pass_reason",
)
EXPORT_COLUMNS = LEADING_COLUMNS + SPEC_COLUMNS + TRAILING_COLUMNS

PASS_REASONS = frozenset(
    {"below_gate", "outsized_edge", "restricted", "stale_reference", "no_bettable_price"}
)
RESULTS = frozenset({"win", "loss", "push", "void"})
_TOL = 1e-9

ClosePlan = Callable[[str, str], bool]  # (game_id, market) -> close snapshot scheduled?


class OpinionLogError(ValueError):
    pass


class NotCloseCapturableError(OpinionLogError):
    """The market has no scheduled close snapshot, so it cannot be logged."""


@dataclass
class Opinion:
    game_id: str
    market: str
    side: str
    line: float | None
    book: str  # where the price was taken (direction-aware best price)
    price_american: float
    p_model: float
    p_market_novig: float
    model_version: str
    ref_quality: str
    devig_method: str = C.DEFAULT_DEVIG_METHOD
    player: str | None = None
    stat: str | None = None
    two_way: bool = True
    w: float | None = None
    p_bet: float | None = None
    tier2_adj: float = 0.0
    ev_pct: float | None = None
    gate_pass: bool | None = None
    stake_u: float = 0.0
    pass_reason: str | None = None
    notes: str = ""
    timestamp: datetime | None = None


def _iso_et(ts: datetime) -> str:
    if ts.tzinfo is None:
        raise OpinionLogError("Timestamps must be timezone-aware")
    return ts.astimezone(ET).isoformat(timespec="seconds")


def _prob(name: str, value: float) -> float:
    v = float(value)
    if not 0.0 < v < 1.0:
        raise OpinionLogError(f"{name}={value!r} must be in (0, 1)")
    return v


class OpinionLog:
    def __init__(self, db_path: str | Path, close_plan: ClosePlan):
        self.db_path = str(db_path)
        self.close_plan = close_plan
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        cols = ", ".join(f"{c} {'TEXT PRIMARY KEY' if c == 'opinion_id' else ''}" for c in EXPORT_COLUMNS)
        self.conn.execute(f"CREATE TABLE IF NOT EXISTS opinions ({cols})")
        self.conn.commit()

    # --- write -------------------------------------------------------------
    def record(self, op: Opinion, *, allow_custom_w: bool = False) -> str:
        row = self._validate(op, allow_custom_w)
        placeholders = ", ".join("?" for _ in EXPORT_COLUMNS)
        self.conn.execute(
            f"INSERT INTO opinions ({', '.join(EXPORT_COLUMNS)}) VALUES ({placeholders})",
            [row.get(c) for c in EXPORT_COLUMNS],
        )
        self.conn.commit()
        return row["opinion_id"]

    def _validate(self, op: Opinion, allow_custom_w: bool) -> dict:
        if not op.two_way:
            raise OpinionLogError("One-way market: no edge is computed off a one-way price (NN4)")
        if not self.close_plan(op.game_id, op.market):
            raise NotCloseCapturableError(
                f"{op.market} for {op.game_id} has no scheduled close snapshot; not logged"
            )
        cls = C.market_class(op.market)
        if cls == C.PROPS and not op.player:
            raise OpinionLogError("Prop opinions need a player")
        if op.market not in C.LINELESS_MARKETS and op.line is None:
            raise OpinionLogError(f"{op.market} needs a line")
        if op.ref_quality not in C.REF_QUALITY_ORDER:
            raise OpinionLogError(f"Unknown ref_quality {op.ref_quality!r}")
        if op.devig_method not in METHODS:
            raise OpinionLogError(f"Unknown devig_method {op.devig_method!r}")
        if not op.model_version:
            raise OpinionLogError("model_version is required")

        p_model = _prob("p_model", op.p_model)
        p_mkt = _prob("p_market_novig", op.p_market_novig)
        default_w = C.BLEND_W[cls]
        w = default_w if op.w is None else float(op.w)
        if not 0.0 <= w <= 1.0:
            raise OpinionLogError(f"w={w} outside [0, 1]")
        if abs(w - default_w) > _TOL and not allow_custom_w:
            raise OpinionLogError(
                f"w={w} differs from the {cls} start weight {default_w}; "
                "pass allow_custom_w=True only for a w fit out-of-sample"
            )
        p_bet = w * p_model + (1.0 - w) * p_mkt
        if op.p_bet is not None and abs(op.p_bet - p_bet) > _TOL:
            raise OpinionLogError(f"p_bet={op.p_bet} != w*p_model + (1-w)*p_market = {p_bet}")

        tier2 = float(op.tier2_adj or 0.0)
        if abs(tier2) > C.TIER2_CAP + _TOL:
            raise OpinionLogError(f"tier2_adj={tier2} exceeds the Tier-2 cap {C.TIER2_CAP}")
        p_final = _prob("p_bet + tier2_adj", p_bet + tier2)

        decimal = american_to_decimal(op.price_american)
        ev = p_final * decimal - 1.0
        if op.ev_pct is not None and abs(op.ev_pct - ev) > 1e-6:
            raise OpinionLogError(f"ev_pct={op.ev_pct} != computed {ev}")

        gate = C.EDGE_GATE[cls]
        ceiling = C.OUTSIZED_EDGE_CEILING.get(cls)
        reason = op.pass_reason
        if reason is not None and reason not in PASS_REASONS:
            raise OpinionLogError(f"Unknown pass_reason {reason!r}")
        if reason is None:
            if ev < gate:
                reason = "below_gate"
            elif ceiling is not None and ev > ceiling:
                reason = "outsized_edge"
        gate_pass = reason is None
        if op.gate_pass is not None and bool(op.gate_pass) != gate_pass:
            raise OpinionLogError(
                f"gate_pass={op.gate_pass} conflicts with ev={ev:.4f}, gate={gate}, reason={reason}"
            )

        stake = float(op.stake_u or 0.0)
        if stake < 0:
            raise OpinionLogError("stake_u cannot be negative")
        if not gate_pass and stake != 0.0:
            raise OpinionLogError("A pass carries stake_u = 0")
        if stake > 0:
            used = self.conn.execute(
                "SELECT COALESCE(SUM(stake_u), 0) FROM opinions WHERE game_id = ?", (op.game_id,)
            ).fetchone()[0]
            if used + stake > C.PER_GAME_CAP_U + _TOL:
                raise OpinionLogError(
                    f"Per-game cap: {used} + {stake} > {C.PER_GAME_CAP_U}u for {op.game_id}"
                )

        ts = op.timestamp or datetime.now(ET)
        return {
            "opinion_id": uuid.uuid4().hex,
            "timestamp_et": _iso_et(ts),
            "game_id": op.game_id,
            "market": op.market,
            "player": op.player,
            "stat": op.stat or C.stat_for_market(op.market),
            "line": op.line,
            "side": op.side,
            "book": op.book,
            "price_american": round(float(op.price_american), 2),
            "two_way": 1,
            "p_model": p_model,
            "p_market_novig": p_mkt,
            "w": w,
            "p_bet": p_bet,
            "tier2_adj": tier2,
            "ev_pct": ev,
            "gate_pass": int(gate_pass),
            "stake_u": stake,
            "notes": op.notes,
            "model_version": op.model_version,
            "ref_quality": op.ref_quality,
            "devig_method": op.devig_method,
            "pass_reason": reason,
        }

    # --- close-out and results ------------------------------------------------
    def _get(self, opinion_id: str) -> sqlite3.Row:
        row = self.conn.execute(
            "SELECT * FROM opinions WHERE opinion_id = ?", (opinion_id,)
        ).fetchone()
        if row is None:
            raise OpinionLogError(f"Unknown opinion_id {opinion_id}")
        return row

    def close_out(self, opinion_id: str, clv: ClvResult) -> None:
        row = self._get(opinion_id)
        if row["clv_method"] is not None:
            raise OpinionLogError(f"{opinion_id} is already closed out")
        quality = worse_quality(row["ref_quality"], clv.ref_quality)
        self.conn.execute(
            """UPDATE opinions SET close_line = ?, close_price = ?, p_close_novig = ?,
               clv_pct = ?, clv_method = ?, close_ts_et = ?, ref_quality = ?
               WHERE opinion_id = ?""",
            (
                clv.close_line,
                round(clv.close_price, 2) if clv.close_price is not None else None,
                clv.p_close_novig,
                clv.clv_pct,
                clv.clv_method,
                _iso_et(clv.close_ts) if clv.close_ts else None,
                quality,
                opinion_id,
            ),
        )
        self.conn.commit()

    def mark_close_missing(self, opinion_id: str) -> None:
        self.close_out(opinion_id, ClvResult(None, None, None, None, MISSING, None, None, None))

    def set_result(self, opinion_id: str, result: str) -> None:
        if result not in RESULTS:
            raise OpinionLogError(f"result must be one of {sorted(RESULTS)}")
        self._get(opinion_id)
        self.conn.execute("UPDATE opinions SET result = ? WHERE opinion_id = ?", (result, opinion_id))
        self.conn.commit()

    # --- read / export -----------------------------------------------------------
    def rows(self) -> list[dict]:
        cur = self.conn.execute(f"SELECT {', '.join(EXPORT_COLUMNS)} FROM opinions ORDER BY timestamp_et, rowid")
        out = []
        for r in cur.fetchall():
            d = dict(r)
            d["two_way"] = bool(d["two_way"])
            d["gate_pass"] = bool(d["gate_pass"])
            out.append(d)
        return out

    def open_rows(self) -> list[dict]:
        return [r for r in self.rows() if r["clv_method"] is None]

    def export_csv(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=EXPORT_COLUMNS)
            writer.writeheader()
            for r in self.rows():
                writer.writerow({k: ("" if v is None else v) for k, v in r.items()})
        return path

    def close(self) -> None:
        self.conn.close()
