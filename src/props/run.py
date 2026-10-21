"""Props lane runner.

  PYTHONPATH=src python -m props.run --db data/market.db --date 2026-10-21 \
      --players nba_sim/data/dt_players_2026-10-21.json \
      --status nba_sim/data/status_2026-10-21.json [--decide-now] [--hold] [--out payload.json]

Games: every event with a props_close job that day (the free-tier archive draw). The market
environment comes from the latest sides decision snapshot; prices from the earliest props
decision snapshot taken at or after the projection (as-of rule). --decide-now takes that
snapshot first (1 credit per game: one market, one region).
"""

from __future__ import annotations

import argparse
import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from market import config as C
from market.model_adapter import _BY_NAME
from market.odds_api import OddsApiClient, estimate_cost
from market.opinion_log import OpinionLog
from market.run_snapshots import _log_all, close_plan_from, run_due
from market.scheduler import PRIORITY, SnapshotJob, _job_id
from market.staking import cap_groups, round_units, stake_units
from market.store import DONE, MarketStore
from market.odds import american_to_decimal

from .adapter import MODEL_VERSION, build_prop_opinions, market_env
from .engine import EngineParams, PlayerIn, simulate_game, summarize


def load_players(players_path: Path, status_path: Path) -> dict[str, list[PlayerIn]]:
    table = json.loads(players_path.read_text())
    st = json.loads(status_path.read_text())
    status = {s["name"]: s["status"] for s in st["statuses"]}
    two_way = set(st.get("two_way", []))
    teams: dict[str, list[PlayerIn]] = {}
    for p in table["players"]:
        teams.setdefault(p["team"], []).append(PlayerIn(
            p["name"], p["team"], p["mpg"], p["pct_start"], p["pts100"], p["poss48"], p["fg3a100"],
            p["fg3pct"], p["fta100"], p["ftpct"], status.get(p["name"], "available"), p["name"] in two_way))
    return teams


def prop_games(store: MarketStore, day: date) -> list[str]:
    return sorted({eid for j in store.jobs(date_et=day.isoformat()) if j.kind == "props_close"
                   for eid in j.event_ids})


def latest_sides(store: MarketStore, day: date):
    done = [j for j in store.jobs(status=DONE, date_et=day.isoformat()) if j.kind == "sides_decision"]
    if not done:
        raise SystemExit("no sides decision snapshot today")
    return store.load_quotes(sorted(done, key=lambda j: j.run_at)[-1].snapshot_id)


def decide_props_now(store: MarketStore, client: OddsApiClient, day: date, event_ids: list[str],
                     now: datetime | None = None, cfg: C.FreeTierSchedule = C.FREE_TIER) -> list[str]:
    now = now or datetime.now(timezone.utc)
    jobs = []
    for eid in event_ids:
        close = next(j for j in store.jobs(date_et=day.isoformat())
                     if j.kind == "props_close" and eid in j.event_ids)
        if close.deadline <= now:
            continue
        jobs.append(SnapshotJob(_job_id("props_decision", now, [eid]), day.isoformat(), "props_decision", now,
                                min(close.deadline, now + timedelta(minutes=cfg.decision_window_min)), (eid,),
                                tuple(cfg.props_markets), tuple(cfg.props_books), "event",
                                estimate_cost(cfg.props_markets, cfg.props_books), PRIORITY["props_decision"],
                                "model decision: prop projection made after the scheduled snapshot"))
    store.upsert_jobs(jobs)
    return run_due(store, client, now)


def project(store: MarketStore, day: date, teams: dict[str, list[PlayerIn]],
            params: EngineParams = EngineParams()) -> dict:
    sides = latest_sides(store, day)
    out = {"model_version": MODEL_VERSION, "slate_date": day.isoformat(),
           "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "games": {}}
    sims = {}
    for eid in prop_games(store, day):
        q = next(q for q in sides if q.event_id == eid)
        h, a = _BY_NAME[q.home_team.upper()], _BY_NAME[q.away_team.upper()]
        env, info = market_env(sides, eid, h, a)
        sim = simulate_game(env, teams[h], teams[a], params)
        sims[eid] = sim
        out["games"][eid] = {
            "matchup": f"{a}@{h}", "env": info, "params": sim["params"],
            "players": {nm: {k: round(v, 2) for k, v in summarize(sp).items()} | {"team": sp["team"]}
                        for nm, sp in sim["players"].items()},
        }
    return out, sims


def record_props(store: MarketStore, log: OpinionLog, day: date, sims: dict, as_of: datetime,
                 hold: bool) -> list[str]:
    msgs: list[str] = []
    snaps = sorted((j for j in store.jobs(status=DONE, date_et=day.isoformat()) if j.kind == "props_decision"),
                   key=lambda j: j.run_at)
    existing = {(r["game_id"], r["market"], r["model_version"], r["player"]) for r in log.rows()}
    for eid, sim in sims.items():
        snap = None
        for j in snaps:
            if eid in j.event_ids:
                qs = store.load_quotes(j.snapshot_id, eid)
                if qs and max(x.fetched_at for x in qs) >= as_of:
                    snap = qs
                    break
        if snap is None:
            msgs.append(f"skipped {eid}: no props snapshot at or after the projection; use --decide-now")
            continue
        rows, skipped = build_prop_opinions(sim, snap, eid, hold=hold)
        msgs += [f"skipped {s}" for s in skipped]
        # Staking: 1/16 Kelly; correlated same-game legs are one position (cap left on the game).
        raw = []
        for r in rows:
            op = r.opinion
            p_bet = C.BLEND_W[C.PROPS] * op.p_model + (1 - C.BLEND_W[C.PROPS]) * op.p_market_novig
            d = american_to_decimal(op.price_american)
            ok = op.pass_reason is None and p_bet * d - 1 >= C.EDGE_GATE[C.PROPS]
            raw.append(stake_units(p_bet, d) if ok else 0.0)
        used = sum(float(x["stake_u"] or 0) for x in log.rows() if x["game_id"] == eid)
        capped = cap_groups(raw, [eid] * len(raw), max(0.0, C.PER_GAME_CAP_U - used))
        for r, s_raw, s in zip(rows, raw, capped):
            if s_raw > 0:
                r.opinion.stake_u = round_units(s)
                r.opinion.notes += f"; stake 1/16 Kelly {s_raw:.2f}u -> {r.opinion.stake_u:.2f}u (game cap)"
        msgs += _log_all(log, [r.opinion for r in rows], existing)
    return msgs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="props.run")
    ap.add_argument("--db", type=Path, required=True)
    ap.add_argument("--date", type=date.fromisoformat, required=True)
    ap.add_argument("--players", type=Path, required=True)
    ap.add_argument("--status", type=Path, required=True)
    ap.add_argument("--decide-now", action="store_true")
    ap.add_argument("--hold", action="store_true", help="log every row as model_hold (record-only)")
    ap.add_argument("--out", type=Path, default=None, help="write the projection payload (no prices)")
    args = ap.parse_args(argv)
    store = MarketStore(args.db)
    teams = load_players(args.players, args.status)
    payload, sims = project(store, args.date, teams)
    as_of = datetime.fromisoformat(payload["generated_at"])
    if args.out:
        args.out.write_text(json.dumps(payload, indent=1))
    for eid, g in payload["games"].items():
        print(f"{g['matchup']}: env {g['env']}")
    if args.decide_now:
        client = OddsApiClient(ledger=store)
        for m in decide_props_now(store, client, args.date, list(sims)):
            print(m)
    log = OpinionLog(args.db, close_plan=close_plan_from(store))
    for m in record_props(store, log, args.date, sims, as_of, args.hold):
        print(m)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
