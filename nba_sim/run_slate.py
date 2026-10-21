"""Run the slate: python run_slate.py data/slate_2026-10-20.json"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from nba_sim import engine, ledger
from nba_sim.config import BANKROLL, FUSION, SIM, PRIORS, SITUATIONAL
from nba_sim.pipeline import load_snapshot


def main(path: str, out_dir: str = "out") -> dict:
    snap = load_snapshot(path)
    games = [engine.price_game(g, snap["teams"]) for g in snap["games"]]
    by_id = {g["game_id"]: (p, g) for p, g in zip(games, snap["games"])}

    bets = [dict(p, game_id=g["game_id"], game=f"{g['away']} @ {g['home']}") for g in games for p in g["plays"]]

    # Ledger: carry prior entries, audit them against tonight's posterior
    entries, audits = [], []
    for e in snap.get("ledger_entries", []):
        gp, graw = by_id[e["game_id"]]
        audit = engine.evaluate_ticket(gp, graw, e)
        flag = None
        if audit["points_better_than_market"] >= 3.0:
            flag = "(unverified)"
        audits.append({**e, "audit": audit, "flag": flag})
        entries.append({**e, "date_label": "Oct 20", "flag": flag})
    for b in bets:
        sel = (f"{b['team']} {b['line']:+g}" if b["market"] == "spread" else
               f"{b['side'].capitalize()} {b['line']:g}" if b["market"] == "total" else f"{b['team']} ML")
        entries.append({"date": snap["slate_date"], "date_label": "Oct 20", "game": b["game"], "market": b["market"],
                        "side": b["side"], "line": b["line"], "price": b["price"], "stake": b["kelly_units"],
                        "selection": sel, "status": "pending", "close_line": None, "close_price": None})
    bet_games = {e["game"] for e in entries}
    passes = [{"date": snap["slate_date"], "date_label": "Oct 20", "game": f"{g['away']} @ {g['home']}"}
              for g in games if not g["plays"] and f"{g['away']} @ {g['home']}" not in bet_games]
    lessons = [
        "Oct 20: Opening-night fusion keeps the structural model at 12% weight on sides and 10% on totals "
        "(σ_model 4.0/6.0 vs σ_market 1.5/2.0); widen model weight only as in-season Kalman variance falls.",
        "Oct 20: Matinee rule is weekend-only per Context doc; BOS @ DET (Tue 3:00 pm) not adjusted. "
        "Backtest a weekday-matinee variant before enabling.",
        "Oct 20: Four Factors, coaching-scheme and lineup-pace feeds not loaded; modules ran neutral. "
        "First 10–15 games of PBP needed before they carry signal.",
    ]
    md = ledger.render(entries, passes, lessons)
    summ = ledger.summarize(entries)

    sharp_alerts = sum(1 for g in games for a in g["market_signals"]["alerts"] if a["level"] in ("RLM", "SHARP"))
    payload = {
        "schema_version": "1.0",
        "model_version": engine.MODEL_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "slate_date": snap["slate_date"],
        "data_snapshot": snap["snapshot"],
        "config": {"bankroll": asdict(BANKROLL), "fusion": asdict(FUSION), "simulation": asdict(SIM),
                   "priors": {k: v for k, v in asdict(PRIORS).items()}, "situational": asdict(SITUATIONAL)},
        "kpis": {
            "games": len(games), "plays": len(bets), "units_recommended": round(sum(b["kelly_units"] for b in bets), 2),
            "max_ev_pct": max((m["ev_pct"] for g in games for m in g["markets"]), default=None),
            "sharp_alerts": sharp_alerts,
            "open_exposure_units": summ["pending_units"] + round(sum(b["kelly_units"] for b in bets), 2),
            "bankroll": summ["current"], "unit_value": summ["current"] * BANKROLL.unit_pct,
        },
        "games": games,
        "bets": bets,
        "ledger": {"summary": summ, "audits": audits, "markdown": md},
    }
    out = Path(out_dir)
    out.mkdir(exist_ok=True)
    (out / f"payload_{snap['slate_date']}.json").write_text(json.dumps(payload, indent=2, default=float))
    (out / "season_ledger.md").write_text(md)
    return payload


if __name__ == "__main__":
    p = main(sys.argv[1] if len(sys.argv) > 1 else "data/slate_2026-10-20.json")
    for g in p["games"]:
        pr = g["projection"]
        print(f"{g['away']} @ {g['home']}: struct {pr['structural']['home_margin']:+.2f}/{pr['structural']['total']:.1f} | "
              f"mkt {pr['market']['home_margin']:+.2f}/{pr['market']['total']:.1f} | post {pr['posterior']['home_margin']:+.2f}/"
              f"{pr['posterior']['total']:.1f} | win {g['simulation']['home_win_prob']:.3f} | {g['recommendation']}")
        for m in g["markets"]:
            print(f"   {m['market']:9s} {m['side']:5s} {str(m['line']):>6s} {m['price']:+5d} @{m['best_book']:10s} "
                  f"p={m['p_win']:.3f} push={m['p_push']:.3f} fair={m['fair_odds']:+5d} EV={m['ev_pct']*100:+.2f}% "
                  f"trig={m['trigger']}")
        for a in g["market_signals"]["alerts"]:
            print("   ALERT", a)
    print(p["ledger"]["audits"])
    print(p["ledger"]["markdown"])
