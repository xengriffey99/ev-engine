"""Flatten the slate payload into dashboard datasets (row-object JSON files)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

from nba_sim import engine
from nba_sim.config import BANKROLL
from nba_sim.pipeline import load_snapshot
from nba_sim.pricing import kelly_units

SCENARIOS = {"base": None, "w25": 0.25, "w50": 0.50}
SCEN_LABEL = {"base": "Opening-night fusion", "w25": "Model weight 25%", "w50": "Model weight 50%"}
TEAM_ABBR = {"home": "home", "away": "away"}


def selection(m: dict, g: dict) -> str:
    if m["market"] == "spread":
        return f"{m['team']} {m['line']:+g}"
    if m["market"] == "total":
        return f"{m['side'].capitalize()} {m['line']:g}"
    return f"{m['team']} ML"


def main(snapshot: str, payload_path: str, out_dir: str = "out/dash"):
    snap = load_snapshot(snapshot)
    payload = json.loads(Path(payload_path).read_text())
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    unit_usd = payload["kpis"]["unit_value"]

    games, markets, alerts, dist, risk, signals = [], [], [], [], [], []
    for g in payload["games"]:
        gid, mu = g["game_id"], f"{g['away']} @ {g['home']}"
        pr, sm = g["projection"], g["simulation"]
        best = max(g["markets"], key=lambda m: m["ev_pct"])
        games.append({
            "game_id": gid, "matchup": mu, "away": g["away"], "home": g["home"],
            "away_name": g["away_name"], "home_name": g["home_name"], "tip_et": g["tip_et"], "venue": g["venue"],
            "recommendation": g["recommendation"],
            "struct_margin": pr["structural"]["home_margin"], "mkt_margin": pr["market"]["home_margin"],
            "post_margin": pr["posterior"]["home_margin"],
            "struct_total": pr["structural"]["total"], "mkt_total": pr["market"]["total"], "post_total": pr["posterior"]["total"],
            "spread_home": pr["market"]["spread_display_home"], "total_line": pr["market"]["total_display"],
            "home_win_prob": sm["home_win_prob"], "sd_margin": sm["sd_margin"], "exp_total": sm["exp_total"],
            "sd_total": sm["sd_total"], "ot_rate": sm["ot_rate"], "pace": pr["structural"]["pace"],
            "w_model_side": pr["posterior"]["w_model_side"], "w_model_total": pr["posterior"]["w_model_total"],
            "signal_shift": "; ".join(pr["posterior"]["signal_shifts"]) or "none",
            "best_selection": selection(best, g), "best_ev_pct": best["ev_pct"],
        })
        for k, a in enumerate(g["market_signals"]["alerts"]):
            alerts.append({"key": f"{gid}|{k}", "game_id": gid, "matchup": mu, "level": a["level"], "market": a["market"], "text": a["text"]})
        for kind, hist in (("margin", sm["margin_hist"]), ("total", sm["total_hist"])):
            for h in hist:
                dist.append({"key": f"{gid}|{kind}|{h['bin']}", "game_id": gid, "kind": kind, "bin": h["bin"], "pct": h["pct"]})
        re = g["risk_environment"]
        for team in (g["away"], g["home"]):
            inj = re["injuries"][team]
            for p in inj["players"]:
                risk.append({"key": f"{gid}|{team}|{p['name']}", "game_id": gid, "team": team, "item": p["name"],
                             "status": p["status"].capitalize(), "detail": p.get("reason", ""),
                             "impact_per100": None})
            risk.append({"key": f"{gid}|{team}|net", "game_id": gid, "team": team, "item": "Net availability impact",
                         "status": "Model", "detail": "EPM vs replacement, minutes-weighted, incl. depth penalty",
                         "impact_per100": inj["net_per100"]})
        for i, n in enumerate(re["situational"]["notes"] or ["No rest, travel or altitude modifier applies"]):
            risk.append({"key": f"{gid}|sit|{i}", "game_id": gid, "team": "Both", "item": "Situational", "status": "Note",
                         "detail": n, "impact_per100": None})
        ms = g["market_signals"]
        for mk in ("spread", "total", "moneyline"):
            s = ms[mk]
            open_v = s.get("open", s.get("away_novig_open"))
            cur_v = s.get("current", s.get("away_novig_now"))
            signals.append({"key": f"{gid}|{mk}", "game_id": gid, "market": mk.capitalize(),
                            "open": open_v, "current": cur_v,
                            "measure": "away no-vig win prob" if mk == "moneyline" else ("home line" if mk == "spread" else "total"),
                            "public_side": s["public"]["label"], "tickets": s["public"]["tickets"], "money": s["public"]["money"],
                            "gap": round(s["public"]["money"] - s["public"]["tickets"], 2),
                            "rlm": "RLM" if s["rlm"]["flag"] else ("Sub-threshold" if s["rlm"]["sub_threshold"] else "No"),
                            "sharp": "Yes" if s["sharp_handle"] else "No"})

    # Kelly execution rows for each fusion-weight scenario
    for scen, w in SCENARIOS.items():
        pls = payload["games"] if w is None else [engine.price_game(gr, snap["teams"], w, w) for gr in snap["games"]]
        for g in pls:
            mu = f"{g['away']} @ {g['home']}"
            for m in g["markets"]:
                ku_q = m["kelly_units"]
                ku_h = kelly_units(m["p_win"], m["price"], m["p_push"], fraction=0.5) if m["eligible"] else 0.0
                markets.append({
                    "key": f"{g['game_id']}|{m['market']}|{m['side']}|{scen}", "scenario": scen, "scenario_label": SCEN_LABEL[scen],
                    "game_id": g["game_id"], "matchup": mu, "market": m["market"].capitalize(), "side": m["side"],
                    "selection": selection(m, g), "book": m["best_book"], "price": m["price"],
                    "p_win": m["p_win"], "p_push": m["p_push"], "fair_odds": m["fair_odds"], "ev_pct": m["ev_pct"],
                    "ev_threshold": m["ev_threshold"], "eligible": "Bet" if m["eligible"] else "Pass",
                    "kelly_q_units": ku_q, "kelly_h_units": ku_h,
                    "stake_q_usd": round(ku_q * unit_usd, 2), "stake_h_usd": round(ku_h * unit_usd, 2),
                    "trigger_price": m["trigger"]["price_at_current_line"], "trigger_line": m["trigger"]["line_at_minus110"],
                })

    k = payload["kpis"]
    base = [m for m in markets if m["scenario"] == "base"]
    best = max(base, key=lambda m: m["ev_pct"])
    kpis = [{"key": "slate", "games": k["games"], "plays": k["plays"], "units_recommended": k["units_recommended"],
             "max_ev_pct": best["ev_pct"], "best_label": f"{best['selection']} {best['price']:+d}",
             "gate_pct": BANKROLL.ev_min_sides_totals, "sharp_alerts": k["sharp_alerts"],
             "open_exposure_units": k["open_exposure_units"], "bankroll": k["bankroll"], "unit_value": k["unit_value"],
             "snapshot_ts": snap["snapshot"]["odds_ts"]}]

    led = []
    sm = payload["ledger"]["summary"]
    for line in payload["ledger"]["markdown"].splitlines():
        if line.startswith("| Oct"):
            c = [x.strip() for x in line.strip("|").split("|")]
            led.append({"key": f"{c[1]}|{c[2]}", "date": c[0], "game": c[1], "selection": c[2], "odds": c[3], "stake": c[4],
                        "result": c[5], "pl": c[6], "closing": c[7], "clv": c[8]})

    files = {"games": games, "markets": markets, "alerts": alerts, "sim_dist": dist, "risk": risk,
             "signals": signals, "kpis": kpis, "ledger": led}
    for name, rows in files.items():
        (out / f"{name}.json").write_text(json.dumps(rows, indent=1))
        print(name, len(rows), (out / f"{name}.json").stat().st_size)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
