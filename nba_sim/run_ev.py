"""EV Engine slate run (sides/totals lane): roster-aware EPM prior + Hollinger pace -> structural
projection -> Monte Carlo distributions -> payload for the opinion log.

    python run_ev.py 2026-10-21

Inputs (all in data/):
  games_<date>.json             slate (event ids, tips) from the Odds API decision snapshot
  epm_dt_<date>.json            team ratings from Dunks & Threes expected EPM + valued injuries
  pace_hollinger_2025-26.json   team pace (prior season), rescaled to the model scale
Output: out/payload_<date>.json (no prices; the opinion log prices it against a decision snapshot).
"""
from __future__ import annotations

import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from nba_sim import engine, montecarlo as mc
from nba_sim.config import PRIORS, SIM

MODEL_VERSION = "nba-sim 0.2.0 (EPM roster prior, Hollinger pace)"
HERE = Path(__file__).parent


def build(day: str) -> dict:
    games = json.loads((HERE / f"data/games_{day}.json").read_text())
    epm = json.loads((HERE / f"data/epm_dt_{day}.json").read_text())
    hp = json.loads((HERE / "data/pace_hollinger_2025-26.json").read_text())
    mean_h = statistics.mean(hp["pace"].values())

    teams = {}
    for abbr, r in epm["teams"].items():
        teams[abbr] = {
            "name": abbr, "rating_prior": "epm", "roster_delta": 0.0,
            "ortg": PRIORS.league_ortg + r["off_rel"], "drtg": PRIORS.league_ortg - r["def_rel"],
            "pace": PRIORS.league_pace * hp["pace"][abbr] / mean_h,
        }
    inj_by_team: dict[str, list] = {}
    for p in epm["injuries"]:
        inj_by_team.setdefault(p["team"], []).append(
            {k: p[k] for k in ("name", "pos_group", "epm", "minutes", "status")})

    out_games = []
    for g in games["games"]:
        h, a = g["home"], g["away"]
        tip = datetime.fromisoformat(g["tip_et"])
        gg = {
            "home": h, "away": a, "tip_local": tip.replace(tzinfo=None).isoformat(),
            "injuries": {h: inj_by_team.get(h, []), a: inj_by_team.get(a, [])},
            # Season debut for all 22 teams: no back-to-back, no 3-in-4. Travel not modeled.
            "fatigue": {"home": {"rest_days": 5, "games_last4": 1, "games_last5": 1, "tz_shift": 0},
                        "away": {"rest_days": 5, "games_last4": 1, "games_last5": 1, "tz_shift": 0}},
        }
        st = engine.structural_projection(gg, teams)
        sim = mc.simulate(st["home_margin"], st["total"])
        summ = mc.summarize(sim)
        pct = lambda x, q: float(np.percentile(x, q))
        out_games.append({
            "game_id": f"{day}-{a}-{h}", "event_id": g["event_id"], "away": a, "home": h, "tip_et": g["tip_et"],
            "projection": {"structural": {
                "home_margin": round(st["home_margin"], 2), "total": round(st["total"], 1),
                "pace": round(st["pace"], 1), "exp_ortg": {k: round(v, 1) for k, v in st["ortg"].items()},
                "net_rating": {k: round(v["net"], 2) for k, v in st["ratings"].items()},
                "injury_net_per100": {k: v["net_per100"] for k, v in st["injuries"].items()},
                "injury_notes": {k: v["notes"] for k, v in st["injuries"].items()},
                "situational": st["situational"]["notes"],
            }},
            "simulation": {
                "n": SIM.n_sims, "seed": SIM.seed,
                "home_win_prob": round(summ["home_win_prob"], 4),
                "exp_margin": round(summ["exp_margin"], 2), "sd_margin": round(summ["sd_margin"], 2),
                "exp_total": round(summ["exp_total"], 1), "sd_total": round(summ["sd_total"], 2),
                "margin_p10_p50_p90": [round(pct(sim["margin"], q), 1) for q in (10, 50, 90)],
                "total_p10_p50_p90": [round(pct(sim["total"], q), 1) for q in (10, 50, 90)],
                "ot_rate": round(summ["ot_rate"], 4),
            },
        })
    return {
        "model_version": MODEL_VERSION, "slate_date": day,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "inputs": {
            "slate": games["source"], "ratings": epm["source_epm"] + f" retrieved {epm['retrieved_epm']}",
            "injuries": epm["source_injuries"], "pace": hp["source"] + f" retrieved {hp['retrieved']}",
            "estimates": [
                "P(play) by status: nba_sim priors (probable 0.85, questionable 0.50, doubtful 0.15)",
                "replacement level EPM -2.0 (nba_sim prior)", "home-court 2.4 pts (APPROX)",
                "pace is 2025-26 (prior season), regressed 20% to league",
                "travel/time zones not modeled; four factors, scheme and lineup pace modules neutral",
            ],
        },
        "games": out_games,
    }


if __name__ == "__main__":
    day = sys.argv[1]
    payload = build(day)
    out = HERE / "out"
    out.mkdir(exist_ok=True)
    path = out / f"payload_{day}.json"
    path.write_text(json.dumps(payload, indent=1))
    for g in payload["games"]:
        s, m = g["projection"]["structural"], g["simulation"]
        print(f"{g['away']}@{g['home']}: margin {s['home_margin']:+.1f} total {s['total']:.1f} "
              f"pace {s['pace']} sd {m['sd_margin']}/{m['sd_total']} inj {s['injury_net_per100']}")
    print(path)
