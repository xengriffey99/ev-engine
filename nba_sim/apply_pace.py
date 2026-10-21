"""Overwrite team pace AND offensive/defensive rating in a slate file from ESPN Hollinger.

Hollinger pace runs on a higher scale than the model's other inputs (league mean ~102 vs
99.6), so it is rescaled by ratio to the league mean:  pace_model = league_pace * H / mean(H).
Efficiency is rescaled the same way:  rating_model = league_ortg * H / mean(H), separately for OFF and
DEF. League means check out: 101.97 pace x 112.75 eff == 99.6 x 115.5 (same league game total).
Usage: python apply_pace.py data/slate_YYYY-MM-DD.json [pace.json] [eff.json]
"""
import json
import statistics
import sys

from nba_sim.config import PRIORS


def apply(slate_path: str, pace_path: str = "data/pace_hollinger_2025-26.json",
          eff_path: str = "data/eff_hollinger_2025-26.json") -> dict:
    ref = json.load(open(pace_path))
    h = ref["pace"]
    mean_h = statistics.mean(h.values())
    slate = json.load(open(slate_path))
    eff = json.load(open(eff_path))
    mo = statistics.mean(v["off"] for v in eff["eff"].values())
    md = statistics.mean(v["def"] for v in eff["eff"].values())
    changed = {}
    for abbr, t in slate["teams"].items():
        if abbr not in h:
            raise KeyError(f"{abbr} missing from {pace_path}")
        old = t["pace"]
        t["pace"] = round(PRIORS.league_pace * h[abbr] / mean_h, 2)
        t.setdefault("provenance", {})["pace"] = (
            f"sourced: ESPN Hollinger Pace Factor 2025-26 ({h[abbr]}), rescaled to model scale "
            f"(x{PRIORS.league_pace}/{mean_h:.2f}); retrieved {ref['retrieved']}")
        e = eff["eff"][abbr]
        o_old, d_old = t["ortg"], t["drtg"]
        t["ortg"] = round(PRIORS.league_ortg * e["off"] / mo, 2)
        t["drtg"] = round(PRIORS.league_ortg * e["def"] / md, 2)
        note = (f"sourced: ESPN Hollinger 2025-26 ({{}}), rescaled to model scale; retrieved {eff['retrieved']}")
        t["provenance"]["ortg"] = note.format(f"OFF EFF {e['off']}")
        t["provenance"]["drtg"] = note.format(f"DEF EFF {e['def']}")
        changed[abbr] = (old, t["pace"], o_old, t["ortg"], d_old, t["drtg"])
    json.dump(slate, open(slate_path, "w"), indent=2)
    return changed


if __name__ == "__main__":
    for k, (po, pn, oo, on, do, dn) in apply(*sys.argv[1:]).items():
        print(f"{k}: pace {po}->{pn} | ortg {oo}->{on} | drtg {do}->{dn}")
