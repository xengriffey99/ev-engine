"""Overwrite team pace in a slate file from ESPN Hollinger Pace Factor.

Hollinger pace runs on a higher scale than the model's other inputs (league mean ~102 vs
99.6), so it is rescaled by ratio to the league mean:  pace_model = league_pace * H / mean(H).
Usage: python apply_pace.py data/slate_YYYY-MM-DD.json [data/pace_hollinger_2025-26.json]
"""
import json
import statistics
import sys

from nba_sim.config import PRIORS


def apply(slate_path: str, pace_path: str = "data/pace_hollinger_2025-26.json") -> dict:
    ref = json.load(open(pace_path))
    h = ref["pace"]
    mean_h = statistics.mean(h.values())
    slate = json.load(open(slate_path))
    changed = {}
    for abbr, t in slate["teams"].items():
        if abbr not in h:
            raise KeyError(f"{abbr} missing from {pace_path}")
        old = t["pace"]
        t["pace"] = round(PRIORS.league_pace * h[abbr] / mean_h, 2)
        t.setdefault("provenance", {})["pace"] = (
            f"sourced: ESPN Hollinger Pace Factor 2025-26 ({h[abbr]}), rescaled to model scale "
            f"(x{PRIORS.league_pace}/{mean_h:.2f}); retrieved {ref['retrieved']}")
        changed[abbr] = (old, t["pace"])
    json.dump(slate, open(slate_path, "w"), indent=2)
    return changed


if __name__ == "__main__":
    for k, (o, n) in apply(*sys.argv[1:]).items():
        print(f"{k}: {o} -> {n}")
