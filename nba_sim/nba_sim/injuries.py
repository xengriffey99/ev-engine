"""Module 5: lineup availability, EPM replacement value and position synergy."""
from __future__ import annotations

from collections import defaultdict

from .config import PRIORS

STATUS_PLAY_PROB = {"out": 0.0, "doubtful": 0.15, "questionable": PRIORS.q_play_prob,
                    "probable": PRIORS.p_play_prob, "available": 1.0}


def injury_impact(players: list[dict], include_season_long: bool = False) -> dict:
    """Net-rating impact (points per 100) for one team.

    players: [{'name','pos_group','epm','minutes','status','season_long': bool}]
      pos_group in {'G','W','B'} (guard, wing, big). Season-long absences are already in the
      roster prior, so they count toward position-synergy depth but not the direct EPM cost
      unless include_season_long=True.
    """
    direct, notes = 0.0, []
    expected_out_by_pos = defaultdict(float)
    for p in players:
        p_out = 1 - STATUS_PLAY_PROB[p["status"]]
        expected_out_by_pos[p["pos_group"]] += p_out
        if p.get("season_long") and not include_season_long:
            continue
        cost = (p["epm"] - PRIORS.replacement_epm) * p["minutes"] / 48 * p_out
        direct -= cost
        if p_out > 0:
            notes.append(f"{p['name']} ({p['status']}, P(out)={p_out:.2f}): {-cost:+.2f}/100")

    synergy = 0.0
    for pos, n in expected_out_by_pos.items():
        if n >= 2:
            pen = PRIORS.pos_synergy_lambda * (n - 1) ** 1.5
            synergy -= pen
            notes.append(f"{pos} depth: {n:.1f} expected out -> synergy {-pen:+.2f}/100")
    return {"net_per100": round(direct + synergy, 3), "direct": round(direct, 3),
            "synergy": round(synergy, 3), "notes": notes}
