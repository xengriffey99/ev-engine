"""Result grading for game markets (spreads, totals) from final scores.

Scores come from ESPN's public scoreboard JSON (free; fetched by the runner, not The Odds API,
so grading costs no credits). Props are not graded here (box scores needed); they are held.
Results grade P/L for the pilot drawdown pause only: the desk is graded on CLV, never win rate.
"""

from __future__ import annotations

import json
import urllib.request
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date

from . import config as C
from .model_adapter import canonical_team

SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={ymd}"


@dataclass(frozen=True)
class Final:
    home: str  # Odds API full name
    away: str
    home_pts: int
    away_pts: int


def parse_scoreboard(body: dict) -> list[Final]:
    out = []
    for ev in body.get("events", []):
        comp = (ev.get("competitions") or [{}])[0]
        status = (comp.get("status") or ev.get("status") or {}).get("type", {})
        if not status.get("completed"):
            continue
        teams = {c.get("homeAway"): c for c in comp.get("competitors", [])}
        if {"home", "away"} - set(teams):
            continue
        try:
            out.append(Final(
                home=canonical_team(teams["home"]["team"]["displayName"]),
                away=canonical_team(teams["away"]["team"]["displayName"]),
                home_pts=int(teams["home"]["score"]), away_pts=int(teams["away"]["score"]),
            ))
        except (KeyError, ValueError):
            continue
    return out


def fetch_finals(day: date, timeout: float = 20.0) -> list[Final]:
    url = SCOREBOARD.format(ymd=day.strftime("%Y%m%d"))
    req = urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": "nba-ev-runner"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return parse_scoreboard(json.loads(resp.read()))


def grade(market: str, side: str, line: float, home: str, away: str, f: Final) -> str:
    margin = f.home_pts - f.away_pts
    if market in C.SPREAD_LIKE_MARKETS:
        if side == home:
            x = margin + line
        elif side == away:
            x = -margin + line
        else:
            raise ValueError(f"{side!r} not in {away} @ {home}")
    elif market in C.TOTAL_LIKE_MARKETS:
        x = (f.home_pts + f.away_pts - line) * (1 if side == "Over" else -1)
    else:
        raise ValueError(f"cannot grade {market}")
    return "win" if x > 0 else "loss" if x < 0 else "push"


def grade_rows(rows: Iterable[dict], teams: dict[str, tuple[str, str]], finals: list[Final]) -> dict[str, str]:
    """opinion_id -> result for ungraded game-market rows whose game is final."""
    by_teams = {(f.home, f.away): f for f in finals}
    out = {}
    for r in rows:
        if r["result"] or r["market"] not in (C.SPREAD_LIKE_MARKETS | C.TOTAL_LIKE_MARKETS):
            continue
        ht = teams.get(r["game_id"])
        f = by_teams.get(ht) if ht else None
        if f is None:
            continue
        out[r["opinion_id"]] = grade(r["market"], r["side"], float(r["line"]), ht[0], ht[1], f)
    return out
