"""Module 4: rest, travel, altitude and script modifiers (fatigue vector F_v)."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from .config import SITUATIONAL as S

ALTITUDE_ARENAS = {"DEN": ("Ball Arena", S.denver_altitude), "UTA": ("Delta Center", S.utah_altitude)}


@dataclass
class FatigueVector:
    rest_days: int
    games_last4: int      # including tonight
    games_last5: int
    tz_shift: int = 0     # +east, -west, hours crossed since last game
    arrival_hour: float | None = None  # local hour of hotel arrival, e.g. 2.5 = 2:30 am

    @property
    def three_in_four(self) -> bool:
        return self.games_last4 >= 3

    @property
    def four_in_five(self) -> bool:
        return self.games_last5 >= 4


def situational_adjustments(home: str, away_fv: FatigueVector, home_fv: FatigueVector,
                            tip_local: datetime, pace: float) -> dict:
    """Return margin (home perspective) and total adjustments with an audit trail."""
    margin, total, notes = 0.0, 0.0, []

    if home in ALTITUDE_ARENAS and away_fv.rest_days <= 1:
        arena, pts = ALTITUDE_ARENAS[home]
        margin += pts
        notes.append(f"{arena} altitude vs visitor on {away_fv.rest_days}d rest: +{pts} home")

    if away_fv.three_in_four or away_fv.four_in_five:
        pts = S.road_fatigue_ortg * pace / 100
        margin -= pts  # visitor scores less -> home margin up
        total += pts
        notes.append(f"Visitor 3-in-4/4-in-5: {S.road_fatigue_ortg} ORtg ({pts:+.2f} pts)")

    if away_fv.tz_shift >= 2 and (away_fv.arrival_hour or 0) >= 1.0:
        margin -= S.eastbound_tz_penalty
        notes.append(f"Visitor eastbound {away_fv.tz_shift} tz, late arrival: {S.eastbound_tz_penalty} visitor")

    is_weekend = tip_local.weekday() >= 5
    early = tip_local.hour + tip_local.minute / 60 < 15.5
    if early and (is_weekend or not S.matinee_weekend_only):
        total += S.matinee_total
        notes.append(f"Matinee before 3:30 pm local: {S.matinee_total} total")
    elif early:
        notes.append("Weekday matinee: weekend-only matinee rule NOT applied (toggle matinee_weekend_only)")

    return {"margin": round(margin, 3), "total": round(total, 3), "notes": notes}
