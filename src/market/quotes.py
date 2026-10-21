"""Quote model and parsing of The Odds API v4 payloads (decimal odds).

Odds are requested in decimal format: the API notes American odds can carry
small rounding differences for some books. American prices are derived.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from .odds import decimal_to_american, implied_prob


def parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    ts = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


@dataclass(frozen=True)
class Quote:
    """One outcome price from one book in one snapshot."""

    snapshot_id: str
    fetched_at: datetime  # UTC, when we pulled it
    event_id: str
    commence_time: datetime  # UTC
    home_team: str
    away_team: str
    book: str
    market: str
    side: str  # team name, "Over" or "Under"
    line: float | None  # outcome point; None for h2h
    price_decimal: float
    player: str | None = None  # props only (outcome description)
    market_last_update: datetime | None = None  # UTC, as reported by the API

    @property
    def price_american(self) -> float:
        return decimal_to_american(self.price_decimal)

    @property
    def implied(self) -> float:
        return implied_prob(self.price_decimal)


def _event_quotes(event: dict[str, Any], snapshot_id: str, fetched_at: datetime) -> list[Quote]:
    out: list[Quote] = []
    commence = parse_ts(event["commence_time"])
    for bm in event.get("bookmakers", []):
        for mk in bm.get("markets", []):
            last_update = parse_ts(mk.get("last_update")) or parse_ts(bm.get("last_update"))
            for oc in mk.get("outcomes", []):
                point = oc.get("point")
                out.append(
                    Quote(
                        snapshot_id=snapshot_id,
                        fetched_at=fetched_at,
                        event_id=event["id"],
                        commence_time=commence,
                        home_team=event["home_team"],
                        away_team=event["away_team"],
                        book=bm["key"],
                        market=mk["key"],
                        side=oc["name"],
                        line=float(point) if point is not None else None,
                        price_decimal=float(oc["price"]),
                        player=oc.get("description"),
                        market_last_update=last_update,
                    )
                )
    return out


def parse_odds_payload(
    payload: list[dict[str, Any]] | dict[str, Any],
    snapshot_id: str,
    fetched_at: datetime,
) -> list[Quote]:
    """Parse a featured-odds list or a single event-odds object."""
    events: Iterable[dict[str, Any]] = [payload] if isinstance(payload, dict) else payload
    quotes: list[Quote] = []
    for ev in events:
        quotes.extend(_event_quotes(ev, snapshot_id, fetched_at))
    return quotes
