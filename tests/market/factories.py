"""Test helpers: build Quote objects from American prices."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from market.odds import american_to_decimal
from market.quotes import Quote

FETCHED = datetime(2026, 10, 22, 16, 0, tzinfo=timezone.utc)  # 12:00 ET
TIP = datetime(2026, 10, 22, 23, 0, tzinfo=timezone.utc)  # 19:00 ET
HOME, AWAY = "Boston Celtics", "New York Knicks"


def q(
    book: str,
    side: str,
    line: float | None,
    american: float,
    *,
    market: str = "player_points",
    player: str | None = "Jayson Tatum",
    event_id: str = "evt1",
    snapshot_id: str = "snap1",
    fetched_at: datetime = FETCHED,
    age_min: float = 1.0,
) -> Quote:
    return Quote(
        snapshot_id=snapshot_id,
        fetched_at=fetched_at,
        event_id=event_id,
        commence_time=TIP,
        home_team=HOME,
        away_team=AWAY,
        book=book,
        market=market,
        side=side,
        line=line,
        price_decimal=american_to_decimal(american),
        player=player,
        market_last_update=fetched_at - timedelta(minutes=age_min),
    )


def game(book, side, line, american, market="spreads", **kw):
    return q(book, side, line, american, market=market, player=None, **kw)
