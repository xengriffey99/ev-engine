"""In-memory stand-in for The Odds API v4 (no network in tests)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from math import ceil


def iso(ts: datetime) -> str:
    return ts.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class FakeOddsApi:
    """Bills like the real API: markets x regions, 10 books per region, empty = free."""

    def __init__(self, events, board_fn, remaining=500, clock=None):
        self.events = events  # list of dicts like the /events payload
        self.board_fn = board_fn  # (event, markets, books, now) -> list of bookmaker dicts
        self.remaining = remaining
        self.used = 0
        self.calls: list[tuple[str, dict]] = []
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.fail_status: int | None = None

    def _headers(self, cost):
        self.remaining -= cost
        self.used += cost
        return {"x-requests-last": str(cost), "x-requests-used": str(self.used),
                "x-requests-remaining": str(self.remaining)}

    def __call__(self, url: str, params: dict):
        self.calls.append((url, dict(params)))
        if self.fail_status:
            status, self.fail_status = self.fail_status, None
            return status, self._headers(0), b'{"message": "rate limited"}'
        path = url.split("api.the-odds-api.com", 1)[1]
        books = params.get("bookmakers", "").split(",") if params.get("bookmakers") else []
        regions = max(1, ceil(len(books) / 10))
        markets = params.get("markets", "").split(",") if params.get("markets") else []
        now = self.clock()
        if path.endswith("/events"):
            return 200, self._headers(0), json.dumps(self.events).encode()
        if path.endswith("/odds") and "/events/" in path:
            event_id = path.split("/events/")[1].split("/")[0]
            ev = next(e for e in self.events if e["id"] == event_id)
            bms = self.board_fn(ev, markets, books, now)
            returned = {m["key"] for b in bms for m in b["markets"]}
            body = {**ev, "bookmakers": bms}
            return 200, self._headers(len(returned) * regions if bms else 0), json.dumps(body).encode()
        if path.endswith("/odds"):
            wanted = set(params.get("eventIds", "").split(",")) if params.get("eventIds") else None
            out = []
            for ev in self.events:
                if wanted and ev["id"] not in wanted:
                    continue
                bms = self.board_fn(ev, markets, books, now)
                if bms:
                    out.append({**ev, "bookmakers": bms})
            cost = len(markets) * regions if out else 0
            return 200, self._headers(cost), json.dumps(out).encode()
        return 404, self._headers(0), b'{"message": "not found"}'
