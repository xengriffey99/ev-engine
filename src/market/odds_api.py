"""The Odds API v4 client with a credit meter and budget guard.

Billing (verified 2026-10-21 against the v4 docs):
  featured /odds          cost = markets x regions (independent of game count)
  /events/{id}/odds       cost = unique markets returned x regions (one game)
  /events/{id}/markets    cost = 1
  /events, /sports        free; empty responses are free
  regions                 every group of up to 10 named bookmakers = 1 region
  historical endpoints    10x, paid plans only (not used on the free tier)

The API key comes from the ODDS_API_KEY environment variable and is redacted
from every error message and log line.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from math import ceil
from typing import Any, Protocol

from . import config as C


class OddsApiError(RuntimeError):
    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status

    @property
    def retryable(self) -> bool:
        """Network errors, 429 and 5xx are transient; other 4xx are not."""
        return self.status is None or self.status == 429 or self.status >= 500


RETRY_STATUS = frozenset({429, 500, 502, 503, 504})


class BudgetExceeded(OddsApiError):
    """The call would overspend the credit budget; it was not made."""


class Ledger(Protocol):
    @property
    def remaining(self) -> int | None: ...

    def record(self, *, ts: datetime, path: str, est_cost: int, cost_last: int | None,
               used: int | None, remaining: int | None) -> None: ...


class MemoryLedger:
    def __init__(self, remaining: int | None = None):
        self._remaining = remaining
        self.entries: list[dict] = []

    @property
    def remaining(self) -> int | None:
        return self._remaining

    def record(self, **kw) -> None:
        self.entries.append(kw)
        if kw.get("remaining") is not None:
            self._remaining = kw["remaining"]


Transport = Callable[[str, dict[str, str]], tuple[int, dict[str, str], bytes]]


def urllib_transport(url: str, params: dict[str, str]) -> tuple[int, dict[str, str], bytes]:
    full = f"{url}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(full, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status, {k.lower(): v for k, v in resp.headers.items()}, resp.read()
    except urllib.error.HTTPError as err:
        return err.code, {k.lower(): v for k, v in err.headers.items()}, err.read()


@dataclass(frozen=True)
class ApiResponse:
    path: str
    body: Any
    fetched_at: datetime
    cost_last: int | None
    used: int | None
    remaining: int | None


def regions_for(bookmakers: Sequence[str]) -> int:
    return max(1, ceil(len(bookmakers) / C.MAX_BOOKS_PER_REGION))


def estimate_cost(markets: Sequence[str], bookmakers: Sequence[str]) -> int:
    return len(markets) * regions_for(bookmakers)


def _int(h: dict[str, str], key: str) -> int | None:
    v = h.get(key)
    try:
        return int(float(v)) if v is not None else None
    except ValueError:
        return None


class OddsApiClient:
    def __init__(
        self,
        api_key: str | None = None,
        *,
        transport: Transport = urllib_transport,
        ledger: Ledger | None = None,
        reserve_credits: int = C.FREE_TIER.reserve_credits,
        host: str = C.API_HOST,
        sport: str = C.SPORT_KEY,
        max_retries: int = 2,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ):
        self._api_key = api_key
        self.transport = transport
        self.ledger = ledger or MemoryLedger()
        self.reserve = reserve_credits
        self.host = host.rstrip("/")
        self.sport = sport
        self.max_retries = max_retries
        self.sleep = sleep
        self.clock = clock

    # --- helpers --------------------------------------------------------------------
    @property
    def api_key(self) -> str:
        key = self._api_key or os.environ.get(C.API_KEY_ENV)
        if not key:
            raise OddsApiError(f"Set the {C.API_KEY_ENV} environment variable")
        return key

    def _redact(self, text: str) -> str:
        key = self._api_key or os.environ.get(C.API_KEY_ENV)
        return text.replace(key, "***") if key else text

    @staticmethod
    def _books(bookmakers: Sequence[str], allow_multi_region: bool) -> str:
        if not bookmakers:
            raise ValueError("Name the bookmakers explicitly (<=10 keeps the call at 1 region)")
        if len(bookmakers) > C.MAX_BOOKS_PER_REGION and not allow_multi_region:
            raise ValueError(
                f"{len(bookmakers)} bookmakers would bill as {regions_for(bookmakers)} regions; "
                "pass allow_multi_region=True to accept the extra cost"
            )
        return ",".join(bookmakers)

    def _guard(self, est_cost: int, protected: bool) -> None:
        remaining = self.ledger.remaining
        if remaining is None or est_cost == 0:
            return
        if est_cost > remaining:
            raise BudgetExceeded(f"Call needs {est_cost} credits; {remaining} remain")
        if not protected and remaining - est_cost < self.reserve:
            raise BudgetExceeded(
                f"Call would leave {remaining - est_cost} credits, under the {self.reserve} reserve"
            )

    def _get(self, path: str, params: dict[str, str], *, est_cost: int, protected: bool) -> ApiResponse:
        self._guard(est_cost, protected)
        url = f"{self.host}{path}"
        query = {**params, "apiKey": self.api_key}
        for attempt in range(self.max_retries + 1):
            try:
                status, headers, body = self.transport(url, query)
            except OSError as err:
                if attempt < self.max_retries:
                    self.sleep(2.0 * (attempt + 1))
                    continue
                raise OddsApiError(self._redact(f"Network error calling {path}: {err}")) from None
            if status in RETRY_STATUS and attempt < self.max_retries:
                self.sleep(2.0 * (attempt + 1))
                continue
            break
        fetched_at = self.clock()
        cost_last, used, remaining = (
            _int(headers, "x-requests-last"), _int(headers, "x-requests-used"),
            _int(headers, "x-requests-remaining"),
        )
        self.ledger.record(ts=fetched_at, path=path, est_cost=est_cost, cost_last=cost_last,
                           used=used, remaining=remaining)
        if status != 200:
            snippet = body[:300].decode("utf-8", "replace")
            raise OddsApiError(self._redact(f"HTTP {status} from {path}: {snippet}"), status=status)
        return ApiResponse(path, json.loads(body or b"null"), fetched_at, cost_last, used, remaining)

    # --- endpoints ---------------------------------------------------------------------
    def events(self, commence_from: datetime | None = None, commence_to: datetime | None = None) -> ApiResponse:
        params: dict[str, str] = {"dateFormat": "iso"}
        if commence_from:
            params["commenceTimeFrom"] = commence_from.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        if commence_to:
            params["commenceTimeTo"] = commence_to.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        return self._get(f"/v4/sports/{self.sport}/events", params, est_cost=0, protected=True)

    def featured_odds(self, markets: Sequence[str], bookmakers: Sequence[str], *,
                      event_ids: Sequence[str] | None = None, protected: bool = False,
                      allow_multi_region: bool = False) -> ApiResponse:
        params = {
            "markets": ",".join(markets),
            "bookmakers": self._books(bookmakers, allow_multi_region),
            "oddsFormat": "decimal",
            "dateFormat": "iso",
        }
        if event_ids:
            params["eventIds"] = ",".join(event_ids)
        return self._get(f"/v4/sports/{self.sport}/odds", params,
                         est_cost=estimate_cost(markets, bookmakers), protected=protected)

    def event_odds(self, event_id: str, markets: Sequence[str], bookmakers: Sequence[str], *,
                   protected: bool = False, allow_multi_region: bool = False) -> ApiResponse:
        params = {
            "markets": ",".join(markets),
            "bookmakers": self._books(bookmakers, allow_multi_region),
            "oddsFormat": "decimal",
            "dateFormat": "iso",
        }
        return self._get(f"/v4/sports/{self.sport}/events/{event_id}/odds", params,
                         est_cost=estimate_cost(markets, bookmakers), protected=protected)

    def event_markets(self, event_id: str, bookmakers: Sequence[str]) -> ApiResponse:
        params = {"bookmakers": self._books(bookmakers, False), "dateFormat": "iso"}
        return self._get(f"/v4/sports/{self.sport}/events/{event_id}/markets", params,
                         est_cost=1, protected=False)
