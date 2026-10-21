from datetime import datetime, timezone

import pytest
from fake_api import FakeOddsApi, iso

from market.odds_api import (
    BudgetExceeded,
    MemoryLedger,
    OddsApiClient,
    OddsApiError,
    estimate_cost,
)
from market.quotes import parse_odds_payload

KEY = "SECRET-KEY-123"
TIP = datetime(2026, 10, 22, 23, 0, tzinfo=timezone.utc)
EVENTS = [{"id": "evt1", "sport_key": "basketball_nba", "commence_time": iso(TIP),
           "home_team": "Boston Celtics", "away_team": "New York Knicks"}]


def board(ev, markets, books, now):
    out = []
    for b in books:
        mk = []
        if "spreads" in markets:
            mk.append({"key": "spreads", "last_update": iso(now), "outcomes": [
                {"name": ev["home_team"], "price": 1.91, "point": -4.5},
                {"name": ev["away_team"], "price": 1.91, "point": 4.5}]})
        if "player_points" in markets and b != "pinnacle":
            mk.append({"key": "player_points", "last_update": iso(now), "outcomes": [
                {"name": "Over", "description": "Jayson Tatum", "price": 1.87, "point": 26.5},
                {"name": "Under", "description": "Jayson Tatum", "price": 1.95, "point": 26.5}]})
        if mk:
            out.append({"key": b, "title": b, "markets": mk})
    return out


def client(api, ledger=None, **kw):
    return OddsApiClient(KEY, transport=api, ledger=ledger or MemoryLedger(), sleep=lambda s: None, **kw)


def test_cost_estimate_uses_ten_books_per_region():
    assert estimate_cost(["spreads", "totals"], ["b"] * 10) == 2
    assert estimate_cost(["spreads", "totals"], ["b"] * 11) == 4
    assert estimate_cost(["player_points"], ["b"] * 9) == 1


def test_featured_call_params_and_credit_meter():
    api = FakeOddsApi(EVENTS, board)
    ledger = MemoryLedger()
    c = client(api, ledger)
    resp = c.featured_odds(["spreads", "totals"], ["pinnacle", "fanduel"], event_ids=["evt1"])
    url, params = api.calls[-1]
    assert url.endswith("/v4/sports/basketball_nba/odds")
    assert params["oddsFormat"] == "decimal"
    assert params["bookmakers"] == "pinnacle,fanduel" and "regions" not in params
    assert params["eventIds"] == "evt1"
    assert resp.cost_last == 2 and resp.remaining == 498
    assert ledger.remaining == 498 and ledger.entries[-1]["est_cost"] == 2


def test_more_than_ten_books_refused_by_default():
    c = client(FakeOddsApi(EVENTS, board))
    with pytest.raises(ValueError):
        c.featured_odds(["spreads"], [f"b{i}" for i in range(11)])


def test_budget_guard_protects_reserve_but_not_closes():
    api = FakeOddsApi(EVENTS, board, remaining=61)
    c = client(api, MemoryLedger(remaining=61), reserve_credits=60)
    with pytest.raises(BudgetExceeded):
        c.featured_odds(["spreads", "totals"], ["pinnacle"])  # would leave 59 < 60
    assert api.calls == []  # never sent
    c.featured_odds(["spreads", "totals"], ["pinnacle"], protected=True)  # a close still runs
    with pytest.raises(BudgetExceeded):
        client(api, MemoryLedger(remaining=1)).featured_odds(["spreads", "totals"], ["pinnacle"], protected=True)


def test_events_endpoint_is_free_and_allowed_at_zero():
    api = FakeOddsApi(EVENTS, board, remaining=0)
    resp = client(api, MemoryLedger(remaining=0)).events()
    assert resp.body[0]["id"] == "evt1" and resp.cost_last == 0


def test_event_odds_parses_props_with_player_and_timestamps():
    api = FakeOddsApi(EVENTS, board)
    resp = client(api).event_odds("evt1", ["player_points"], ["fanduel", "draftkings"])
    quotes = parse_odds_payload(resp.body, "snapX", resp.fetched_at)
    assert len(quotes) == 4
    under = next(x for x in quotes if x.book == "fanduel" and x.side == "Under")
    assert under.player == "Jayson Tatum" and under.line == 26.5
    assert under.price_decimal == 1.95 and round(under.price_american) == -105
    assert under.market_last_update is not None and under.fetched_at.tzinfo is not None


def test_retries_429_then_succeeds():
    api = FakeOddsApi(EVENTS, board)
    api.fail_status = 429
    resp = client(api).featured_odds(["spreads"], ["pinnacle"])
    assert len(api.calls) == 2 and resp.body


@pytest.mark.parametrize("status", [500, 502, 503, 504])
def test_retries_transient_5xx(status):
    api = FakeOddsApi(EVENTS, board)
    api.fail_status = status  # live 2026-10-21: one props call got HTTP 502
    resp = client(api).event_odds("evt1", ["player_points"], ["fanduel"])
    assert len(api.calls) == 2 and resp.body


def test_persistent_5xx_raises_retryable_error():
    def always_502(url, params):
        return 502, {}, b"Internal Server Error"
    c = OddsApiClient(KEY, transport=always_502, sleep=lambda s: None)
    with pytest.raises(OddsApiError) as exc:
        c.featured_odds(["spreads"], ["pinnacle"])
    assert exc.value.status == 502 and exc.value.retryable


def test_client_errors_are_not_retryable():
    calls = []
    def bad_request(url, params):
        calls.append(1)
        return 422, {}, b'{"message": "Invalid market"}'
    c = OddsApiClient(KEY, transport=bad_request, sleep=lambda s: None)
    with pytest.raises(OddsApiError) as exc:
        c.featured_odds(["nope"], ["pinnacle"])
    assert len(calls) == 1 and not exc.value.retryable


def test_api_key_never_appears_in_errors(monkeypatch):
    def broken(url, params):
        return 401, {}, f"bad key {params['apiKey']}".encode()
    c = OddsApiClient(KEY, transport=broken, sleep=lambda s: None)
    with pytest.raises(OddsApiError) as exc:
        c.featured_odds(["spreads"], ["pinnacle"])
    assert KEY not in str(exc.value) and "***" in str(exc.value)


def test_missing_key_is_a_clear_error(monkeypatch):
    monkeypatch.delenv("ODDS_API_KEY", raising=False)
    c = OddsApiClient(transport=FakeOddsApi(EVENTS, board))
    with pytest.raises(OddsApiError, match="ODDS_API_KEY"):
        c.featured_odds(["spreads"], ["pinnacle"])
