import pytest
from factories import AWAY, HOME, game

from market.fees import effective_decimal, effective_price, fee_note
from market.odds import decimal_to_american
from market.selection import best_price


def test_kalshi_taker_fee_at_even_money():
    # Buy at 50c: fee 0.07 * 0.5 * 0.5 = 1.75c -> cost 51.75c for $1 -> about -107
    d = effective_decimal("kalshi", 2.0)
    assert d == pytest.approx(1 / 0.5175)
    assert decimal_to_american(d) == pytest.approx(-107.25, abs=0.01)


def test_kalshi_fee_shrinks_toward_the_tails():
    edge = lambda d: d - effective_decimal("kalshi", d)
    assert edge(2.0) > 0 and effective_decimal("kalshi", 1.25) < 1.25


def test_commission_on_net_winnings():
    assert effective_decimal("prophetx", 2.0) == pytest.approx(1.98)
    assert effective_decimal("novig", 1.909) == pytest.approx(1.909)  # 0% pregame, confirmed 2026-10-22


def test_sportsbooks_are_unchanged():
    assert effective_decimal("draftkings", 1.91) == 1.91
    assert effective_decimal("fanduel", 2.5) == 2.5


def test_selection_ranks_net_of_fees():
    # Kalshi posts +100 (net ~ -107); DraftKings posts -105: DraftKings is the better real price.
    board = [game("kalshi", AWAY, 4.5, +100), game("draftkings", AWAY, 4.5, -105)]
    pick = best_price(board, event_id="evt1", market="spreads", side=AWAY, line=4.5)
    assert pick.book == "draftkings"
    raw = best_price(board, event_id="evt1", market="spreads", side=AWAY, line=4.5, net_of_fees=False)
    assert raw.book == "kalshi"


def test_fee_note():
    q = game("kalshi", HOME, -4.5, +100)
    assert fee_note(q) == "posted=+100; fee=kalshi_taker_0.07p(1-p)"
    assert fee_note(game("fanduel", HOME, -4.5, -110)) == ""
    assert effective_price(q) < q.price_decimal
