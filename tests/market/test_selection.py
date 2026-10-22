"""Direction-aware selection. The UNDER tests are the Phase 0 acceptance item:
an UNDER is never priced off the OVER's best odds."""

import random
from datetime import timedelta

import pytest
from factories import AWAY, FETCHED, HOME, game, q

from market.devig import OneWayMarketError
from market.fees import effective_price
from market.selection import (
    MixedSnapshotError,
    NoQuoteError,
    best_price,
    reference_prob,
    two_way_pairs,
    worse_quality,
)

TATUM = "Jayson Tatum"


def _buggy_max_odds(quotes):
    """The known bug pattern: max odds regardless of side."""
    return max(quotes, key=lambda x: x.price_decimal)


def _prop_board():
    return [
        q("fanduel", "Over", 24.5, +130),  # best price on the whole board
        q("fanduel", "Under", 24.5, -160),
        q("draftkings", "Over", 24.5, +120),
        q("draftkings", "Under", 24.5, -150),
        q("betmgm", "Under", 24.5, -145),  # best UNDER
        q("betmgm", "Over", 24.5, +115),
    ]


# --- UNDER is never priced off the OVER ---------------------------------------------
def test_under_never_uses_the_overs_best_odds():
    board = _prop_board()
    pick = best_price(board, event_id="evt1", market="player_points", side="Under", line=24.5, player=TATUM)
    assert pick.side == "Under"
    assert pick.book == "betmgm"
    assert pick.price_american == pytest.approx(-145)
    # Guard against regressions to the known bug: it would have taken the +130 OVER.
    bug = _buggy_max_odds(board)
    assert bug.side == "Over" and bug.price_american == pytest.approx(130)
    assert pick.price_decimal < bug.price_decimal


def test_under_raises_instead_of_falling_back_to_over():
    overs_only = [q("fanduel", "Over", 24.5, +130), q("draftkings", "Over", 24.5, +120)]
    with pytest.raises(NoQuoteError):
        best_price(overs_only, event_id="evt1", market="player_points", side="Under", line=24.5, player=TATUM)


def test_under_never_uses_a_different_line():
    board = [
        q("fanduel", "Under", 23.5, -105),  # better price, wrong line
        q("draftkings", "Under", 24.5, -115),
    ]
    pick = best_price(board, event_id="evt1", market="player_points", side="Under", line=24.5, player=TATUM)
    assert pick.line == 24.5 and pick.price_american == pytest.approx(-115)


def test_under_never_uses_a_different_player():
    board = [
        q("fanduel", "Under", 24.5, +105, player="Jaylen Brown"),
        q("draftkings", "Under", 24.5, -115),
    ]
    pick = best_price(board, event_id="evt1", market="player_points", side="Under", line=24.5, player=TATUM)
    assert pick.player == TATUM and pick.price_american == pytest.approx(-115)


def test_total_under_never_uses_total_over():
    board = [
        game("fanduel", "Over", 221.5, +105, market="totals"),
        game("fanduel", "Under", 221.5, -125, market="totals"),
        game("draftkings", "Under", 221.5, -118, market="totals"),
    ]
    pick = best_price(board, event_id="evt1", market="totals", side="Under", line=221.5)
    assert pick.side == "Under" and pick.price_american == pytest.approx(-118)


def test_spread_dog_never_uses_favorites_price():
    board = [
        game("fanduel", HOME, -4.5, +100),  # favorite at a plus price (best number)
        game("fanduel", AWAY, 4.5, -120),
        game("draftkings", AWAY, 4.5, -112),
    ]
    pick = best_price(board, event_id="evt1", market="spreads", side=AWAY, line=4.5)
    assert pick.side == AWAY and pick.price_american == pytest.approx(-112)


def test_randomized_selection_is_always_same_side_and_best_for_that_side():
    rng = random.Random(2026)
    books = ["fanduel", "draftkings", "betmgm", "betrivers", "kalshi", "novig"]
    for _ in range(500):
        board = []
        for b in books:
            for side in ("Over", "Under"):
                if rng.random() < 0.8:
                    line = rng.choice([23.5, 24.5])
                    board.append(q(b, side, line, rng.choice([-1, 1]) * rng.uniform(100, 200)))
        side = rng.choice(["Over", "Under"])
        line = rng.choice([23.5, 24.5])
        same = [x for x in board if x.side == side and x.line == line]
        if not same:
            with pytest.raises(NoQuoteError):
                best_price(board, event_id="evt1", market="player_points", side=side, line=line, player=TATUM)
            continue
        pick = best_price(board, event_id="evt1", market="player_points", side=side, line=line, player=TATUM)
        assert pick.side == side and pick.line == line
        assert effective_price(pick) == max(effective_price(x) for x in same)
        raw = best_price(board, event_id="evt1", market="player_points", side=side, line=line,
                         player=TATUM, net_of_fees=False)
        assert raw.side == side and raw.price_decimal == max(x.price_decimal for x in same)


def test_stale_quotes_are_skipped():
    board = [
        q("fanduel", "Under", 24.5, -105, age_min=30),  # better but stale
        q("draftkings", "Under", 24.5, -115, age_min=1),
    ]
    pick = best_price(
        board, event_id="evt1", market="player_points", side="Under", line=24.5, player=TATUM,
        as_of=FETCHED, max_age=timedelta(minutes=10),
    )
    assert pick.book == "draftkings"


def test_mixed_snapshots_rejected():
    board = [
        q("fanduel", "Under", 24.5, -105, snapshot_id="noon"),
        q("draftkings", "Under", 24.5, -115, snapshot_id="close"),
    ]
    with pytest.raises(MixedSnapshotError):
        best_price(board, event_id="evt1", market="player_points", side="Under", line=24.5, player=TATUM)


def test_bettable_book_filter():
    board = [game("pinnacle", AWAY, 4.5, +102), game("draftkings", AWAY, 4.5, -108)]
    pick = best_price(board, event_id="evt1", market="spreads", side=AWAY, line=4.5, books={"draftkings"})
    assert pick.book == "draftkings"


# --- Two-way pairs and the reference price -------------------------------------
def test_pairs_never_cross_books():
    # Book A posts only the Over, book B only the Under: no two-way market exists.
    board = [q("fanduel", "Over", 24.5, +130), q("draftkings", "Under", 24.5, +110)]
    pairs, one_way = two_way_pairs(board, event_id="evt1", market="player_points", side="Under", line=24.5, player=TATUM)
    assert pairs == []
    assert sorted(one_way) == ["draftkings", "fanduel"]
    with pytest.raises(OneWayMarketError):
        reference_prob(board, event_id="evt1", market="player_points", side="Under", line=24.5, player=TATUM)


def test_spread_pairs_use_opposite_line():
    board = [game("pinnacle", HOME, -4.5, -105), game("pinnacle", AWAY, 4.5, -105)]
    pairs, _ = two_way_pairs(board, event_id="evt1", market="spreads", side=AWAY, line=4.5)
    assert len(pairs) == 1 and pairs[0].other_quote.side == HOME and pairs[0].other_quote.line == -4.5


def test_sides_reference_prefers_pinnacle():
    board = [
        game("pinnacle", HOME, -4.5, -110), game("pinnacle", AWAY, 4.5, +100),
        game("kalshi", HOME, -4.5, -102), game("kalshi", AWAY, 4.5, -102),
        game("fanduel", HOME, -4.5, -110), game("fanduel", AWAY, 4.5, -110),
    ]
    ref = reference_prob(board, event_id="evt1", market="spreads", side=AWAY, line=4.5)
    assert ref.ref_quality == "sharp" and ref.books == ("pinnacle",)
    assert ref.primary_book == "pinnacle"
    assert 0.47 < ref.p < 0.50  # +100 vs -110: the dog side under 50%


def test_sides_reference_falls_back_to_exchange_then_retail():
    ex = [
        game("kalshi", HOME, -4.5, -102), game("kalshi", AWAY, 4.5, -102),
        game("fanduel", HOME, -4.5, -110), game("fanduel", AWAY, 4.5, -110),
    ]
    assert reference_prob(ex, event_id="evt1", market="spreads", side=AWAY, line=4.5).ref_quality == "exchange"
    retail = ex[2:]
    assert reference_prob(retail, event_id="evt1", market="spreads", side=AWAY, line=4.5).ref_quality == "single_retail"


def test_dk_feed_books_count_as_one_opinion():
    board = [
        q("draftkings", "Over", 24.5, -110), q("draftkings", "Under", 24.5, -110),
        q("betmgm", "Over", 24.5, -110), q("betmgm", "Under", 24.5, -110),
    ]
    ref = reference_prob(board, event_id="evt1", market="player_points", side="Over", line=24.5, player=TATUM)
    assert ref.groups == ("dk_feed",)
    assert ref.ref_quality == "single_retail"
    board += [q("fanduel", "Over", 24.5, -120), q("fanduel", "Under", 24.5, +100)]
    ref = reference_prob(board, event_id="evt1", market="player_points", side="Over", line=24.5, player=TATUM)
    assert ref.ref_quality == "consensus_retail" and ref.primary_book == "fanduel"
    assert ref.groups == ("dk_feed", "fanduel")


def test_props_reference_excludes_exchanges():
    board = [q("kalshi", "Over", 24.5, -102), q("kalshi", "Under", 24.5, -102)]
    with pytest.raises(NoQuoteError):
        reference_prob(board, event_id="evt1", market="player_points", side="Over", line=24.5, player=TATUM)


def test_reference_is_direction_consistent():
    board = [q("fanduel", "Over", 24.5, -130), q("fanduel", "Under", 24.5, +105)]
    over = reference_prob(board, event_id="evt1", market="player_points", side="Over", line=24.5, player=TATUM)
    under = reference_prob(board, event_id="evt1", market="player_points", side="Under", line=24.5, player=TATUM)
    assert over.p + under.p == pytest.approx(1.0, abs=1e-12)
    assert over.p > 0.5 > under.p


def test_worse_quality():
    assert worse_quality("sharp", "single_retail") == "single_retail"
    assert worse_quality("exchange", "sharp") == "exchange"
    assert worse_quality(None, "sharp") == "sharp"


def test_consensus_blends_pinnacle_and_exchange_equally():
    board = [q("pinnacle", "Over", 220.5, -110, market="totals", player=None), q("pinnacle", "Under", 220.5, -110, market="totals", player=None),
             q("novig", "Over", 220.5, -125, market="totals", player=None), q("novig", "Under", 220.5, +105, market="totals", player=None)]
    kw = dict(event_id=board[0].event_id, market="totals", side="Over", line=220.5)
    base = reference_prob(board, **kw)
    cons = reference_prob(board, consensus=True, **kw)
    ex = reference_prob([x for x in board if x.book == "novig"], **kw)
    assert base.p == pytest.approx(0.5)
    assert cons.p == pytest.approx(0.5 * base.p + 0.5 * ex.p)
    assert cons.ref_quality == "sharp" and set(cons.books) == {"pinnacle", "novig"}


def test_consensus_is_noop_without_exchange_and_never_for_props():
    board = [q("pinnacle", "Over", 220.5, -110, market="totals", player=None), q("pinnacle", "Under", 220.5, -112, market="totals", player=None)]
    kw = dict(event_id=board[0].event_id, market="totals", side="Over", line=220.5)
    assert reference_prob(board, consensus=True, **kw).p == pytest.approx(reference_prob(board, **kw).p)
