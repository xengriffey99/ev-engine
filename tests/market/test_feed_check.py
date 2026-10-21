from factories import q

from market.feed_check import feed_agreement


def _pair(book, player, line, over, under, snap="s1"):
    return [q(book, "Over", line, over, player=player, snapshot_id=snap),
            q(book, "Under", line, under, player=player, snapshot_id=snap)]


def test_identical_books_flagged_as_one_feed():
    quotes = []
    for i in range(60):
        p = f"P{i}"
        quotes += _pair("draftkings", p, 20.5, -115, -105) + _pair("betmgm", p, 20.5, -115, -105)
    res = feed_agreement(quotes, "draftkings", "betmgm")
    assert res.n_compared == 60 and res.same_line_rate == 1.0 and res.same_price_rate == 1.0
    assert res.same_feed() is True


def test_independent_books_not_flagged():
    quotes = []
    for i in range(60):
        p = f"P{i}"
        quotes += _pair("fanduel", p, 20.5, -120, -110) + _pair("draftkings", p, 21.5, -110, -110)
    res = feed_agreement(quotes, "fanduel", "draftkings")
    assert res.same_line_rate == 0.0 and res.same_feed() is False


def test_small_sample_returns_none():
    quotes = _pair("draftkings", "P", 20.5, -115, -105) + _pair("betmgm", "P", 20.5, -115, -105)
    assert feed_agreement(quotes, "draftkings", "betmgm").same_feed() is None


def test_one_sided_boards_are_not_compared():
    quotes = [q("draftkings", "Over", 20.5, -115, player="P")] + _pair("betmgm", "P", 20.5, -115, -105)
    assert feed_agreement(quotes, "draftkings", "betmgm").n_compared == 0
