"""Do two books share one prop feed?

Spec section 5 counts DraftKings, Caesars and MGM as one opinion. The research
doc sources that to NFL props, so Phase 0 tests it on our own NBA snapshots:
if two books post the same line and the same prices on most shared players,
they are one opinion in the consensus (config.FEED_GROUPS).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from .quotes import Quote
from .selection import same_line


@dataclass(frozen=True)
class FeedAgreement:
    book_a: str
    book_b: str
    n_compared: int  # (snapshot, event, market, player) keys both books priced
    same_line_rate: float
    same_price_rate: float  # same line AND both prices within tolerance

    def same_feed(self, *, min_n: int = 50, line_thr: float = 0.90, price_thr: float = 0.80) -> bool | None:
        """None until the sample is big enough to say."""
        if self.n_compared < min_n:
            return None
        return self.same_line_rate >= line_thr and self.same_price_rate >= price_thr


def feed_agreement(
    quotes: Iterable[Quote], book_a: str, book_b: str, *, price_tol: float = 0.005
) -> FeedAgreement:
    """Compare each book's main (Over/Under) line per player in each snapshot."""
    boards: dict[tuple, dict[str, dict[str, Quote]]] = {}
    for q in quotes:
        if q.book not in (book_a, book_b) or q.player is None or q.side not in ("Over", "Under"):
            continue
        key = (q.snapshot_id, q.event_id, q.market, q.player)
        boards.setdefault(key, {}).setdefault(q.book, {})[q.side] = q

    n = same_l = same_p = 0
    for books in boards.values():
        a, b = books.get(book_a), books.get(book_b)
        if not a or not b or len(a) < 2 or len(b) < 2:
            continue
        n += 1
        if same_line(a["Over"].line, b["Over"].line) and same_line(a["Under"].line, b["Under"].line):
            same_l += 1
            if all(abs(a[s].price_decimal - b[s].price_decimal) <= price_tol for s in ("Over", "Under")):
                same_p += 1
    return FeedAgreement(book_a, book_b, n, same_l / n if n else 0.0, same_p / n if n else 0.0)
