"""Direction-aware price selection and the reference (no-vig) price.

Two different jobs, kept apart on purpose:

1. best_price(): the price you would BET. Best payout for the exact
   (event, market, player, side, line). It never looks at the other side.
   The known bug (NBA_CONTEXT.md section 12) took max odds regardless of
   side, so an UNDER got priced off the OVER's best odds.

2. reference_prob(): the market's no-vig probability. De-vigs each book's
   OWN two-way pair (same book, same line, same snapshot), then averages
   independent opinions. Never pairs a best Over from book A with a best
   Under from book B: that builds a fake low-hold market.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta

from . import config as C
from .devig import OneWayMarketError, devig_decimal
from .fees import effective_price
from .quotes import Quote


class NoQuoteError(LookupError):
    """No price exists for the exact side and line requested."""


class MixedSnapshotError(ValueError):
    """Quotes from different snapshots were mixed in one selection."""


_LINE_TOL = 1e-9


def same_line(a: float | None, b: float | None) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return abs(a - b) < _LINE_TOL


def counterpart_line(market: str, line: float | None) -> float | None:
    """Spreads: the other team's line is -line. Totals/props: same line."""
    if line is None:
        return None
    return -line if market in C.SPREAD_LIKE_MARKETS else line


def _check_single_snapshot(quotes: Sequence[Quote]) -> None:
    snaps = {q.snapshot_id for q in quotes}
    if len(snaps) > 1:
        raise MixedSnapshotError(f"Quotes span {len(snaps)} snapshots; select within one")


def _matches(q: Quote, event_id: str, market: str, player: str | None) -> bool:
    return q.event_id == event_id and q.market == market and q.player == player


def _fresh(q: Quote, as_of: datetime | None, max_age: timedelta | None) -> bool:
    if as_of is None or max_age is None:
        return True
    if q.market_last_update is None:
        return False
    return as_of - q.market_last_update <= max_age


def best_price(
    quotes: Iterable[Quote],
    *,
    event_id: str,
    market: str,
    side: str,
    line: float | None,
    player: str | None = None,
    books: Iterable[str] | None = None,
    as_of: datetime | None = None,
    max_age: timedelta | None = None,
    net_of_fees: bool = True,
) -> Quote:
    """Best price for exactly this side and line. Never falls back to another side.

    Ranked by the price actually received (net of exchange fees) unless
    net_of_fees=False. Use fees.effective_price(q) for the price to log.
    """
    allowed = set(books) if books is not None else None
    candidates = [
        q
        for q in quotes
        if _matches(q, event_id, market, player)
        and q.side == side
        and same_line(q.line, line)
        and (allowed is None or q.book in allowed)
        and _fresh(q, as_of, max_age)
    ]
    if not candidates:
        raise NoQuoteError(
            f"No {side} {line} price for {market} {player or ''} in event {event_id}".strip()
        )
    _check_single_snapshot(candidates)
    price = effective_price if net_of_fees else (lambda q: q.price_decimal)
    return sorted(candidates, key=lambda q: (-price(q), q.book))[0]


@dataclass(frozen=True)
class TwoWayPair:
    book: str
    side_quote: Quote  # the side asked for
    other_quote: Quote  # the other side, same book, same snapshot


def two_way_pairs(
    quotes: Iterable[Quote],
    *,
    event_id: str,
    market: str,
    side: str,
    line: float | None,
    player: str | None = None,
) -> tuple[list[TwoWayPair], list[str]]:
    """Per-book complete two-way pairs, plus the books that were one-way."""
    rel = [q for q in quotes if _matches(q, event_id, market, player)]
    _check_single_snapshot(rel)
    other_line = counterpart_line(market, line)
    by_book: dict[str, list[Quote]] = {}
    for q in rel:
        by_book.setdefault(q.book, []).append(q)

    pairs: list[TwoWayPair] = []
    one_way: list[str] = []
    for book, qs in sorted(by_book.items()):
        mine = [q for q in qs if q.side == side and same_line(q.line, line)]
        others = [q for q in qs if q.side != side and same_line(q.line, other_line)]
        if not mine and not others:
            continue  # book does not post this line at all
        if len(mine) == 1 and len(others) == 1:
            pairs.append(TwoWayPair(book, mine[0], others[0]))
        elif len(mine) > 1 or len(others) > 1:
            raise ValueError(f"{book}: ambiguous outcomes for {market} {side} {line}")
        else:
            one_way.append(book)
    return pairs, one_way


@dataclass(frozen=True)
class ReferencePrice:
    p: float  # no-vig probability of the requested side
    ref_quality: str
    method: str
    books: tuple[str, ...]
    groups: tuple[str, ...]
    primary_book: str
    primary_price_decimal: float  # requested side's price at the primary book
    line: float | None
    one_way_books: tuple[str, ...]
    stale_books: tuple[str, ...]


_EXCHANGE_PREF = ("kalshi", "novig", "prophetx", "polymarket_us")


def _primary(books: Sequence[str], tier: str) -> str:
    if tier == "sharp":
        return "pinnacle"
    if tier == "exchange":
        for b in _EXCHANGE_PREF:
            if b in books:
                return b
    if "fanduel" in books:  # FanDuel originates its own props (section 5)
        return "fanduel"
    return sorted(books)[0]


def reference_prob(
    quotes: Iterable[Quote],
    *,
    event_id: str,
    market: str,
    side: str,
    line: float | None,
    player: str | None = None,
    method: str = C.DEFAULT_DEVIG_METHOD,
    as_of: datetime | None = None,
    max_age: timedelta | None = None,
    consensus: bool = False,
) -> ReferencePrice:
    """No-vig probability of `side` from the best available reference tier.

    consensus=True (sides/totals decision pricing only; approved 2026-10-22): when Pinnacle AND at
    least one exchange post a fresh two-way pair on the same line, the reference is the equal-weight
    mean of the Pinnacle de-vig and the mean exchange de-vig (weights are a placeholder until fit
    out of sample). An exchange price that beats Pinnacle's fair is then information, not edge.
    CLV grading keeps consensus=False (Pinnacle-only close) so history stays comparable.

    Sides/totals: Pinnacle -> exchanges -> retail consensus (section 14).
    Props: retail two-way only; exchanges are not a prop de-vig anchor.
    Feed-sharing books (DK/MGM/Caesars) count as one opinion.
    """
    quotes = list(quotes)
    pairs, one_way = two_way_pairs(
        quotes, event_id=event_id, market=market, side=side, line=line, player=player
    )
    stale = [
        p.book
        for p in pairs
        if not (_fresh(p.side_quote, as_of, max_age) and _fresh(p.other_quote, as_of, max_age))
    ]
    pairs = [p for p in pairs if p.book not in stale]
    if not pairs:
        if one_way:
            raise OneWayMarketError(
                f"Only one-way prices for {market} {side} {line}: {', '.join(one_way)}"
            )
        raise NoQuoteError(f"No fresh two-way reference for {market} {side} {line}")

    cls = C.market_class(market)
    if cls == C.SIDES_TOTALS:
        tiers = [
            ("sharp", [p for p in pairs if p.book in C.SHARP_BOOKS]),
            ("exchange", [p for p in pairs if p.book in C.EXCHANGE_BOOKS]),
            (
                "retail",
                [p for p in pairs if p.book not in C.SHARP_BOOKS | C.EXCHANGE_BOOKS],
            ),
        ]
    else:
        tiers = [
            (
                "retail",
                [p for p in pairs if p.book not in C.SHARP_BOOKS | C.EXCHANGE_BOOKS],
            )
        ]

    tier, chosen = next(((t, ps) for t, ps in tiers if ps), (None, []))
    if not chosen:
        raise NoQuoteError(f"No usable reference tier for {market} {side} {line}")

    if consensus and cls == C.SIDES_TOTALS and tier == "sharp":
        exch = [p for p in pairs if p.book in C.EXCHANGE_BOOKS]
        if exch:
            chosen = chosen + exch
    by_group: dict[str, list[float]] = {}
    for pr in chosen:
        res = devig_decimal([pr.side_quote.price_decimal, pr.other_quote.price_decimal], method)
        by_group.setdefault(C.opinion_group(pr.book), []).append(res.probs[0])
    if consensus and tier == "sharp" and len(by_group) > 1:
        sharp_p = sum(by_group[C.opinion_group("pinnacle")]) / len(by_group[C.opinion_group("pinnacle")])
        ex = [x for g, v in by_group.items() if g != C.opinion_group("pinnacle") for x in v]
        p = 0.5 * sharp_p + 0.5 * (sum(ex) / len(ex))
    else:
        group_means = [sum(v) / len(v) for v in by_group.values()]
        p = sum(group_means) / len(group_means)

    if tier == "retail":
        quality = "consensus_retail" if len(by_group) >= 2 else "single_retail"
    else:
        quality = tier

    books = tuple(sorted(pr.book for pr in chosen))
    primary = _primary(books, tier)
    primary_quote = next(pr.side_quote for pr in chosen if pr.book == primary)
    return ReferencePrice(
        p=p,
        ref_quality=quality,
        method=method,
        books=books,
        groups=tuple(sorted(by_group)),
        primary_book=primary,
        primary_price_decimal=primary_quote.price_decimal,
        line=line,
        one_way_books=tuple(one_way),
        stale_books=tuple(stale),
    )


def worse_quality(a: str | None, b: str | None) -> str | None:
    """The lower of two reference qualities (None-safe)."""
    if a is None:
        return b
    if b is None:
        return a
    order = C.REF_QUALITY_ORDER
    return a if order.index(a) >= order.index(b) else b
