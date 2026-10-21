"""Closing line value against the de-vigged close (NN1).

CLV per unit staked: $\\text{CLV} = p_{close} \\cdot d_{bet} - 1$, where
$p_{close}$ is the closing no-vig probability of the bet AS PLACED (same side,
same line) and $d_{bet}$ is the decimal price taken. Equivalent to
$(p_{close} - \\pi_{bet}) / \\pi_{bet}$.

When the closing line moved off the bet line:
- sides/totals: the close is shifted to the bet line with a discretized normal
  (pushes handled) and labeled `line_shift_normal_est`. Its spread parameter is
  a PLACEHOLDER until fit in Phase 1b.
- props: not priced until the Phase 4 prop distribution exists
  (`line_moved_unpriced`, clv_pct left empty). No estimate is presented as sourced.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta
from math import erf, floor, sqrt

from . import config as C
from .devig import OneWayMarketError
from .odds import american_to_decimal, decimal_to_american
from .quotes import Quote
from .selection import NoQuoteError, ReferencePrice, reference_prob, same_line

EXACT = "exact_line"
SHIFTED = "line_shift_normal_est"
UNPRICED = "line_moved_unpriced"
MISSING = "close_missing"


@dataclass(frozen=True)
class ClvResult:
    clv_pct: float | None
    p_close_novig: float | None  # closing no-vig prob of the bet as placed
    close_line: float | None
    close_price: float | None  # American, primary reference book, bet side, close line
    clv_method: str
    ref_quality: str | None
    devig_method: str | None
    close_ts: datetime | None


def clv_exact(p_close_novig: float, bet_price_american: float) -> float:
    return p_close_novig * american_to_decimal(bet_price_american) - 1.0


# --- Line shift (sides/totals only) ---------------------------------------------
def _phi(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def bet_threshold(market: str, side: str, line: float) -> float:
    """Express a bet as 'X > c' in the bet side's own frame.

    Totals: Over L -> X = total, c = L; Under L -> X = -total, c = -L.
    Spreads: team at line l -> X = that team's margin, c = -l.
    """
    if market in C.TOTAL_LIKE_MARKETS:
        if side == "Over":
            return line
        if side == "Under":
            return -line
        raise ValueError(f"Totals side must be Over/Under, got {side!r}")
    if market in C.SPREAD_LIKE_MARKETS:
        return -line
    raise ValueError(f"Line shift not defined for {market!r}")


def win_push(c: float, mu: float, sigma: float) -> tuple[float, float]:
    """P(X > c) and P(X == c) for integer X = round(Y), Y ~ N(mu, sigma^2)."""
    if abs(c - round(c)) < 1e-9:  # integer line: push possible
        k = round(c)
        upper = _phi((k + 0.5 - mu) / sigma)
        return 1.0 - upper, upper - _phi((k - 0.5 - mu) / sigma)
    k = floor(c)
    return 1.0 - _phi((k + 0.5 - mu) / sigma), 0.0


def p_no_push(c: float, mu: float, sigma: float) -> float:
    w, p = win_push(c, mu, sigma)
    return w / (1.0 - p)


def solve_mu(c_close: float, p_close: float, sigma: float) -> float:
    lo, hi = c_close - 12.0 * sigma, c_close + 12.0 * sigma
    for _ in range(200):  # p_no_push increases with mu
        mid = 0.5 * (lo + hi)
        if p_no_push(c_close, mid, sigma) < p_close:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def shift_prob(
    p_close: float, close_line: float, bet_line: float, *, market: str, side: str, sigma: float
) -> float:
    """Closing no-vig prob at the close line -> prob at the bet line (no-push basis)."""
    c_close = bet_threshold(market, side, close_line)
    c_bet = bet_threshold(market, side, bet_line)
    mu = solve_mu(c_close, p_close, sigma)
    return p_no_push(c_bet, mu, sigma)


# --- Full CLV from a close snapshot -----------------------------------------------
def _try_ref(quotes, **kw) -> ReferencePrice | None:
    try:
        return reference_prob(quotes, **kw)
    except (NoQuoteError, OneWayMarketError):
        return None


def compute_clv(
    close_quotes: Iterable[Quote],
    *,
    event_id: str,
    market: str,
    side: str,
    bet_line: float | None,
    bet_price_american: float,
    player: str | None = None,
    method: str = C.DEFAULT_DEVIG_METHOD,
    sigma: float | None = None,
    max_age: timedelta | None = timedelta(minutes=C.MAX_REF_STALENESS_MIN),
) -> ClvResult:
    quotes = list(close_quotes)
    if not quotes:
        return ClvResult(None, None, None, None, MISSING, None, None, None)
    close_ts = max(q.fetched_at for q in quotes)
    common = dict(
        event_id=event_id, market=market, side=side, player=player,
        method=method, as_of=close_ts, max_age=max_age,
    )

    exact = _try_ref(quotes, line=bet_line, **common)
    if exact is not None:
        return ClvResult(
            clv_pct=clv_exact(exact.p, bet_price_american),
            p_close_novig=exact.p,
            close_line=bet_line,
            close_price=decimal_to_american(exact.primary_price_decimal),
            clv_method=EXACT,
            ref_quality=exact.ref_quality,
            devig_method=method,
            close_ts=close_ts,
        )

    # Line moved: find the closing line(s) posted for this side.
    lines = sorted(
        {
            q.line
            for q in quotes
            if q.event_id == event_id and q.market == market and q.side == side
            and q.player == player and q.line is not None
            and not same_line(q.line, bet_line)
        }
    )
    refs = [r for r in (_try_ref(quotes, line=ln, **common) for ln in lines) if r is not None]
    if bet_line is None or not refs:
        return ClvResult(None, None, None, None, MISSING, None, None, close_ts)

    order = C.REF_QUALITY_ORDER
    best = min(refs, key=lambda r: (order.index(r.ref_quality), abs(r.line - bet_line)))
    close_price = decimal_to_american(best.primary_price_decimal)

    shift_key = (
        "spreads" if market in C.SPREAD_LIKE_MARKETS
        else "totals" if market in C.TOTAL_LIKE_MARKETS
        else None
    )
    if shift_key is None:  # props (and anything without a margin model)
        return ClvResult(None, None, best.line, close_price, UNPRICED, best.ref_quality, method, close_ts)

    s = sigma if sigma is not None else C.CLV_SHIFT_SIGMA[shift_key]
    p_bet_line = shift_prob(best.p, best.line, bet_line, market=market, side=side, sigma=s)
    return ClvResult(
        clv_pct=clv_exact(p_bet_line, bet_price_american),
        p_close_novig=p_bet_line,
        close_line=best.line,
        close_price=close_price,
        clv_method=SHIFTED,
        ref_quality=best.ref_quality,
        devig_method=method,
        close_ts=close_ts,
    )
