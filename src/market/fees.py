"""Net-of-fee prices for exchanges.

The Odds API shows exchange prices before trading fees. Ranking or computing EV
on the posted price overstates every exchange edge, so the price you actually
get is used everywhere a bet price matters (selection, EV, CLV).

Sources (checked 2026-10-21):
- Kalshi taker fee per contract: $0.07 \\cdot P (1 - P)$, rounded up to the next cent
  per order (RotoWire fee guide, updated 2026-10-20). Buying at P costs $P + f$
  for a $1 payout: $d_{net} = 1 / (P + 0.07 P (1-P))$. Per-order rounding adds up
  to 1 cent per order on top and is ignored here (small for orders of 10+ contracts).
- ProphetX: commission on net winnings, 1% or 2% depending on the source
  (Legal Sports Report, 2026-10-06). 2% used: conservative.
- Novig: UNVERIFIED. LSR says no fee on pregame trades; RotoWire says commission
  on net winnings. 2% assumed (conservative) until confirmed in the app.
Commission on net winnings c: $d_{net} = 1 + (d - 1)(1 - c)$.
Reference (de-vig) prices are not fee-adjusted: fees change what you receive,
not the market's probability.
"""

from __future__ import annotations

from .quotes import Quote

KALSHI_TAKER_RATE = 0.07
NET_WINNINGS_COMMISSION = {"prophetx": 0.02, "novig": 0.02}
FEE_SOURCE = {
    "kalshi": "kalshi_taker_0.07p(1-p)",
    "prophetx": "commission_2pct_net_winnings",
    "novig": "commission_2pct_net_winnings_UNVERIFIED",
}


def effective_decimal(book: str, decimal: float) -> float:
    if book == "kalshi":
        p = 1.0 / decimal
        return 1.0 / (p + KALSHI_TAKER_RATE * p * (1.0 - p))
    c = NET_WINNINGS_COMMISSION.get(book)
    if c is not None:
        return 1.0 + (decimal - 1.0) * (1.0 - c)
    return decimal


def effective_price(q: Quote) -> float:
    """Decimal price actually received at q.book, net of fees."""
    return effective_decimal(q.book, q.price_decimal)


def fee_note(q: Quote) -> str:
    src = FEE_SOURCE.get(q.book)
    if src is None:
        return ""
    from .odds import decimal_to_american

    return f"posted={decimal_to_american(q.price_decimal):+.0f}; fee={src}"
