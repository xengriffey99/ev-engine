"""Module 6: market microstructure. Consensus lines, RLM, handle vs tickets, steam, key numbers."""
from __future__ import annotations

import statistics
from scipy.stats import norm

from .config import MARKET, SIM
from .pricing import devig

MARGIN_SD = (2 * SIM.team_sd ** 2 * (1 - SIM.rho)) ** 0.5
TOTAL_SD = (2 * SIM.team_sd ** 2 * (1 + SIM.rho)) ** 0.5


def fair_home_margin(home_spread: float, home_price: float, away_price: float) -> float:
    """Market-implied mean home margin from one book's two-sided spread price.

    home_spread is the home team's handicap (e.g. DET +1.5 -> +1.5). Home covers when
    margin > -home_spread, so mu = -home_spread + sd * Phi^-1(p_home_fair).
    """
    p_home, _ = devig(home_price, away_price)
    return -home_spread + MARGIN_SD * norm.ppf(p_home)


def fair_total(line: float, over_price: float, under_price: float) -> float:
    p_over, _ = devig(over_price, under_price)
    return line + TOTAL_SD * norm.ppf(p_over)


def consensus(books: dict) -> dict:
    """Median of book-level fair margins/totals plus best available prices per side."""
    margins = [fair_home_margin(b["home_spread"], b["home_spread_price"], b["away_spread_price"])
               for b in books.values() if b.get("home_spread") is not None]
    totals = [fair_total(b["total"], b["over_price"], b["under_price"])
              for b in books.values() if b.get("total") is not None]
    ml_home = [devig(b["home_ml"], b["away_ml"])[0] for b in books.values() if b.get("home_ml") is not None]
    return {
        "home_margin": statistics.median(margins) if margins else None,
        "total": statistics.median(totals) if totals else None,
        "home_win_prob_novig": statistics.median(ml_home) if ml_home else None,
        "spread_display": statistics.median([b["home_spread"] for b in books.values() if b.get("home_spread") is not None]),
        "total_display": statistics.median([b["total"] for b in books.values() if b.get("total") is not None]),
    }


def detect_rlm(open_line: float, current_line: float, public_side_pct: float, public_side_sign: int) -> dict:
    """Generic RLM test on a scalar line.

    public_side_sign: +1 if the public side benefits when the line goes UP (Under on a total,
    the home side on a home-spread line), -1 otherwise (Over; the away side).
    Books normally move a line to make the public side WORSE. RLM is the opposite: the line moves
    to make the public side BETTER, i.e. (current - open) * public_side_sign > 0, which implies
    the money that moved it sits on the other side.
    """
    move = current_line - open_line
    toward_public_benefit = move * public_side_sign > 0
    flag = public_side_pct >= MARKET.rlm_public_threshold and toward_public_benefit
    return {"flag": bool(flag), "move": round(move, 2), "public_pct": public_side_pct,
            "sub_threshold": bool(toward_public_benefit and not flag)}


def sharp_handle(money_pct: float, ticket_pct: float) -> bool:
    return money_pct > MARKET.sharp_money_min and ticket_pct < MARKET.sharp_ticket_max


def key_numbers_crossed(open_line: float, current_line: float) -> list[float]:
    """Signed home-spread path from open to current; a key number +/-k is crossed when it lies on
    the path, excluding the opening value itself. A favourite flip passes through zero."""
    lo, hi = sorted((open_line, current_line))
    hits = set()
    for k in MARKET.key_numbers:
        for v in (k, -k):
            if lo <= v <= hi and v != open_line:
                hits.add(k)
    return sorted(hits)


def detect_steam(snapshots: list[dict]) -> bool:
    """snapshots: [{'ts': minutes, 'book': str, 'line': float}] ordered by ts."""
    if not snapshots:
        return False
    for s in snapshots:
        window = [x for x in snapshots if 0 <= x["ts"] - s["ts"] <= MARKET.steam_window_min]
        moved = {x["book"] for x in window if abs(x["line"] - s["line"]) >= MARKET.steam_points}
        if len(moved) >= MARKET.steam_books:
            return True
    return False
