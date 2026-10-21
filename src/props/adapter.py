"""Prop engine -> opinion rows (props lane).

- Market environment: home margin and total implied by the de-vigged sides decision snapshot
  (best reference tier, power de-vig): mu = line-implied centre shifted by the de-vigged price,
  mu_margin = -L_home + sd x Phi^-1(p_home_cover), mu_total = L + sd x Phi^-1(p_over).
- One opinion per player and stat, at his MAIN line (most books two-way, then nearest 50%): no
  alternate ladders, so no alternate-under lines (section 11).
- p_bet = 0.50 x p_model + 0.50 x p_market (props start weight); gate 3.5%; 1/16 Kelly.
- Restricted types are logged as passes (pass_reason restricted), never priced as bets.
- Every prop reference on The Odds API is retail (section 14) and is labelled as such.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from scipy.stats import norm

from market import config as C
from market.devig import OneWayMarketError
from market.fees import effective_price, fee_note
from market.odds import decimal_to_american
from market.opinion_log import Opinion
from market.quotes import Quote
from market.selection import NoQuoteError, best_price, reference_prob

from .engine import MarketEnv, PlayerIn, prob_over, restriction, summarize

MODEL_VERSION = "props-v1 (D&T rates, market env)"
STAT_MARKET = {"points": "player_points"}


def norm_name(name: str) -> str:
    n = name.lower()
    n = re.sub(r"\b(jr|sr|ii|iii|iv)\b\.?", "", n)
    return re.sub(r"[^a-z]", "", n)


def market_env(quotes: Sequence[Quote], event_id: str, home_abbr: str, away_abbr: str,
               sd_margin: float = C.CLV_SHIFT_SIGMA["spreads"],
               sd_total: float = C.CLV_SHIFT_SIGMA["totals"]) -> tuple[MarketEnv, dict]:
    qs = [q for q in quotes if q.event_id == event_id]
    if not qs:
        raise NoQuoteError(f"{event_id} not in the sides snapshot")
    home = qs[0].home_team

    def centre(market, side):
        lines = sorted({q.line for q in qs if q.market == market and q.side == side and q.book == "pinnacle"})
        cands = lines or sorted({q.line for q in qs if q.market == market and q.side == side})
        best = None
        for ln in cands:
            try:
                r = reference_prob(qs, event_id=event_id, market=market, side=side, line=ln)
            except (NoQuoteError, OneWayMarketError):
                continue
            key = (C.REF_QUALITY_ORDER.index(r.ref_quality), abs(r.p - 0.5))
            if best is None or key < best[0]:
                best = (key, ln, r)
        if best is None:
            raise NoQuoteError(f"{event_id} {market}: no two-way reference")
        return best[1], best[2]

    ls, rs = centre("spreads", home)
    lt, rt = centre("totals", "Over")
    mu_m = float(-ls + sd_margin * norm.ppf(rs.p))
    mu_t = float(lt + sd_total * norm.ppf(rt.p))
    env = MarketEnv(home_abbr, away_abbr, mu_margin=mu_m, mu_total=mu_t, sd_margin=sd_margin, sd_total=sd_total)
    info = {"spread_line_home": ls, "p_home_cover": rs.p, "spread_ref": rs.ref_quality,
            "total_line": lt, "p_over": rt.p, "total_ref": rt.ref_quality,
            "mu_margin": round(mu_m, 2), "mu_total": round(mu_t, 2)}
    return env, info


@dataclass
class PropRow:
    opinion: Opinion
    summary: dict
    alt: dict  # the other side, for reporting


def main_lines(quotes: Sequence[Quote], event_id: str, market: str) -> dict[str, float]:
    """Player -> main line: the line with the most books posting both sides, then nearest 50%."""
    by = defaultdict(set)
    for q in quotes:
        if q.event_id == event_id and q.market == market and q.player and q.line is not None:
            by[(q.player, q.line)].add(q.side)
    best: dict[str, tuple] = {}
    for (pl, ln), sides in by.items():
        if not {"Over", "Under"} <= sides:
            continue
        try:
            r = reference_prob(quotes, event_id=event_id, market=market, side="Over", line=ln, player=pl)
        except (NoQuoteError, OneWayMarketError):
            continue
        key = (len(r.books), -abs(r.p - 0.5))
        if pl not in best or key > best[pl][0]:
            best[pl] = (key, ln)
    return {pl: v[1] for pl, v in best.items()}


def build_prop_opinions(sim: dict, quotes: Sequence[Quote], event_id: str, *, stat: str = "points",
                        books: frozenset[str] | None = None, hold: bool = False,
                        method: str = C.DEFAULT_DEVIG_METHOD) -> tuple[list[PropRow], list[str]]:
    books = books if books is not None else C.BETTABLE_BOOKS
    market = STAT_MARKET[stat]
    qs = [q for q in quotes if q.event_id == event_id and q.market == market]
    if not qs:
        return [], [f"{event_id}: no {market} quotes in the props snapshot"]
    snap_time = max(q.fetched_at for q in qs)
    by_norm = {norm_name(nm): sp for nm, sp in sim["players"].items()}
    w = C.BLEND_W[C.PROPS]
    rows: list[PropRow] = []
    skipped: list[str] = []
    for player, line in sorted(main_lines(qs, event_id, market).items()):
        sp = by_norm.get(norm_name(player))
        if sp is None:
            skipped.append(f"{player}: not in the player table")
            continue
        summ = summarize(sp)
        pl: PlayerIn = sp["player"]
        p_over_model = prob_over(sp["points"], line)
        cands = {}
        for side, pm in (("Over", p_over_model), ("Under", 1.0 - p_over_model)):
            try:
                ref = reference_prob(qs, event_id=event_id, market=market, side=side, line=line,
                                     player=player, method=method)
                pick = best_price(qs, event_id=event_id, market=market, side=side, line=line,
                                  player=player, books=books)
            except (NoQuoteError, OneWayMarketError):
                continue
            p_bet = w * pm + (1 - w) * ref.p
            cands[side] = (p_bet * effective_price(pick) - 1.0, pm, ref, pick)
        if not cands:
            skipped.append(f"{player} {line}: no two-way reference with a bettable price")
            continue
        side = max(cands, key=lambda s: cands[s][0])
        ev, pm, ref, pick = cands[side]
        reason = restriction(market, side, summ["exp_minutes"], pl.two_way, is_main_line=True)
        pass_reason = "restricted" if reason else ("model_hold" if hold else None)
        fee = fee_note(pick)
        op = Opinion(
            game_id=event_id, market=market, side=side, line=line, book=pick.book,
            price_american=decimal_to_american(effective_price(pick)),
            p_model=min(max(pm, 1e-4), 1 - 1e-4), p_market_novig=ref.p,
            model_version=MODEL_VERSION, ref_quality=ref.ref_quality, devig_method=method,
            player=player, stat=stat, pass_reason=pass_reason, timestamp=snap_time,
            notes=(f"median={summ['median']:.0f} mean={summ['mean']:.1f} sd={summ['sd']:.1f} "
                   f"exp_min={summ['exp_minutes']:.1f} p_play={summ['p_play']:.2f} status={pl.status}; "
                   f"ref_books={','.join(ref.books)} (retail reference); bet_state={C.BET_STATE}"
                   + (f"; restricted: {reason}" if reason else "") + (f"; {fee}" if fee else "")),
        )
        other = "Under" if side == "Over" else "Over"
        rows.append(PropRow(op, summ, {other: cands[other][0]} if other in cands else {}))
    return rows, skipped
