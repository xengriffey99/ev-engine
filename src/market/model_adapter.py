"""Model output -> opinion rows (sides/totals lane).

Any model plugs in through GameProjection: a mean and spread for the home
margin and the game total, plus the time the projection was made. Two sources:

- nba_sim 0.1.0 payloads (`payload_<date>.json`, built 2026-10-20 in this
  project): uses the STRUCTURAL projection. Its posterior already fuses the
  market (w_model ~0.10-0.12); feeding that in as p_model and then blending
  with the market again at w = 0.40 would count the market twice.
- nba-forecast v1 (Kalman): build GameProjection from its predicted margin,
  total and error spread in a few lines once the repo is attached.

p_model is on the same no-push basis as the de-vigged market: for a bet
"X > c" with X = round(Y), Y ~ N(mu, sd^2),
$p = P(X > c) / (1 - P(X = c))$.

As-of rule (NN9): a projection made after the decision snapshot is refused,
so no opinion uses information the logged price did not have.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from zoneinfo import ZoneInfo

from . import config as C
from .clv import p_no_push
from .devig import OneWayMarketError
from .fees import effective_price, fee_note
from .odds import decimal_to_american
from .opinion_log import Opinion
from .quotes import Quote, parse_ts
from .selection import NoQuoteError, best_price, reference_prob

ET = ZoneInfo("America/New_York")

# Odds API team names (full names) by standard abbreviation.
TEAMS = {
    "ATL": "Atlanta Hawks", "BOS": "Boston Celtics", "BKN": "Brooklyn Nets",
    "CHA": "Charlotte Hornets", "CHI": "Chicago Bulls", "CLE": "Cleveland Cavaliers",
    "DAL": "Dallas Mavericks", "DEN": "Denver Nuggets", "DET": "Detroit Pistons",
    "GSW": "Golden State Warriors", "HOU": "Houston Rockets", "IND": "Indiana Pacers",
    "LAC": "Los Angeles Clippers", "LAL": "Los Angeles Lakers", "MEM": "Memphis Grizzlies",
    "MIA": "Miami Heat", "MIL": "Milwaukee Bucks", "MIN": "Minnesota Timberwolves",
    "NOP": "New Orleans Pelicans", "NYK": "New York Knicks", "OKC": "Oklahoma City Thunder",
    "ORL": "Orlando Magic", "PHI": "Philadelphia 76ers", "PHX": "Phoenix Suns",
    "POR": "Portland Trail Blazers", "SAC": "Sacramento Kings", "SAS": "San Antonio Spurs",
    "TOR": "Toronto Raptors", "UTA": "Utah Jazz", "WAS": "Washington Wizards",
}
_ALIASES = {
    "BRK": "BKN", "CHO": "CHA", "GS": "GSW", "NO": "NOP", "NOR": "NOP", "NY": "NYK",
    "PHO": "PHX", "SA": "SAS", "UTAH": "UTA", "WSH": "WAS", "LA CLIPPERS": "LAC",
}
_BY_NAME = {v.upper(): k for k, v in TEAMS.items()}


def canonical_team(name: str) -> str:
    """Abbreviation or name -> the Odds API full name. Unknown names raise."""
    key = name.strip().upper()
    key = _ALIASES.get(key, key)
    if key in TEAMS:
        return TEAMS[key]
    if key in _BY_NAME:
        return TEAMS[_BY_NAME[key]]
    raise KeyError(f"Unknown team {name!r}")


@dataclass(frozen=True)
class GameProjection:
    model_game_id: str
    date_et: date
    home: str  # Odds API full name
    away: str
    mu_home_margin: float
    sd_margin: float
    mu_total: float
    sd_total: float
    model_version: str
    as_of: datetime  # when the projection was made (UTC-aware)
    source: str = ""


def from_nba_sim_payload(payload: dict) -> list[GameProjection]:
    """nba_sim 0.1.0 payload -> projections (structural, pre-fusion)."""
    slate = date.fromisoformat(payload["slate_date"])
    as_of = parse_ts(payload["generated_at"])
    out = []
    for g in payload["games"]:
        st, sim = g["projection"]["structural"], g["simulation"]
        out.append(GameProjection(
            model_game_id=g["game_id"], date_et=slate,
            home=canonical_team(g["home"]), away=canonical_team(g["away"]),
            mu_home_margin=float(st["home_margin"]), sd_margin=float(sim["sd_margin"]),
            mu_total=float(st["total"]), sd_total=float(sim["sd_total"]),
            model_version=payload["model_version"], as_of=as_of, source="nba_sim structural",
        ))
    return out


def p_model(proj: GameProjection, market: str, side: str, line: float) -> float:
    """Model probability of the bet winning, conditional on no push."""
    if market in C.SPREAD_LIKE_MARKETS:
        if side == proj.home:
            return p_no_push(-line, proj.mu_home_margin, proj.sd_margin)
        if side == proj.away:
            return p_no_push(-line, -proj.mu_home_margin, proj.sd_margin)
        raise ValueError(f"{side!r} is not in {proj.away} @ {proj.home}")
    if market in C.TOTAL_LIKE_MARKETS:
        if side == "Over":
            return p_no_push(line, proj.mu_total, proj.sd_total)
        if side == "Under":
            return p_no_push(-line, -proj.mu_total, proj.sd_total)
        raise ValueError(f"Totals side must be Over/Under, got {side!r}")
    raise ValueError(f"No model probability for {market!r}")


def match_events(projections: Sequence[GameProjection], quotes: Iterable[Quote]) -> dict[str, str]:
    """model_game_id -> Odds API event id, by ET date and team names."""
    events = {
        (q.commence_time.astimezone(ET).date(), q.home_team, q.away_team): q.event_id for q in quotes
    }
    out = {}
    for p in projections:
        eid = events.get((p.date_et, p.home, p.away))
        if eid:
            out[p.model_game_id] = eid
    return out


@dataclass(frozen=True)
class _Candidate:
    side: str
    line: float
    ref_p: float
    ref_quality: str
    ref_books: tuple[str, ...]
    pick: Quote
    p_model: float
    ev: float


def _candidates(home: str, away: str, pm_fn, event_id: str, market: str, quotes: list[Quote],
                books: frozenset[str], method: str) -> list[_Candidate]:
    anchor_side = home if market in C.SPREAD_LIKE_MARKETS else "Over"
    lines = sorted({
        q.line for q in quotes
        if q.event_id == event_id and q.market == market and q.side == anchor_side and q.line is not None
    })
    w = C.BLEND_W[C.market_class(market)]
    found: list[_Candidate] = []
    for ln in lines:
        if market in C.SPREAD_LIKE_MARKETS:
            pairs = [(home, ln), (away, -ln)]
        else:
            pairs = [("Over", ln), ("Under", ln)]
        for side, line in pairs:
            kw = dict(event_id=event_id, market=market, side=side, line=line)
            try:
                ref = reference_prob(quotes, method=method, **kw)
                pick = best_price(quotes, books=books, **kw)
            except (NoQuoteError, OneWayMarketError):
                continue
            pm = pm_fn(market, side, line, ref.p)
            p_bet = w * pm + (1.0 - w) * ref.p
            found.append(_Candidate(side, line, ref.p, ref.ref_quality, ref.books, pick, pm,
                                    p_bet * effective_price(pick) - 1.0))
    if not found:
        return []
    order = C.REF_QUALITY_ORDER
    best_tier = min(order.index(c.ref_quality) for c in found)
    return [c for c in found if order.index(c.ref_quality) == best_tier]


def build_opinions(
    projections: Sequence[GameProjection],
    decision_quotes: Sequence[Quote],
    *,
    books: frozenset[str] | None = None,
    markets: Sequence[str] = ("spreads", "totals"),
    method: str = C.DEFAULT_DEVIG_METHOD,
) -> tuple[list[Opinion], list[str]]:
    """One opinion per game and market: the best-EV side at a best-tier reference line.

    Passes are included (the log marks them); skipped games are reported, never silent.
    """
    books = books if books is not None else C.BETTABLE_BOOKS
    quotes = list(decision_quotes)
    skipped: list[str] = []
    if not quotes:
        return [], ["decision snapshot is empty"]
    snap_time = max(q.fetched_at for q in quotes)
    ids = match_events(projections, quotes)
    opinions: list[Opinion] = []
    for proj in projections:
        eid = ids.get(proj.model_game_id)
        if eid is None:
            skipped.append(f"{proj.model_game_id}: no matching event in the decision snapshot")
            continue
        if proj.as_of > snap_time:
            skipped.append(
                f"{proj.model_game_id}: projection made {proj.as_of.isoformat()} after the decision "
                f"snapshot {snap_time.isoformat()} (as-of rule)"
            )
            continue
        for market in markets:
            cands = _candidates(proj.home, proj.away,
                                lambda m, s, ln, _ref, p=proj: p_model(p, m, s, ln),
                                eid, market, quotes, books, method)
            if not cands:
                skipped.append(f"{proj.model_game_id} {market}: no two-way reference with a bettable price")
                continue
            c = max(cands, key=lambda x: (x.ev, effective_price(x.pick)))
            sd = proj.sd_margin if market in C.SPREAD_LIKE_MARKETS else proj.sd_total
            mu = proj.mu_home_margin if market in C.SPREAD_LIKE_MARKETS else proj.mu_total
            fee = fee_note(c.pick)
            opinions.append(Opinion(
                game_id=eid, market=market, side=c.side, line=c.line, book=c.pick.book,
                price_american=decimal_to_american(effective_price(c.pick)),  # net of fees
                p_model=c.p_model, p_market_novig=c.ref_p,
                model_version=proj.model_version, ref_quality=c.ref_quality, devig_method=method,
                timestamp=snap_time,
                notes=(f"model_game_id={proj.model_game_id}; source={proj.source}; mu={mu:.2f}; "
                       f"sd={sd:.2f}; ref_books={','.join(c.ref_books)}; bet_state={C.BET_STATE}"
                       + (f"; {fee}" if fee else "")),
            ))
    return opinions, skipped


BASELINE_VERSION = "market-baseline-v0"


def build_baseline_opinions(
    decision_quotes: Sequence[Quote],
    *,
    books: frozenset[str] | None = None,
    markets: Sequence[str] = ("spreads", "totals"),
    method: str = C.DEFAULT_DEVIG_METHOD,
) -> tuple[list[Opinion], list[str]]:
    """No-model baseline: p_model = p_market for every game in the snapshot.

    Each row is the best usable price at the decision snapshot, so its CLV measures
    what line shopping alone is worth against the close: the bar every model must
    clear. A gate pass here means a usable book beat the de-vigged reference by the
    gate (a stale copy). Tagged model_version=market-baseline-v0; never graded as a
    model and never counted toward promotion.
    """
    books = books if books is not None else C.BETTABLE_BOOKS
    quotes = list(decision_quotes)
    if not quotes:
        return [], ["decision snapshot is empty"]
    snap_time = max(q.fetched_at for q in quotes)
    games = {q.event_id: (q.home_team, q.away_team) for q in quotes}
    opinions: list[Opinion] = []
    skipped: list[str] = []
    for eid, (home, away) in sorted(games.items()):
        for market in markets:
            if not any(q.event_id == eid and q.market == market for q in quotes):
                continue  # market not in this snapshot (spreads-only day)
            cands = _candidates(home, away, lambda m, s, ln, ref_p: ref_p, eid, market, quotes, books, method)
            if not cands:
                skipped.append(f"{eid} {market}: no two-way reference with a bettable price")
                continue
            c = max(cands, key=lambda x: (x.ev, effective_price(x.pick)))
            fee = fee_note(c.pick)
            opinions.append(Opinion(
                game_id=eid, market=market, side=c.side, line=c.line, book=c.pick.book,
                price_american=decimal_to_american(effective_price(c.pick)),
                p_model=c.ref_p, p_market_novig=c.ref_p,
                model_version=BASELINE_VERSION, ref_quality=c.ref_quality, devig_method=method,
                timestamp=snap_time,
                notes=(f"baseline: p_model = p_market (no model), excluded from promotion; "
                       f"{away} @ {home}; ref_books={','.join(c.ref_books)}; bet_state={C.BET_STATE}"
                       + (f"; {fee}" if fee else "")),
            ))
    return opinions, skipped
