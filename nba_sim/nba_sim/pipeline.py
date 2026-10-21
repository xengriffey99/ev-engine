"""Data ingestion adapters. Network calls are isolated here so the engine runs on snapshots.

- OddsAPIClient: The Odds API v4 (free tier). credits per call = markets x regions.
- NBAStatsClient: play-by-play, advanced box and lineups via the `nba_api` package.
- InjuryReportClient: official NBA injury-report PDFs.
- SplitsLoader: public ticket / handle splits from a vendor CSV or manual entry.
"""
from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

ODDS_BASE = "https://api.the-odds-api.com/v4"
INJURY_PDF = "https://ak-static.cms.nba.com/referee/injury/Injury-Report_{date}_{hhmm}{ampm}.pdf"


@dataclass
class CreditBudget:
    monthly_cap: int
    used: int = 0

    def can_spend(self, credits: int, reserve_for_closing: int = 0) -> bool:
        return self.used + credits + reserve_for_closing <= self.monthly_cap

    def spend(self, credits: int):
        self.used += credits


class OddsAPIClient:
    def __init__(self, api_key: str | None = None, budget: CreditBudget | None = None):
        self.key = api_key or os.environ.get("ODDS_API_KEY")
        self.budget = budget or CreditBudget(monthly_cap=int(os.environ.get("ODDS_API_MONTHLY_CAP", "500")))

    def nba_odds(self, markets=("h2h", "spreads", "totals"), regions=("us",), closing: bool = False) -> list:
        import requests  # imported lazily so the engine runs without network deps
        cost = len(markets) * len(regions)
        reserve = 0 if closing else 3 * 3  # keep credits for closing pulls
        if not self.budget.can_spend(cost, reserve):
            raise RuntimeError("Odds API credit budget exhausted for non-closing pull")
        r = requests.get(f"{ODDS_BASE}/sports/basketball_nba/odds", timeout=20, params={
            "apiKey": self.key, "regions": ",".join(regions), "markets": ",".join(markets),
            "oddsFormat": "american", "dateFormat": "iso"})
        r.raise_for_status()
        self.budget.spend(int(r.headers.get("x-requests-last", cost)))
        return r.json()

    @staticmethod
    def to_book_rows(events: list) -> dict:
        """Normalize to {game_key: {book: {home_spread, home_spread_price, ... }}}."""
        out = {}
        for ev in events:
            key = f"{ev['away_team']}@{ev['home_team']}"
            books = out.setdefault(key, {})
            for bk in ev.get("bookmakers", []):
                row = books.setdefault(bk["key"], {})
                for mk in bk["markets"]:
                    oc = {o["name"]: o for o in mk["outcomes"]}
                    if mk["key"] == "spreads":
                        row["home_spread"] = oc[ev["home_team"]]["point"]
                        row["home_spread_price"] = oc[ev["home_team"]]["price"]
                        row["away_spread_price"] = oc[ev["away_team"]]["price"]
                    elif mk["key"] == "totals":
                        row["total"] = oc["Over"]["point"]
                        row["over_price"], row["under_price"] = oc["Over"]["price"], oc["Under"]["price"]
                    elif mk["key"] == "h2h":
                        row["home_ml"], row["away_ml"] = oc[ev["home_team"]]["price"], oc[ev["away_team"]]["price"]
        return out


class NBAStatsClient:
    """Thin wrapper over nba_api endpoints (pip install nba_api)."""

    def play_by_play(self, game_id: str):
        from nba_api.stats.endpoints import playbyplayv3
        return playbyplayv3.PlayByPlayV3(game_id=game_id).get_data_frames()[0]

    def advanced_box(self, game_id: str):
        from nba_api.stats.endpoints import boxscoreadvancedv3
        return boxscoreadvancedv3.BoxScoreAdvancedV3(game_id=game_id).get_data_frames()

    def lineups(self, season: str = "2026-27", group_quantity: int = 5):
        from nba_api.stats.endpoints import leaguedashlineups
        return leaguedashlineups.LeagueDashLineups(season=season, group_quantity=group_quantity,
                                                   measure_type_detailed_defense="Advanced").get_data_frames()[0]


class InjuryReportClient:
    @staticmethod
    def url(ts: datetime) -> str:
        hhmm = ts.strftime("%I_%M")
        return INJURY_PDF.format(date=ts.strftime("%Y-%m-%d"), hhmm=hhmm, ampm=ts.strftime("%p"))

    def fetch_and_parse(self, ts: datetime) -> list[dict]:
        import io
        import pdfplumber
        import requests
        r = requests.get(self.url(ts), timeout=20)
        r.raise_for_status()
        rows = []
        with pdfplumber.open(io.BytesIO(r.content)) as pdf:
            for page in pdf.pages:
                for table in page.extract_tables():
                    rows.extend(table)
        return rows  # downstream: map to {'name','team','status','reason'}


class SplitsLoader:
    FIELDS = ("game", "market", "side", "ticket_pct", "money_pct", "line_at_snapshot", "ts")

    def load_csv(self, path: str | Path) -> list[dict]:
        with open(path, newline="") as f:
            return [dict(r) for r in csv.DictReader(f)]


def load_snapshot(path: str | Path) -> dict:
    with open(path) as f:
        return json.load(f)
