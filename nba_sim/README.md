# nba-sim 0.1.0

NBA betting simulation model: structural projection (8 modules) → Kalman fusion with the no-vig
market line → 10,000-run Monte Carlo → EV gate → quarter-Kelly sizing → JSON payload, dashboard
datasets and the season ledger.

## Run

```bash
pip install numpy scipy pandas pytest            # + requests, nba_api, pdfplumber for live feeds
python run_slate.py data/slate_2026-10-20.json   # writes out/payload_<date>.json and out/season_ledger.md
python export_dashboard.py data/slate_2026-10-20.json out/payload_2026-10-20.json   # out/dash/*.json
python -m pytest -q tests
```

## Layout

| Path | Module |
| --- | --- |
| `nba_sim/config.py` | Bankroll, EV gates, situational constants (from the Project Context doc), fusion and sim settings |
| `nba_sim/pace.py` | 1 Possessions, matchup pace, lineup-weighted pace with shrinkage |
| `nba_sim/four_factors.py` | 2 Four Factors matrix, opponent-adjusted ORtg, rating regression |
| `nba_sim/coaching.py` | 3 Shot-profile xPPS, scheme-vs-handler matrix, minutes caps |
| `nba_sim/situational.py` | 4 Fatigue vector, altitude, 3-in-4 / 4-in-5, matinee (weekend-only flag) |
| `nba_sim/injuries.py` | 5 EPM replacement value, Questionable expectation, position-synergy penalty |
| `nba_sim/market.py` | 6 No-vig consensus, RLM, handle vs tickets, steam, key numbers |
| `nba_sim/kalman.py` | Team-strength Kalman filter and model/market fusion |
| `nba_sim/montecarlo.py` | 7 Bivariate-normal integer scores, overtime, OT-calibrated totals |
| `nba_sim/pricing.py` | 8 Odds math, devig (multiplicative/power), EV, push-adjusted fractional Kelly |
| `nba_sim/garbage.py` | Garbage-time filter (Q4, leader win prob > 98%) |
| `nba_sim/backtest.py` | Layer 2: log loss, Brier skill, ECE, CLV beat rate, ROI, go-live gates, walk-forward splits |
| `nba_sim/pipeline.py` | Adapters: The Odds API v4 (credit budget), nba_api, injury PDFs, splits CSV |
| `nba_sim/ledger.py` | Grading, P/L, ROI, CLV, Project ledger template rendering |
| `nba_sim/engine.py` | Slate engine and payload assembly |

## Inputs to replace before relying on totals

Five of six team paces, PHI's split ratings, roster deltas and player EPM values in
`data/slate_2026-10-20.json` are marked `approximate` / `analyst estimate` in each team's
`provenance`. The structural totals run 1–5.5 points under the market on all three games; treat that
as a calibration fault until sourced paces are loaded.

## Live data

Set `ODDS_API_KEY` (and optionally `ODDS_API_MONTHLY_CAP`). `OddsAPIClient.to_book_rows()` returns the
same `books` shape the snapshot file uses, so the engine runs unchanged on live pulls.
