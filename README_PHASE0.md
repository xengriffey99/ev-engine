# NBA EV Engine v2 — Phase 0 (market layer) v1.1

Close capture, de-vig, direction-aware price selection, CLV, the opinion log, and the
model-to-log wiring. Spec: NBA_CONTEXT.md §4–§6, §9 and the §14 amendments
(approved 2026-10-21). Paper trading only.

## Drop-in
Copy `src/market/`, `scripts/`, `tests/market/` and `tests/conftest.py` into `nba-forecast/`
(or run this folder as-is). Standard library only, Python 3.11+; `pytest` for the tests.
If the repo imports via a `src.` prefix, change `import market` to `import src.market`
and drop `tests/conftest.py`.

```
pytest tests/market -q          # 152 tests
```

## Set up the runner (one command)
Mac/Linux:
```
scripts/setup_runner.sh              # betting from Maine (default)
scripts/setup_runner.sh --state MA   # betting from Massachusetts
scripts/setup_runner.sh --dry-run    # preview, change nothing
```
Windows (untested; send the output if it fails):
```
powershell -ExecutionPolicy Bypass -File scripts\setup_runner.ps1 -State ME
```
Setup runs the tests, asks for the Odds API key with hidden input, stores it outside the repo
(mode 600 on Mac/Linux, encrypted to your account on Windows), installs the schedule below,
and checks the key with the free /events call (0 credits).

| ET time | Command | Credits |
|---|---|---|
| 09:00 daily | `plan` (free /events call, stores the day's snapshot jobs) | 0 |
| every 5 min | `run-due` (12:00 decision, T-5 closes, props archive) | ~14/day |
| 12:15 daily | `opinions-today` (reads `data/payload_<date>.json` if the model wrote one) | 0 |
| 02:30 daily | `close-out` (closes and CLV on open opinions) | 0 |
| 02:45 daily | `export` → `data/opinion_log.csv` | 0 |

The machine must be awake 09:00 to ~01:00 ET. A job not run before its deadline is marked
`missed` (a late close is not a close). Logs: `data/logs/runner_<YYYY-MM>.log`.
By hand: `scripts/nba_ev.sh status`.

## Model wiring
`model_adapter.py` turns any model into opinion rows through `GameProjection`
(home-margin mean and spread, total mean and spread, `as_of` time).
- **nba_sim 0.1.0** (`payload_<date>.json`): wired. Uses the STRUCTURAL projection; the
  payload's posterior already fuses the market, so blending it again would count the market twice.
- **nba-forecast v1 Kalman**: build `GameProjection` from its margin/total output once the repo is attached.

Per game and market it evaluates both sides at the best-tier reference line (Pinnacle when
available), takes the best price at a book you can legally use (`NBA_EV_BET_STATE`), and logs the
best-EV side, passes included. A projection made after the decision snapshot is refused (as-of rule).
The model must write its payload before 12:00 ET.

## Venues (checked 2026-10-21)
| State | Bettable books priced on the free tier | Paid tier adds |
|---|---|---|
| ME (default) | DraftKings, Kalshi, Novig, ProphetX | Caesars |
| MA | DraftKings, FanDuel, BetMGM, theScore Bet, Bally Bet, Novig, ProphetX | Caesars, Fanatics |
Kalshi sports contracts are blocked in MA by court order. Sportsbooks geofence on where you are,
not where you live.

## Modules
| Module | Job |
|---|---|
| `devig.py` | Multiplicative, power (default), Shin, additive. One-way markets raise `OneWayMarketError` |
| `selection.py` | `best_price()` for the exact side and line (never the other side); `reference_prob()` de-vigs each book's own two-way pair, tiers Pinnacle → exchanges → retail, counts DK/MGM/Caesars as one opinion, excludes exchanges from prop references |
| `clv.py` | CLV = p_close × decimal − 1. Exact line, or a labeled line shift for sides/totals; props with a moved line stay unpriced until Phase 4 |
| `opinion_log.py` | SQLite store + Excel-ready CSV. Enforces two-way only, close-scheduled only, blend, Tier-2 cap, EV, gates, pass reasons, 2u per-game cap |
| `model_adapter.py` | Model projections → opinion rows (see above) |
| `odds_api.py` | The Odds API v4 client: decimal odds, ≤10 books per call (1 region), credit meter, budget guard with a reserve that never blocks a close |
| `scheduler.py` | Free-tier Option A plan: 12:00 ET decision (T-60 for early tips), T-5 close per tip cluster, props archive on 2 seeded-random games |
| `feed_check.py` | Tests whether two books share a prop feed on our own snapshots |
| `run_snapshots.py` | CLI: `plan`, `run-due`, `opinions`, `close-out`, `export`, `status` |

## Opinion log columns
`opinion_id`, then §9 in spec order, then `model_version`, `ref_quality`, `devig_method`,
`close_ts_et`, `clv_method`, `pass_reason`.
- `p_close_novig` is the closing no-vig probability of the bet as placed (same side, same line).
- `ref_quality` is the worse of the decision and close references.
- `clv_method`: `exact_line` | `line_shift_normal_est` | `line_moved_unpriced` | `close_missing`.
- `stake_u` is 0 on every row until Phase 2 adds 1/16 Kelly sizing; `gate_pass` still marks the bets.

## Still placeholders
- `config.CLV_SHIFT_SIGMA` (12 pts margin, 18 pts total) for line-moved CLV: fit in Phase 1b.
- Staking, restriction filter, outsized-edge ceiling value: Phases 2 and 4.
