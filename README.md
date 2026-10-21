# EV Engine — NBA market layer

Paper-trading research code. Captures NBA odds snapshots on The Odds API free tier,
de-vigs two-way markets, logs every opinion (passes included), and grades each one on
closing line value against the de-vigged close. No real-money action.

## How it runs
GitHub Actions (`.github/workflows/runner.yml`) fires every 30 minutes. Each run loops
minute by minute for 55 minutes, so coverage is continuous from 08:45 to 02:00 ET:

| ET | What happens | Credits |
|---|---|---|
| first tick of the day | plan the day's snapshot jobs (free `/events` call) | 0 |
| 12:00 (T-60 for early tips) | decision snapshot: spreads + totals, all games | 2 |
| 12:00 | props archive: `player_points` on 2 seeded-random games | 2 |
| T-5 per tip time | close snapshot | 2 each |
| after each close | CLV on the opinions for those games | 0 |

On a tight month the planner drops the props archive first, then goes spreads-only
(decision and closes together), and flags upgrade triggers as run warnings.

**Data privacy.** This repo is public, but The Odds API terms forbid republishing raw odds.
All prices are stored AES-256 encrypted on the `data` branch (`state.tar.gz.enc`).
`STATUS.md` on that branch shows counts only.

## One-time setup
1. Add two repository secrets (Settings → Secrets and variables → Actions):
   - `ODDS_API_KEY`: your The Odds API key
   - `DATA_PASSPHRASE`: a long random passphrase (keep a copy; it decrypts your data)
2. Optional repository variable `NBA_EV_BET_STATE` = `ME` (default) or `MA`.
3. Actions tab → nba-ev-runner → Run workflow (or wait for the next :07/:37).

Scheduled workflows in public repos pause after 60 days without repository activity;
the runner's pushes to `data` keep it active, but check the Actions tab if runs stop.

## Read your data
```
DATA_PASSPHRASE=... scripts/decrypt_state.sh state/
```
Gives `state/market.db` (all snapshots) and `state/opinion_log.csv` (Excel-ready).

## Model input
Opinions come from two sources:
- `market-baseline-v0` (automatic): p_model = p_market at the best usable price. Measures what
  line shopping alone is worth against the close. Never graded as a model.
- Model payloads: commit `payloads/payload_<YYYY-MM-DD>.json` before 12:00 ET. Trim nba_sim
  output first so no prices land in the public repo:
  `python scripts/trim_payload.py out/payload_<date>.json payloads/`

## Local runner (alternative to Actions)
`scripts/setup_runner.sh` (Mac/Linux cron) or `scripts/setup_runner.ps1` (Windows). See
`README_PHASE0.md` for module details.

## Tests
```
pip install pytest && python -m pytest tests -q
```
