#!/usr/bin/env bash
# Wrapper the scheduler calls. Loads the key from a 600-permission file outside
# the repo, sets PYTHONPATH and the betting state, logs every run.
#   scripts/nba_ev.sh run-due | plan | close-out | export | status | opinions-today
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONF_DIR="${NBA_EV_CONF_DIR:-$HOME/.config/nba-ev}"
KEY_FILE="$CONF_DIR/odds_api_key"
STATE_FILE="$CONF_DIR/bet_state"
PYTHON="${NBA_EV_PYTHON:-python3}"

cd "$REPO"
mkdir -p data/logs
LOG="data/logs/runner_$(TZ=America/New_York date +%Y-%m).log"

if [[ ! -r "$KEY_FILE" ]]; then
  echo "$(date -u +%FT%TZ) ERROR: no key file at $KEY_FILE; run scripts/setup_runner.sh" | tee -a "$LOG" >&2
  exit 2
fi
ODDS_API_KEY="$(tr -d '[:space:]' < "$KEY_FILE")"
export ODDS_API_KEY
export NBA_EV_BET_STATE="$(cat "$STATE_FILE" 2>/dev/null || echo ME)"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"

cmd="${1:-status}"
shift || true
if [[ "$cmd" == "opinions-today" ]]; then
  payload="data/payload_$(TZ=America/New_York date +%F).json"
  if [[ ! -f "$payload" ]]; then
    echo "$(date -u +%FT%TZ) opinions-today: no model payload at $payload (run the model before 12:00 ET)" >> "$LOG"
    exit 0
  fi
  set -- opinions --payload "$payload"
else
  set -- "$cmd" "$@"
fi

run() {
  echo "=== $(date -u +%FT%TZ) $* (state=$NBA_EV_BET_STATE)"
  "$PYTHON" -m market.run_snapshots "$@"
}
if [[ -t 1 ]]; then
  run "$@" 2>&1 | tee -a "$LOG"   # by hand: show it and log it
else
  run "$@" >> "$LOG" 2>&1         # from the scheduler: log only
fi
