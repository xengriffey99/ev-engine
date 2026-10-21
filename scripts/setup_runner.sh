#!/usr/bin/env bash
# One-time setup of the Phase 0 snapshot runner on macOS or Linux.
#
#   scripts/setup_runner.sh                 # prompts for the key (hidden), installs the schedule
#   scripts/setup_runner.sh --state MA      # betting from Massachusetts instead of Maine
#   scripts/setup_runner.sh --dry-run       # show what would change, change nothing
#
# What it does:
#   1. Checks Python 3.11+ and runs the test suite.
#   2. Stores your Odds API key in ~/.config/nba-ev/odds_api_key (mode 600, outside the repo,
#      never in crontab or shell history).
#   3. Installs a marked block in your crontab (re-running replaces it, never duplicates).
#   4. Verifies the key with the free /events call (0 credits).
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONF_DIR="${NBA_EV_CONF_DIR:-$HOME/.config/nba-ev}"
PYTHON="${NBA_EV_PYTHON:-python3}"
CRONTAB_CMD="${NBA_EV_CRONTAB:-crontab}"   # overridable for testing
STATE="ME"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --state) STATE="$(echo "$2" | tr '[:lower:]' '[:upper:]')"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
    *) echo "Unknown option $1" >&2; exit 1 ;;
  esac
done
[[ "$STATE" == "ME" || "$STATE" == "MA" ]] || { echo "--state must be ME or MA" >&2; exit 1; }

say() { printf '%s\n' "$*"; }

# 1. Python and tests ---------------------------------------------------------------
"$PYTHON" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' \
  || { say "Python 3.11+ is required (found $("$PYTHON" -V 2>&1))."; exit 1; }
if "$PYTHON" -c 'import pytest' 2>/dev/null; then
  (cd "$REPO" && "$PYTHON" -m pytest tests/market -q -p no:cacheprovider) \
    || { say "Tests failed; not installing the schedule."; exit 1; }
else
  say "pytest not installed; skipping tests (pip install pytest to run them)."
fi

# Timezone: cron runs on the machine clock, and the schedule below is written in ET.
if [[ "$(date +%z)" != "$(TZ=America/New_York date +%z)" ]]; then
  say "WARNING: this machine is not on US Eastern time ($(date +%Z)). The schedule assumes ET."
fi

# 2. Key ---------------------------------------------------------------------------------
KEY_FILE="$CONF_DIR/odds_api_key"
if [[ $DRY_RUN -eq 1 ]]; then
  say "[dry-run] would store the key at $KEY_FILE (mode 600) and bet state $STATE"
elif [[ -s "$KEY_FILE" && -z "${NBA_EV_REPLACE_KEY:-}" ]]; then
  say "Key already stored at $KEY_FILE (set NBA_EV_REPLACE_KEY=1 to replace it)."
else
  mkdir -p "$CONF_DIR" && chmod 700 "$CONF_DIR"
  if [[ -n "${ODDS_API_KEY:-}" ]]; then
    key="$ODDS_API_KEY"
  else
    read -r -s -p "Paste your Odds API key (input hidden): " key; echo
  fi
  [[ -n "$key" ]] || { say "No key entered."; exit 1; }
  ( umask 077; printf '%s\n' "$key" > "$KEY_FILE" )
  unset key
  say "Key stored at $KEY_FILE (mode 600)."
fi
[[ $DRY_RUN -eq 1 ]] || { mkdir -p "$CONF_DIR"; echo "$STATE" > "$CONF_DIR/bet_state"; }
chmod +x "$REPO/scripts/nba_ev.sh" 2>/dev/null || true

# 3. Schedule (ET) ----------------------------------------------------------------------
W="$REPO/scripts/nba_ev.sh"
BLOCK=$(cat <<EOF
# >>> nba-ev runner >>>  (managed by scripts/setup_runner.sh; times are ET)
0 9 * * *     "$W" plan
*/5 * * * *   "$W" run-due
15 12 * * *   "$W" opinions-today
30 2 * * *    "$W" close-out
45 2 * * *    "$W" export
# <<< nba-ev runner <<<
EOF
)
current="$($CRONTAB_CMD -l 2>/dev/null || true)"
cleaned="$(printf '%s\n' "$current" | awk '/# >>> nba-ev runner >>>/{skip=1} !skip{print} /# <<< nba-ev runner <<</{skip=0}')"
new="$(printf '%s\n%s\n' "$cleaned" "$BLOCK" | sed '/^$/N;/^\n$/D')"
if [[ $DRY_RUN -eq 1 ]]; then
  say "[dry-run] crontab would contain:"; printf '%s\n' "$new"
  exit 0
fi
printf '%s\n' "$new" | $CRONTAB_CMD -
say "Schedule installed (crontab -l to view)."

# 4. Verify with the free /events call --------------------------------------------------
RUN_LOG="$REPO/data/logs/runner_$(TZ=America/New_York date +%Y-%m).log"
if "$W" plan; then
  tail -n 3 "$RUN_LOG"
  say "Key works. Today's snapshot plan is stored (0 credits used)."
else
  tail -n 3 "$RUN_LOG" 2>/dev/null || true
  say "The free /events check failed (see above). The schedule is installed; fix the key or"
  say "network and run: scripts/nba_ev.sh plan"
  exit 1
fi

cat <<'EOF'

Keep this machine awake from 09:00 to ~01:00 ET; cron does not run while it sleeps,
and a missed T-5 close counts as missed.
  macOS (plugged in):  sudo pmset -c sleep 0     and   sudo pmset repeat wakeorpoweron MTWRFSU 08:55:00
  Linux:               disable suspend, or use an always-on machine
Logs: data/logs/runner_<YYYY-MM>.log    Opinion log: data/opinion_log.csv (written nightly)
EOF
