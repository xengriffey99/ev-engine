#!/usr/bin/env bash
# Minute loop for one GitHub Actions run. Each run covers MINUTES; runs start every
# 30 minutes and queue behind each other, so coverage is continuous while the
# schedule fires. Outside 08:45-02:00 ET the first tick exits at once.
# State is saved after every tick that changed something.
set -uo pipefail
MINUTES="${1:-55}"
EVENT="${2:-schedule}"
end=$((SECONDS + MINUTES * 60))
export PYTHONPATH=src

while (( SECONDS < end )); do
  out="$(python -m market.run_snapshots --db data/market.db tick --payload-dir payloads 2>&1)"
  code=$?
  if (( code == 10 )); then
    if [[ "$EVENT" == "schedule" ]]; then
      echo "$(date -u +%FT%TZ) outside the active window (08:45-02:00 ET); exiting"
      break
    fi
    sleep 300  # dispatched (chained) run: wait for the window instead of exiting
    continue
  fi
  if [[ -n "$out" ]]; then
    printf '%s\n' "$out" | sed "s/^/$(date -u +%H:%M:%SZ) /"
    while IFS= read -r line; do
      case "$line" in
        MISSED*close*|ERROR*close*) echo "::error::$line"; echo "$line" >> /tmp/close_alert ;;
        *"Upgrade trigger"*|*"Quota limits coverage"*|SKIPPED*|MISSED*|ERROR*) echo "::warning::$line" ;;
      esac
    done <<< "$out"
    scripts/ci_state.sh save || echo "::warning::state save failed; will retry next change"
  fi
  (( code != 0 )) && echo "::warning::tick exited with $code"
  sleep $(( 60 - 10#$(date +%S) ))
done
