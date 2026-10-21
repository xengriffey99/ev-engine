#!/usr/bin/env bash
# Encrypted runner state on the `data` branch (GitHub Actions).
#   scripts/ci_state.sh restore   # data branch -> data/market.db (fresh DB if none)
#   scripts/ci_state.sh save      # data/ -> AES-256 ciphertext + STATUS.md on the data branch
#
# The Odds API terms forbid publishing raw odds as downloadable files, and this repo
# is public, so prices only ever leave the runner encrypted. STATUS.md holds counts only.
# The data branch is one force-pushed commit, so the repo does not grow with history.
set -euo pipefail

: "${DATA_PASSPHRASE:?DATA_PASSPHRASE secret is not set; refusing to read or write state}"
REPO_URL="https://x-access-token:${GITHUB_TOKEN:?}@github.com/${GITHUB_REPOSITORY:?}.git"
ENC=(-aes-256-cbc -pbkdf2 -iter 200000 -md sha256)
mkdir -p data

case "${1:-}" in
  restore)
    tmp="$(mktemp -d)"
    if git clone -q --depth 1 --branch data "$REPO_URL" "$tmp" 2>/dev/null && [[ -f "$tmp/state.tar.gz.enc" ]]; then
      openssl enc -d "${ENC[@]}" -pass env:DATA_PASSPHRASE -in "$tmp/state.tar.gz.enc" | tar -xz -C data
      echo "state restored: $(du -h data/market.db | cut -f1)"
    else
      echo "no saved state yet; starting a fresh database"
    fi
    rm -rf "$tmp"
    ;;
  save)
    [[ -f data/market.db ]] || { echo "nothing to save"; exit 0; }
    PYTHONPATH=src python -m market.run_snapshots --db data/market.db export --out data/opinion_log.csv >/dev/null
    tmp="$(mktemp -d)"
    tar -cz -C data market.db opinion_log.csv \
      | openssl enc "${ENC[@]}" -salt -pass env:DATA_PASSPHRASE -out "$tmp/state.tar.gz.enc"
    PYTHONPATH=src python -m market.run_snapshots --db data/market.db summary --out "$tmp/STATUS.md" >/dev/null
    cat > "$tmp/README.md" <<'EOF'
Runner state for the NBA EV Engine. `state.tar.gz.enc` is AES-256 encrypted (prices from
The Odds API may not be republished). Decrypt locally with scripts/decrypt_state.sh.
EOF
    (
      cd "$tmp"
      git init -q -b data
      git add state.tar.gz.enc STATUS.md README.md
      git -c user.name="nba-ev-runner" -c user.email="nba-ev-runner@users.noreply.github.com" \
        commit -q -m "runner state $(date -u +%FT%TZ)"
      for attempt in 1 2 3; do
        git push -q -f "$REPO_URL" data:data && break
        sleep $((attempt * 5))
      done
    )
    rm -rf "$tmp"
    echo "state saved"
    ;;
  *)
    echo "usage: $0 restore|save" >&2; exit 1 ;;
esac
