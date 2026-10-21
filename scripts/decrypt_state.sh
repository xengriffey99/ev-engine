#!/usr/bin/env bash
# Pull and decrypt the runner state on your own machine.
#   DATA_PASSPHRASE=... scripts/decrypt_state.sh [out_dir]
# Writes out_dir/market.db and out_dir/opinion_log.csv (default: ./state).
set -euo pipefail
: "${DATA_PASSPHRASE:?set DATA_PASSPHRASE (the value stored in the repo Actions secrets)}"
OUT="${1:-state}"
REPO="${NBA_EV_REPO:-https://github.com/xengriffey99/ev-engine.git}"
tmp="$(mktemp -d)"
git clone -q --depth 1 --branch data "$REPO" "$tmp"
mkdir -p "$OUT"
openssl enc -d -aes-256-cbc -pbkdf2 -iter 200000 -md sha256 -pass env:DATA_PASSPHRASE \
  -in "$tmp/state.tar.gz.enc" | tar -xz -C "$OUT"
rm -rf "$tmp"
echo "Decrypted to $OUT/ (market.db, opinion_log.csv)"
