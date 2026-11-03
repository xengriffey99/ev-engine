#!/usr/bin/env python3
"""CLV by side, lane and market from the decrypted opinion log. Reporting only.

  DATA_PASSPHRASE=... scripts/decrypt_state.sh state/
  python scripts/clv_by_side.py state/opinion_log.csv [--model-version-prefix sidequest]

Prices stay on your machine: this reads the local CSV and prints counts and CLV summaries.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from market.clv_summary import report  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", help="state/opinion_log.csv from scripts/decrypt_state.sh")
    ap.add_argument("--model-version-prefix", default="", help="e.g. sidequest to keep only the side-quest lanes")
    args = ap.parse_args()
    print(report(args.csv, args.model_version_prefix))


if __name__ == "__main__":
    main()
