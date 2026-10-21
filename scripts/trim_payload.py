"""Strip an nba_sim payload down to projections before committing it to the public repo.

  python scripts/trim_payload.py out/payload_2026-10-22.json payloads/

Keeps only what the opinion log needs (teams, structural margin/total, spreads, timestamps).
Drops markets, prices, books and ledger data: odds never go into the public repo in plaintext.
"""

import json
import sys
from pathlib import Path


def trim(payload: dict) -> dict:
    return {
        "model_version": payload["model_version"],
        "generated_at": payload["generated_at"],
        "slate_date": payload["slate_date"],
        "games": [
            {
                "game_id": g["game_id"],
                "away": g["away"],
                "home": g["home"],
                "projection": {"structural": {
                    "home_margin": g["projection"]["structural"]["home_margin"],
                    "total": g["projection"]["structural"]["total"],
                }},
                "simulation": {
                    "sd_margin": g["simulation"]["sd_margin"],
                    "sd_total": g["simulation"]["sd_total"],
                },
            }
            for g in payload["games"]
        ],
    }


if __name__ == "__main__":
    src, out_dir = Path(sys.argv[1]), Path(sys.argv[2] if len(sys.argv) > 2 else "payloads")
    data = trim(json.loads(src.read_text()))
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"payload_{data['slate_date']}.json"
    dest.write_text(json.dumps(data, indent=2) + "\n")
    print(dest)
