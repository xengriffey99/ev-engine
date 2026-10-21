"""Season ledger: grade tickets, compute P/L, ROI and CLV beat rate, render the Project template."""
from __future__ import annotations

from datetime import date

from .pricing import american_to_decimal

MARKET_LABEL = {"spread": "Spreads", "total": "Totals (Over/Under)", "moneyline": "Moneylines", "prop": "Player Props"}


def grade(entry: dict, final_home: int | None, final_away: int | None) -> dict:
    if final_home is None or final_away is None:
        return {**entry, "status": entry.get("status", "pending")}
    margin, total = final_home - final_away, final_home + final_away
    m, s, line = entry["market"], entry["side"], entry.get("line")
    if m == "spread":
        v = (margin + line) if s == "home" else (-margin + line)
    elif m == "total":
        v = (total - line) if s == "over" else (line - total)
    else:
        v = margin if s == "home" else -margin
    status = "win" if v > 0 else "loss" if v < 0 else "push"
    return {**entry, "status": status}


def pl_units(e: dict) -> float:
    if e["status"] == "win":
        return round(e["stake"] * (american_to_decimal(e["price"]) - 1), 2)
    if e["status"] == "loss":
        return -e["stake"]
    return 0.0


def clv_beat(e: dict) -> bool | None:
    if e.get("close_line") is None:
        return None
    if e["market"] in ("spread",):
        return e["line"] > e["close_line"] if e["line"] != e["close_line"] else (
            american_to_decimal(e["price"]) > american_to_decimal(e["close_price"]))
    if e["market"] == "total":
        better = e["line"] > e["close_line"] if e["side"] == "under" else e["line"] < e["close_line"]
        return better if e["line"] != e["close_line"] else american_to_decimal(e["price"]) > american_to_decimal(e["close_price"])
    return american_to_decimal(e["price"]) > american_to_decimal(e["close_price"])


def summarize(entries: list[dict], starting: float = 10_000.0, unit_pct: float = 0.01) -> dict:
    settled = [e for e in entries if e["status"] in ("win", "loss", "push")]
    staked = sum(e["stake"] for e in settled if e["status"] != "push")
    pl = sum(pl_units(e) for e in settled)
    clv = [clv_beat(e) for e in entries if e.get("stake", 0) > 0]
    clv_known = [c for c in clv if c is not None]
    rec = {}
    for mk in MARKET_LABEL:
        es = [e for e in settled if e["market"] == mk]
        rec[mk] = {"w": sum(e["status"] == "win" for e in es), "l": sum(e["status"] == "loss" for e in es),
                   "p": sum(e["status"] == "push" for e in es), "pl": round(sum(pl_units(e) for e in es), 2)}
    bankroll = starting * (1 + pl * unit_pct)  # flat-unit approximation on starting unit size
    return {
        "starting": starting, "current": bankroll, "staked": staked, "pl": pl,
        "roi": (100 * pl / staked) if staked else 0.0,
        "clv_rate": (100 * sum(clv_known) / len(clv_known)) if clv_known else None,
        "pending_units": sum(e["stake"] for e in entries if e["status"] == "pending"),
        "by_market": rec,
    }


def render(entries: list[dict], passes: list[dict], lessons: list[str], season: str = "2026–27") -> str:
    s = summarize(entries)
    clv = f"{s['clv_rate']:.1f}%" if s["clv_rate"] is not None else "--%"
    lines = [
        f"# Season {season} Bankroll & Performance Ledger", "",
        "### Season Summary",
        f"- Starting Bankroll: ${s['starting']:,.2f}",
        f"- Current Bankroll: ${s['current']:,.2f}",
        f"- Total Units Staked: {s['staked']:.2f}u",
        f"- Net Profit/Loss: {s['pl']:+.2f}u",
        f"- Overall ROI%: {s['roi']:.1f}%",
        f"- Closing Line Value (CLV) Beat Rate: {clv}",
        f"- Open Exposure (pending): {s['pending_units']:.2f}u", "",
        "### Track Record by Market",
    ]
    for mk, label in MARKET_LABEL.items():
        r = s["by_market"][mk]
        lines.append(f"- {label}: {r['w']}-{r['l']}-{r['p']} ({r['pl']:+.2f}u)")
    lines += ["", "### Bet Log Table",
              "| Date | Game | Selection | Odds | Stake (u) | Result | P/L (u) | Closing Line | CLV Beat? |",
              "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |"]
    rows = []
    for p in passes:
        rows.append((p["date"], f"| {p['date_label']} | {p['game']} | Pass | N/A | 0.0u | N/A | 0.0u | N/A | N/A |"))
    for e in entries:
        res = e["status"].capitalize()
        plv = "--" if e["status"] == "pending" else f"{pl_units(e):+.2f}u"
        close = "--" if e.get("close_line") is None else f"{e['close_line']:+g} ({e['close_price']:+d})"
        c = clv_beat(e)
        cb = "--" if c is None else ("Yes" if c else "No")
        flag = f" {e['flag']}" if e.get("flag") else ""
        rows.append((e["date"], f"| {e['date_label']} | {e['game']} | {e['selection']}{flag} | {e['price']:+d} | {e['stake']:.1f}u | {res} | {plv} | {close} | {cb} |"))
    lines += [r for _, r in sorted(rows, key=lambda x: x[0])]
    lines += ["", "### Key Model Lessons & Adjustments"] + [f"- {l}" for l in lessons]
    return "\n".join(lines) + "\n"
