"""Layer 2: walk-forward backtesting and calibration.

Each record is one model opinion logged before tip (bet or not):
  {date, game, market: spread|total|moneyline, side, p_model, p_market_novig, outcome: 1/0/None(push),
   price_american, entry_line, close_line, close_price_american, bet: bool, stake_units}
"""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from .pricing import american_to_decimal, implied_prob


def log_loss(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def calibration_table(p: np.ndarray, y: np.ndarray, bins: int = 10) -> tuple[list[dict], float]:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    rows, ece = [], 0.0
    for b in range(bins):
        m = idx == b
        if not m.any():
            continue
        pred, obs = float(p[m].mean()), float(y[m].mean())
        rows.append({"bin": f"{edges[b]:.1f}-{edges[b+1]:.1f}", "n": int(m.sum()), "pred": round(pred, 3), "obs": round(obs, 3)})
        ece += m.mean() * abs(pred - obs)
    return rows, float(ece)


def clv_beat(rec: dict) -> bool | None:
    """True when our entry beats the close. Spreads/totals by points (then price), ML by no-vig prob."""
    if rec.get("close_line") is None and rec.get("close_price_american") is None:
        return None
    if rec["market"] == "moneyline":
        return implied_prob(rec["close_price_american"]) > implied_prob(rec["price_american"])
    sign = rec.get("line_sign", 1)  # +1 when a higher line is better for our side (dog spread, under)
    diff = (rec["entry_line"] - rec["close_line"]) * sign
    if diff != 0:
        return diff > 0
    return implied_prob(rec["close_price_american"]) > implied_prob(rec["price_american"])


def evaluate(records: list[dict]) -> dict:
    graded = [r for r in records if r.get("outcome") is not None]
    p = np.array([r["p_model"] for r in graded], dtype=float)
    pm = np.array([r["p_market_novig"] for r in graded], dtype=float)
    y = np.array([r["outcome"] for r in graded], dtype=float)
    out = {"n_opinions": len(records), "n_graded": len(graded)}
    if len(graded):
        out.update({
            "log_loss_model": log_loss(p, y), "log_loss_market": log_loss(pm, y),
            "brier_model": brier(p, y), "brier_market": brier(pm, y),
        })
        out["brier_skill_vs_market"] = 1 - out["brier_model"] / out["brier_market"]
        out["calibration"], out["ece"] = calibration_table(p, y)

    by_mkt = defaultdict(lambda: {"bets": 0, "w": 0, "l": 0, "p": 0, "staked": 0.0, "pl": 0.0})
    clv_hits, clv_n = 0, 0
    for r in records:
        if not r.get("bet"):
            continue
        m = by_mkt[r["market"]]
        m["bets"] += 1
        stake = r["stake_units"]
        if r.get("outcome") is None and r.get("result") == "push":
            m["p"] += 1
        elif r.get("outcome") == 1:
            m["w"] += 1; m["staked"] += stake; m["pl"] += stake * (american_to_decimal(r["price_american"]) - 1)
        elif r.get("outcome") == 0:
            m["l"] += 1; m["staked"] += stake; m["pl"] -= stake
        c = clv_beat(r)
        if c is not None:
            clv_n += 1; clv_hits += int(c)
    for m in by_mkt.values():
        m["roi_pct"] = 100 * m["pl"] / m["staked"] if m["staked"] else None
    out["by_market"] = dict(by_mkt)
    out["clv_beat_rate"] = clv_hits / clv_n if clv_n else None
    out["gates"] = acceptance_gates(out, clv_n)
    return out


def acceptance_gates(res: dict, clv_n: int) -> dict:
    """Go-live gates. A model that fails any gate stays in paper mode."""
    g = {
        "brier_skill_ge_0": res.get("brier_skill_vs_market", -1) >= 0,
        "ece_le_0.03": res.get("ece", 1) <= 0.03,
        "clv_beat_ge_52pct": (res.get("clv_beat_rate") or 0) >= 0.52,
        "clv_sample_ge_250": clv_n >= 250,
    }
    g["all_pass"] = all(g.values())
    return g


def walk_forward_splits(dates: list[str], train_days: int = 60, test_days: int = 7):
    """Yield (train_dates, test_dates) windows that never look ahead."""
    uniq = sorted(set(dates))
    i = train_days
    while i < len(uniq):
        yield uniq[i - train_days:i], uniq[i:i + test_days]
        i += test_days
