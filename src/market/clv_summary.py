"""CLV by side / lane / market from the decrypted opinion log (added 2026-11-03, reporting only).

Closes are captured by the runner and CLV is computed at T-5 (clv.py). This module only summarizes
`state/opinion_log.csv` (from scripts/decrypt_state.sh). It answers the audit's arbiter question: does CLV hold up
separately for Unders and Overs (spec lift criterion: CLV >= 0 for each side)?

CLV is a fraction, clv = p_close_novig * decimal_price - 1. Graded rows are those with a closing probability:
  exact_line            line unchanged, exact CLV
  line_shift_normal_est line moved, estimated by a normal shift (sides/totals only)
Excluded from the mean: close_missing, line_moved_unpriced, and rows not yet closed. Counts are always reported.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from math import sqrt
from pathlib import Path
from typing import Iterable, Sequence

from .clv import EXACT, MISSING, SHIFTED, UNPRICED

GRADED_METHODS = frozenset({EXACT, SHIFTED})


def load_rows(path: str | Path) -> list[dict]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def _num(x) -> float | None:
    try:
        return float(x) if x not in ("", None) else None
    except ValueError:
        return None


def cluster_mean_se(values: Sequence[float], clusters: Sequence[str]) -> tuple[float, float]:
    """Mean with a game-clustered SE (rows in one game are one position, NN7)."""
    n = len(values)
    if n == 0:
        return float("nan"), float("nan")
    m = sum(values) / n
    agg: dict[str, float] = defaultdict(float)
    for v, c in zip(values, clusters):
        agg[c] += v - m
    g = len(agg)
    if g < 2:
        return m, float("nan")
    return m, sqrt(sum(s * s for s in agg.values()) * g / (g - 1)) / n


def summarize(rows: Iterable[dict], by: Sequence[str]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        groups[tuple(r.get(k, "") for k in by)].append(r)
    out = []
    for key, rs in sorted(groups.items()):
        graded = [r for r in rs if r.get("clv_method") in GRADED_METHODS and _num(r.get("clv_pct")) is not None]
        vals = [_num(r["clv_pct"]) for r in graded]
        m, se = cluster_mean_se(vals, [r.get("game_id", "") for r in graded])
        out.append(dict(
            key=key, n_logged=len(rs),
            n_closed=sum(1 for r in rs if r.get("clv_method")),
            n_missing=sum(1 for r in rs if r.get("clv_method") in (MISSING, UNPRICED)),
            n_graded=len(graded), n_exact=sum(1 for r in graded if r["clv_method"] == EXACT),
            mean_clv=m, se=se, t=(m / se if se and se == se and se > 0 else float("nan")),
            share_pos=(sum(1 for v in vals if v > 0) / len(vals) if vals else float("nan"))))
    return out


def _f(x: float, scale: float = 1.0, fmt: str = "{:+.2f}") -> str:
    return "n/a" if x != x else fmt.format(x * scale)


def render_md(rows: Sequence[dict], by: Sequence[str], title: str) -> str:
    head = " / ".join(by)
    lines = [f"#### {title}", "",
             f"| {head} | logged | closed | missing | graded (exact) | mean CLV (pts) | SE (clustered) | t | share > 0 |",
             "|---|---|---|---|---|---|---|---|---|"]
    for g in summarize(rows, by):
        lines.append(f"| {' / '.join(g['key']) or 'all'} | {g['n_logged']} | {g['n_closed']} | {g['n_missing']} | "
                     f"{g['n_graded']} ({g['n_exact']}) | {_f(g['mean_clv'], 100)} | {_f(g['se'], 100, '{:.2f}')} | "
                     f"{_f(g['t'], 1, '{:+.1f}')} | {_f(g['share_pos'], 100, '{:.0f}%')} |")
    return "\n".join(lines) + "\n"


def report(path: str | Path, model_version_prefix: str = "") -> str:
    rows = load_rows(path)
    if model_version_prefix:
        rows = [r for r in rows if (r.get("model_version") or "").startswith(model_version_prefix)]
    parts = [render_md(rows, ("side",), "CLV by side"),
             render_md(rows, ("model_version", "side"), "CLV by lane and side"),
             render_md(rows, ("market", "side"), "CLV by market and side")]
    return "\n".join(parts)
