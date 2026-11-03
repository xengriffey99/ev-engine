"""Directional-drift and single-outcome-saturation checks (added 2026-11-03, reporting only).

Why: the 11/3 direction audit found the prop card all Unders (17 straight through 11/1). The cause is a model mean
that sits about 0.1-0.2 units below the market-implied mean, amplified by P(side) / EV selection on right-skewed
counts. These checks make the drift visible every night. They FLAG and REPORT; they never add or force a pick (NN8).

Pure functions, no I/O. Verdicts: OK | LOW_N | DRIFT | SATURATED (and ASYMMETRIC_STAGE per funnel stage).
Counts are modelled as NB(mean, var = vm * mean); at a model mean ON the line, P(Under) is 52.7-56.1% at vm 1.3
for lines 2.5-12.5, so the null for a model-on-the-line pool is not 50%. Pass the pool share as the null when
testing a selection stage.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isnan, sqrt
from typing import Iterable, Mapping, Sequence

from scipy.stats import binomtest, nbinom, norm


# ---------- 1. Under share vs a null ----------------------------------------------------------------------------
@dataclass(frozen=True)
class DriftResult:
    n: int
    k_under: int
    share: float
    null: float
    p_value: float
    verdict: str


def under_share_test(sides: Sequence[str], null_share: float = 0.5, alpha: float = 0.01,
                     min_n: int = 10, saturation: float = 0.90) -> DriftResult:
    """Two-sided exact binomial test of the Under share against a null, p = 2 Pr[Bin(n, p0) >= k].

    LOW_N is a verdict, never an OK. SATURATED means one side holds >= `saturation` of >= `min_n` picks.
    """
    bad = [s for s in sides if s not in ("Over", "Under")]
    if bad:
        raise ValueError(f"unknown side values: {sorted(set(bad))}")
    n = len(sides)
    k = sum(1 for s in sides if s == "Under")
    share = k / n if n else float("nan")
    if n < min_n:
        return DriftResult(n, k, share, null_share, float("nan"), "LOW_N")
    p = float(binomtest(k, n, null_share, alternative="two-sided").pvalue)
    if share >= saturation or (1 - share) >= saturation:
        return DriftResult(n, k, share, null_share, p, "SATURATED")
    return DriftResult(n, k, share, null_share, p, "DRIFT" if p < alpha else "OK")


def clustered_share(sides: Sequence[str], clusters: Sequence[str]) -> tuple[float, float]:
    """Under share with a cluster-robust SE (clusters = game ids; same-game rows are one position, NN7)."""
    if len(sides) != len(clusters):
        raise ValueError("sides and clusters must align")
    agg: dict[str, list[int]] = {}
    for s, c in zip(sides, clusters):
        a = agg.setdefault(c, [0, 0])
        a[0] += 1 if s == "Under" else 0
        a[1] += 1
    g = len(agg)
    n = sum(v[1] for v in agg.values())
    k = sum(v[0] for v in agg.values())
    p = k / n
    if g < 2:
        return p, float("nan")
    var = sum((v[0] - p * v[1]) ** 2 for v in agg.values()) * g / (g - 1) / n ** 2
    return p, sqrt(var)


# ---------- 2. Funnel symmetry ----------------------------------------------------------------------------------
def funnel_report(stages: Mapping[str, Sequence[str]], jump_pts: float = 0.10, alpha: float = 0.05) -> list[dict]:
    """Under share per stage; each stage is z-tested against the FIRST stage (the pool).

    A stage is ASYMMETRIC_STAGE when its Under share exceeds the previous stage's by >= jump_pts and differs from
    the pool at `alpha`. Stages must be passed in funnel order.
    """
    names = list(stages)
    pool = stages[names[0]]
    n0, k0 = len(pool), sum(1 for s in pool if s == "Under")
    out: list[dict] = []
    prev = None
    for nm in names:
        s = stages[nm]
        n, k = len(s), sum(1 for x in s if x == "Under")
        share = k / n if n else float("nan")
        z = pval = float("nan")
        if n and n0:
            pp = (k + k0) / (n + n0)
            if 0 < pp < 1:
                se = sqrt(pp * (1 - pp) * (1 / n + 1 / n0))
                z = (share - k0 / n0) / se
                pval = 2 * (1 - norm.cdf(abs(z)))
        flag = ""
        if prev is not None and n >= 5 and share - prev >= jump_pts and not isnan(pval) and pval < alpha:
            flag = "ASYMMETRIC_STAGE"
        if n < 5:
            flag = flag or "LOW_N"
        out.append(dict(stage=nm, n=n, share=share, delta_prev=None if prev is None else share - prev,
                        z_vs_pool=z, p_vs_pool=pval, flag=flag))
        prev = share
    return out


# ---------- 3. Candidate-pool symmetry --------------------------------------------------------------------------
def missing_side_keys(candidates: Iterable[Mapping]) -> list[tuple]:
    """Keys (game, player, market, line) evaluated on only one side. A symmetric pool prices both sides first."""
    seen: dict[tuple, set] = {}
    for c in candidates:
        key = (c["game"], c.get("player"), c["market"], c["line"])
        seen.setdefault(key, set()).add(c["side"])
    return sorted((k for k, v in seen.items() if v != {"Over", "Under"}), key=str)


# ---------- 4. Baselines ----------------------------------------------------------------------------------------
def baseline_gap(wins_on_disagreements: int, n_disagreements: int, z_crit: float = 1.96) -> dict:
    """Paired sign test of the model against a one-sided baseline (always-Under or always-Over).

    Only rows where the model's side differs from the baseline's can separate them. There the model wins W and the
    baseline wins m - W, so the win-count difference is 2W - m with SD sqrt(m) under "model adds nothing":
    z = (2W - m) / sqrt(m). 10/30: model Overs went 40-37 (m = 77), diff +3, z = 0.34.
    """
    m = n_disagreements
    if m == 0:
        return dict(diff=0, z=float("nan"), verdict="LOW_N")
    diff = 2 * wins_on_disagreements - m
    z = diff / sqrt(m)
    return dict(diff=diff, z=z, verdict="DISTINGUISHABLE" if abs(z) >= z_crit else "NOT_DISTINGUISHABLE")


# ---------- 5. Mean offset vs the market-implied mean -----------------------------------------------------------
def mean_offset_report(model_minus_market: Sequence[float], t_crit: float = 3.0, min_n: int = 30) -> dict:
    """Average of (model mean - market-implied mean). A persistent negative value is the structural Under lean."""
    n = len(model_minus_market)
    if n < min_n:
        return dict(n=n, mean=float("nan"), t=float("nan"), verdict="LOW_N")
    m = sum(model_minus_market) / n
    var = sum((x - m) ** 2 for x in model_minus_market) / (n - 1)
    t = m / sqrt(var / n) if var > 0 else float("inf")
    return dict(n=n, mean=m, t=t, verdict="DRIFT" if abs(t) >= t_crit else "OK")


# ---------- 6. Skew null ----------------------------------------------------------------------------------------
def p_under_mean_on_line(line: float, vm: float = 1.3) -> float:
    """P(Under) for NB(mean = line, var = vm * mean), no push, .5 lines."""
    r = line / (vm - 1.0)
    return 1.0 - float(nbinom.sf(int(line // 1), r, r / (r + line)))


# ---------- 7. Nightly report block (markdown, reporting only) --------------------------------------------------
def _pct(x: float) -> str:
    return "n/a" if x != x else f"{100 * x:.0f}%"


def direction_block_md(title: str, pool_sides: Sequence[str], *, selected_sides: Sequence[str] | None = None,
                       stages: Mapping[str, Sequence[str]] | None = None, pool_clusters: Sequence[str] | None = None,
                       baseline: tuple[int, int] | None = None,
                       mean_gaps: Sequence[float] | None = None) -> str:
    """Markdown 'direction block' for a nightly grade or card. Every check returns a verdict string.

    pool_sides: model side on every priced row (the pool). selected_sides: the published rows, tested against the
    POOL share (not 50%). stages: ordered funnel {stage: sides}. baseline: (model wins, rows) on the rows where the
    model side differs from always-Under. mean_gaps: per-row (model mean - market-implied mean).
    """
    pool = under_share_test(pool_sides, null_share=0.5, min_n=1)
    rows = [("Pool Under share", f"{pool.k_under}/{pool.n} = {_pct(pool.share)}", "reference")]
    if pool_clusters is not None:
        p, se = clustered_share(pool_sides, pool_clusters)
        rows.append(("Pool share, game-clustered", f"{_pct(p)} +/- {100 * se:.1f} pts", "reference"))
    if selected_sides is not None:
        r = under_share_test(selected_sides, null_share=pool.share)
        pv = "n/a (n too small)" if r.p_value != r.p_value else f"{r.p_value:.3g}"
        rows.append(("Selected Under share vs pool null",
                     f"{r.k_under}/{r.n} = {_pct(r.share)} (null {_pct(r.null)}, p = {pv})", r.verdict))
    if baseline is not None:
        b = baseline_gap(*baseline)
        rows.append(("Model vs always-Under (paired)", f"diff {b['diff']:+d} wins on {baseline[1]} rows, z = {b['z']:.2f}",
                     b["verdict"]))
    if mean_gaps is not None:
        g = mean_offset_report(list(mean_gaps))
        rows.append(("Mean offset (model - market mean)", f"{g['mean']:+.2f} units, t = {g['t']:.1f}, n = {g['n']}",
                     g["verdict"]))
    lines = [f"### Direction block: {title}", "", "| Check | Value | Verdict |", "|---|---|---|"]
    lines += [f"| {a} | {b} | {c} |" for a, b, c in rows]
    if stages:
        lines += ["", "| Funnel stage | n | Under share | vs previous | p vs pool | Flag |", "|---|---|---|---|---|---|"]
        for s in funnel_report(stages):
            d = "-" if s["delta_prev"] is None else f"{100 * s['delta_prev']:+.0f} pts"
            pv = "-" if s["p_vs_pool"] != s["p_vs_pool"] else f"{s['p_vs_pool']:.3g}"
            lines.append(f"| {s['stage']} | {s['n']} | {_pct(s['share'])} | {d} | {pv} | {s['flag'] or 'ok'} |")
    return "\n".join(lines) + "\n"
