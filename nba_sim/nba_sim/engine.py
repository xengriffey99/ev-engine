"""Slate engine: structural projection -> market fusion -> Monte Carlo -> EV / Kelly -> payload."""
from __future__ import annotations

import math
from datetime import datetime

from . import four_factors as ff
from . import coaching, injuries, market, montecarlo as mc, pace, situational
from .config import BANKROLL, FUSION, PRIORS, SIM
from .kalman import fuse_side, fuse_total
from .pricing import (american_to_decimal, ev_pct, ev_threshold, fair_american, kelly_units,
                      prob_to_american)

MODEL_VERSION = "nba-sim 0.1.0 (opening-night priors)"


def _team_ratings(t: dict, inj_net: float) -> dict:
    """Last-season team ratings are regressed toward the league (offseason carryover).
    A roster-aware prior (t["rating_prior"] == "epm": summed expected player EPM, already
    shrunk career projections) is used as-is, so it is not shrunk twice."""
    lg = PRIORS.league_ortg
    reg = (lambda r: r) if t.get("rating_prior") == "epm" else ff.regress_rating
    off = reg(t["ortg"]) + 0.5 * t["roster_delta"] + 0.5 * inj_net
    deff = reg(t["drtg"]) - 0.5 * t["roster_delta"] - 0.5 * inj_net
    return {"off": off, "def": deff, "net": off - deff, "pace": pace.regress_pace(t["pace"]), "lg": lg}


def structural_projection(g: dict, teams: dict) -> dict:
    h, a = g["home"], g["away"]
    inj_h = injuries.injury_impact(g["injuries"].get(h, []))
    inj_a = injuries.injury_impact(g["injuries"].get(a, []))
    rh, ra = _team_ratings(teams[h], inj_h["net_per100"]), _team_ratings(teams[a], inj_a["net_per100"])

    game_pace = pace.matchup_pace(rh["pace"], ra["pace"])
    game_pace = pace.lineup_weighted_pace([], game_pace)  # lineup feed not loaded on opening night

    ff_h, ffc_h = ff.factor_edge(None, None, None)          # factor feed not loaded -> neutral
    ff_a, ffc_a = ff.factor_edge(None, None, None)
    tac_h = coaching.shot_profile_adj(None, None) + coaching.scheme_adj(None, None)
    tac_a = coaching.shot_profile_adj(None, None) + coaching.scheme_adj(None, None)

    ortg_h = ff.expected_ortg(rh["off"], ra["def"], factor_adj=ff_h + tac_h)
    ortg_a = ff.expected_ortg(ra["off"], rh["def"], factor_adj=ff_a + tac_a)

    fv = g["fatigue"]
    sit = situational.situational_adjustments(
        h, situational.FatigueVector(**fv["away"]), situational.FatigueVector(**fv["home"]),
        datetime.fromisoformat(g["tip_local"]), game_pace)

    pts_h, pts_a = ortg_h * game_pace / 100, ortg_a * game_pace / 100
    margin = pts_h - pts_a + PRIORS.hca_points + sit["margin"]
    total = pts_h + pts_a + sit["total"]
    return {
        "home_margin": margin, "total": total, "pace": game_pace,
        "home_pts": (total + margin) / 2, "away_pts": (total - margin) / 2,
        "ortg": {h: ortg_h, a: ortg_a},
        "ratings": {h: rh, a: ra},
        "injuries": {h: inj_h, a: inj_a},
        "situational": sit,
        "modules_neutral": ["four_factors (feed not loaded)", "coaching/scheme (feed not loaded)",
                            "lineup pace (feed not loaded)"],
    }


def market_signals(g: dict, cons: dict) -> dict:
    o, s = g["open"], g["splits"]
    sig = {}
    # spread on home-line scale: home side benefits when the home line goes UP
    sp_side = s["spread"]["side"]
    rlm_sp = market.detect_rlm(o["home_spread"], cons["spread_display"], s["spread"]["tickets"], +1 if sp_side == "home" else -1)
    sig["spread"] = {"open": o["home_spread"], "current": cons["spread_display"], "rlm": rlm_sp,
                     "sharp_handle": market.sharp_handle(s["spread"]["money"], s["spread"]["tickets"]),
                     "public": s["spread"], "keys_crossed": market.key_numbers_crossed(o["home_spread"], cons["spread_display"])}
    tot_side = s["total"]["side"]
    rlm_t = market.detect_rlm(o["total"], cons["total_display"], s["total"]["tickets"], +1 if tot_side == "under" else -1)
    sig["total"] = {"open": o["total"], "current": cons["total_display"], "rlm": rlm_t,
                    "sharp_handle": market.sharp_handle(s["total"]["money"], s["total"]["tickets"]), "public": s["total"]}
    # moneyline: track away no-vig prob; away benefits when its price lengthens
    from .pricing import devig
    p_away_open = devig(o["away_ml"], o["home_ml"])[0]
    p_away_now = 1 - cons["home_win_prob_novig"]
    ml_side = s["moneyline"]["side"]
    # public-benefit direction: public side's price getting longer = its win prob falling
    move = p_away_now - p_away_open
    public_benefit = (move < 0) if ml_side == "away" else (move > 0)
    sig["moneyline"] = {"away_novig_open": round(p_away_open, 4), "away_novig_now": round(p_away_now, 4),
                        "rlm": {"flag": bool(public_benefit and s["moneyline"]["tickets"] >= 0.65),
                                "sub_threshold": bool(public_benefit and s["moneyline"]["tickets"] < 0.65),
                                "move_prob": round(move, 4), "public_pct": s["moneyline"]["tickets"]},
                        "sharp_handle": market.sharp_handle(s["moneyline"]["money"], s["moneyline"]["tickets"]),
                        "public": s["moneyline"]}
    alerts = []
    for mkt in ("spread", "total", "moneyline"):
        r = sig[mkt]["rlm"]
        pub = sig[mkt]["public"]
        if r["flag"]:
            alerts.append({"level": "RLM", "market": mkt, "text": f"Reverse line move: {pub['tickets']:.0%} of tickets on {pub['label']}, yet the number moved in their favour; money sits on the other side"})
        elif r["sub_threshold"]:
            alerts.append({"level": "WATCH", "market": mkt, "text": f"Number moved in favour of {pub['label']} with {pub['tickets']:.0%} of tickets there; sub-threshold (RLM needs 65%)"})
        if sig[mkt]["sharp_handle"]:
            alerts.append({"level": "SHARP", "market": mkt, "text": f"Money {pub['money']:.0%} vs tickets {pub['tickets']:.0%} on {pub['label']}"})
    if sig["spread"]["keys_crossed"]:
        flip = (o["home_spread"] < 0) != (cons["spread_display"] < 0)
        fav_open = g["home"] if o["home_spread"] < 0 else g["away"]
        fav_now = g["home"] if cons["spread_display"] < 0 else g["away"]
        txt = (f"{g['home']} line {o['home_spread']:+g} → {cons['spread_display']:+g}; key numbers crossed: "
               f"{', '.join(str(k) for k in sig['spread']['keys_crossed'])}")
        if flip:
            txt += f"; favourite flipped {fav_open} → {fav_now}"
        alerts.append({"level": "KEY", "market": "spread", "text": txt})
    sig["alerts"] = alerts
    return sig


def signal_shifts(g: dict, sig: dict) -> dict:
    """Posterior nudges from microstructure flags (uncalibrated, small, auditable)."""
    dm, dt, notes = 0.0, 0.0, []
    # direction of the listed split side: +1 raises the total / home margin
    t_dir = +1 if sig["total"]["public"]["side"] == "over" else -1
    s_dir = +1 if sig["spread"]["public"]["side"] == "home" else -1
    if sig["total"]["rlm"]["flag"]:            # money is opposite the public side
        dt -= t_dir * FUSION.rlm_shift_points
        notes.append(f"total {-t_dir * FUSION.rlm_shift_points:+.1f} (RLM vs {sig['total']['public']['label']})")
    if sig["total"]["sharp_handle"]:           # money is on the listed side
        dt += t_dir * FUSION.sharp_shift_points
        notes.append(f"total {t_dir * FUSION.sharp_shift_points:+.1f} (handle vs tickets)")
    if sig["spread"]["rlm"]["flag"]:
        dm -= s_dir * FUSION.rlm_shift_points
        notes.append(f"home margin {-s_dir * FUSION.rlm_shift_points:+.1f} (spread RLM)")
    if sig["spread"]["sharp_handle"]:
        dm += s_dir * FUSION.sharp_shift_points
        notes.append(f"home margin {s_dir * FUSION.sharp_shift_points:+.1f} (spread handle vs tickets)")
    return {"margin": dm, "total": dt, "notes": notes}


def _offers(g: dict) -> list[dict]:
    out = []
    for book, b in g["books"].items():
        if b.get("home_spread") is not None:
            out += [{"market": "spread", "side": "home", "line": b["home_spread"], "price": b["home_spread_price"], "book": book},
                    {"market": "spread", "side": "away", "line": -b["home_spread"], "price": b["away_spread_price"], "book": book}]
        if b.get("total") is not None:
            out += [{"market": "total", "side": "over", "line": b["total"], "price": b["over_price"], "book": book},
                    {"market": "total", "side": "under", "line": b["total"], "price": b["under_price"], "book": book}]
        if b.get("home_ml") is not None:
            out += [{"market": "moneyline", "side": "home", "line": None, "price": b["home_ml"], "book": book},
                    {"market": "moneyline", "side": "away", "line": None, "price": b["away_ml"], "book": book}]
    return out


def _probs(sim: dict, o: dict) -> tuple[float, float]:
    if o["market"] == "spread":
        hs = o["line"] if o["side"] == "home" else -o["line"]
        c = mc.cover_probs(sim, hs)
        return (c["home"], c["push"]) if o["side"] == "home" else (c["away"], c["push"])
    if o["market"] == "total":
        t = mc.total_probs(sim, o["line"])
        return t[o["side"]], t["push"]
    pw = float((sim["margin"] > 0).mean())
    return (pw, 0.0) if o["side"] == "home" else (1 - pw, 0.0)


def _trigger(sim: dict, o: dict, thr: float) -> dict:
    """Worst price at the current line, and worst line at -110, that still clear the EV gate."""
    pw, pp = _probs(sim, o)
    pl = 1 - pw - pp
    d_req = 1 + (thr + pl) / pw if pw > 0 else float("inf")
    price_req = None
    if d_req < 50:  # round toward the better price so the trigger always clears the gate
        exact = (d_req - 1) * 100 if d_req >= 2 else -100 / (d_req - 1)
        price_req = math.ceil(exact)
        if -100 < price_req < 100:
            price_req = 100
    line_req = None
    if o["market"] in ("spread", "total"):
        step = 0.5
        direction = +1 if (o["market"] == "spread" or o["side"] == "under") else -1
        for i in range(0, 41):
            cand = dict(o, line=o["line"] + direction * step * i, price=-110)
            w, p = _probs(sim, cand)
            if ev_pct(w, -110, p) >= thr:
                line_req = cand["line"]
                break
    return {"price_at_current_line": price_req, "line_at_minus110": line_req}


def _sd_for_weight(w_model: float, sd_market: float) -> float:
    """Model SD that gives the model weight w_model against a market of SD sd_market."""
    return sd_market * ((1 - w_model) / w_model) ** 0.5


def price_game(g: dict, teams: dict, w_model_side: float | None = None, w_model_total: float | None = None) -> dict:
    """w_model_*: optional override of the model's fusion weight (sensitivity scenarios)."""
    h, a = g["home"], g["away"]
    struct = structural_projection(g, teams)
    cons = market.consensus(g["books"])
    sig = market_signals(g, cons)
    sd_ms = _sd_for_weight(w_model_side, FUSION.sd_market_side) if w_model_side else FUSION.sd_model_side
    sd_mt = _sd_for_weight(w_model_total, FUSION.sd_market_total) if w_model_total else FUSION.sd_model_total
    side = fuse_side(struct["home_margin"], cons["home_margin"], sd_model=sd_ms)
    tot = fuse_total(struct["total"], cons["total"], sd_model=sd_mt)
    shift = signal_shifts(g, sig)
    post_margin, post_total = side["mu"] + shift["margin"], tot["mu"] + shift["total"]

    sim = mc.simulate(post_margin, post_total)
    summ = mc.summarize(sim)

    best = {}
    for o in _offers(g):
        pw, pp = _probs(sim, o)
        ev = ev_pct(pw, o["price"], pp)
        key = (o["market"], o["side"])
        if key not in best or ev > best[key]["ev_pct"]:
            thr = ev_threshold(o["market"])
            best[key] = {
                "market": o["market"], "side": o["side"], "team": (h if o["side"] == "home" else a) if o["market"] != "total" else None,
                "line": o["line"], "best_book": o["book"], "price": o["price"],
                "p_win": round(pw, 4), "p_push": round(pp, 4), "fair_odds": fair_american(pw, pp),
                "ev_pct": round(ev, 4), "ev_threshold": thr, "eligible": ev >= thr,
                "kelly_units": kelly_units(pw, o["price"], pp) if ev >= thr else 0.0,
                "kelly_units_raw": kelly_units(pw, o["price"], pp),
                "trigger": _trigger(sim, o, thr),
            }
    markets_out = sorted(best.values(), key=lambda r: (r["market"], r["side"]))
    plays = [m for m in markets_out if m["eligible"]]
    # one play per correlated group (spread/ML same team); keep the higher EV
    plays.sort(key=lambda m: -m["ev_pct"])
    chosen, used = [], set()
    for p in plays:
        grp = ("side", p["team"]) if p["market"] in ("spread", "moneyline") else ("total", p["side"])
        if grp in used:
            continue
        used.add(grp); chosen.append(p)

    return {
        "game_id": g["game_id"], "away": a, "home": h, "away_name": teams[a]["name"], "home_name": teams[h]["name"],
        "tip_et": g["tip_et"], "venue": g["venue"], "network": g.get("network"),
        "risk_environment": {
            "injuries": {t: {"players": g["injuries"].get(t, []), **struct["injuries"][t]} for t in (a, h)},
            "situational": struct["situational"],
            "rest": g["fatigue"],
        },
        "projection": {
            "structural": {"home_margin": round(struct["home_margin"], 2), "total": round(struct["total"], 1),
                           "home_pts": round(struct["home_pts"], 1), "away_pts": round(struct["away_pts"], 1),
                           "pace": round(struct["pace"], 1),
                           "exp_ortg": {k: round(v, 1) for k, v in struct["ortg"].items()},
                           "net_prior": {k: round(v["net"], 2) for k, v in struct["ratings"].items()},
                           "modules_neutral": struct["modules_neutral"]},
            "market": {"home_margin": round(cons["home_margin"], 2), "total": round(cons["total"], 1),
                       "spread_display_home": cons["spread_display"], "total_display": cons["total_display"],
                       "home_win_prob_novig": round(cons["home_win_prob_novig"], 4)},
            "posterior": {"home_margin": round(post_margin, 2), "total": round(post_total, 1),
                          "w_model_side": round(side["w_model"], 3), "w_model_total": round(tot["w_model"], 3),
                          "signal_shifts": shift["notes"]},
            "model_minus_market": {"home_margin": round(struct["home_margin"] - cons["home_margin"], 2),
                                   "total": round(struct["total"] - cons["total"], 1)},
        },
        "simulation": {"n": SIM.n_sims, **{k: round(v, 4) for k, v in summ.items()},
                       "margin_hist": mc.histogram(sim["margin"], -40, 40, 2),
                       "total_hist": mc.histogram(sim["total"], 170, 290, 3)},
        "markets": markets_out,
        "market_signals": sig,
        "plays": chosen,
        "recommendation": "BET" if chosen else "PASS",
    }


def evaluate_ticket(game_payload: dict, g: dict, ticket: dict) -> dict:
    """Price an existing ledger ticket against tonight's posterior (for audit and CLV preview)."""
    sim = mc.simulate(game_payload["projection"]["posterior"]["home_margin"], game_payload["projection"]["posterior"]["total"])
    o = {"market": ticket["market"], "side": ticket["side"], "line": ticket["line"], "price": ticket["price"]}
    pw, pp = _probs(sim, o)
    cur = game_payload["projection"]["market"]["spread_display_home"]
    cur_side_line = cur if ticket["side"] == "home" else -cur
    return {"p_win": round(pw, 4), "p_push": round(pp, 4), "ev_pct": round(ev_pct(pw, ticket["price"], pp), 4),
            "current_consensus_line_same_side": cur_side_line,
            "points_better_than_market": round(ticket["line"] - cur_side_line, 2)}
