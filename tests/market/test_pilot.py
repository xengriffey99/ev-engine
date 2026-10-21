import pytest

from market import pilot
from market.results import Final, grade, grade_rows, parse_scoreboard

DAY = __import__("datetime").date(2026, 10, 23)
OK = {"novig_fee": {"passed": True, "evidence": "x"}, "home_lean": {"passed": True, "evidence": "y"}}


def row(oid, *, day="2026-10-23", market="spreads", model="nba-sim 0.3.0", gate=True, stake=0.4,
        clv=None, clv_method=None, result=None, game="g1", side="Boston Celtics", line=-4.5, price=-105):
    return {"opinion_id": oid, "timestamp_et": f"{day}T13:20:00-04:00", "game_id": game, "market": market,
            "model_version": model, "gate_pass": gate, "stake_u": stake, "clv_pct": clv,
            "clv_method": clv_method, "result": result, "side": side, "line": line, "book": "novig",
            "price_american": price}


def closed_prior_slate(n=20, missing=0):
    return [row(f"p{i}", day="2026-10-22", gate=False, stake=0, clv=0.0,
                clv_method="close_missing" if i < missing else "exact_line") for i in range(n)]


def test_inactive_until_every_entry_check_passes(tmp_path):
    db = tmp_path / "m.db"
    rows = closed_prior_slate() + [row("a")]
    msgs = pilot.record_pilot(db, rows, DAY, manual={"home_lean": OK["home_lean"]})
    assert "novig_fee" in msgs[0]
    b = pilot.picks(db, DAY)
    assert b[0]["status"] == "skipped" and b[0]["pilot_stake_u"] == 0
    assert pilot.record_pilot(db, rows, DAY, manual=OK) == []  # a skipped candidate is not re-sized


def test_close_capture_check_needs_95_percent():
    assert not pilot.entry_checks(closed_prior_slate(20, missing=2), DAY, OK)["close_capture"]["passed"]
    assert pilot.entry_checks(closed_prior_slate(20, missing=1), DAY, OK)["close_capture"]["passed"]
    assert not pilot.entry_checks([], DAY, OK)["close_capture"]["passed"]


def test_sizing_half_stake_per_bet_cap_daily_cap(tmp_path):
    db = tmp_path / "m.db"
    rows = closed_prior_slate() + [
        row("a", stake=0.59, game="g1"), row("b", stake=0.39, game="g2"), row("c", stake=0.28, game="g3"),
        row("d", stake=0.27, game="g4"), row("e", stake=0.23, game="g5"), row("f", stake=0.22, game="g6"),
        row("t", market="totals", game="g7"),  # totals are not in the pilot
        row("p", model="props-v1", market="player_points", game="g8"),  # nor props
        row("z", gate=False, stake=0.0, game="g9"),
    ]
    msgs = pilot.record_pilot(db, rows, DAY, manual=OK)
    picks = {b["opinion_id"]: b for b in pilot.picks(db, DAY)}
    assert set(picks) == {"a", "b", "c", "d", "e", "f"}
    assert all(b["pilot_stake_u"] <= 0.25 for b in picks.values())
    assert sum(b["pilot_stake_u"] for b in picks.values()) <= 1.0 + 1e-9
    assert picks["a"]["pilot_stake_u"] >= picks["f"]["pilot_stake_u"]
    assert len(msgs) == 6 and all(m.startswith("PILOT PICK") for m in msgs)
    assert pilot.record_pilot(db, rows, DAY, manual=OK) == []  # idempotent


def test_kill_switch_on_negative_clv_at_50(tmp_path):
    db = tmp_path / "m.db"
    con = pilot._conn(db)
    rows = []
    for i in range(50):
        rows.append(row(f"m{i}", clv=-0.01 if i % 3 else 0.004, clv_method="exact_line", game=f"g{i}"))
        rows.append(row(f"b{i}", model="market-baseline-v0", clv=0.0, clv_method="exact_line", game=f"g{i}"))
        con.execute("INSERT INTO pilot_bets (opinion_id, date_et, pilot_stake_u, status) VALUES (?,?,?,?)",
                    (f"m{i}", "2026-10-23", 0.1, "placed"))
    con.commit()
    ks = pilot.kill_switch(con, rows)
    assert ks["n"] == 50 and ks["mean_clv"] < 0 and ks["t_vs_baseline"] < -1 and ks["stop"]


def test_drawdown_pause(tmp_path):
    db = tmp_path / "m.db"
    con = pilot._conn(db)
    rows = closed_prior_slate()
    for i in range(21):
        rows.append(row(f"l{i}", result="loss", game=f"g{i}", day="2026-10-22"))
        con.execute("INSERT INTO pilot_bets (opinion_id, date_et, pilot_stake_u, price_american, status) "
                    "VALUES (?,?,?,?,?)", (f"l{i}", "2026-10-22", 0.25, -110, "placed"))
    con.commit()
    st = pilot.status(db, rows, DAY, manual=OK)
    assert st["pnl_u"] == pytest.approx(-5.25) and st["drawdown_paused"] and not st["active"]


SCOREBOARD = {"events": [{"competitions": [{"status": {"type": {"completed": True}}, "competitors": [
    {"homeAway": "home", "score": "110", "team": {"displayName": "LA Clippers"}},
    {"homeAway": "away", "score": "104", "team": {"displayName": "Sacramento Kings"}}]}]},
    {"competitions": [{"status": {"type": {"completed": False}}, "competitors": []}]}]}


def test_scoreboard_parse_and_grade():
    finals = parse_scoreboard(SCOREBOARD)
    assert finals == [Final("Los Angeles Clippers", "Sacramento Kings", 110, 104)]
    f = finals[0]
    assert grade("spreads", "Los Angeles Clippers", -5.0, f.home, f.away, f) == "win"
    assert grade("spreads", "Sacramento Kings", 5.0, f.home, f.away, f) == "loss"
    assert grade("spreads", "Los Angeles Clippers", -6.0, f.home, f.away, f) == "push"
    assert grade("totals", "Under", 227.5, f.home, f.away, f) == "win"
    rows = [row("x", game="e1", side="Sacramento Kings", line=5.0), row("y", game="e2")]
    out = grade_rows(rows, {"e1": (f.home, f.away)}, finals)
    assert out == {"x": "loss"}
