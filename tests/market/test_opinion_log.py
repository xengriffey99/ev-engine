import csv
from datetime import datetime, timezone

import pytest
from factories import AWAY, HOME, game, q

from market import config as C
from market.clv import EXACT, MISSING, ClvResult, compute_clv
from market.opinion_log import (
    EXPORT_COLUMNS,
    SPEC_COLUMNS,
    NotCloseCapturableError,
    Opinion,
    OpinionLog,
    OpinionLogError,
)
from market.selection import best_price, reference_prob

TS = datetime(2026, 10, 22, 16, 5, tzinfo=timezone.utc)
SPEC_SECTION_9 = (
    "timestamp_et, game_id, market, player, stat, line, side, book, price_american, two_way, "
    "p_model, p_market_novig, w, p_bet, tier2_adj, ev_pct, gate_pass, stake_u, close_line, "
    "close_price, p_close_novig, clv_pct, result, notes"
)


@pytest.fixture
def log(tmp_path):
    lg = OpinionLog(tmp_path / "log.db", close_plan=lambda game_id, market: True)
    yield lg
    lg.close()


def _side(**kw):
    base = dict(
        game_id="evt1", market="spreads", side=AWAY, line=4.5, book="draftkings",
        price_american=-105, p_model=0.56, p_market_novig=0.50,
        model_version="v1-kalman", ref_quality="sharp", timestamp=TS,
    )
    base.update(kw)
    return Opinion(**base)


def test_columns_keep_spec_order_with_extras_at_the_edges():
    assert SPEC_COLUMNS == tuple(c.strip() for c in SPEC_SECTION_9.split(","))
    assert EXPORT_COLUMNS[0] == "opinion_id"
    assert EXPORT_COLUMNS[1:25] == SPEC_COLUMNS
    assert EXPORT_COLUMNS[25:] == (
        "model_version", "ref_quality", "devig_method", "close_ts_et", "clv_method", "pass_reason",
    )


def test_blend_ev_and_gate_are_computed(log):
    oid = log.record(_side())
    row = log.rows()[0]
    assert row["opinion_id"] == oid
    assert row["w"] == pytest.approx(0.40)
    assert row["p_bet"] == pytest.approx(0.4 * 0.56 + 0.6 * 0.50)  # 0.524
    assert row["ev_pct"] == pytest.approx(0.524 * (1 + 100 / 105) - 1)  # +2.30%
    assert row["gate_pass"] is False and row["pass_reason"] == "below_gate"  # under 2.5%
    log.record(_side(p_model=0.60))  # p_bet 0.54 -> EV +5.43%
    row2 = log.rows()[1]
    assert row2["ev_pct"] == pytest.approx(0.54 * (1 + 100 / 105) - 1)
    assert row2["gate_pass"] is True and row2["pass_reason"] is None


def test_pass_is_logged_below_gate(log):
    # p_bet 0.524 at -110 -> EV = 0.38%: below the 2.5% sides gate.
    log.record(_side(price_american=-110))
    row = log.rows()[0]
    assert row["gate_pass"] is False
    assert row["pass_reason"] == "below_gate"
    assert row["stake_u"] == 0


def test_props_use_props_weight_and_gate(log):
    log.record(Opinion(
        game_id="evt1", market="player_points", side="Under", line=24.5, book="betmgm",
        price_american=-110, p_model=0.60, p_market_novig=0.52, player="Jayson Tatum",
        model_version="phase0-test", ref_quality="consensus_retail", timestamp=TS,
    ))
    row = log.rows()[0]
    assert row["w"] == pytest.approx(0.50)
    assert row["stat"] == "points"
    assert row["p_bet"] == pytest.approx(0.56)
    assert row["ev_pct"] == pytest.approx(0.56 * (1 + 100 / 110) - 1)  # 6.9% >= 3.5%
    assert row["gate_pass"] is True


def test_one_way_rejected(log):
    with pytest.raises(OpinionLogError):
        log.record(_side(two_way=False))


def test_market_without_close_snapshot_rejected(tmp_path):
    lg = OpinionLog(tmp_path / "x.db", close_plan=lambda g, m: m != "spreads")
    with pytest.raises(NotCloseCapturableError):
        lg.record(_side())
    lg.close()


def test_wrong_weight_rejected_unless_fit(log):
    with pytest.raises(OpinionLogError):
        log.record(_side(w=0.6))
    log.record(_side(w=0.6, price_american=-110), allow_custom_w=True)


def test_inconsistent_p_bet_rejected(log):
    with pytest.raises(OpinionLogError):
        log.record(_side(p_bet=0.6))


def test_tier2_cap_enforced(log):
    with pytest.raises(OpinionLogError):
        log.record(_side(tier2_adj=0.025))
    log.record(_side(tier2_adj=-0.02))
    assert log.rows()[0]["tier2_adj"] == pytest.approx(-0.02)


def test_gate_conflict_and_stake_on_pass_rejected(log):
    with pytest.raises(OpinionLogError):
        log.record(_side(price_american=-110, gate_pass=True))
    with pytest.raises(OpinionLogError):
        log.record(_side(price_american=-110, stake_u=0.5))


def test_forced_pass_reason(log):
    log.record(_side(pass_reason="stale_reference"))
    row = log.rows()[0]
    assert row["gate_pass"] is False and row["pass_reason"] == "stale_reference"
    with pytest.raises(OpinionLogError):
        log.record(_side(pass_reason="vibes"))


def test_outsized_edge_ceiling(log, monkeypatch):
    monkeypatch.setitem(C.OUTSIZED_EDGE_CEILING, C.SIDES_TOTALS, 0.01)
    log.record(_side())  # EV ~2.3% > 1% ceiling
    assert log.rows()[0]["pass_reason"] in {"below_gate", "outsized_edge"}
    log.record(_side(p_model=0.70))
    assert log.rows()[1]["pass_reason"] == "outsized_edge"


def test_per_game_cap(log):
    log.record(_side(p_model=0.70, stake_u=1.5))
    with pytest.raises(OpinionLogError):
        log.record(_side(p_model=0.70, stake_u=0.6))
    log.record(_side(p_model=0.70, stake_u=0.5))


def test_timestamps_are_et_with_offset(log):
    log.record(_side())
    assert log.rows()[0]["timestamp_et"] == "2026-10-22T12:05:00-04:00"


def test_close_out_and_export(log, tmp_path):
    oid = log.record(_side())
    close_ts = datetime(2026, 10, 22, 22, 55, tzinfo=timezone.utc)
    log.close_out(oid, ClvResult(0.031, 0.54, 4.5, -118.0, EXACT, "exchange", "power", close_ts))
    row = log.rows()[0]
    assert row["clv_pct"] == pytest.approx(0.031)
    assert row["close_ts_et"] == "2026-10-22T18:55:00-04:00"
    assert row["ref_quality"] == "exchange"  # worse of sharp (open) and exchange (close)
    with pytest.raises(OpinionLogError):
        log.close_out(oid, ClvResult(0.0, 0.5, 4.5, -110.0, EXACT, "sharp", "power", close_ts))
    log.set_result(oid, "win")

    path = log.export_csv(tmp_path / "opinion_log.csv")
    with path.open() as fh:
        reader = csv.DictReader(fh)
        assert tuple(reader.fieldnames) == EXPORT_COLUMNS
        rows = list(reader)
    assert len(rows) == 1 and rows[0]["result"] == "win"


def test_missed_close_stays_in_log(log):
    oid = log.record(_side())
    log.mark_close_missing(oid)
    row = log.rows()[0]
    assert row["clv_method"] == MISSING and row["clv_pct"] is None
    assert len(log.rows()) == 1


def test_end_to_end_under_ev_uses_the_under_price(log):
    """Acceptance: an UNDER's EV and CLV come from UNDER prices only."""
    noon = [
        q("fanduel", "Over", 24.5, +130), q("fanduel", "Under", 24.5, -160),
        q("draftkings", "Over", 24.5, +120), q("draftkings", "Under", 24.5, -150),
        q("betmgm", "Over", 24.5, +115), q("betmgm", "Under", 24.5, -145),
    ]
    kw = dict(event_id="evt1", market="player_points", side="Under", line=24.5, player="Jayson Tatum")
    pick = best_price(noon, books=C.BETTABLE_BY_STATE["MA"], **kw)
    # From Maine only DraftKings of these three is legal: the best usable Under is worse.
    me_pick = best_price(noon, books=C.BETTABLE_BY_STATE["ME"], **kw)
    assert me_pick.book == "draftkings" and me_pick.price_american == pytest.approx(-150)
    ref = reference_prob(noon, **kw)
    oid = log.record(Opinion(
        game_id="evt1", market="player_points", side="Under", line=24.5, book=pick.book,
        price_american=pick.price_american, p_model=0.62, p_market_novig=ref.p,
        player="Jayson Tatum", model_version="phase0-test", ref_quality=ref.ref_quality, timestamp=TS,
    ))
    row = log.rows()[0]
    assert row["book"] == "betmgm" and row["price_american"] == pytest.approx(-145)
    p_final = 0.5 * 0.62 + 0.5 * ref.p
    assert row["ev_pct"] == pytest.approx(p_final * (1 + 100 / 145) - 1)
    # The buggy version would have used +130 and shown a far larger edge.
    assert row["ev_pct"] < p_final * 2.30 - 1

    close = [
        q("fanduel", "Over", 24.5, +120, snapshot_id="close"),
        q("fanduel", "Under", 24.5, -150, snapshot_id="close"),
    ]
    clv = compute_clv(close, event_id="evt1", market="player_points", side="Under", bet_line=24.5,
                      bet_price_american=row["price_american"], player="Jayson Tatum")
    log.close_out(oid, clv)
    closed = log.rows()[0]
    assert closed["clv_method"] == EXACT
    assert closed["close_price"] == pytest.approx(-150)  # the UNDER's closing price
    assert closed["p_close_novig"] > 0.5
