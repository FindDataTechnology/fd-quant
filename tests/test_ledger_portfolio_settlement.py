"""组合级结算（add-portfolio-intent-contract tasks 4.1-4.4）。

四条纪律，每条都有对应的拒绝用例：
1. 加权收益 + 现金部分分开报（现金收益记零，不摊进标的收益）；
2. 整期原子：任一腿不覆盖 → 整次拒绝，不落半行；
3. 幂等：重复结算不产生第二行；
4. 不得以单票结果拼装——组合结算只走自己的表与入口。
"""
from __future__ import annotations

import pytest

from fd_quant.ledger.errors import LedgerError, SettlementRefused
from fd_quant.ledger.store import LedgerStore
from fd_quant.portfolio import build_intent

DATES = ["2026-06-30", "2026-07-01", "2026-07-02"]
BENCH = {"benchmark_dates": DATES, "benchmark_prices": [100.0, 100.0, 101.0]}
KW = {**BENCH, "as_of": "2026-07-02", "benchmark_symbol": "000300", "holding_days": 2}


@pytest.fixture
def store(tmp_path):
    return LedgerStore(tmp_path / "ledger.db")


def _intent(cash=0.2, subjects=("A", "B")):
    return build_intent(
        [{"as_of": "2026-06-30", "subject": s, "side": "buy", "conviction": 0.8} for s in subjects],
        {"kind": "equal_weight", "cash_floor": cash},
        as_of="2026-06-30", producer={"pack": "p", "skill": "k"},
    )


def _record(store, intent=None, key="p-1", caller="c1"):
    return store.record(
        caller=caller, as_of="2026-06-30", portfolio_key=key, pack="p", skill="k",
        payload=intent or _intent(), record_kind="portfolio",
    )["decision_id"]


def _legs(a_exit=12.0, b_exit=22.0):
    return {"A": (DATES, [10.0, 11.0, a_exit]), "B": (DATES, [20.0, 20.0, b_exit])}


# ------------------------------------------------------- 4.1 加权收益与现金


def test_weighted_return_and_cash_are_reported_separately(store):
    """手算：等权 cash 0.2 → 各 0.4；A 12/10-1=0.2，B 22/20-1=0.1；
    raw = 0.4*0.2 + 0.4*0.1 = 0.12；现金 0.2 收益记 0；基准 101/100-1=0.01；alpha=0.11。"""
    did = _record(store)
    out = store.settle_portfolio(decision_id=did, legs=_legs(), **KW)
    s = out["settlement"]
    assert s["raw_return"] == pytest.approx(0.12)
    assert s["invested_weight"] == pytest.approx(0.8)
    assert s["cash_weight"] == pytest.approx(0.2)
    assert s["cash_return"] == 0.0
    assert s["benchmark_return"] == pytest.approx(0.01)
    assert s["alpha"] == pytest.approx(0.11)
    assert s["leg_returns"] == {"A": pytest.approx(0.2), "B": pytest.approx(0.1)}


def test_weights_come_from_the_recorded_intent(store):
    """权重只认记录里那份：调用方无从另传一份，故"依据哪份权重"只有一个答案。"""
    did = _record(store, _intent(cash=0.5))
    out = store.settle_portfolio(decision_id=did, legs=_legs(), **KW)
    assert out["settlement"]["invested_weight"] == pytest.approx(0.5)
    assert out["settlement"]["cash_weight"] == pytest.approx(0.5)


def test_record_is_marked_settled(store):
    did = _record(store)
    store.settle_portfolio(decision_id=did, legs=_legs(), **KW)
    assert store.get(did)["status"] == "settled"


# ------------------------------------------------------- 4.2 整期原子


def test_a_short_leg_refuses_the_whole_settlement(store):
    """一只腿缺 2 天 → 整次拒绝，记录仍 pending，且不落任何组合结算行。"""
    did = _record(store)
    short = {
        "A": (DATES, [10.0, 11.0, 12.0]),
        "B": (DATES[:1], [20.0]),          # 只有入场日
    }
    with pytest.raises(SettlementRefused) as ei:
        store.settle_portfolio(decision_id=did, legs=short, **KW)
    assert "missing_trading_days" in str(ei.value) or "insufficient coverage" in str(ei.value)
    assert store.get(did)["status"] == "pending"


def test_missing_leg_is_refused_not_assumed_zero(store):
    """少一条腿不能当"这只没持仓"处理——那是把输入缺失读成结论。"""
    did = _record(store)
    with pytest.raises(SettlementRefused, match="missing price series"):
        store.settle_portfolio(decision_id=did, legs={"A": (DATES, [10.0, 11.0, 12.0])}, **KW)


def test_extra_leg_is_refused(store):
    did = _record(store)
    legs = _legs()
    legs["C"] = (DATES, [5.0, 5.0, 5.0])
    with pytest.raises(SettlementRefused, match="missing weight for leg"):
        store.settle_portfolio(decision_id=did, legs=legs, **KW)


def test_entry_date_must_be_on_the_aligned_axis(store):
    did = _record(store)
    legs = {"A": (DATES[1:], [11.0, 12.0]), "B": (DATES[1:], [20.0, 22.0])}
    with pytest.raises(SettlementRefused, match="not on the aligned trading axis"):
        store.settle_portfolio(decision_id=did, legs=legs, **KW)


def test_non_positive_endpoint_is_refused(store):
    did = _record(store)
    legs = _legs(a_exit=0.0)
    with pytest.raises(SettlementRefused, match="non-positive"):
        store.settle_portfolio(decision_id=did, legs=legs, **KW)


# ------------------------------------------------------- 4.3 幂等


def test_repeat_settlement_is_idempotent(store):
    did = _record(store)
    first = store.settle_portfolio(decision_id=did, legs=_legs(), **KW)
    second = store.settle_portfolio(decision_id=did, legs=_legs(), **KW)
    assert first["created"] is True
    assert second["created"] is False and second["reason"] == "duplicate_settlement"
    assert second["settlement"]["raw_return"] == first["settlement"]["raw_return"]


def test_reflection_difference_does_not_rewrite_the_row(store):
    did = _record(store)
    store.settle_portfolio(decision_id=did, legs=_legs(), reflection="原反思", **KW)
    again = store.settle_portfolio(decision_id=did, legs=_legs(), reflection="新反思", **KW)
    assert again["reflection_ignored"] is True
    assert again["settlement"]["reflection"] == "原反思"


# --------------------------------------------- 4.4 不得以单票结果拼装


def test_portfolio_settlement_has_its_own_table(store):
    """组合结算落在 portfolio_settlements，与单票 settlements 物理分开。"""
    import sqlite3

    did = _record(store)
    store.settle_portfolio(decision_id=did, legs=_legs(), **KW)
    con = sqlite3.connect(store.db_path)
    assert con.execute("SELECT COUNT(*) FROM settlements").fetchone()[0] == 0
    assert con.execute("SELECT COUNT(*) FROM portfolio_settlements").fetchone()[0] == 1
    con.close()


def test_no_api_composes_a_portfolio_from_single_settlements(store):
    """结构上不存在这样的入口：没有可选 weights 参数，也就无法从单票结果反推组合收益。"""
    import inspect

    sig = inspect.signature(LedgerStore.settle_portfolio)
    assert "weights" not in sig.parameters
    assert not hasattr(store, "settle_from_singles")


def test_single_record_cannot_be_settled_as_portfolio(store):
    from fd_quant.ledger.store import LedgerStore as _L  # noqa: F401

    did = store.record(
        caller="c1", as_of="2026-06-30", symbol="A", pack="p", skill="k",
        payload={
            "schema_version": "1", "as_of": "2026-06-30", "subject": "A", "side": "buy",
            "basis": "t",
            "evidence": [{"evidence_id": "e", "label": "l", "value": 1, "unit": "u",
                          "as_of": "2026-06-30", "source": "s"}],
            "producer": {"pack": "p", "skill": "k"},
        },
    )["decision_id"]
    with pytest.raises(LedgerError, match="only settles portfolio records"):
        store.settle_portfolio(decision_id=did, legs=_legs(), **KW)


def test_zero_aligned_points_reports_missing_days(store):
    did = _record(store)
    legs = {"A": (["2026-06-30"], [10.0]), "B": (["2026-06-30"], [20.0])}
    with pytest.raises(SettlementRefused) as ei:
        store.settle_portfolio(decision_id=did, legs=legs, **KW)
    assert "insufficient coverage" in str(ei.value)
