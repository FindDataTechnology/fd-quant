"""账本待结算列举（add-investment-trading-fleet tasks 1.1-1.5）。

这个入口存在的唯一理由：`retrieve()` 只回**已结算**记录，所以"该结什么"此前无路可查，
定时复盘只能靠记忆 decision_id。下面五条各钉一种会坏掉的方式。
"""
from __future__ import annotations

import json

import pytest

from fd_quant.ledger import tools as lt
from fd_quant.ledger.store import LedgerStore
from fd_quant.portfolio import build_intent

PROD = {"pack": "p", "skill": "k"}


def _rating(subject="600519", as_of="2026-06-30", side="buy"):
    return {"schema_version": "1", "as_of": as_of, "subject": subject, "side": side,
            "basis": "t",
            "evidence": [{"evidence_id": "e", "label": "l", "value": 1, "unit": "u",
                          "as_of": as_of, "source": "s"}],
            "producer": PROD}


@pytest.fixture
def store(tmp_path):
    return LedgerStore(tmp_path / "l.db")


def _record(store, caller="c1", symbol="600519", as_of="2026-06-30", pack="p"):
    # payload.producer 必须与记录的 pack/skill 一致（账本强制，不替调用方挑一个当真）
    payload = {**_rating(symbol, as_of), "producer": {"pack": pack, "skill": "k"}}
    return store.record(caller=caller, as_of=as_of, symbol=symbol, pack=pack, skill="k",
                        payload=payload)["decision_id"]


# ------------------------------------------------------- 1.1 分页与总数


def test_pagination_union_equals_total_without_duplicates(store):
    ids = {_record(store, symbol=s) for s in ("A1", "A2", "A3")}
    page1 = store.list_pending(caller="c1", limit=2, offset=0, today="2026-07-20")
    page2 = store.list_pending(caller="c1", limit=2, offset=2, today="2026-07-20")

    assert page1["pagination"]["total"] == 3
    seen = [i["decision_id"] for i in page1["items"] + page2["items"]]
    assert len(seen) == 3 and len(set(seen)) == 3
    assert set(seen) == ids


def test_filters_by_pack_and_record_kind(store):
    _record(store, symbol="A1", pack="p1")
    _record(store, symbol="A2", pack="p2")
    intent = build_intent([{"as_of": "2026-06-30", "subject": "A1", "side": "buy",
                            "conviction": 0.8}],
                          {"kind": "equal_weight", "cash_floor": 0.2},
                          as_of="2026-06-30", producer={"pack": "p1", "skill": "k"})
    store.record(caller="c1", as_of="2026-06-30", portfolio_key="k1", pack="p1", skill="k",
                 payload=intent, record_kind="portfolio")

    only_p1 = store.list_pending(caller="c1", pack="p1", today="2026-07-20")
    assert only_p1["pagination"]["total"] == 2

    only_pf = store.list_pending(caller="c1", record_kind="portfolio", today="2026-07-20")
    assert only_pf["pagination"]["total"] == 1
    assert only_pf["items"][0]["record_kind"] == "portfolio"


def test_as_of_from_filters_older_records(store):
    _record(store, symbol="A1", as_of="2026-05-01")
    _record(store, symbol="A2", as_of="2026-06-30")
    got = store.list_pending(caller="c1", as_of_from="2026-06-01", today="2026-07-20")
    assert got["pagination"]["total"] == 1
    assert got["items"][0]["as_of"] == "2026-06-30"


def test_settled_records_are_not_listed(store):
    did = _record(store, symbol="A1", as_of="2026-06-30")
    dates = ["2026-06-30", "2026-07-01", "2026-07-02"]
    store.settle(decision_id=did, symbol_dates=dates, symbol_prices=[10.0, 11.0, 12.0],
                 benchmark_dates=dates, benchmark_prices=[100.0, 100.0, 101.0],
                 as_of="2026-07-02", benchmark_symbol="000300", holding_days=2)
    got = store.list_pending(caller="c1", today="2026-07-20")
    assert got["pagination"]["total"] == 0


# ------------------------------------------------------- 1.2 未到期 ≠ 不存在


def test_not_yet_due_record_is_listed_and_flagged(store):
    """未到期不是"查不到"：它必须在清单里，且标 matured=false + 缺口天数。"""
    _record(store, symbol="A1", as_of="2026-07-18")
    got = store.list_pending(caller="c1", holding_days=5, today="2026-07-19")
    assert got["pagination"]["total"] == 1
    item = got["items"][0]
    assert item["matured"] is False
    assert item["due_estimate"] == "2026-07-23"
    assert item["missing_trading_days_estimate"] > 0


def test_due_record_is_flagged_matured(store):
    _record(store, symbol="A1", as_of="2026-06-01")
    got = store.list_pending(caller="c1", holding_days=5, today="2026-07-19")
    assert got["items"][0]["matured"] is True
    assert got["items"][0]["missing_trading_days_estimate"] == 0


def test_items_carry_payload_and_identity(store):
    did = _record(store, symbol="A1")
    item = store.list_pending(caller="c1", today="2026-07-20")["items"][0]
    assert item["decision_id"] == did
    assert item["payload"]["subject"] == "A1"
    assert item["status"] == "pending"


# ------------------------------------------------------- 1.3 调用者隔离


def test_callers_only_see_their_own_pending(store):
    _record(store, caller="c1", symbol="A1")
    _record(store, caller="c1", symbol="A2")
    _record(store, caller="c2", symbol="B1")

    one = store.list_pending(caller="c1", today="2026-07-20")
    two = store.list_pending(caller="c2", today="2026-07-20")
    assert one["pagination"]["total"] == 2
    assert two["pagination"]["total"] == 1
    assert {i["payload"]["subject"] for i in one["items"]} == {"A1", "A2"}
    assert {i["payload"]["subject"] for i in two["items"]} == {"B1"}


# ------------------------------------------------------- 1.4 遍历中结算


def test_settling_mid_pagination_does_not_duplicate_or_skip(store):
    ids = [_record(store, symbol=s, as_of=f"2026-06-0{i}") for i, s in
           enumerate(("A1", "A2", "A3", "A4"), start=1)]
    dates = ["2026-06-01", "2026-06-02", "2026-06-03", "2026-06-04"]
    page1 = store.list_pending(caller="c1", limit=2, offset=0, today="2026-07-20")
    assert page1["pagination"]["total"] == 4

    # 结算第一页的第一条
    first = page1["items"][0]["decision_id"]
    store.settle(decision_id=first, symbol_dates=dates, symbol_prices=[10.0, 11.0, 12.0, 13.0],
                 benchmark_dates=dates, benchmark_prices=[100.0] * 4, as_of="2026-06-04",
                 benchmark_symbol="000300", holding_days=2)

    page2 = store.list_pending(caller="c1", limit=2, offset=2, today="2026-07-20")
    assert page2["pagination"]["total"] == 3
    later = [i["decision_id"] for i in page2["items"]]
    assert first not in later
    assert len(set(later)) == len(later)


# ------------------------------------------------------- 1.5 不动点内语义


def test_retrieve_still_hides_future_settlements(store):
    """既有纪律：as_of 之后才结算的记录不出现在 retrieve 结果里。"""
    did = _record(store, symbol="A1", as_of="2026-06-01")
    dates = ["2026-06-01", "2026-06-02", "2026-06-03"]
    store.settle(decision_id=did, symbol_dates=dates, symbol_prices=[10.0, 11.0, 12.0],
                 benchmark_dates=dates, benchmark_prices=[100.0] * 3, as_of="2026-06-03",
                 benchmark_symbol="000300", holding_days=2)
    before = store.retrieve(caller="c1", symbol="A1", as_of="2026-06-02")
    after = store.retrieve(caller="c1", symbol="A1", as_of="2026-06-03")
    assert before["same_symbol"] == []
    assert len(after["same_symbol"]) == 1
    # 列举入口 meanwhile 看到的是 0 条（已结算）
    assert store.list_pending(caller="c1", today="2026-07-20")["pagination"]["total"] == 0


def test_tool_wrapper_returns_envelope_and_total(store, monkeypatch):
    monkeypatch.setenv("FD_QUANT_LEDGER_DB", str(store.db_path))
    monkeypatch.setenv("FD_QUANT_CALLER", "c1")   # 工具层用 resolve_caller()，与 store 层一致
    _record(store, symbol="A1")
    out = lt.list_pending_decisions(as_of="2026-07-20")
    assert out["tool"] == "list_pending_decisions"
    assert out["result"]["pagination"]["total"] == 1
    json.dumps(out, allow_nan=False, ensure_ascii=False)
