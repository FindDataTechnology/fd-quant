"""阈值口径结算（generalize-decision-contract 任务 4.2–4.4）。

三条不可退让：
1. 数据不足记 `insufficient_data`，**不记 miss**——把"不知道"当成"判断错了"会让
   复盘系统性低估准确率，而且再也查不出来；
2. 数据不足**照样落行**，不让记录停在 pending 而不给原因；
3. 序列由调用方传入，账本只留指纹，不留客户原始数据。
"""

from datetime import datetime, timezone

import pytest

from fd_quant.ledger.errors import LedgerError, UnknownDecision
from fd_quant.ledger.store import LedgerStore


@pytest.fixture
def store(tmp_path):
    return LedgerStore(tmp_path / "ledger.db")


def record(store, *, rule="metric_threshold", subject="store-0413"):
    payload = {
        "schema_version": "2",
        "as_of": "2026-05-01",
        "subject": subject,
        "subject_kind": "store",
        "action": "expand",
        "preconditions": [],
        "settlement_rule_id": rule,
        "expected_outcome": {"metric_ref": subject, "comparator": "gte",
                             "threshold": 0.42,
                             "horizon": {"unit": "month", "count": 3},
                             "settlement_source": "caller_supplied_series"},
        "basis": "门店月均流水连续三个月高于保本线。",
        "evidence": [],
        "producer": {"pack": "wire-erp", "skill": "store-expansion"},
    }
    return store.record(
        caller="alice", as_of="2026-05-01", symbol=subject,
        pack="wire-erp", skill="store-expansion", payload=payload,
        evidence=[],
    )["decision_id"]


DATES = ["2026-05-31", "2026-06-30", "2026-07-31"]


def test_hit_when_threshold_met(store):
    did = record(store)

    out = store.settle_metric_threshold(
        decision_id=did, dates=DATES, values=[0.38, 0.41, 0.45],
        as_of="2026-08-01", comparator="gte", threshold=0.42,
    )

    assert out["verdict"] == "hit"
    assert out["observed_value"] == 0.45
    assert out["rule_id"] == "metric_threshold"
    assert out["created"] is True


def test_miss_when_threshold_not_met(store):
    did = record(store)

    out = store.settle_metric_threshold(
        decision_id=did, dates=DATES, values=[0.38, 0.40, 0.41],
        as_of="2026-08-01", comparator="gte", threshold=0.42,
    )

    assert out["verdict"] == "miss"


def test_insufficient_data_is_not_miss(store):
    """观察窗内全是空值 → insufficient_data，且**照样落行**。"""
    did = record(store)

    out = store.settle_metric_threshold(
        decision_id=did, dates=DATES, values=[None, None, None],
        as_of="2026-08-01", comparator="gte", threshold=0.42,
    )

    assert out["verdict"] == "insufficient_data"
    assert out["verdict"] != "miss"
    assert out["observed_value"] is None
    assert out["status"] == "settled", "不得停在 pending 而不给原因"


def test_no_points_in_window_is_insufficient_data(store):
    did = record(store)

    out = store.settle_metric_threshold(
        decision_id=did, dates=["2026-05-31"], values=[0.9],
        as_of="2026-08-01", comparator="gte", threshold=0.42,
        window_end="2026-05-01",   # 窗口结束早于唯一的数据点 → 窗内无点
    )

    assert out["verdict"] == "insufficient_data"


def test_uses_latest_point_within_window(store):
    """取窗口内最后一个点，不看窗口外的未来值。"""
    did = record(store)

    out = store.settle_metric_threshold(
        decision_id=did, dates=DATES, values=[0.30, 0.50, 0.20],
        as_of="2026-08-01", comparator="gte", threshold=0.42,
        window_end="2026-06-30",
    )

    assert out["observed_value"] == 0.50
    assert out["verdict"] == "hit"


def test_settlement_is_idempotent(store):
    did = record(store)
    first = store.settle_metric_threshold(
        decision_id=did, dates=["2026-06-30"], values=[0.45], as_of="2026-08-01",
        comparator="gte", threshold=0.42)
    second = store.settle_metric_threshold(
        decision_id=did, dates=["2026-06-30"], values=[0.10], as_of="2026-08-01",
        comparator="gte", threshold=0.42)

    assert first["created"] is True
    assert second["created"] is False
    assert second["reason"] == "duplicate_settlement"
    assert second["verdict"] == first["verdict"], "既有结算不被改写"


def test_fingerprint_is_stored_and_stable(store):
    did = record(store)
    out = store.settle_metric_threshold(
        decision_id=did, dates=DATES, values=[0.38, 0.41, 0.45],
        as_of="2026-08-01", comparator="gte", threshold=0.42)

    assert out["series_fingerprint"]
    assert len(out["series_fingerprint"]) == 32

    # 同一序列指纹一致；改一个值就变
    from fd_quant.ledger.store import _series_fingerprint
    a = _series_fingerprint([("2026-05-31", 0.38), ("2026-06-30", 0.41)])
    b = _series_fingerprint([("2026-05-31", 0.38), ("2026-06-30", 0.41)])
    c = _series_fingerprint([("2026-05-31", 0.38), ("2026-06-30", 0.42)])
    assert a == b
    assert a != c


def test_future_window_refused(store):
    did = record(store)

    with pytest.raises(LedgerError) as exc:
        store.settle_metric_threshold(
            decision_id=did, dates=DATES, values=[0.5], as_of="2026-08-01",
            comparator="gte", threshold=0.42, window_end="2026-12-31")

    assert "未来" in str(exc.value)


def test_illegal_comparator_refused_with_legal_set(store):
    did = record(store)

    with pytest.raises(LedgerError) as exc:
        store.settle_metric_threshold(
            decision_id=did, dates=DATES, values=[0.5], as_of="2026-08-01",
            comparator="≈", threshold=0.42)

    assert "gte" in str(exc.value)


def test_unknown_decision_refused(store):
    with pytest.raises(UnknownDecision):
        store.settle_metric_threshold(
            decision_id="nope", dates=["2026-06-30"], values=[0.5],
            as_of="2026-08-01", comparator="gte", threshold=0.42)


@pytest.mark.parametrize("comparator,value,threshold,expect", [
    ("lt", 0.3, 0.4, "hit"),
    ("lte", 0.4, 0.4, "hit"),
    ("gt", 0.4, 0.4, "miss"),
    ("gte", 0.4, 0.4, "hit"),
    ("eq", 0.4, 0.4, "hit"),
    ("neq", 0.4, 0.4, "miss"),
])
def test_all_comparators(store, comparator, value, threshold, expect):
    did = record(store, subject=f"store-{comparator}")

    out = store.settle_metric_threshold(
        decision_id=did, dates=["2026-06-30"], values=[value],
        as_of="2026-08-01", comparator=comparator, threshold=threshold)

    assert out["verdict"] == expect
