"""经营结算的隔离与数据不出 cell（generalize-decision-contract 任务 5.1 / 5.2）。

两条不可退让：
1. `caller` 隔离对经营结算与证券记录**同等生效**——另一个 caller 结算同一
   metric_ref，谁也看不见谁；
2. 账本行只含 `metric_ref` + 序列指纹 + 结论。客户原始数据行**不出 cell**，
   账本里没有任何一列能装下它。
"""

import pytest

from fd_quant.ledger.store import LedgerStore


@pytest.fixture
def store(tmp_path):
    return LedgerStore(tmp_path / "ledger.db")


def v2_payload(subject: str, metric_ref: str) -> dict:
    return {
        "schema_version": "2", "as_of": "2026-05-01",
        "subject": subject, "subject_kind": "store", "action": "expand",
        "preconditions": [{"claim": "月均流水 ≥ 38 万", "metric_ref": metric_ref,
                           "comparator": "gte", "threshold": 380000,
                           "as_of": "2026-05-01"}],
        "settlement_rule_id": "metric_threshold",
        "expected_outcome": {"metric_ref": metric_ref, "comparator": "gte",
                             "threshold": 0.42,
                             "horizon": {"unit": "month", "count": 3},
                             "settlement_source": "caller_supplied_series"},
        "basis": "前提已核对。", "evidence": [],
        "producer": {"pack": "wire-erp", "skill": "store-expansion"},
    }


def record_for(store, caller, subject):
    return store.record(caller=caller, as_of="2026-05-01", symbol=subject,
                        pack="wire-erp", skill="store-expansion",
                        payload=v2_payload(subject, "cell/store_gross_margin"),
                        evidence=[])["decision_id"]


def test_business_settlements_are_caller_isolated(store):
    """两个 caller 对同一 metric_ref 各结算一次，互不可见。"""
    a = record_for(store, "customer-A", "store-A")
    b = record_for(store, "customer-B", "store-B")
    for did in (a, b):
        store.settle_metric_threshold(
            decision_id=did, dates=["2026-05-31"], values=[0.50],
            as_of="2026-06-30", comparator="gte", threshold=0.42)

    rows_a = store.retrieve(symbol="store-A", as_of="2026-12-31", n_same=5,
                            m_cross=0, caller="customer-A")
    rows_b = store.retrieve(symbol="store-B", as_of="2026-12-31", n_same=5,
                            m_cross=0, caller="customer-B")

    assert [x["decision_id"] for x in rows_a["same_symbol"]] == [a]
    assert [x["decision_id"] for x in rows_b["same_symbol"]] == [b]
    all_a = {x["decision_id"] for x in rows_a["same_symbol"] + rows_a["cross_symbol"]}
    assert b not in all_a, "customer-A 不得看到 customer-B 的结算"
    all_b = {x["decision_id"] for x in rows_b["same_symbol"] + rows_b["cross_symbol"]}
    assert a not in all_b


def test_ledger_row_has_no_customer_raw_values(store):
    """落库行只含 metric_ref + 指纹 + 结论；检查每一列确认没有客户原始值列。"""
    did = record_for(store, "customer-A", "store-A")
    store.settle_metric_threshold(
        decision_id=did, dates=["2026-05-31", "2026-06-30"], values=[0.50, 0.45],
        as_of="2026-06-30", comparator="gte", threshold=0.42)

    import sqlite3
    con = sqlite3.connect(store.db_path)
    con.row_factory = sqlite3.Row
    row = dict(con.execute(
        "SELECT * FROM settlements WHERE decision_id = ?", (did,)).fetchone())
    con.close()

    # 指纹在、结论在
    assert row["series_fingerprint"], "指纹是复核依据，必须在"
    assert row["verdict"] == "hit"
    assert row["metric_ref"] == "cell/store_gross_margin"
    # 客户原始序列的逐点值没有落库列可装：值只有结算时点的 observed_value
    assert row["observed_value"] == pytest.approx(0.45)
    # 粗查：任何列都不该是"整条序列"（列表/长 JSON）
    longish = [k for k, v in row.items()
               if isinstance(v, str) and len(v) > 200]
    assert longish == [], f"疑似原始数据落库：{longish}"
