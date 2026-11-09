"""两种结算口径在聚合里不得合算（generalize-decision-contract 任务 8.2）。

价格口径问"跑赢基准多少"，阈值口径问"指标有没有达到阈值"。把它们合成一个胜率，
那个胜率就没有主语了——而且 `insufficient_data`（不知道）会被平摊成"判断错了"。
"""

from datetime import datetime, timezone

import pytest

from fd_quant.ledger.store import LedgerStore

TODAY = datetime.now(timezone.utc).date().isoformat()


@pytest.fixture
def store(tmp_path):
    return LedgerStore(tmp_path / "ledger.db")


def _record(store, *, subject, rule):
    payload = {
        "schema_version": "2",
        "as_of": "2026-05-01",
        "subject": subject,
        "subject_kind": "store",
        "action": "expand",
        "preconditions": [],
        "settlement_rule_id": rule,
        "basis": "前提已核对。",
        "evidence": [],
        "producer": {"pack": "wire-erp", "skill": "store-expansion"},
        "expected_outcome": {
            "metric_ref": subject, "comparator": "gte", "threshold": 0.42,
            "horizon": {"unit": "month", "count": 3},
            "settlement_source": "caller_supplied_series",
        },
    }
    return store.record(caller="alice", as_of="2026-05-01", symbol=subject,
                        pack="wire-erp", skill="store-expansion",
                        payload=payload, evidence=[])["decision_id"]


def _settle_threshold(store, did, value):
    return store.settle_metric_threshold(
        decision_id=did, dates=["2026-05-31"], values=[value],
        as_of="2026-06-30", comparator="gte", threshold=0.42)


def test_threshold_settlements_are_split_into_their_own_rule_bucket(store):
    hit_did = _record(store, subject="store-a", rule="metric_threshold")
    _settle_threshold(store, hit_did, 0.50)
    miss_did = _record(store, subject="store-b", rule="metric_threshold")
    _settle_threshold(store, miss_did, 0.10)

    out = store.stats_by_pack(caller="alice", as_of=TODAY)
    row = out["rows"][0]

    bucket = row["by_rule"]["metric_threshold"]
    assert bucket["samples"] == 2
    assert bucket["hits"] == 1 and bucket["misses"] == 1
    assert bucket["hit_rate"] == pytest.approx(0.5)


def test_threshold_hits_do_not_leak_into_alpha_win_rate(store):
    """只有阈值口径的结算时，alpha 相关字段必须为 null 且 alpha_samples=0。

    若把无 alpha 的行当 0 参与均值，这里会得到 mean_alpha=-0.0 与 win_rate=0，
    看起来像"这个 pack 全输了"，而实际是"这个 pack 用的是另一种口径"。
    """
    did = _record(store, subject="store-c", rule="metric_threshold")
    _settle_threshold(store, did, 0.50)

    row = store.stats_by_pack(caller="alice", as_of=TODAY)["rows"][0]

    assert row["samples"] == 1, "总样本数仍应包含阈值口径"
    assert row["alpha_samples"] == 0
    assert row["mean_alpha"] is None
    assert row["win_rate"] is None, "不得把阈值口径算成 0 胜率"
    assert row["wins"] == 0


def test_insufficient_data_is_counted_separately_not_as_miss(store):
    """insufficient_data 既不是命中也不是未命中；命中率的分母不含它。"""
    for i, value in enumerate((0.50, 0.10, None)):
        did = _record(store, subject=f"store-d{i}", rule="metric_threshold")
        _settle_threshold(store, did, value)

    bucket = store.stats_by_pack(caller="alice", as_of=TODAY)["rows"][0]["by_rule"][
        "metric_threshold"]

    assert bucket["samples"] == 3
    assert bucket["hits"] == 1
    assert bucket["misses"] == 1
    assert bucket["insufficient_data"] == 1
    assert bucket["hit_rate"] == pytest.approx(0.5), "分母只含 hit+miss"


def test_all_insufficient_means_null_hit_rate(store):
    did = _record(store, subject="store-e", rule="metric_threshold")
    _settle_threshold(store, did, None)

    bucket = store.stats_by_pack(caller="alice", as_of=TODAY)["rows"][0]["by_rule"][
        "metric_threshold"]

    assert bucket["insufficient_data"] == 1
    assert bucket["hit_rate"] is None, "没有结论时命中率无定义，不是 0"


def test_mixed_rules_appear_as_two_buckets(store):
    """同一 pack 两种口径并存时，各占一桶，互不稀释。"""
    t_did = _record(store, subject="store-f", rule="metric_threshold")
    _settle_threshold(store, t_did, 0.50)

    # 价格口径记录（v1 Rating）
    rating = {
        "schema_version": "1", "as_of": "2026-05-01", "subject": "600519.SS",
        "side": "buy", "conviction": 0.6, "price_target": None,
        "target_evidence_ids": [], "basis": "理由",
        "evidence": [], "producer": {"pack": "wire-erp", "skill": "store-expansion"},
    }
    a_did = store.record(caller="alice", as_of="2026-05-01", symbol="600519.SS",
                         pack="wire-erp", skill="store-expansion",
                         payload=rating, evidence=[])["decision_id"]
    # 价格口径默认持有 5 个交易日 → 需要 6 个对齐价点，给足再结算。
    axis = ["2026-05-04", "2026-05-05", "2026-05-06", "2026-05-07", "2026-05-08",
            "2026-05-11"]
    dates = ["2026-05-01"] + axis
    store.settle(decision_id=a_did, symbol_dates=dates,
                 symbol_prices=[100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 110.0],
                 benchmark_dates=dates,
                 benchmark_prices=[100.0] * 7, as_of="2026-06-30",
                 benchmark_symbol="SPY")

    row = store.stats_by_pack(caller="alice", as_of=TODAY)["rows"][0]
    by_rule = row["by_rule"]

    assert set(by_rule) == {"alpha_vs_benchmark", "metric_threshold"}
    assert by_rule["alpha_vs_benchmark"]["alpha_samples"] == 1
    assert by_rule["metric_threshold"]["hits"] == 1
    # 顶层 alpha 指标只由价格口径贡献：与价格口径桶逐值一致，且不等于把两种口径
    # 混算出来的数（关系断言，不写死具体收益，避免因价格点调整而误报）。
    alpha_bucket = by_rule["alpha_vs_benchmark"]
    assert row["alpha_samples"] == 1
    assert row["mean_alpha"] == pytest.approx(alpha_bucket["mean_alpha"])
    assert row["win_rate"] == pytest.approx(alpha_bucket["win_rate"])
    assert row["mean_alpha"] > 0, "样本应为正收益，用于区分'被阈值口径稀释'的 0"
