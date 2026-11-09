"""前提分拣的完整性（add-decision-consistency-and-premise-watch 任务 3.1 / 3.6）。

唯一不可退让的性质：**两条计数之和等于前提总数**。
少一条而不报，那条前提就等于"已监控"——前提失效永远不会响，而所有人以为有人在看。
"""

import pytest

from fd_quant.compute.premise_watch import (
    REASON_BAD_COMPARATOR,
    REASON_BAD_THRESHOLD,
    REASON_MISSING_AS_OF,
    REASON_MISSING_METRIC,
    REASON_METRIC_NOT_RESOLVABLE,
    REASON_NO_WINDOW,
    REASON_NOT_A_MAPPING,
    classify_preconditions,
)


def pc(**kw):
    base = {
        "claim": "毛利率不低于 42%",
        "metric_ref": "financials.gross_margin",
        "comparator": "gte",
        "threshold": 0.42,
        "as_of": "2026-11-10",
    }
    base.update(kw)
    return base


AS_OF = "2026-11-10"
HORIZON = ["2026-11-10", "2027-05-10"]


def test_full_precondition_is_monitorable():
    out = classify_preconditions(as_of=AS_OF, preconditions=[pc()], time_horizon=HORIZON)

    assert out["counts"] == {"total": 1, "monitorable": 1, "unmonitorable": 0}
    assert out["complete"] is True
    entry = out["monitorable"][0]
    assert entry["metric_ref"] == "financials.gross_margin"
    assert entry["window"] == HORIZON


def test_missing_metric_ref_goes_to_unmonitorable():
    out = classify_preconditions(as_of=AS_OF, preconditions=[pc(metric_ref="  ")], time_horizon=HORIZON)

    assert out["unmonitorable"][0]["reason"] == REASON_MISSING_METRIC
    assert out["monitorable"] == []


def test_unresolvable_metric_flagged_when_allowlist_given():
    out = classify_preconditions(
        [pc(metric_ref="ghost.metric")], as_of=AS_OF, time_horizon=HORIZON,
        resolvable_metrics=["financials.gross_margin"],
    )

    assert out["unmonitorable"][0]["reason"] == REASON_METRIC_NOT_RESOLVABLE


def test_invalid_comparator_rejected_with_legal_set():
    out = classify_preconditions(as_of=AS_OF, preconditions=[pc(comparator="≈")], time_horizon=HORIZON)

    entry = out["unmonitorable"][0]
    assert entry["reason"] == REASON_BAD_COMPARATOR
    assert "gte" in entry["detail"], "错误必须给出合法集合，不能只说非法"


def test_boolean_threshold_is_not_a_number():
    """Python 里 True 是 int 的子类——不排除它会让阈值 1 混进来。"""
    out = classify_preconditions(as_of=AS_OF, preconditions=[pc(threshold=True)], time_horizon=HORIZON)

    assert out["unmonitorable"][0]["reason"] == REASON_BAD_THRESHOLD


def test_missing_as_of_rejected():
    out = classify_preconditions(as_of=AS_OF, preconditions=[pc(as_of="")], time_horizon=HORIZON)

    assert out["unmonitorable"][0]["reason"] == REASON_MISSING_AS_OF


def test_no_window_is_refused_not_defaulted_to_infinite():
    """不给 time_horizon 就不猜窗口——无限窗永远不会触发，比不可监控更糟。"""
    out = classify_preconditions(as_of=AS_OF, preconditions=[pc()])

    assert out["unmonitorable"][0]["reason"] == REASON_NO_WINDOW
    assert "不默认取无限窗" in out["unmonitorable"][0]["detail"]


def test_non_mapping_item_is_reported_not_skipped():
    out = classify_preconditions(as_of=AS_OF, preconditions=["毛利率要好"], time_horizon=HORIZON)

    assert out["unmonitorable"][0]["reason"] == REASON_NOT_A_MAPPING


def test_counts_sum_equals_total_for_mixed_input():
    items = [
        pc(),                                              # 可监控
        pc(metric_ref=None),                               # 缺指标
        pc(comparator="between"),                          # 比较子非法
        "不是对象",                                          # 非对象
        pc(),                                              # 可监控
    ]
    out = classify_preconditions(items, as_of=AS_OF, time_horizon=HORIZON)

    c = out["counts"]
    assert c["total"] == 5
    assert c["monitorable"] == 2
    assert c["unmonitorable"] == 3
    assert c["monitorable"] + c["unmonitorable"] == c["total"]
    assert out["complete"] is True


def test_index_refers_to_original_position():
    items = [pc(), pc(metric_ref=None)]
    out = classify_preconditions(items, as_of=AS_OF, time_horizon=HORIZON)

    assert out["monitorable"][0]["index"] == 0
    assert out["unmonitorable"][0]["index"] == 1


def test_empty_preconditions_is_trivially_complete():
    out = classify_preconditions(as_of=AS_OF, preconditions=[], time_horizon=HORIZON)

    assert out["counts"]["total"] == 0
    assert out["complete"] is True


def test_window_must_be_ordered():
    out = classify_preconditions(
        [pc()], as_of=AS_OF, time_horizon=["2027-05-10", "2026-11-10"],
    )

    assert out["unmonitorable"][0]["reason"] == REASON_NO_WINDOW


@pytest.mark.parametrize("comparator", ["lt", "lte", "gt", "gte", "eq", "neq"])
def test_all_legal_comparators_accepted(comparator):
    out = classify_preconditions(as_of=AS_OF, preconditions=[pc(comparator=comparator)], time_horizon=HORIZON)

    assert out["counts"]["monitorable"] == 1


def test_as_of_required_and_future_premise_refused():
    with pytest.raises(TypeError):
        classify_preconditions([])  # type: ignore[call-arg]

    with pytest.raises(ValueError) as exc:
        classify_preconditions([pc(as_of="2099-01-01")], as_of=AS_OF,
                               time_horizon=HORIZON)
    assert "2099-01-01" in str(exc.value)


def test_evidence_returned():
    out = classify_preconditions([pc()], as_of=AS_OF, time_horizon=HORIZON)

    assert out["as_of"] == AS_OF
    ids = [e["evidence_id"] for e in out["evidence"]]
    assert len(ids) == 2 and len(set(ids)) == 2
    assert all("preconditions_monitorable" in i for i in ids)
    assert all(e["source"].startswith("fd-quant:") for e in out["evidence"])
