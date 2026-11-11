"""决策一致性检出（add-decision-consistency-and-premise-watch 任务 2.x）。

守住四件：
1. 三类冲突各自归类，不混同（尤其"同证据反向结论"必须与"单纯方向相反"分开）；
2. 时序演进不算冲突；
3. 引用不可解的记录被排除，不参与判定；
4. 超限显式拒绝，绝不静默截断后报"无冲突"。
"""

import pytest

from fd_quant.compute.consistency import (
    DIRECTION_CONFLICT,
    EVIDENCE_CONFLICT,
    PREMISE_CONFLICT,
    comparators_conflict,
    detect_conflicts,
)


AS_OF = "2026-11-10"


def rec(**kw):
    base = {
        "decision_id": "d1",
        "as_of": "2026-11-10",
        "subject_kind": "instrument",
        "subject": "600519.SS",
        "action": "buy",
        "evidence": [{"evidence_id": "e1"}],
        "target_evidence_ids": ["e1"],
        "producer": {"pack": "wire-equity-research", "skill": "equity-research-thesis"},
    }
    base.update(kw)
    return base


# --- 方向互斥 ---------------------------------------------------------------

def test_same_period_opposite_direction_is_a_conflict():
    out = detect_conflicts(as_of=AS_OF, records=[
        rec(decision_id="a", action="buy"),
        rec(decision_id="b", action="sell", evidence=[{"evidence_id": "e2"}],
            target_evidence_ids=["e2"]),
    ])

    assert out["counts"] == {DIRECTION_CONFLICT: 1}
    assert out["conflicts"][0]["parties"][0]["decision_id"] == "a"


def test_neutral_actions_do_not_conflict():
    out = detect_conflicts(as_of=AS_OF, records=[
        rec(decision_id="a", action="buy"),
        rec(decision_id="b", action="hold"),
    ])

    assert out["conflicts"] == []


def test_review_is_absence_not_neutrality():
    """review 是「还没形成判断」，不参与方向判定（与账本 review≠hold 同一条纪律）。"""
    out = detect_conflicts(as_of=AS_OF, records=[
        rec(decision_id="a", action="buy"),
        rec(decision_id="b", action="review"),
    ])

    assert out["conflicts"] == []
    assert out["temporal_evolution"] == []


# --- 证据反向引用单独归类 ---------------------------------------------------

def test_shared_evidence_opposite_conclusion_is_its_own_kind():
    out = detect_conflicts(as_of=AS_OF, records=[
        rec(decision_id="a", action="buy"),
        rec(decision_id="b", action="sell"),   # 同一 e1
    ])

    assert out["counts"] == {EVIDENCE_CONFLICT: 1}
    assert out["conflicts"][0]["shared_evidence_ids"] == ["e1"]


# --- 时序演进不算冲突 -------------------------------------------------------

def test_different_as_of_is_temporal_evolution_not_conflict():
    out = detect_conflicts(as_of=AS_OF, records=[
        rec(decision_id="a", as_of="2026-03-01", action="buy"),
        rec(decision_id="b", as_of="2026-06-01", action="sell"),
    ])

    assert out["conflicts"] == []
    assert len(out["temporal_evolution"]) == 1
    assert out["counts"] == {}


def test_overlapping_windows_still_count_as_conflict():
    out = detect_conflicts(as_of=AS_OF, records=[
        rec(decision_id="a", as_of="2026-03-01", action="buy",
            window=["2026-01-01", "2026-12-31"]),
        rec(decision_id="b", as_of="2026-06-01", action="sell",
            window=["2026-06-01", "2026-12-31"]),
    ])

    assert len(out["conflicts"]) == 1


# --- 引用不可解 -------------------------------------------------------------

def test_unresolvable_reference_excluded_from_judging():
    out = detect_conflicts(as_of=AS_OF, records=[
        rec(decision_id="a", action="buy"),
        rec(decision_id="bad", action="sell", target_evidence_ids=["ghost"]),
    ])

    assert out["conflicts"] == []
    assert out["unresolvable"][0]["decision_id"] == "bad"
    assert out["unresolvable"][0]["unresolved_ids"] == ["ghost"]


# --- 前提矛盾 ---------------------------------------------------------------

def test_contradictory_thresholds_detected():
    out = detect_conflicts(as_of=AS_OF, records=[
        rec(decision_id="a", preconditions=[
            {"metric_ref": "financials.gross_margin", "comparator": "gt", "threshold": 0.5}]),
        rec(decision_id="b", action="buy", preconditions=[
            {"metric_ref": "financials.gross_margin", "comparator": "lt", "threshold": 0.4}],
            evidence=[{"evidence_id": "e2"}], target_evidence_ids=["e2"]),
    ])

    assert out["counts"].get(PREMISE_CONFLICT) == 1
    assert out["conflicts"][0]["metric_ref"] == "financials.gross_margin"


@pytest.mark.parametrize("c1,t1,c2,t2,expect", [
    ("gt", 5, "lt", 3, True),      # 5<x<3 空
    ("gt", 5, "lt", 6, False),     # 有交集
    ("gte", 5, "lte", 5, False),   # 恰好等于 5
    ("gt", 5, "lte", 5, True),     # 边界不闭合
    ("eq", 5, "gt", 5, True),
    ("eq", 5, "gte", 5, False),
    ("eq", 5, "eq", 5, False),
    ("eq", 5, "eq", 6, True),
    ("neq", 5, "eq", 5, True),
    ("neq", 5, "gt", 1, False),
])
def test_comparator_interval_logic(c1, t1, c2, t2, expect):
    assert comparators_conflict(c1, t1, c2, t2) is expect


# --- 上限与确定性 -----------------------------------------------------------

def test_over_limit_is_refused_not_truncated():
    with pytest.raises(ValueError) as exc:
        detect_conflicts(as_of=AS_OF, records=[rec(decision_id=str(i)) for i in range(5)], max_records=4)

    assert "不静默截断" in str(exc.value)


def test_output_is_deterministic():
    payload = [
        rec(decision_id="a", action="buy"),
        rec(decision_id="b", action="sell"),
        rec(decision_id="c", as_of="2026-01-01", action="sell"),
    ]
    first = detect_conflicts(payload, as_of=AS_OF)
    second = detect_conflicts(payload, as_of=AS_OF)

    assert first == second


def test_no_adjudication_fields_present():
    """输出里不得出现建议、排序或"谁更可信"。"""
    out = detect_conflicts(as_of=AS_OF, records=[rec(decision_id="a"), rec(decision_id="b", action="sell")])

    flat = str(out)
    for banned in ("suggest", "recommend", "winner", "confidence_rank"):
        assert banned not in flat


# --- 计算面合规（quant-compute-plane）----------------------------------------

def test_as_of_is_required():
    with pytest.raises(TypeError):
        detect_conflicts([])  # type: ignore[call-arg]


def test_future_records_are_refused_and_named():
    """晚于口径日的记录必须拒绝并点名，不得静默丢弃。

    静默丢弃会制造一个"当时并不存在"的同行评审：用未来的结论去判当时的冲突，
    结论看起来完全正常，但时间线是错的。
    """
    payload = [
        rec(decision_id="a", action="buy"),
        rec(decision_id="future", as_of="2099-01-01", action="sell"),
    ]
    with pytest.raises(ValueError) as exc:
        detect_conflicts(payload, as_of="2026-11-10")

    assert "2099-01-01" in str(exc.value)
    assert "拒绝" in str(exc.value)


def test_evidence_and_input_range_returned():
    out = detect_conflicts([rec()], as_of="2026-11-10")

    assert out["as_of"] == "2026-11-10"
    assert out["input_range"] == ["2026-11-10", "2026-11-10"]
    # 统一外壳：快照型工具的输入起止为 null（本工具吃记录集合，不吃单条序列）
    assert out["input_start"] is None and out["input_end"] is None
    ids = [e["evidence_id"] for e in out["evidence"]]
    assert len(ids) == len(set(ids))
    assert all("conflicts" in i for i in ids)
    assert all(e["as_of"] == "2026-11-10" for e in out["evidence"])
    assert all(e["source"].startswith("fd-quant:") for e in out["evidence"])
    assert all(set(e) == {"evidence_id", "label", "value", "unit", "as_of", "source"}
               for e in out["evidence"])


# --- 方向判不了的记录必须点名（不是"没冲突"）-----------------------------------

def test_unregistered_subject_kind_is_named_not_silently_clean():
    """未登记的 subject_kind 会让方向判定整个失效。

    此前的行为：两条 buy/sell 记录默默进了 alive，`records_judged=2`、
    `conflicts=[]` —— 一份**什么都没扫的绿报告**。现在必须点名，并让
    `direction_judged` 与 `records_judged` 拉开差距。
    """
    out = detect_conflicts(as_of=AS_OF, records=[
        rec(decision_id="a", action="buy", subject_kind="stock"),
        rec(decision_id="b", action="sell", subject_kind="stock",
            evidence=[{"evidence_id": "e2"}], target_evidence_ids=["e2"]),
    ])

    assert out["conflicts"] == []
    assert out["direction_judged"] == 0
    assert len(out["unjudged"]) == 2
    assert {u["decision_id"] for u in out["unjudged"]} == {"a", "b"}
    assert "not on a registered axis" in out["unjudged"][0]["reason"]
    # 计数器必须自证"没真判"：judged 含进来了的，direction_judged 才是判了的
    assert out["records_judged"] == 2


def test_neutral_actions_are_judged_not_unjudged():
    """hold / review 是**有意的中性**（极性 None 但登记在案），不算判不了。"""
    out = detect_conflicts(as_of=AS_OF, records=[
        rec(decision_id="a", action="hold"),
        rec(decision_id="b", action="review", price_target=None),
    ])

    assert out["unjudged"] == []
    assert out["direction_judged"] == 2


def test_unregistered_kind_still_participates_in_premise_conflicts():
    """判不了方向 ≠ 踢出分析：前提矛盾不依赖极性，未登记 kind 也要能检出。"""
    out = detect_conflicts(as_of=AS_OF, records=[
        rec(decision_id="a", action="buy", subject_kind="stock", preconditions=[
            {"metric_ref": "m.roe", "comparator": "gt", "threshold": 5}]),
        rec(decision_id="b", action="sell", subject_kind="stock",
            evidence=[{"evidence_id": "e2"}], target_evidence_ids=["e2"], preconditions=[
                {"metric_ref": "m.roe", "comparator": "lt", "threshold": 3}]),
    ])

    assert out["counts"].get(PREMISE_CONFLICT) == 1
    assert len(out["unjudged"]) == 2
