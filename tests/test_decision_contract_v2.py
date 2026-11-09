"""`Decision` v2 契约（generalize-decision-contract 任务 1.x / 2.x）。

三条必须成立：
1. v1 `Rating` / `TargetWeights` 不受影响，继续可校验；
2. `action` 越界必须指名该 `subject_kind` 的合法集合（只说"非法"等于让人去猜）；
3. 证券类必须给可核前提；经营类可空，给了就逐条可核。
"""

import pytest

from fd_quant.contracts.decision import (
    ACTIONS_BY_KIND,
    Decision,
    check_decision,
)
from fd_quant.contracts.models import check_rating


def decision(**kw):
    base = {
        "schema_version": "2",
        "as_of": "2026-11-10",
        "subject": "600519.SS",
        "subject_kind": "instrument",
        "action": "buy",
        "preconditions": [{
            "claim": "毛利率不低于 42%",
            "metric_ref": "financials.gross_margin",
            "comparator": "gte",
            "threshold": 0.42,
            "as_of": "2026-11-10",
        }],
        "settlement_rule_id": "alpha_vs_benchmark",
        "basis": "经营现金流同比转正，估值中枢高于当期读数。",
        "evidence": [{"evidence_id": "e1", "label": "毛利率", "value": 0.45,
                      "unit": "%", "as_of": "2026-10-31",
                      "source": "mcp__fd-quant__compute_profitability"}],
        "producer": {"pack": "wire-equity-research", "skill": "equity-research-thesis"},
    }
    base.update(kw)
    return base


def paths(errs):
    return [e["path"] for e in errs]


# --- 通过 ---------------------------------------------------------------

def test_instrument_decision_passes():
    assert check_decision(decision()) == []


def test_business_decision_passes():
    d = decision(subject="store-0413", subject_kind="store", action="expand",
                 settlement_rule_id="metric_threshold",
                 expected_outcome={"metric_ref": "cell/store_gross_margin",
                                   "comparator": "gte", "threshold": 0.42,
                                   "horizon": {"unit": "month", "count": 6},
                                   "settlement_source": "caller_supplied_series"})
    assert check_decision(d) == []


def test_dataclass_roundtrip():
    d = Decision(as_of="2026-11-10", subject="store-1", subject_kind="store",
                 action="expand", basis="b", settlement_rule_id="metric_threshold",
                 preconditions=[], producer={"pack": "p", "skill": "s"},
                 expected_outcome={"metric_ref": "cell/margin", "comparator": "gte",
                                   "threshold": 0.4,
                                   "horizon": {"unit": "month", "count": 6},
                                   "settlement_source": "caller_supplied_series"})
    assert check_decision(d.to_dict()) == []


# --- action 分派 ---------------------------------------------------------

def test_action_is_dispatched_by_subject_kind():
    errs = check_decision(decision(subject_kind="store", action="strong_buy"))

    assert paths(errs) == ["action"]
    assert "expand" in errs[0]["reason"], "错误必须给出该 kind 的合法集合"


def test_every_kind_has_at_least_one_legal_action():
    for kind, actions in ACTIONS_BY_KIND.items():
        assert actions, kind


def test_review_is_legal_for_instrument_but_not_business():
    assert check_decision(decision(action="review")) == []
    errs = check_decision(decision(subject_kind="pricing", action="review"))
    assert paths(errs) == ["action"]


# --- 前提 ---------------------------------------------------------------

def test_instrument_requires_preconditions():
    errs = check_decision(decision(preconditions=[]))

    assert "preconditions" in paths(errs)


def test_business_decision_may_omit_preconditions():
    assert check_decision(decision(subject_kind="store", action="close",
                                   preconditions=[])) == []


@pytest.mark.parametrize("missing", ["claim", "metric_ref", "comparator", "threshold", "as_of"])
def test_each_precondition_field_is_required(missing):
    pc = {"claim": "c", "metric_ref": "m", "comparator": "gte",
          "threshold": 1, "as_of": "2026-11-10"}
    pc.pop(missing)
    errs = check_decision(decision(preconditions=[pc]))

    assert any(missing in p for p in paths(errs)), errs


def test_boolean_threshold_rejected():
    pc = {"claim": "c", "metric_ref": "m", "comparator": "gte",
          "threshold": True, "as_of": "2026-11-10"}
    errs = check_decision(decision(preconditions=[pc]))

    assert "preconditions[0].threshold" in paths(errs)


# --- 结算口径 -----------------------------------------------------------

def test_metric_threshold_requires_expected_outcome():
    errs = check_decision(decision(settlement_rule_id="metric_threshold"))

    assert any("expected_outcome" in p for p in paths(errs))


def test_settlement_source_must_be_caller_supplied():
    d = decision(settlement_rule_id="metric_threshold",
                 expected_outcome={"metric_ref": "m", "comparator": "gte",
                                   "threshold": 1, "horizon": {"unit": "month", "count": 3},
                                   "settlement_source": "ledger_fetches_itself"})
    errs = check_decision(d)

    assert any("settlement_source" in p for p in paths(errs))


def test_unknown_settlement_rule_rejected():
    errs = check_decision(decision(settlement_rule_id="vibes"))

    assert "settlement_rule_id" in paths(errs)


# --- 与 v1 的关系 -------------------------------------------------------

def test_v1_rating_still_validates_independently():
    rating = {
        "schema_version": "1", "as_of": "2026-11-10", "subject": "600519.SS",
        "side": "buy", "conviction": 0.6, "price_target": 1800.0,
        "target_evidence_ids": ["e1"], "basis": "理由",
        "evidence": [{"evidence_id": "e1", "label": "x", "value": 1,
                      "unit": "元", "as_of": "2026-11-10", "source": "s"}],
        "producer": {"pack": "p", "skill": "s"},
    }
    assert check_rating(rating) == []


def test_v2_does_not_require_v1_fields():
    """v2 没有 side / price_target，不应因此报错。"""
    errs = check_decision(decision())

    assert "side" not in paths(errs)
    assert "price_target" not in paths(errs)


def test_schema_version_must_be_two():
    errs = check_decision(decision(schema_version="1"))

    assert "schema_version" in paths(errs)
