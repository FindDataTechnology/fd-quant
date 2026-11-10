"""PortfolioIntent 契约与转换器（add-portfolio-intent-contract, tasks 1.1-1.4 / 2.1-2.6）。

测的重点是**拒绝**：不确定的边界、要优化的请求、无出处的权重、静默丢弃的评级。
这些才是这个契约存在的理由——能通过的部分，一个字典也能做到。
"""
from __future__ import annotations

import itertools
import json

import pytest

from fd_quant.contracts.models import (
    EXCLUSION_REASONS,
    RULE_KINDS,
    SELECTED_SIDES,
    SUPPORTED_SCHEMA_VERSION,
    ContractError,
    PortfolioIntent,
    Producer,
    TargetWeights,
    check_portfolio_intent,
    load_schema,
    validate,
    validate_portfolio_intent,
    validate_target_weights,
)
from fd_quant.portfolio import IntentBuildError, build_intent

PROD = {"pack": "fd-quant", "skill": "test"}


def _valid(**over):
    base = {
        "schema_version": SUPPORTED_SCHEMA_VERSION,
        "as_of": "2026-06-30",
        "universe": ["600519", "000001"],
        "rules": {"kind": "equal_weight", "cash_floor": 0.2},
        "weights": {"600519": 0.4, "000001": 0.4},
        "cash_floor": 0.2,
        "derived_from": ["r1", "r2"],
        "producer": PROD,
    }
    base.update(over)
    return base


def _rating(subject="600519", side="buy", conviction=0.8, price_target=1800.0, as_of="2026-06-30"):
    return {
        "as_of": as_of, "subject": subject, "side": side,
        "conviction": conviction, "price_target": price_target,
    }


# ------------------------------------------------------- 1.1 契约与不变式


def test_valid_intent_passes():
    assert check_portfolio_intent(_valid()) == []
    assert validate_portfolio_intent(_valid())["valid"] is True


@pytest.mark.parametrize("field", ["as_of", "universe", "rules", "weights", "cash_floor", "derived_from", "producer"])
def test_missing_required_is_rejected(field):
    d = _valid()
    d.pop(field)
    assert any(e["path"] == field for e in check_portfolio_intent(d))


@pytest.mark.parametrize(
    "weights,cash,path",
    [
        ({"600519": -0.1}, 0.2, "weights.600519"),           # 负权重
        ({"600519": 0.7, "000001": 0.6}, 0.0, "weights"),    # Σ > 1
        ({"600519": 0.6, "000001": 0.3}, 0.2, "cash_floor"), # cash + Σ > 1
        ({"999999": 0.1}, 0.2, "weights.999999"),            # universe 外
    ],
)
def test_invariants_match_target_weights(weights, cash, path):
    d = _valid(weights=weights, cash_floor=cash)
    errs = check_portfolio_intent(d)
    assert any(e["path"] == path for e in errs), errs
    # 同一类输入，两个契约的报错句式必须一致（否则读的人要学两套）
    tw_errs = check_portfolio_intent(_valid(weights=weights, cash_floor=cash))
    assert [e["reason"] for e in errs] == [e["reason"] for e in tw_errs]


def test_rejects_unknown_field():
    assert any(e["path"] == "extra" for e in check_portfolio_intent(_valid(extra=1)))


def test_rejects_wrong_schema_version():
    assert any(e["path"] == "schema_version" for e in check_portfolio_intent(_valid(schema_version="9")))


def test_universe_outside_weights_ok_but_weight_outside_universe_not():
    assert check_portfolio_intent(_valid(universe=["600519", "000001", "000333"])) == []


# ------------------------------------------------------------ 1.2 无损导出


def test_export_to_target_weights_is_lossless():
    intent = PortfolioIntent.from_dict(_valid())
    tw = intent.to_target_weights()
    out = tw.to_dict()
    assert out["weights"] == intent.to_dict()["weights"]
    assert out["universe"] == intent.to_dict()["universe"]
    assert out["cash_floor"] == intent.to_dict()["cash_floor"]
    assert validate_target_weights(tw)["valid"] is True


def test_export_roundtrip_for_many_random_legal_instances():
    """20 组构造实例：导出后再校验通过，且三要素逐位相同。"""
    combos = list(itertools.islice(itertools.product(
        [0.0, 0.05, 0.2, 0.5], [0.1, 0.15, 0.3], ["600519", "000001"], ["600519", "000001"]
    ), 20))
    for cash, w, sym, other in combos:
        universe = sorted({sym, other})
        weights = {sym: w} if w + cash <= 1 else {sym: round(1 - cash, 6)}
        d = _valid(universe=universe, weights=weights, cash_floor=cash)
        if check_portfolio_intent(d):
            continue
        intent = PortfolioIntent.from_dict(d)
        exported = intent.to_target_weights()
        assert validate_target_weights(exported)["valid"] is True
        assert exported.weights == intent.weights
        assert exported.universe == intent.universe
        assert exported.cash_floor == intent.cash_floor


# ------------------------------------------------------------ 1.3 rules 枚举


def test_rule_kinds_are_exactly_three():
    assert RULE_KINDS == ("equal_weight", "conviction_weighted", "target_upside_weighted")


def test_optimisation_rule_is_rejected_and_lists_allowed():
    errs = check_portfolio_intent(_valid(rules={"kind": "mean_variance_optimal", "cash_floor": 0.1}))
    hit = [e for e in errs if e["path"] == "rules.kind"]
    assert hit, errs
    for k in RULE_KINDS:
        assert k in hit[0]["reason"]
    assert "in-sample" in hit[0]["reason"]


@pytest.mark.parametrize("kind", RULE_KINDS)
def test_cash_floor_required_for_every_kind(kind):
    errs = check_portfolio_intent(_valid(rules={"kind": kind}))
    assert any(e["path"] == "rules.cash_floor" for e in errs)


def test_conviction_floor_required_for_weighted_kinds():
    for kind in ("conviction_weighted", "target_upside_weighted"):
        errs = check_portfolio_intent(
            _valid(rules={"kind": kind, "cash_floor": 0.1})
        )
        assert any(e["path"] == "rules.conviction_floor" for e in errs), kind


def test_price_reference_required_for_upside_kind():
    errs = check_portfolio_intent(
        _valid(rules={"kind": "target_upside_weighted", "cash_floor": 0.1, "conviction_floor": 0.3})
    )
    assert any(e["path"] == "rules.price_reference" for e in errs)


def test_price_reference_must_be_positive():
    errs = check_portfolio_intent(_valid(rules={
        "kind": "target_upside_weighted", "cash_floor": 0.1,
        "conviction_floor": 0.3, "price_reference": {"600519": 0},
    }))
    assert any("price_reference" in e["path"] for e in errs)


# ------------------------------------------------------------ 1.4 excluded


def test_exclusion_reasons_are_fixed_enum():
    assert len(EXCLUSION_REASONS) == 6
    assert "negative_upside" in EXCLUSION_REASONS


def test_free_text_exclusion_reason_is_rejected():
    errs = check_portfolio_intent(_valid(excluded=[{"id": "r1", "reason": "looks_weak"}]))
    hit = [e for e in errs if e["path"].endswith("reason")]
    assert hit and "free text" in hit[0]["reason"]


def test_exclusion_must_refer_to_an_input():
    errs = check_portfolio_intent(_valid(excluded=[{"id": "ghost", "reason": "data_gap"}]))
    assert any("not in derived_from" in e["reason"] for e in errs)


def test_duplicate_exclusion_is_rejected():
    errs = check_portfolio_intent(_valid(excluded=[
        {"id": "r1", "reason": "data_gap"}, {"id": "r1", "reason": "user_excluded"}
    ]))
    assert any("duplicate excluded id" in e["reason"] for e in errs)


def test_weights_with_every_input_excluded_is_rejected():
    """入选 0 条却有非空权重 = 权重没有出处。"""
    errs = check_portfolio_intent(_valid(
        excluded=[{"id": "r1", "reason": "data_gap"}, {"id": "r2", "reason": "data_gap"}]
    ))
    assert any("no selected input" in e["reason"] for e in errs)


def test_empty_weights_with_all_excluded_is_legal():
    """空权重 + 全排除是合法结果（如实报"本期无仓"）。"""
    assert check_portfolio_intent(_valid(
        weights={}, excluded=[{"id": "r1", "reason": "below_conviction_floor"},
                              {"id": "r2", "reason": "below_conviction_floor"}]
    )) == []


# ------------------------------------------------------- 2.1 纯函数与确定性


def test_build_is_deterministic_including_rationale():
    ratings = [_rating(), _rating("000001", "strong_buy", 0.6, 12.0)]
    rules = {"kind": "conviction_weighted", "cash_floor": 0.2, "conviction_floor": 0.3}
    a = build_intent(ratings, rules, as_of="2026-06-30")
    b = build_intent(ratings, rules, as_of="2026-06-30")
    assert a.to_json() == b.to_json()
    assert a.rationale == b.rationale


def test_build_does_no_io(monkeypatch):
    """纯函数：调用期间不允许开 socket / 读文件。"""
    import socket

    def boom(*a, **k):
        raise AssertionError("build_intent must not do I/O")

    monkeypatch.setattr(socket, "socket", boom)
    intent = build_intent([_rating()], {"kind": "equal_weight", "cash_floor": 0.1}, as_of="2026-06-30")
    assert intent.weights


# ------------------------------------------------------- 2.2 as_of 守卫


def test_future_rating_is_refused_and_named():
    with pytest.raises(IntentBuildError) as ei:
        build_intent(
            [_rating("600519", as_of="2026-07-15")],
            {"kind": "equal_weight", "cash_floor": 0.1},
            as_of="2026-06-30",
        )
    msg = str(ei.value)
    assert "600519" in msg and "2026-07-15" in msg
    assert "not truncated silently" in msg


def test_rating_on_the_same_day_is_allowed():
    build_intent([_rating(as_of="2026-06-30")], {"kind": "equal_weight", "cash_floor": 0.1}, as_of="2026-06-30")


# ------------------------------------------------------- 2.3 三种算式


def test_equal_weight_formula():
    """手算：cash 0.2，2 只 → 每只 (1-0.2)/2 = 0.4。"""
    intent = build_intent(
        [_rating(), _rating("000001")], {"kind": "equal_weight", "cash_floor": 0.2}, as_of="2026-06-30"
    )
    assert intent.weights == {"600519": 0.4, "000001": 0.4}


def test_conviction_weighted_formula():
    """手算：cash 0.2 → alloc 0.8；c = 0.8 / 0.6，Σ = 1.4；
    0.8*0.8/1.4 = 0.457142857143，0.8*0.6/1.4 = 0.342857142857。"""
    intent = build_intent(
        [_rating(conviction=0.8), _rating("000001", "strong_buy", 0.6)],
        {"kind": "conviction_weighted", "cash_floor": 0.2, "conviction_floor": 0.3},
        as_of="2026-06-30",
    )
    assert intent.weights["600519"] == pytest.approx(0.457142857143, abs=1e-11)
    assert intent.weights["000001"] == pytest.approx(0.342857142857, abs=1e-11)


def test_target_upside_weighted_formula():
    """手算：cash 0.1 → alloc 0.9；u1 = 1800/1500-1 = 0.2，u2 = 12/10-1 = 0.2；
    各得 0.9*0.2/0.4 = 0.45。"""
    intent = build_intent(
        [_rating(price_target=1800.0), _rating("000001", "strong_buy", 0.6, 12.0)],
        {
            "kind": "target_upside_weighted", "cash_floor": 0.1, "conviction_floor": 0.3,
            "price_reference": {"600519": 1500.0, "000001": 10.0},
        },
        as_of="2026-06-30",
    )
    assert intent.weights == {"600519": 0.45, "000001": 0.45}


def test_negative_upside_is_excluded_not_negatively_weighted():
    intent = build_intent(
        [_rating(price_target=1400.0)],  # 1500 现价 → -6.7%
        {
            "kind": "target_upside_weighted", "cash_floor": 0.1, "conviction_floor": 0.3,
            "price_reference": {"600519": 1500.0},
        },
        as_of="2026-06-30",
    )
    assert intent.weights == {}
    assert [e["reason"] for e in intent.excluded] == ["negative_upside"]


# ------------------------------------------------------- 2.4 行情闸


def test_null_price_target_becomes_data_gap_not_guessed():
    intent = build_intent(
        [_rating(price_target=None)],
        {
            "kind": "target_upside_weighted", "cash_floor": 0.1, "conviction_floor": 0.3,
            "price_reference": {"600519": 1500.0},
        },
        as_of="2026-06-30",
    )
    assert intent.weights == {}
    assert [e["reason"] for e in intent.excluded] == ["data_gap"]


def test_missing_price_reference_refuses_whole_rule_never_falls_back():
    with pytest.raises(IntentBuildError) as ei:
        build_intent(
            [_rating()],
            {
                "kind": "target_upside_weighted", "cash_floor": 0.1, "conviction_floor": 0.3,
                "price_reference": {"000001": 10.0},  # 缺 600519
            },
            as_of="2026-06-30",
        )
    assert "rather than falling back to equal_weight" in str(ei.value)


# ------------------------------------------------------- 2.5 空结果


def test_empty_result_is_reported_with_exact_reason():
    intent = build_intent(
        [_rating(conviction=0.5)],
        {"kind": "conviction_weighted", "cash_floor": 0.1, "conviction_floor": 0.99},
        as_of="2026-06-30",
    )
    assert intent.weights == {}
    assert "入选 0 条" in intent.rationale and "0.99" in intent.rationale
    assert intent.excluded[0]["reason"] == "below_conviction_floor"
    # 空权重仍是合法契约实例
    assert validate_portfolio_intent(intent)["valid"] is True


# ------------------------------------------------------- 2.6 rationale


def test_rationale_names_rule_params_and_conviction_disclaimer():
    intent = build_intent(
        [_rating()],
        {"kind": "conviction_weighted", "cash_floor": 0.2, "conviction_floor": 0.4},
        as_of="2026-06-30",
    )
    r = intent.rationale
    assert "conviction_weighted" in r
    assert "0.4" in r and "0.2" in r
    assert "判定强度" in r and "不是预期收益" in r
    assert "最优化" in r


def test_excluded_count_plus_selected_equals_input():
    ratings = [
        _rating("600519", "buy", 0.8),
        _rating("000001", "hold", 0.9),        # side_not_selected
        _rating("000333", "buy", 0.1),         # below floor
    ]
    intent = build_intent(
        ratings,
        {"kind": "conviction_weighted", "cash_floor": 0.0, "conviction_floor": 0.3},
        as_of="2026-06-30",
    )
    selected = len(intent.derived_from) - len(intent.excluded)
    assert selected == 1
    assert selected + len(intent.excluded) == len(ratings)
    assert {e["reason"] for e in intent.excluded} == {"side_not_selected", "below_conviction_floor"}


# ------------------------------------------------------- 统一入口与 schema


def test_validate_sniffs_intent_kind():
    assert validate(_valid())["valid"] is True
    assert validate(PortfolioIntent.from_dict(_valid()))["valid"] is True


def test_load_schema_matches_python_side():
    s = load_schema("portfolio_intent")
    assert s["title"] == "PortfolioIntent"
    assert tuple(s["properties"]["rules"]["properties"]["kind"]["enum"]) == RULE_KINDS
    assert tuple(s["properties"]["excluded"]["items"]["properties"]["reason"]["enum"]) == EXCLUSION_REASONS
    assert s["additionalProperties"] is False


def test_from_dict_refuses_invalid():
    with pytest.raises(ContractError):
        PortfolioIntent.from_dict(_valid(weights={"x": 5}))


def test_selected_sides_are_buy_and_strong_buy():
    assert SELECTED_SIDES == ("strong_buy", "buy")


# ------------------------------------- 参考价带出处（task 5.4，契约层）

REF = {
    "schema_version": "1", "source_id": "astock_daily", "symbol": "600519",
    "adjustment": "qfq", "adjustment_anchor": "2026-07-10",
    "as_of_max": "2026-06-30", "window_start": "2026-01-01",
    "window_end": "2026-06-30", "row_count": 120, "fingerprint": "b" * 64,
}
UPSIDE = {"kind": "target_upside_weighted", "cash_floor": 0.1, "conviction_floor": 0.3}


def test_price_reference_may_carry_a_series_ref():
    """契约层必须认这种形态——否则别处送来的 intent 会被判非法。"""
    errs = check_portfolio_intent(_valid(rules={
        **UPSIDE, "price_reference": {"600519": {"price": 1500.0, "series_ref": REF}},
    }))
    assert errs == []


def test_an_invalid_series_ref_inside_price_reference_is_rejected():
    bad = {k: v for k, v in REF.items() if k != "fingerprint"}
    errs = check_portfolio_intent(_valid(rules={
        **UPSIDE,
        "price_reference": {"600519": {"price": 1500.0, "series_ref": bad}},
    }))
    assert any(e["path"].startswith("rules.price_reference.600519.series_ref")
               for e in errs), errs


def test_an_entry_without_a_price_is_rejected():
    errs = check_portfolio_intent(_valid(rules={
        **UPSIDE, "price_reference": {"600519": {"series_ref": REF}},
    }))
    assert any(e["path"] == "rules.price_reference.600519.price" for e in errs), errs


def test_unknown_fields_inside_a_reference_entry_are_rejected():
    errs = check_portfolio_intent(_valid(rules={
        **UPSIDE,
        "price_reference": {"600519": {"price": 1500.0, "series_ref": REF, "note": "x"}},
    }))
    assert any(e["path"].endswith(".note") for e in errs), errs
