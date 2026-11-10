"""端到端：评级 → 意图 → 权重 → 落账 → 组合结算 → 聚合（task 5.3 的计算侧一半）。

这条链是本 change 的存在理由：没有它，`Rating` 与 `TargetWeights` 之间那一步只存在于
人的临场算术里。这里逐段证明它真的接上了，且每一步都能被第三方照着重算。

引擎侧另一半（把导出权重喂 `replay_weights` 跑出成交与净值）在 fd-paper-trading 的
`tests/unit/test_portfolio_intent_consumption.py`——那边才装得下交易引擎。
"""
from __future__ import annotations

import pytest

from fd_quant.contracts.models import (
    PortfolioIntent,
    TargetWeights,
    validate_portfolio_intent,
    validate_target_weights,
)
from fd_quant.ledger.store import LedgerStore
from fd_quant.portfolio import build_intent

# 三条构造评级：两只入选、一只因 side 被排除。
RATINGS = [
    {"as_of": "2026-06-30", "subject": "600519", "side": "strong_buy",
     "conviction": 0.8, "price_target": 1800.0},
    {"as_of": "2026-06-30", "subject": "000001", "side": "buy",
     "conviction": 0.6, "price_target": 12.0},
    {"as_of": "2026-06-30", "subject": "000333", "side": "hold",
     "conviction": 0.5, "price_target": 70.0},
]
PRODUCER = {"pack": "wire-equity-research", "skill": "portfolio-construction"}


def _intent(kind="equal_weight", cash=0.2, **extra):
    rules = {"kind": kind, "cash_floor": cash, **extra}
    return build_intent(RATINGS, rules, as_of="2026-06-30",
                        producer=PRODUCER, rebalance={"every_n_bars": 10})


# --------------------------------------------------- 链的每一段都能独立复核


def test_chain_from_ratings_to_executable_weights():
    intent = _intent()
    assert validate_portfolio_intent(intent)["valid"] is True

    # 三段出处都可解析：derived_from 三条 = 2 入选 + 1 排除
    assert len(intent.derived_from) == 3
    assert len(intent.excluded) == 1
    assert intent.excluded[0]["reason"] == "side_not_selected"
    assert len(intent.derived_from) - len(intent.excluded) == len(intent.weights)

    # 无损导出到执行侧契约
    tw = intent.to_target_weights()
    assert isinstance(tw, TargetWeights)
    assert validate_target_weights(tw)["valid"] is True
    assert tw.weights == intent.weights
    assert tw.cash_floor == intent.cash_floor
    assert tw.universe == intent.universe


def test_third_party_can_recompute_the_weights_from_the_instance_alone():
    """只凭 rules + 入选标的，照 D2 的算式重算，必须与实例里的权重逐位一致。"""
    intent = _intent(kind="conviction_weighted", cash=0.2, conviction_floor=0.3)
    conv = {"600519": 0.8, "000001": 0.6}
    alloc = 1.0 - intent.cash_floor
    total = sum(conv.values())
    recomputed = {s: round(alloc * c / total, 12) for s, c in conv.items()}
    assert intent.weights == recomputed


def test_upside_rule_uses_explicit_reference_and_records_its_caliber_day():
    intent = _intent(
        kind="target_upside_weighted", cash=0.1, conviction_floor=0.3,
        price_reference={"600519": 1500.0, "000001": 10.0},
        price_reference_as_of="2026-06-30",
    )
    # u = 1800/1500-1 = 0.2；12/10-1 = 0.2 → 等分 0.9
    assert intent.weights == {"600519": 0.45, "000001": 0.45}
    assert "price_reference_as_of" in intent.rules


# --------------------------------------------------- 落账 → 结算 → 聚合


@pytest.fixture
def store(tmp_path):
    return LedgerStore(tmp_path / "ledger.db")


DATES = ["2026-06-30", "2026-07-01", "2026-07-02"]


def _settle(store, did, legs=None, **kw):
    return store.settle_portfolio(
        decision_id=did,
        legs=legs or {
            "600519": (DATES, [1500.0, 1600.0, 1800.0]),
            "000001": (DATES, [10.0, 11.0, 12.0]),
        },
        benchmark_dates=DATES, benchmark_prices=[100.0, 100.0, 101.0],
        as_of="2026-07-02", benchmark_symbol="000300",
        holding_days=2, **kw,
    )


def test_intent_lands_in_the_ledger_and_settles_at_portfolio_level(store):
    intent = _intent()
    rec = store.record(caller="c1", as_of="2026-06-30", portfolio_key="2026-06-30-eq",
                       pack=PRODUCER["pack"], skill=PRODUCER["skill"],
                       payload=intent, record_kind="portfolio")
    assert rec["created"] is True

    out = _settle(store, rec["decision_id"])
    s = out["settlement"]
    # 等权 0.4/0.4：600519 1800/1500-1=0.2；000001 12/10-1=0.2 → 0.4*0.2+0.4*0.2 = 0.16
    assert s["raw_return"] == pytest.approx(0.16)
    assert s["invested_weight"] == pytest.approx(0.8)
    assert s["cash_weight"] == pytest.approx(0.2)
    assert s["cash_return"] == 0.0
    assert s["benchmark_return"] == pytest.approx(0.01)
    assert s["alpha"] == pytest.approx(0.15)
    assert set(s["leg_returns"]) == {"600519", "000001"}
    assert store.get(rec["decision_id"])["status"] == "settled"


def test_every_weight_is_traceable_to_a_named_rating(store):
    intent = _intent()
    for sym in intent.weights:
        assert sym in intent.universe
    # 权重里的每个标的都能追到至少一条 derived_from 覆盖的评级标的
    subjects = {r["subject"] for r in RATINGS if r["side"] in ("buy", "strong_buy")}
    assert set(intent.weights) == subjects


def test_aggregation_reports_single_and_portfolio_separately(store):
    """同一 pack 下既有单票结算也有组合结算 → 平均 alpha 必须分两组，不混算。"""
    intent = _intent()
    p = store.record(caller="c1", as_of="2026-06-30", portfolio_key="k1",
                     pack=PRODUCER["pack"], skill=PRODUCER["skill"],
                     payload=intent, record_kind="portfolio")
    _settle(store, p["decision_id"])

    single = store.record(
        caller="c1", as_of="2026-06-30", symbol="600519",
        pack=PRODUCER["pack"], skill=PRODUCER["skill"],
        payload={"schema_version": "1", "as_of": "2026-06-30", "subject": "600519",
                 "side": "buy", "basis": "t",
                 "evidence": [{"evidence_id": "e", "label": "l", "value": 1, "unit": "u",
                               "as_of": "2026-06-30", "source": "s"}],
                 "producer": PRODUCER},
    )
    store.settle(
        decision_id=single["decision_id"],
        symbol_dates=DATES, symbol_prices=[1500.0, 1600.0, 1800.0],
        benchmark_dates=DATES, benchmark_prices=[100.0, 100.0, 101.0],
        as_of="2026-07-02", benchmark_symbol="000300", holding_days=2,
    )

    stats = store.stats_by_pack(caller="c1", as_of="2026-07-02", packs=[PRODUCER["pack"]])
    assert "record_kind" in stats["group_by"]
    kinds = {r["record_kind"] for r in stats["rows"]}
    assert kinds == {"single", "portfolio"}, stats["rows"]
    for row in stats["rows"]:
        assert row["samples"] == 1
    # 两类各自一组，互不平均：组合 alpha 0.15，单票 alpha 0.19
    by_kind = {r["record_kind"]: r for r in stats["rows"]}
    assert by_kind["portfolio"]["mean_alpha"] == pytest.approx(0.15)
    assert by_kind["single"]["mean_alpha"] == pytest.approx(0.19)


def test_portfolio_intent_survives_the_roundtrip_and_stays_valid(store):
    intent = _intent(kind="conviction_weighted", cash=0.2, conviction_floor=0.3)
    rec = store.record(caller="c1", as_of="2026-06-30", portfolio_key="k2",
                       pack=PRODUCER["pack"], skill=PRODUCER["skill"],
                       payload=intent, record_kind="portfolio")
    got = store.get(rec["decision_id"])["payload"]
    assert PortfolioIntent.from_dict(got).to_dict() == intent.to_dict()
    assert validate_portfolio_intent(got)["valid"] is True


# --------------------------------------------------- 参考价带出处（task 5.4）

REF_OK = {
    "schema_version": "1", "source_id": "astock_daily", "symbol": "600519",
    "adjustment": "qfq", "adjustment_anchor": "2026-07-10",
    "as_of_max": "2026-06-30", "window_start": "2026-01-01",
    "window_end": "2026-06-30", "row_count": 120,
    "fingerprint": "a" * 64,
}


def test_reference_price_may_carry_its_series_ref():
    """参考价可以带出处：`{price, series_ref}`。

    出处随 `rules` 逐字进 intent，所以第三方能回答"这个参考价是哪一路、哪天的水位"——
    这正是 task 5.4 要的那半件事（4.1 落地后不再需要占位）。
    """
    intent = _intent(
        kind="target_upside_weighted", cash=0.1, conviction_floor=0.3,
        price_reference={
            "600519": {"price": 1500.0, "series_ref": REF_OK},
            "000001": 10.0,          # 老写法并存：不带出处也照旧工作
        },
        price_reference_as_of="2026-06-30",
    )
    # 与只给裸数字时权重逐位一致——带出处不改变算式
    assert intent.weights == {"600519": 0.45, "000001": 0.45}
    assert intent.rules["price_reference"]["600519"]["series_ref"]["source_id"] == "astock_daily"


def test_a_ref_with_the_wrong_shape_is_refused_not_ignored():
    """带了出处但不合法 → 拒绝。静默退回"只有价格"会让报告看不出差别。"""
    from fd_quant.portfolio import IntentBuildError

    bad = dict(REF_OK)
    bad.pop("fingerprint")
    with pytest.raises(IntentBuildError, match="series_ref"):
        _intent(kind="target_upside_weighted", cash=0.1, conviction_floor=0.3,
                price_reference={"600519": {"price": 1500.0, "series_ref": bad},
                                 "000001": 10.0})


def test_a_ref_without_a_price_is_refused_not_treated_as_a_gap():
    """有出处没价格 = 参考价条目坏了，不是"行情缺口"。

    两者处置不同：缺口可以在下一轮补上（本轮排除该标的），条目坏则说明上游
    拼错了对象——静默降级成 data_gap 会让"谁把条目写错了"永远查不出来。
    """
    from fd_quant.portfolio import IntentBuildError

    with pytest.raises(IntentBuildError, match="price_reference"):
        _intent(kind="target_upside_weighted", cash=0.1, conviction_floor=0.3,
                price_reference={"600519": {"series_ref": REF_OK},
                                 "000001": 10.0})
