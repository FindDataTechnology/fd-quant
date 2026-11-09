"""投研交易舱端到端（add-investment-trading-fleet task 5.1/5.2/5.3）。

跑的是**真实链路**，不打桩：
研究判定 → 组合构建（显式规则）→ 风控闸 → 落账 → 到期结算 → 聚合分两类。

五个岗位的人格稿住在 paas 的 fleet 种子（本仓看不到），但它们调用的每一个能力都在这里。
本文件证明"这些能力拼起来能走完一轮，且在缺前置项时确实不行动"。
"""
from __future__ import annotations

import pytest

from fd_quant.ledger.store import LedgerStore
from fd_quant.portfolio import build_intent
from fd_quant.risk import STATUS_BREACHED, STATUS_OK, STATUS_UNAVAILABLE, evaluate_portfolio_gate

PACK, SKILL = "wire-equity-research", "portfolio-construction"
# 等权两只各 0.4 → 上限设 0.45 才"通过"。夹具若设成 0.35，闸会正确地拦住，
# 那不是闸的 bug 而是夹具没算清楚（第一版就踩了）。
THRESHOLDS = {"per_symbol_max": 0.45, "industry_max": 0.6,
              "drawdown_max": 0.12, "daily_loss_max": 0.04, "max_staleness_days": 30}

RATINGS = [
    {"as_of": "2026-06-30", "subject": "600519", "side": "strong_buy",
     "conviction": 0.8, "price_target": 1800.0},
    {"as_of": "2026-06-30", "subject": "000001", "side": "buy",
     "conviction": 0.6, "price_target": 12.0},
    {"as_of": "2026-06-30", "subject": "000333", "side": "hold",
     "conviction": 0.5, "price_target": 70.0},
]
DATES = ["2026-06-30", "2026-07-01", "2026-07-02"]


@pytest.fixture
def ledger(tmp_path):
    return LedgerStore(tmp_path / "l.db")


def _run_round(ledger, *, thresholds=THRESHOLDS, as_of="2026-06-30",
               price_as_of_max="2026-06-30", sectors=None, equities=None):
    """一轮：研究 → 组合 → 风控闸 → 落账 → 结算 → 聚合。返回各段产出。"""
    # 1) 研究产出（此处就是上面那三条判定）
    # 2) 组合岗：按显式规则压权重
    intent = build_intent(RATINGS, {"kind": "equal_weight", "cash_floor": 0.2},
                          as_of=as_of, producer={"pack": PACK, "skill": SKILL})
    # 3) 执行岗：过闸（这是"不许乱下单"的唯一闸门）
    gate = evaluate_portfolio_gate(
        weights=intent.weights, thresholds=thresholds,
        sectors=sectors if sectors is not None else {s: "sector-" + s[0] for s in intent.weights},
        equity_series=equities,
        as_of=as_of, price_as_of_max=price_as_of_max,
    )
    # 4) 只有 ok 才落账（不可用与触达都等于不行动）
    rec = None
    if gate["status"] == STATUS_OK:
        rec = ledger.record(caller="c1", as_of=as_of, portfolio_key=f"{as_of}-eq",
                            pack=PACK, skill=SKILL, payload=intent,
                            record_kind="portfolio")
    return {"intent": intent, "gate": gate, "record": rec}


def _settle(ledger, did):
    return ledger.settle_portfolio(
        decision_id=did,
        legs={"600519": (DATES, [1500.0, 1600.0, 1800.0]),
              "000001": (DATES, [10.0, 11.0, 12.0])},
        benchmark_dates=DATES, benchmark_prices=[100.0, 100.0, 101.0],
        as_of="2026-07-02", benchmark_symbol="000300", holding_days=2,
    )


# ------------------------------------------------- 5.1 一轮完整走完


def test_a_full_round_completes_and_trades_nothing_real(ledger):
    out = _run_round(ledger)
    assert out["gate"]["status"] == STATUS_OK
    assert out["record"]["created"] is True
    assert len(out["intent"].weights) == 2          # 第三只因 side=hold 未入选
    assert [e["reason"] for e in out["intent"].excluded] == ["side_not_selected"]


def test_round_settles_at_portfolio_level_with_cash_broken_out(ledger):
    rec = _run_round(ledger)["record"]
    s = _settle(ledger, rec["decision_id"])["settlement"]
    # 等权 0.4/0.4：600519 1800/1500-1=0.2；000001 12/10-1=0.2 → 0.16
    assert s["raw_return"] == pytest.approx(0.16)
    assert s["invested_weight"] == pytest.approx(0.8)
    assert s["cash_weight"] == pytest.approx(0.2)
    assert s["cash_return"] == 0.0
    assert s["alpha"] == pytest.approx(0.15)


def test_review_sees_the_record_through_the_pending_list(ledger):
    rec = _run_round(ledger)["record"]
    pending = ledger.list_pending(caller="c1", today="2026-07-01")
    assert pending["pagination"]["total"] == 1
    item = pending["items"][0]
    assert item["decision_id"] == rec["decision_id"]
    assert item["record_kind"] == "portfolio"      # 复盘岗据此选组合结算入口
    assert item["matured"] is False


# ------------------------------------------------- 5.2 缺前置项就不行动


def test_missing_threshold_blocks_the_round(ledger):
    """阈值未配置 → 闸返回 unavailable → 零落账（执行岗不行动）。"""
    out = _run_round(ledger, thresholds={"per_symbol_max": 0.35})
    assert out["gate"]["status"] == STATUS_UNAVAILABLE
    assert "industry_max" in out["gate"]["inputs"]["missing"]
    assert out["record"] is None
    assert ledger.row_counts()["decisions"] == 0


def test_breached_gate_blocks_the_round(ledger):
    """单票超限 → breached → 零落账。"""
    out = _run_round(ledger, thresholds={**THRESHOLDS, "per_symbol_max": 0.1})
    assert out["gate"]["status"] == STATUS_BREACHED
    kinds = [i["kind"] for i in out["gate"]["items"]]
    assert "per_symbol" in kinds
    assert out["record"] is None
    assert ledger.row_counts()["decisions"] == 0


def test_stale_price_blocks_the_round(ledger):
    """价格水位过旧 → breached，且报两侧日期。"""
    out = _run_round(ledger, as_of="2026-08-15", price_as_of_max="2026-06-30")
    assert out["gate"]["status"] == STATUS_BREACHED
    stale = [i for i in out["gate"]["items"] if i["kind"] == "price_staleness"]
    assert stale and "2026-08-15" in stale[0]["note"] and "2026-06-30" in stale[0]["note"]
    assert out["record"] is None


def test_undetermined_sector_does_not_silently_pass_or_block(ledger):
    """行业归属缺失：该项不可判定，其余判据照常 → 闸可以是 ok，但不可判定项可见。"""
    out = _run_round(ledger, sectors={},
                     equities=[("2026-06-01", 100.0), ("2026-06-02", 100.5)])
    assert out["gate"]["status"] == STATUS_OK
    assert any(u["kind"] == "industry_concentration" for u in out["gate"]["undetermined"])


def test_drawdown_blocks_the_round(ledger):
    equities = [("2026-06-01", 100.0), ("2026-06-02", 130.0), ("2026-06-03", 100.0)]
    out = _run_round(ledger, equities=equities)
    assert out["gate"]["status"] == STATUS_BREACHED
    assert any(i["kind"] == "drawdown" for i in out["gate"]["items"])


# ------------------------------------------------- 5.3 聚合分两类


def test_aggregation_keeps_portfolio_and_single_apart(ledger):
    rec = _run_round(ledger)["record"]
    _settle(ledger, rec["decision_id"])
    single = ledger.record(
        caller="c1", as_of="2026-06-30", symbol="600519", pack=PACK, skill=SKILL,
        payload={"schema_version": "1", "as_of": "2026-06-30", "subject": "600519",
                 "side": "buy", "basis": "t",
                 "evidence": [{"evidence_id": "e", "label": "l", "value": 1, "unit": "u",
                               "as_of": "2026-06-30", "source": "s"}],
                 "producer": {"pack": PACK, "skill": SKILL}},
    )
    ledger.settle(decision_id=single["decision_id"], symbol_dates=DATES,
                  symbol_prices=[1500.0, 1600.0, 1800.0], benchmark_dates=DATES,
                  benchmark_prices=[100.0, 100.0, 101.0], as_of="2026-07-02",
                  benchmark_symbol="000300", holding_days=2)

    stats = ledger.stats_by_pack(caller="c1", as_of="2026-07-02", packs=[PACK])
    by_kind = {r["record_kind"]: r for r in stats["rows"]}
    assert set(by_kind) == {"single", "portfolio"}
    assert by_kind["portfolio"]["mean_alpha"] == pytest.approx(0.15)
    assert by_kind["single"]["mean_alpha"] == pytest.approx(0.19)


def test_whole_round_is_reproducible(ledger, tmp_path):
    """同输入跑第二轮：判定与权重逐位相同（确定性链路）。"""
    a = _run_round(ledger)["intent"]
    b = build_intent(RATINGS, {"kind": "equal_weight", "cash_floor": 0.2},
                     as_of="2026-06-30", producer={"pack": PACK, "skill": SKILL})
    assert a.to_json() == b.to_json()
