"""边界回归（task 5.3）：持有期参数化、基准显式覆盖、反思缺失、空账本查询。

四类边界各有断言。它们**不复用**前面的夹具意图：这里的数值刻意取手算容易核对的
整数比（+10% / +2% / +20%），使"三天、五天、十天的窗口各自算对了"是肉眼可验的，
而"实现对但窗口取错"这种错误藏不住。
"""
from __future__ import annotations

import json
import math

import pytest

import ledgerdata as ld
from fd_quant.ledger import LedgerStore, SettlementRefused

#: 单用户本地账本场景的测试调用者（隔离新增维度后必须显式给出）
CALLER = "test-caller"

SYMBOL = "600519.SS"


@pytest.fixture()
def store(tmp_path) -> LedgerStore:
    return LedgerStore(tmp_path / "ledger.db")


def _record(store: LedgerStore, *, as_of: str = "2025-01-06", symbol: str = SYMBOL,
            pack: str = ld.PACK, skill: str = ld.SKILL) -> dict:
    return store.record(caller=CALLER, as_of=as_of, symbol=symbol, pack=pack, skill=skill,
                        payload=ld.rating(as_of, symbol, pack=pack, skill=skill))


# ── 边界 1：持有期参数化（3 / 5 / 10 日）─────────────────────────


@pytest.mark.parametrize("holding_days", [3, 5, 10])
def test_holding_period_is_parameterisable(store, holding_days):
    """持有期 3 / 5 / 10 天各自结算成功，且结算日与持有天数逐例正确。

    价格夹具刻意设为"入场首价 100、出场日 110"：daily 平台段使**只有出场日**的
    价格是 110，故代理价 = 100、出场价 = 110 的唯一解释是"窗口取到了第 N 个交易日"。
    """
    r = _record(store)
    n = holding_days + 1
    sym_dates = ld.dates(n)
    sym_prices = [100.0] * holding_days + [110.0]
    bench_dates = ld.dates(n)
    bench_prices = [1000.0] * holding_days + [1020.0]   # +2%
    s = store.settle(decision_id=r["decision_id"], symbol_dates=sym_dates,
                     symbol_prices=sym_prices, benchmark_dates=bench_dates,
                     benchmark_prices=bench_prices, as_of=sym_dates[-1],
                     holding_days=holding_days)
    assert s["holding_days"] == holding_days
    assert s["settlement_date"] == sym_dates[holding_days]
    assert s["entry_price"] == 100.0 and s["exit_price"] == 110.0
    assert math.isclose(s["raw_return"], 0.10, abs_tol=1e-12)
    assert math.isclose(s["benchmark_return"], 0.02, abs_tol=1e-12)
    assert math.isclose(s["alpha"], 0.08, abs_tol=1e-12)


@pytest.mark.parametrize("holding_days", [3, 5, 10])
def test_each_holding_period_needs_exactly_its_own_window(store, holding_days):
    """反之：每个持有期都**拒**掉只差一个交易日的窗（缺口恰 1 天）。"""
    r = _record(store)
    short_dates = ld.dates(holding_days)     # 差 1 点
    with pytest.raises(SettlementRefused) as ei:
        store.settle(decision_id=r["decision_id"], symbol_dates=short_dates,
                     symbol_prices=[100.0] * holding_days,
                     benchmark_dates=short_dates, benchmark_prices=[1000.0] * holding_days,
                     as_of=short_dates[-1], holding_days=holding_days)
    assert ei.value.missing_trading_days == 1


def test_three_ten_and_five_day_settlements_coexist_and_are_grouped_apart(store):
    """同一条决策的不同持有期**不是**同一件事：3 日与 10 日在聚合里分档（不混进同一均值）。

    （同一 decision 只能结算一次，故这里用三条不同决策、同一 pack。）
    """
    for i, hd in enumerate([3, 5, 10]):
        as_of = f"2025-01-0{i + 1}"
        r = _record(store, as_of=as_of)
        n = hd + 1
        dates = ld.dates(n, start=as_of)
        store.settle(decision_id=r["decision_id"], symbol_dates=dates,
                     symbol_prices=[100.0] * hd + [110.0],
                     benchmark_dates=dates, benchmark_prices=[1000.0] * (hd + 1),
                     as_of=dates[-1], holding_days=hd)
    out = store.stats_by_pack(caller=CALLER, as_of="2025-12-31", group_by_holding=True,
                              holding_days=[3, 5, 10])
    rows = {g["holding_days"]: g for g in out["rows"] if g["pack"] == ld.PACK}
    assert set(rows) == {3, 5, 10}
    assert all(rows[hd]["samples"] == 1 for hd in (3, 5, 10))
    assert all(rows[hd]["mean_alpha"] == pytest.approx(0.10) for hd in (3, 5, 10))


def test_default_holding_period_is_five_trading_days(store):
    """默认持有期 5 个交易日（不传 holding_days 时）——且库级可配。"""
    r = _record(store)
    dates = ld.dates(6)
    s = store.settle(decision_id=r["decision_id"], symbol_dates=dates,
                     symbol_prices=[100.0] * 5 + [110.0],
                     benchmark_dates=dates, benchmark_prices=[1000.0] * 6,
                     as_of=dates[-1])
    assert s["holding_days"] == 5
    assert store.holding_days == 5

    custom = LedgerStore(store.db_path.parent / "custom.db", holding_days=3)
    r2 = _record(custom)
    d3 = ld.dates(4)
    s2 = custom.settle(decision_id=r2["decision_id"], symbol_dates=d3,
                       symbol_prices=[100.0] * 3 + [110.0], benchmark_dates=d3,
                       benchmark_prices=[1000.0] * 4, as_of=d3[-1])
    assert s2["holding_days"] == 3


# ── 边界 2：基准显式覆盖 ────────────────────────────────────────


def test_explicit_benchmark_overrides_the_suffix_mapping(store):
    """显式基准优先于后缀映射，且被标注为 explicit（可审计"基准是谁定的"）。"""
    r = _record(store)
    dates = ld.dates(6)
    s = store.settle(decision_id=r["decision_id"], symbol_dates=dates,
                     symbol_prices=[100.0] * 5 + [110.0],
                     benchmark_dates=dates, benchmark_prices=[2000.0] * 5 + [2200.0],
                     as_of=dates[-1], benchmark_symbol="000300.SS")
    assert s["benchmark_symbol"] == "000300.SS"
    assert s["benchmark_mapping"] == "explicit"
    assert s["benchmark_return"] == pytest.approx(0.10, abs=1e-12)
    assert s["alpha"] == pytest.approx(0.0, abs=1e-12)


def test_explicit_benchmark_can_be_a_symbol_the_suffix_map_does_not_know(store):
    """显式基准可以是映射表里没有的代码（映射只是默认值，不是白名单）。"""
    r = _record(store)
    dates = ld.dates(6)
    s = store.settle(decision_id=r["decision_id"], symbol_dates=dates,
                     symbol_prices=[100.0] * 5 + [105.0],
                     benchmark_dates=dates, benchmark_prices=[1.0] * 6,
                     as_of=dates[-1], benchmark_symbol="CUSTOM-INDEX")
    assert s["benchmark_symbol"] == "CUSTOM-INDEX"
    assert s["benchmark_return"] == pytest.approx(0.0, abs=1e-12)
    assert s["alpha"] == pytest.approx(0.05, abs=1e-12)


def test_benchmark_mapping_is_recorded_on_the_settlement_row_not_only_returned(store):
    """基准是怎么定的落在**库行**里（重开库后仍能回答"这条用的是什么基准"）。"""
    r = _record(store)
    dates = ld.dates(6)
    store.settle(decision_id=r["decision_id"], symbol_dates=dates,
                 symbol_prices=[100.0] * 6, benchmark_dates=dates,
                 benchmark_prices=[1000.0] * 6, as_of=dates[-1],
                 benchmark_symbol="^HSI")
    reopened = LedgerStore(store.db_path)
    assert reopened.get(r["decision_id"])["settlement"]["benchmark_mapping"] == "explicit"
    assert reopened.get(r["decision_id"])["settlement"]["benchmark_symbol"] == "^HSI"

    # 不指定基准的对照组：映射来源也落在库里
    r2 = _record(store, symbol="000001.SZ", as_of="2025-02-06")
    d2 = ld.dates(6, start="2025-02-06")
    store.settle(decision_id=r2["decision_id"], symbol_dates=d2,
                 symbol_prices=[100.0] * 6, benchmark_dates=d2,
                 benchmark_prices=[1000.0] * 6, as_of=d2[-1])
    assert LedgerStore(store.db_path).get(r2["decision_id"])["settlement"][
        "benchmark_mapping"] == "suffix:.SZ"


# ── 边界 3：反思缺失 ────────────────────────────────────────────


def test_missing_reflection_does_not_block_settlement(store):
    """反思缺失不阻塞结算；结算字段仍完整（两者互不依赖）。"""
    r = _record(store)
    dates = ld.dates(6)
    s = store.settle(decision_id=r["decision_id"], symbol_dates=dates,
                     symbol_prices=[100.0] * 5 + [110.0], benchmark_dates=dates,
                     benchmark_prices=[1000.0] * 6, as_of=dates[-1])
    assert s["reflection"] is None
    for key in ("raw_return", "benchmark_return", "alpha", "holding_days",
                "settlement_date", "entry_price", "exit_price"):
        assert s[key] is not None, key
    assert store.get(r["decision_id"])["settlement"]["reflection"] is None


def test_reflection_present_and_absent_coexist_in_one_ledger(store):
    """同一账本里带反思与不带反思的结算并存，聚合口径不受影响（数字不因文本成败而变）。"""
    a = _record(store, symbol="600519.SS", as_of="2025-01-06")
    b = _record(store, symbol="000001.SZ", as_of="2025-02-06")
    d1, d2 = ld.dates(6), ld.dates(6, start="2025-02-06")
    store.settle(decision_id=a["decision_id"], symbol_dates=d1,
                 symbol_prices=[100.0] * 5 + [110.0], benchmark_dates=d1,
                 benchmark_prices=[1000.0] * 6, as_of=d1[-1],
                 reflection="带反思的一条。")
    store.settle(decision_id=b["decision_id"], symbol_dates=d2,
                 symbol_prices=[100.0] * 5 + [110.0], benchmark_dates=d2,
                 benchmark_prices=[1000.0] * 6, as_of=d2[-1])
    stats = store.stats_by_pack(caller=CALLER, as_of="2025-12-31")
    row = next(g for g in stats["rows"] if g["pack"] == ld.PACK)
    assert row["samples"] == 2
    assert row["mean_alpha"] == pytest.approx(0.10)
    assert "带反思的一条。" not in json.dumps(stats, ensure_ascii=False)
    # 反思只在点内检索里带出，且两条各带自己的那一份
    items = store.retrieve(caller=CALLER, symbol="600519.SS", as_of="2025-12-31")
    reflections = {x["settlement"]["reflection"] for x in items["same_symbol"]}
    reflections |= {x["settlement"]["reflection"] for x in items["cross_symbol"]}
    assert reflections == {"带反思的一条。", None}


def test_reflection_can_be_added_only_at_first_settlement(store):
    """反思不可后补（追加语义）：首次结算没传就保持 null，重复结算不会把它写上。"""
    r = _record(store)
    dates = ld.dates(6)
    store.settle(decision_id=r["decision_id"], symbol_dates=dates,
                 symbol_prices=[100.0] * 6, benchmark_dates=dates,
                 benchmark_prices=[1000.0] * 6, as_of=dates[-1])
    again = store.settle(decision_id=r["decision_id"], symbol_dates=dates,
                         symbol_prices=[100.0] * 6, benchmark_dates=dates,
                         benchmark_prices=[1000.0] * 6, as_of=dates[-1],
                         reflection="事后补的反思。")
    assert again["reflection"] is None
    assert again["reflection_ignored"] is True
    assert store.get(r["decision_id"])["settlement"]["reflection"] is None


# ── 边界 4：空账本查询 ──────────────────────────────────────────


def test_empty_ledger_retrieval_returns_empty_arrays(store):
    """空账本检索：两个空数组、计数为 0、不报错。"""
    out = store.retrieve(caller=CALLER, symbol=SYMBOL, as_of="2025-12-31", n_same=5, m_cross=3)
    assert out["same_symbol"] == [] and out["cross_symbol"] == []
    assert out["counts"] == {"same_symbol": 0, "cross_symbol": 0, "total": 0}
    json.dumps(out, allow_nan=False)


def test_empty_ledger_stats_with_no_pack_filter_returns_no_rows(store):
    """空账本、未指定 pack → 无组可出行（行集为空但结构完整）。"""
    out = store.stats_by_pack(caller=CALLER, as_of="2025-12-31")
    assert out["rows"] == []
    assert out["registered_packs"] == []
    assert out["counts"] == {"rows": 0, "settled_rows_in_scope": 0}
    json.dumps(out, allow_nan=False)


def test_empty_ledger_stats_with_requested_packs_returns_zero_rows(store):
    """空账本、显式请求 pack → 每个 pack 一行 samples=0（"从未落账"同样显式可见）。"""
    out = store.stats_by_pack(caller=CALLER, as_of="2025-12-31", packs=["pack-a", "pack-b", "pack-c"])
    assert [g["pack"] for g in out["rows"]] == ["pack-a", "pack-b", "pack-c"]
    assert all(g["samples"] == 0 and g["mean_alpha"] is None for g in out["rows"])


def test_empty_ledger_row_counts_and_get(store):
    """空账本的行数与单查：计数全 0，`get` 回 None（不是异常）。"""
    assert store.row_counts() == {"decisions": 0, "settlements": 0, "pending": 0, "settled": 0}
    assert store.get("dec-anything") is None


def test_settling_into_an_empty_ledger_is_refused_by_name(store):
    """空账本上结算 → 点名"账本里没有这条 decision"（不是 KeyError/None 崩溃）。"""
    from fd_quant.ledger import UnknownDecision

    dates = ld.dates(6)
    with pytest.raises(UnknownDecision) as ei:
        store.settle(decision_id="dec-ghost", symbol_dates=dates,
                     symbol_prices=[100.0] * 6, benchmark_dates=dates,
                     benchmark_prices=[1000.0] * 6, as_of=dates[-1])
    assert "dec-ghost" in str(ei.value)
    assert store.row_counts()["settlements"] == 0


def test_empty_ledger_tool_calls_are_all_usable(tmp_path):
    """工具面在空账本上也全部可用（零配置 + 空结果，不要求先有数据）。"""
    from fd_quant.ledger import tools

    db = str(tmp_path / "empty.db")
    assert tools.retrieve_decisions(symbol=SYMBOL, as_of="2025-01-01", db_path=db)[
        "result"]["counts"]["total"] == 0
    stats = tools.summarize_pack_performance(as_of="2025-01-01", db_path=db,
                                             packs=["pack-a"])
    assert stats["result"]["rows"][0]["samples"] == 0
