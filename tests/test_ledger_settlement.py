"""到期结算：基准映射、对齐与覆盖、alpha 口径、幂等、状态流转（tasks 2.1–2.5）。

跨 change 对拍（task 2.3）在这一份里：账本的 `raw_return` 与批 1.1 组合统计工具
`fd_quant.compute.portfolio.summarize_return` 的 `total_return` 用同一首末价定义，
断言两者在 1e-9 内一致。对拍的是 **total_return**（不是 `annualized_return`）：
账本算的是持有窗内的绝对收益，年化是另一个口径；选同口径字段比才有意义。
"""
from __future__ import annotations

import json
import math

import pytest

import ledgerdata as ld
from fd_quant.compute.portfolio import summarize_return
from fd_quant.ledger import LedgerStore, SettlementRefused, UnknownDecision
from fd_quant.ledger.settlement import (
    DEFAULT_BENCHMARK_MAP,
    build_series,
    check_holding_days,
    resolve_benchmark,
    settlement_window,
)

SYMBOL = "600519.SS"
BENCH = "000001.SS"


@pytest.fixture()
def store(tmp_path) -> LedgerStore:
    return LedgerStore(tmp_path / "ledger.db")


def _record(store: LedgerStore, *, symbol: str = SYMBOL, as_of: str = "2025-01-06",
            side: str = "buy", pack: str = ld.PACK, skill: str = ld.SKILL) -> dict:
    return store.record(
        as_of=as_of, symbol=symbol, pack=pack, skill=skill,
        payload=ld.rating(as_of, symbol, side=side, pack=pack, skill=skill),
    )


def _settle(store: LedgerStore, decision_id: str, *, n: int = 6, sym_p0: float = 100.0,
            sym_p1: float = 110.0, bench_p0: float = 3000.0, bench_p1: float = 3060.0,
            as_of: str | None = None, holding_days: int | None = None,
            benchmark_symbol: str | None = None, reflection: str | None = None,
            benchmark_dates=None) -> dict:
    sym = ld.flat_then_move(sym_p0, sym_p1, n)
    bench = ld.flat_then_move(bench_p0, bench_p1, n)
    return store.settle(
        decision_id=decision_id, symbol_dates=sym["dates"], symbol_prices=sym["prices"],
        benchmark_dates=bench["dates"] if benchmark_dates is None else benchmark_dates,
        benchmark_prices=bench["prices"], as_of=as_of or sym["dates"][-1],
        benchmark_symbol=benchmark_symbol, holding_days=holding_days, reflection=reflection,
    )


# ── 2.1 基准映射 ────────────────────────────────────────────────


def test_benchmark_map_by_suffix(store):
    """task 2.1 / spec Scenario「基准映射按后缀生效」：600519.SS 未指定基准 → 000001.SS，且结果标注。"""
    assert DEFAULT_BENCHMARK_MAP[".SS"] == "000001.SS"
    assert DEFAULT_BENCHMARK_MAP[".SZ"] == "399001.SZ"
    assert DEFAULT_BENCHMARK_MAP[".HK"] == "^HSI"
    assert DEFAULT_BENCHMARK_MAP[""] == "SPY"

    r = _record(store, symbol="600519.SS")
    s = _settle(store, r["decision_id"])
    assert s["benchmark_symbol"] == "000001.SS"
    assert s["benchmark_mapping"] == "suffix:.SS"


@pytest.mark.parametrize(("symbol", "expected", "mapping"), [
    ("600519.SS", "000001.SS", "suffix:.SS"),
    ("000001.SZ", "399001.SZ", "suffix:.SZ"),
    ("0700.HK", "^HSI", "suffix:.HK"),
    ("SPY", "SPY", "default:no-suffix"),
    ("AAPL", "SPY", "default:no-suffix"),
    ("1306.T", "^N225", "suffix:.T"),
])
def test_resolve_benchmark_table(symbol, expected, mapping):
    """每一条映射都逐例钉住（含 design.md 直接借用的 `.T→^N225` 与无后缀兜底）。"""
    got = resolve_benchmark(symbol)
    assert got == {"benchmark_symbol": expected, "mapping": mapping}


def test_explicit_benchmark_always_wins(store):
    """spec：「显式指定基准时以显式值为准」——显式优先于后缀映射。"""
    r = _record(store, symbol="600519.SS")
    s = _settle(store, r["decision_id"], benchmark_symbol="^HSI")
    assert s["benchmark_symbol"] == "^HSI"
    assert s["benchmark_mapping"] == "explicit"


def test_benchmark_map_is_configurable(tmp_path):
    """映射表可配置：换一套表就换一套基准（不必改代码）。"""
    custom = {".SS": "999999.SS", "": "QQQ"}
    s = LedgerStore(tmp_path / "custom.db", benchmark_map=custom)
    r = _record(s)
    got = _settle(s, r["decision_id"])
    assert got["benchmark_symbol"] == "999999.SS"
    assert got["benchmark_mapping"] == "suffix:.SS"

    # 映射表里没有的后缀 → 落到无后缀默认键
    assert resolve_benchmark("600519.US", mapping=custom) == {
        "benchmark_symbol": "QQQ", "mapping": "default:no-suffix"
    }


def test_missing_suffix_without_a_default_is_refused_and_says_so():
    """映射表既无后缀匹配、也无默认键 → 拒绝并告诉调用方显式传基准。"""
    with pytest.raises(SettlementRefused) as ei:
        resolve_benchmark("600519.SS", mapping={".SZ": "399001.SZ"})
    assert "benchmark_symbol" in str(ei.value)


def test_longest_suffix_wins_when_suffixes_overlap():
    """后缀匹配取最长：`.SS` 与 `.S` 同时在表里时用 .SS（不让短后缀悄悄吃掉长后缀）。"""
    got = resolve_benchmark("600519.SS", mapping={".S": "WRONG", ".SS": "000001.SS"})
    assert got["benchmark_symbol"] == "000001.SS"


def test_explicit_benchmark_of_blank_string_is_refused():
    """显式基准传空白字符串 → 拒绝（空字符串不是"没传"）。"""
    with pytest.raises(SettlementRefused):
        resolve_benchmark("600519.SS", explicit="   ")


# ── 2.2 对齐与覆盖 ──────────────────────────────────────────────


def test_three_day_series_for_a_five_day_hold_is_refused_with_the_gap(store):
    """task 2.2 / spec Scenario「数据不足拒绝结算」：传 3 日序列（持有期 5）被拒且错误含缺口天数。"""
    r = _record(store)
    three = ld.series([100.0, 101.0, 102.0])
    with pytest.raises(SettlementRefused) as ei:
        store.settle(decision_id=r["decision_id"], symbol_dates=three["dates"],
                     symbol_prices=three["prices"], benchmark_dates=three["dates"],
                     benchmark_prices=three["prices"], as_of=three["dates"][-1],
                     holding_days=5)
    assert "missing 3 trading days" in str(ei.value)
    assert ei.value.missing_trading_days == 3
    # 拒绝不留痕：无结算行、状态仍 pending
    assert store.row_counts() == {"decisions": 1, "settlements": 0, "pending": 1, "settled": 0}


def test_five_day_hold_over_a_six_point_axis_passes(store):
    """task 2.2：传足 6 点（入场 + 5 交易日）通过。"""
    r = _record(store)
    s = _settle(store, r["decision_id"], n=6)
    assert s["holding_days"] == 5
    assert s["settlement_date"] == ld.dates(6)[-1]


def test_exactly_five_points_for_a_five_day_hold_is_refused_by_one_day(store):
    """边界：5 点只够 4 个交易日 → 缺口恰 1 天（off-by-one 在错误文本里可见）。"""
    r = _record(store)
    with pytest.raises(SettlementRefused) as ei:
        _settle(store, r["decision_id"], n=5)
    assert ei.value.missing_trading_days == 1
    assert "missing 1 trading days" in str(ei.value)


def test_dates_are_intersected_across_the_two_series(store):
    """对齐 = 标的口期 ∩ 基准日期：基准缺一天就少一个交易日，覆盖不足即拒。

    夹具：标的 6 天、基准缺中间一天 → 交集 5 点 → 持有期 5 需要 6 点 → 缺口 1 天。
    """
    r = _record(store)
    sym = ld.flat_then_move(100.0, 110.0, 6)
    bench_dates = [d for d in sym["dates"] if d != sym["dates"][2]]
    bench_prices = [3000.0] * 5
    with pytest.raises(SettlementRefused) as ei:
        store.settle(decision_id=r["decision_id"], symbol_dates=sym["dates"],
                     symbol_prices=sym["prices"], benchmark_dates=bench_dates,
                     benchmark_prices=bench_prices, as_of=sym["dates"][-1])
    assert ei.value.missing_trading_days == 1
    assert "aligned prices" in str(ei.value)


def test_misaligned_series_reports_the_gap_in_trading_days(store):
    """design.md 的风险项：基准序列错位两天 → 缺口按交易日计并点名。"""
    r = _record(store)
    sym = ld.flat_then_move(100.0, 110.0, 8)
    bench = ld.flat_then_move(3000.0, 3100.0, 8)
    # 基准少了前两个点（等价于"基准只从第三天起有数据"）
    with pytest.raises(SettlementRefused) as ei:
        store.settle(decision_id=r["decision_id"], symbol_dates=sym["dates"],
                     symbol_prices=sym["prices"], benchmark_dates=bench["dates"][2:],
                     benchmark_prices=bench["prices"][2:], as_of=sym["dates"][-1])
    assert "entry date" in str(ei.value)
    assert ei.value.missing_trading_days is None  # 这不是"缺口"，是入场日不在轴上


def test_entry_date_must_be_on_the_aligned_axis(store):
    """入场日（决策的 as_of）不在对齐轴上 → 拒绝（不静默顺延入场日）。"""
    r = _record(store, as_of="2025-01-06")
    sym = ld.series([100.0] * 6, start="2025-01-07")  # 标的从 01-07 起，缺 01-06
    bench = ld.series([3000.0] * 6, start="2025-01-07")
    with pytest.raises(SettlementRefused) as ei:
        store.settle(decision_id=r["decision_id"], symbol_dates=sym["dates"],
                     symbol_prices=sym["prices"], benchmark_dates=bench["dates"],
                     benchmark_prices=bench["prices"], as_of=sym["dates"][-1])
    assert "2025-01-06" in str(ei.value) and "entry date" in str(ei.value)


def test_future_price_points_are_refused_and_the_first_one_is_named(store):
    """晚于 as_of 的价格点被拒并点名首个越界日期（与计算面同款：不静默截断）。"""
    r = _record(store)
    sym = ld.flat_then_move(100.0, 110.0, 8)
    bench = ld.flat_then_move(3000.0, 3100.0, 8)
    with pytest.raises(SettlementRefused) as ei:
        store.settle(decision_id=r["decision_id"], symbol_dates=sym["dates"],
                     symbol_prices=sym["prices"], benchmark_dates=bench["dates"],
                     benchmark_prices=bench["prices"], as_of=sym["dates"][4])
    assert sym["dates"][5] in str(ei.value)


def test_null_price_on_the_window_endpoints_is_refused(store):
    """端点价格缺失（null）→ 拒绝并点名日期与角色（缺失不能进算术）。"""
    r = _record(store)
    sym = ld.series([100.0, 100.0, 100.0, 100.0, 100.0, None])
    bench = ld.flat_then_move(3000.0, 3000.0, 6)
    with pytest.raises(SettlementRefused) as ei:
        store.settle(decision_id=r["decision_id"], symbol_dates=sym["dates"],
                     symbol_prices=sym["prices"], benchmark_dates=bench["dates"],
                     benchmark_prices=bench["prices"], as_of=sym["dates"][-1])
    assert "null" in str(ei.value) and sym["dates"][-1] in str(ei.value)


def test_series_shape_errors_name_the_lengths():
    """形状错误点名两侧长度（不是"非法输入"这种无操作性的文本）。"""
    with pytest.raises(SettlementRefused) as ei:
        build_series(["2025-01-06", "2025-01-07"], [1.0], name="symbol_prices")
    assert "length mismatch" in str(ei.value) and "2" in str(ei.value) and "1" in str(ei.value)

    with pytest.raises(SettlementRefused) as ei2:
        build_series([], [], name="symbol_prices")
    assert "empty" in str(ei2.value)

    with pytest.raises(SettlementRefused) as ei3:
        build_series(["2025-01-07", "2025-01-06"], [1.0, 2.0], name="symbol_prices")
    assert "strictly increasing" in str(ei3.value)

    with pytest.raises(SettlementRefused) as ei4:
        build_series(["2025-01-06"], [float("nan")], name="symbol_prices")
    assert "finite" in str(ei4.value)


def test_non_positive_endpoint_price_is_refused(store):
    """非正端点价格拒绝（简单收益的分母/几何口径需要正价）。"""
    r = _record(store)
    sym = ld.series([100.0, 100.0, 100.0, 100.0, 100.0, 0.0])
    bench = ld.flat_then_move(3000.0, 3000.0, 6)
    with pytest.raises(SettlementRefused) as ei:
        store.settle(decision_id=r["decision_id"], symbol_dates=sym["dates"],
                     symbol_prices=sym["prices"], benchmark_dates=bench["dates"],
                     benchmark_prices=bench["prices"], as_of=sym["dates"][-1])
    assert "> 0" in str(ei.value)


def test_holding_days_must_be_a_positive_integer():
    for bad in (0, -1, 2.5, True, "5"):
        with pytest.raises(SettlementRefused):
            check_holding_days(bad)
    assert check_holding_days(5) == 5


# ── 2.3 alpha 与跨 change 对拍 ──────────────────────────────────


def test_alpha_is_raw_minus_benchmark_over_the_holding_window(store):
    """task 2.3：原始收益 / 基准收益 / alpha / 持有天数 / 结算日。

    夹具手算：标的 100→110（+10%），基准 3000→3060（+2%），alpha = 8%。
    """
    r = _record(store)
    s = _settle(store, r["decision_id"], holding_days=5)
    assert math.isclose(s["raw_return"], 0.10, rel_tol=0, abs_tol=1e-12)
    assert math.isclose(s["benchmark_return"], 0.02, rel_tol=0, abs_tol=1e-12)
    assert math.isclose(s["alpha"], 0.08, rel_tol=0, abs_tol=1e-12)
    assert math.isclose(s["alpha"], s["raw_return"] - s["benchmark_return"],
                        rel_tol=0, abs_tol=1e-15)
    assert s["holding_days"] == 5
    assert s["settlement_date"] == ld.dates(6)[-1]
    assert s["entry_price"] == 100.0 and s["exit_price"] == 110.0
    assert s["status"] == "settled"


def test_alpha_cross_checked_against_the_compute_plane_return_atom(store):
    """task 2.3 跨 change 对拍：账本原始收益 vs `summarize_return.total_return`（1e-9）。

    对拍字段：`total_return`（全期总收益 = 末值/首值 - 1）——与账本的持有窗原始收益
    同一首末价定义。**不**对拍 `annualized_return`：那是年化口径，与"持有 5 个交易日的
    绝对收益"不是同一个量，拿它比会把两个定义混为一谈。
    """
    r = _record(store, symbol="000001.SZ")
    sym = ld.flat_then_move(200.0, 187.0, 6)     # 跌 6.5%
    bench = ld.flat_then_move(3000.0, 3090.0, 6)  # 涨 3%
    s = store.settle(decision_id=r["decision_id"], symbol_dates=sym["dates"],
                     symbol_prices=sym["prices"], benchmark_dates=bench["dates"],
                     benchmark_prices=bench["prices"], as_of=sym["dates"][-1],
                     benchmark_symbol="399001.SZ")
    # 对拍的是"首末两点"这一组同一口径的输入（平台段不改变 total_return，也不影响
    # 账本的持有窗收益——两者都只看首末价与区间数）
    cross = summarize_return([sym["dates"][0], sym["dates"][-1]],
                             [sym["prices"][0], sym["prices"][-1]],
                             as_of=sym["dates"][-1])
    assert cross["total_return"] is not None
    assert cross["intervals"] == 1
    assert abs(s["raw_return"] - cross["total_return"]) < 1e-9, (s["raw_return"], cross)
    # 基准侧同样对拍（基准收益与账本用的是同一原子）
    bwin = summarize_return([bench["dates"][0], bench["dates"][-1]],
                            [bench["prices"][0], bench["prices"][-1]],
                            as_of=bench["dates"][-1])
    assert abs(s["benchmark_return"] - bwin["total_return"]) < 1e-9
    assert abs(s["alpha"] - (cross["total_return"] - bwin["total_return"])) < 1e-9


def test_alpha_is_exactly_zero_when_both_legs_move_identically(store):
    """标的与基准同幅上涨 → alpha 恰为 0（不是 1e-17 那种"看上去像 0"）。"""
    r = _record(store)
    s = _settle(store, r["decision_id"], sym_p0=100.0, sym_p1=110.0,
                bench_p0=100.0, bench_p1=110.0)
    assert s["alpha"] == 0.0


def test_settlement_window_is_a_pure_function():
    """结算窗的算术可脱离库验证（纯函数层的存在理由）。"""
    sym = ld.flat_then_move(50.0, 55.0, 6)
    bench = ld.flat_then_move(100.0, 101.0, 6)
    out = settlement_window(
        entry_date=sym["dates"][0], symbol_dates=sym["dates"], symbol_values=sym["prices"],
        benchmark_dates=bench["dates"], benchmark_values=bench["prices"],
        as_of=sym["dates"][-1], holding_days=5, benchmark_symbol=BENCH,
    )
    assert out["exit_date"] == sym["dates"][5]
    assert math.isclose(out["raw_return"], 0.1, abs_tol=1e-12)
    assert math.isclose(out["benchmark_return"], 0.01, abs_tol=1e-12)
    assert math.isclose(out["alpha"], 0.09, abs_tol=1e-12)
    assert out["aligned_points_available"] == 6


# ── 2.4 结算幂等 ────────────────────────────────────────────────


def test_repeated_settlement_returns_the_existing_result(store):
    """task 2.4 / spec Scenario「重复结算幂等」：不产生第二条结算行，返回既有结果。"""
    r = _record(store)
    first = _settle(store, r["decision_id"])
    assert first["created"] is True and first["reason"] == "settled"
    after_first = store.row_counts()
    assert after_first["settlements"] == 1

    second = _settle(store, r["decision_id"])
    assert second["created"] is False
    assert second["reason"] == "duplicate_settlement"
    assert store.row_counts() == after_first
    for key in ("settlement_date", "raw_return", "benchmark_return", "alpha", "holding_days",
                "settled_at", "benchmark_symbol", "benchmark_mapping"):
        assert second[key] == first[key], key


def test_repeated_settlement_ignores_a_different_reflection_and_says_so(store):
    """追加语义：已落结算不被改写；新反思被忽略且结果里明示（reflection_ignored）。"""
    r = _record(store)
    _settle(store, r["decision_id"], reflection="第一次反思。")
    again = _settle(store, r["decision_id"], reflection="第二次反思（不应生效）。")
    assert again["reflection"] == "第一次反思。"
    assert again["reflection_ignored"] is True
    assert again["created"] is False


def test_settlement_of_an_unknown_decision_is_refused(store):
    """结算不存在的 decision → UnknownDecision（点名该 id，不是 KeyError）。"""
    sym = ld.flat_then_move(100.0, 110.0, 6)
    with pytest.raises(UnknownDecision) as ei:
        store.settle(decision_id="dec-nope", symbol_dates=sym["dates"],
                     symbol_prices=sym["prices"], benchmark_dates=sym["dates"],
                     benchmark_prices=sym["prices"], as_of=sym["dates"][-1])
    assert "dec-nope" in str(ei.value)


# ── 2.5 状态流转 ────────────────────────────────────────────────


def test_status_transitions_from_pending_to_settled_and_fields_are_queryable(store):
    """task 2.5：结算前后各查一次，断言状态与结算字段。"""
    r = _record(store)
    before = store.get(r["decision_id"])
    assert before["status"] == "pending"
    assert before["settlement"] is None
    assert store.retrieve(symbol=SYMBOL, as_of="2099-01-01")["counts"]["total"] == 0

    _settle(store, r["decision_id"], reflection="持有期内无新信息，判定未变。")
    after = store.get(r["decision_id"])
    assert after["status"] == "settled"
    s = after["settlement"]
    assert s is not None
    assert s["settlement_date"] == ld.dates(6)[-1]
    for key in ("settled_at", "entry_price", "exit_price", "raw_return",
                "benchmark_symbol", "benchmark_mapping", "benchmark_return", "alpha",
                "holding_days"):
        assert s[key] is not None, key
    assert s["reflection"] == "持有期内无新信息，判定未变。"
    assert after["payload"] == before["payload"]     # 判定快照未因结算而变
    assert after["created_at"] == before["created_at"]


def test_settlement_result_carries_the_decision_and_ledger_counts(store):
    """结算返回把"这是哪条决策"与"账本行数"一并给出（幂等核对的可观察事实）。"""
    r = _record(store)
    s = _settle(store, r["decision_id"])
    assert s["decision_id"] == r["decision_id"]
    assert s["decision_as_of"] == "2025-01-06"
    assert s["symbol"] == SYMBOL and s["pack"] == ld.PACK and s["skill"] == ld.SKILL
    assert s["counts"] == {"decisions": 1, "settlements": 1, "pending": 0, "settled": 1}


def test_settlement_output_is_json_serialisable_without_nan(store):
    """spec: 返回必须 `json.dumps(..., allow_nan=False)` 通过（无 NaN / 无 numpy 标量）。"""
    r = _record(store)
    s = _settle(store, r["decision_id"], reflection="一段反思。")
    text = json.dumps(s, allow_nan=False)
    assert "NaN" not in text and "Infinity" not in text
    item = store.get(r["decision_id"])
    json.dumps(item, allow_nan=False)
    json.dumps(store.retrieve(symbol=SYMBOL, as_of="2099-01-01"), allow_nan=False)
    json.dumps(store.stats_by_pack(as_of="2099-01-01"), allow_nan=False)


def test_settled_at_is_recorded_but_not_asserted_for_equality_across_runs(store):
    """时间戳可记录但不得跨运行相等断言：显式 `now` 才是确定性来源。"""
    a = _record(store, symbol="600519.SS")
    s_a = _settle(store, a["decision_id"], as_of=ld.dates(6)[-1])
    assert s_a["settled_at"] and s_a["settled_at"].endswith("+00:00")

    b = _record(store, symbol="000001.SZ")
    sym = ld.flat_then_move(100.0, 110.0, 6)
    bench = ld.flat_then_move(3000.0, 3060.0, 6)
    s_b = store.settle(decision_id=b["decision_id"], symbol_dates=sym["dates"],
                       symbol_prices=sym["prices"], benchmark_dates=bench["dates"],
                       benchmark_prices=bench["prices"], as_of=sym["dates"][-1],
                       now="2025-01-14T00:00:00+00:00")
    assert s_b["settled_at"] == "2025-01-14T00:00:00+00:00"
    # 注入的 now 只影响时间戳，不影响任何被比较的数字
    assert math.isclose(s_b["alpha"], 0.08, abs_tol=1e-12)
