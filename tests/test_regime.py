"""regime 检测族测试：spec Scenario 逐条对拍、边界与拒绝语义、确定性。

夹具**手工构造**（不用随机游走）：慢速 regime 断言的是"三条判据各自触发与不触发"，
随机序列无法保证方向与幅度落在阈值的哪一侧——只有构造序列能让断言说的是"定义如此"
而不是"这次运气如此"。关键定义另就地写一份**独立**朴素实现逐点对拍（这是唯一能发现
"实现与定义一起写错"的方式）。
"""
from __future__ import annotations

import inspect
import json
import math

import pytest

import quantdata as qd
from fd_quant.compute import all_tools, base, tools_by_group
from fd_quant.compute.regime import (
    CALIBRATION_NOTE,
    detect_regime_slow,
    detect_shock_fast,
    document_regime_defaults,
)

#: spec 6.3 要求的注记原文（测试里写死字面量：即便实现把常量改坏，这条也会红）。
NOTE = "默认值来自美股公开量级，A 股需在策略层调参"

SLOW = "detect_regime_slow"
FAST = "detect_shock_fast"
DOC = "document_regime_defaults"


# ── 夹具 ─────────────────────────────────────────────────────────


def _trend_down_with_deep_drawdown(n_up: int = 160, n_down: int = 100) -> list[float]:
    """先温和上涨、其后持续下跌：短均线必然跌破长均线，且回撤远超 10%。"""
    px = [100.0]
    for _ in range(n_up - 1):
        px.append(px[-1] * 1.004)
    for _ in range(n_down):
        px.append(px[-1] * 0.988)
    return px


def _calm_then_vol_spike(n: int, *, level: float = 0.10, spike: float = 0.50) -> list[float]:
    """前段恒定的低波动 + 末值骤升：Z 的分母极小、分子很大，Z 判据必然触发。"""
    return [level] * (n - 1) + [spike]


def _strictly_falling_vol(n: int, *, start: float = 0.30, end: float = 0.10) -> list[float]:
    """单调下降的波动率：末值低于滚动均值 → Z 为负，Z 判据必然不触发。"""
    step = (start - end) / (n - 1)
    return [start - step * i for i in range(n)]


def _evidence_value(result: dict, evidence_id: str):
    for e in result["evidence"]:
        if e["evidence_id"] == evidence_id:
            return e["value"]
    raise AssertionError(
        f"no evidence {evidence_id!r}; have {[e['evidence_id'] for e in result['evidence']]}"
    )


# ── 独立朴素实现（不调用被测代码的任何内部函数） ───────────────────


def _naive_sma(vals, w):
    return sum(float(x) for x in vals[len(vals) - w:]) / w


def _naive_drawdown(vals):
    peak = max(float(x) for x in vals)
    return (peak - float(vals[-1])) / peak


def _naive_z(vals, w):
    win = [float(x) for x in vals[len(vals) - w:]]
    mean = sum(win) / w
    var = sum((x - mean) ** 2 for x in win) / w
    return (win[-1] - mean) / math.sqrt(var)


def _naive_window_return(vals, n_):
    return float(vals[-1]) / float(vals[len(vals) - 1 - n_]) - 1.0


def _naive_jump(vol, w):
    prev = [float(x) for x in vol[len(vol) - 1 - w:len(vol) - 1]]
    return float(vol[-1]) / (sum(prev) / w)


# ── spec Scenario：慢速 regime 输出状态与证据 ─────────────────────


def test_slow_regime_risk_off_on_falling_trend_deep_drawdown_high_vol_z():
    """spec Scenario 6.1：传入"趋势向下 + 回撤超阈 + 波动率 Z 高"的序列 →
    返回 `risk_off` 且证据列出各判据的触发值。"""
    px = _trend_down_with_deep_drawdown()
    ds = qd.dates(len(px))
    vol = _calm_then_vol_spike(len(px))
    r = detect_regime_slow(ds, px, vol, as_of=ds[-1])

    assert r["regime"] == "risk_off"
    assert r["triggered_count"] == 3

    # 三条判据各自：触发、触发值在阈值之外（不是靠标签说触发）
    assert r["criteria"]["trend"]["triggered"] is True
    assert r["criteria"]["trend"]["value"] < 0.0
    assert r["criteria"]["drawdown"]["triggered"] is True
    assert r["criteria"]["drawdown"]["value"] > 0.10
    assert r["criteria"]["volatility_z"]["triggered"] is True
    assert r["criteria"]["volatility_z"]["value"] > 2.0
    for name in ("trend", "drawdown", "volatility_z"):
        assert r["criteria"][name]["status"] == "ok", name

    # 证据里逐条列出触发值（值与 criteria 同源，不是另算一遍）
    for name in ("trend", "drawdown", "volatility_z"):
        ev = _evidence_value(r, f"{SLOW}.{name}_trigger_value")
        assert ev == pytest.approx(r["criteria"][name]["value"], rel=1e-15, abs=1e-15)
    assert _evidence_value(r, f"{SLOW}.triggered_count") == 3

    # 生效参数与逐判据 warmup 回显（慢速默认含 200 日长均线）
    assert r["params"] == {
        "trend_window": 60, "trend_ma_window": 200, "drawdown_threshold": 0.10,
        "z_threshold": 2.0, "z_window": 60,
    }
    assert r["warmup"] == {"trend": 199, "drawdown": 0, "volatility_z": 59}
    assert r["volatility_source"] == "caller_supplied"


def test_slow_regime_criteria_match_independent_naive_implementation():
    """关键定义对拍：趋势 / 回撤 / Z 三条判据与就地写的朴素实现逐点一致。"""
    px = _trend_down_with_deep_drawdown()
    ds = qd.dates(len(px))
    vol = _calm_then_vol_spike(len(px))
    tw, tmw, zw = 60, 200, 60
    r = detect_regime_slow(ds, px, vol, as_of=ds[-1])

    assert r["criteria"]["trend"]["value"] == pytest.approx(
        _naive_sma(px, tw) / _naive_sma(px, tmw) - 1.0, rel=1e-12, abs=1e-15
    )
    assert r["criteria"]["drawdown"]["value"] == pytest.approx(
        _naive_drawdown(px), rel=1e-12, abs=1e-15
    )
    assert r["criteria"]["volatility_z"]["value"] == pytest.approx(
        _naive_z(vol, zw), rel=1e-12, abs=1e-15
    )


def test_slow_regime_risk_on_needs_positive_trend_and_no_vol_trigger():
    """决策规则第二支：趋势向上且 Z 未触发 → risk_on。"""
    n = 260
    ds = qd.dates(n)
    px = [100.0 + 0.1 * i for i in range(n)]
    r = detect_regime_slow(ds, px, _strictly_falling_vol(n), as_of=ds[-1])
    assert r["regime"] == "risk_on"
    assert r["criteria"]["trend"]["value"] > 0.0
    assert r["criteria"]["drawdown"]["value"] == 0.0  # 逐日新高 → 回撤恰为 0
    assert r["criteria"]["volatility_z"]["triggered"] is False


def test_slow_regime_trend_down_without_drawdown_is_neutral():
    """决策规则第一支要求回撤**也**触发：纯趋势向下 → neutral，不是 risk_off。"""
    n = 260
    ds = qd.dates(n)
    px = [100.0 - 0.05 * i for i in range(n)]  # 缓慢下跌，但回撤幅度小
    r = detect_regime_slow(ds, px, _strictly_falling_vol(n), as_of=ds[-1],
                           drawdown_threshold=0.9)  # 抬高阈值使回撤判据不触发
    assert r["criteria"]["trend"]["triggered"] is True
    assert r["criteria"]["drawdown"]["triggered"] is False
    assert r["regime"] == "neutral"


def test_slow_regime_up_trend_with_vol_spike_is_neutral_not_risk_off():
    """回撤未触发时 risk_off 无从成立（规则要求"回撤 + 另一条"）；Z 触发挡住 risk_on。"""
    n = 260
    ds = qd.dates(n)
    px = [100.0 + 0.1 * i for i in range(n)]
    r = detect_regime_slow(ds, px, _calm_then_vol_spike(n), as_of=ds[-1])
    assert r["criteria"]["drawdown"]["triggered"] is False
    assert r["criteria"]["volatility_z"]["triggered"] is True
    assert r["regime"] == "neutral"
    assert r["triggered_count"] == 1


# ── spec Scenario：波动率序列由调用方提供 ─────────────────────────


def test_slow_regime_refuses_without_volatility_naming_the_input():
    """spec Scenario 6.1：未提供波动率序列而调用慢速 regime →
    报错指明缺失的输入，不自行取数。"""
    ds = qd.dates(80)
    px = [100.0] * 80
    with pytest.raises(base.ComputeError) as ei:
        detect_regime_slow(ds, px, as_of=ds[-1])
    msg = str(ei.value)
    assert "volatility" in msg
    assert "required" in msg
    assert "never computes" in msg  # 明说不自行取数/合成


def test_slow_regime_refuses_explicit_none_volatility():
    ds = qd.dates(80)
    with pytest.raises(base.ComputeError, match="volatility"):
        detect_regime_slow(ds, [100.0] * 80, None, as_of=ds[-1])


def test_fast_regime_refuses_without_volatility_naming_the_input():
    ds = qd.dates(80)
    with pytest.raises(base.ComputeError, match="volatility"):
        detect_shock_fast(ds, [100.0] * 80, as_of=ds[-1])


def test_volatility_rule_is_stated_in_docstring_and_tool_description():
    """波动率由调用方供给这条规则必须在 docstring 与工具描述里可见（同一处事实源）。"""
    for name, fn in ((SLOW, detect_regime_slow), (FAST, detect_shock_fast)):
        doc = fn.__doc__ or ""
        assert "由调用方提供" in doc, name
        assert "不自行计算" in doc, name
    by_name = {t.name: t for t in all_tools()}
    for name in (SLOW, FAST):
        assert "由调用方提供" in by_name[name].doc, name


# ── spec Scenario：快速风险关闭独立触发 ───────────────────────────


def test_shock_fast_triggers_and_echoes_recommendation_params():
    """spec Scenario 6.2：传入 N 日内跌幅超阈的序列 → 快速检测返回触发，
    且建议中给出降仓幅度与冷却期（参数可配）——用**非默认**值验证回显不是常量。"""
    n = 85
    ds = qd.dates(n)
    px = [100.0] * 80
    for _ in range(5):
        px.append(px[-1] * 0.98)  # 5 日 -9.6%，超过默认 8%
    vol = [0.20] * 80 + [0.60] * 5  # 末值 / 前 20 位均值 = 3.0，超过默认 1.5
    r = detect_shock_fast(
        ds, px, vol, as_of=ds[-1],
        vol_window=20, de_risk_fraction=0.35, cooldown_days=7,
    )

    assert r["triggered"] is True
    assert r["shock_triggered"] is True
    assert r["shock_return"] == pytest.approx(_naive_window_return(px, 5), rel=1e-12)
    assert r["volatility_jump_ratio"] == pytest.approx(_naive_jump(vol, 20), rel=1e-12)
    assert r["volatility_jump_triggered"] is True
    assert r["status"] == "ok"

    rec = r["recommendation"]
    assert rec["de_risk_fraction"] == 0.35  # 回显传入值，不是默认 0.5
    assert rec["cooldown_days"] == 7  # 回显传入值，不是默认 5
    assert rec["action"] == "reduce_position"
    assert rec["basis"]  # 建议带依据
    assert r["params"]["de_risk_fraction"] == 0.35
    assert r["params"]["cooldown_days"] == 7
    assert _evidence_value(r, f"{FAST}.de_risk_fraction") == 0.35
    assert _evidence_value(r, f"{FAST}.cooldown_days") == 7


def test_shock_fast_price_shock_alone_triggers():
    """两个信号各自独立成立（快线任一信号即触发）。"""
    n = 10
    ds = qd.dates(n)
    px = [100.0] * 5 + [100.0 * (0.90 ** k) for k in range(1, 6)]
    vol = [0.2] * n  # 波动率恒定 → 跳变比 = 1.0，不触发
    r = detect_shock_fast(ds, px, vol, as_of=ds[-1], vol_window=3)
    assert r["shock_triggered"] is True
    assert r["volatility_jump_triggered"] is False
    assert r["triggered"] is True


def test_shock_fast_vol_jump_alone_triggers():
    """价格未跌但波动率跳升 → 独立触发（建议给出降仓幅度与冷却期）。"""
    n = 30
    ds = qd.dates(n)
    px = [100.0] * n
    vol = [0.10] * (n - 1) + [0.40]
    r = detect_shock_fast(ds, px, vol, as_of=ds[-1], vol_window=10)
    assert r["shock_return"] == 0.0
    assert r["shock_triggered"] is False
    assert r["volatility_jump_triggered"] is True
    assert r["triggered"] is True
    assert r["recommendation"]["action"] == "reduce_position"


def test_shock_fast_no_trigger_is_hold_with_negative_evidence():
    """未触发 → action = hold，且值原样回显（不伪造一个"建议"）。"""
    n = 30
    ds = qd.dates(n)
    r = detect_shock_fast(ds, [100.0] * n, [0.10] * n, as_of=ds[-1], vol_window=10)
    assert r["triggered"] is False
    assert r["recommendation"]["action"] == "hold"
    assert r["recommendation"]["de_risk_fraction"] == 0.5
    assert r["recommendation"]["cooldown_days"] == 5
    assert _evidence_value(r, f"{FAST}.triggered") is False


# ── spec 6.3：默认参数文档化 ─────────────────────────────────────


def test_defaults_note_is_in_return_and_docstring():
    """spec 6.3：工具描述与返回都带"默认值来自美股公开量级，A 股需在策略层调参"。"""
    n = 20
    ds = qd.dates(n)
    r = document_regime_defaults(as_of=ds[-1], dates=ds)

    assert r["calibration_note"] == NOTE
    assert NOTE in json.dumps(r, ensure_ascii=False, allow_nan=False)
    assert NOTE in (document_regime_defaults.__doc__ or "")
    assert CALIBRATION_NOTE == NOTE
    # 逐检测器各带一份（调用方不必知道顶层键）
    for name in (SLOW, FAST):
        assert r["tools"][name]["calibration_note"] == NOTE
    assert NOTE in _evidence_value(r, f"{DOC}.calibration_note")


def test_defaults_match_the_detector_signatures():
    """回显的默认值就是签名里的默认值（单一事实源，不会与实现漂移）。"""
    ds = qd.dates(5)
    r = document_regime_defaults(as_of=ds[-1], dates=ds)
    for name, fn in ((SLOW, detect_regime_slow), (FAST, detect_shock_fast)):
        expected = {
            p.name: p.default
            for p in inspect.signature(fn).parameters.values()
            if p.default is not inspect.Parameter.empty and p.default is not None
        }
        assert r["tools"][name]["params"] == expected, name
        # 参数确实是校准旋钮（不是把 dates/volatility 这类输入也列进来）
        assert "dates" not in expected and "volatility" not in expected
        assert expected, f"{name}: no documented defaults"
    # 两族的默认值都覆盖 spec 列的旋钮
    assert r["tools"][SLOW]["params"]["trend_window"] == 60
    assert r["tools"][SLOW]["params"]["drawdown_threshold"] == 0.10
    assert r["tools"][SLOW]["params"]["z_threshold"] == 2.0
    assert r["tools"][FAST]["params"]["shock_window"] == 5


def test_document_regime_defaults_ignores_dates_but_does_not_excuse_future_data():
    """`dates` 只为统一调用形状：可省略；但给了就越界拒绝并点名首个越界日期。"""
    ds = qd.dates(10)
    without = document_regime_defaults(as_of=ds[-1])
    assert without["input_start"] is None and without["input_length"] is None
    assert without["tools"] == document_regime_defaults(as_of=ds[-1], dates=ds)["tools"]

    with pytest.raises(base.ComputeError) as ei:
        document_regime_defaults(as_of=ds[4], dates=ds)
    assert ds[5] in str(ei.value)  # 首个越界日期


def test_document_regime_defaults_is_a_snapshot_with_empty_params_and_warmup():
    """快照型：无数值输入、无生效参数、无序列输出。"""
    ds = qd.dates(3)
    r = document_regime_defaults(as_of=ds[-1], dates=ds)
    assert r["params"] == {} and r["warmup"] == {}
    assert r["evidence"] and r["as_of"] == ds[-1]


# ── 边界：窗口长于序列 / 恰好等于阈值 / 退化输入 ──────────────────


def test_slow_regime_series_shorter_than_ma_window_reports_insufficient_data():
    """序列短于长均线窗 → 趋势判据 insufficient_data（不崩、不抛），其余判据照常出数。"""
    n = 30
    ds = qd.dates(n)
    px = [100.0 - 1.0 * i for i in range(n)]  # 深度回撤，但历史不足
    r = detect_regime_slow(ds, px, [0.1] * n, as_of=ds[-1], trend_window=20, z_window=10)

    assert r["criteria"]["trend"]["status"] == "insufficient_data"
    assert r["criteria"]["trend"]["value"] is None
    assert r["criteria"]["trend"]["triggered"] is False
    assert r["criteria"]["drawdown"]["status"] == "ok"
    assert r["criteria"]["drawdown"]["triggered"] is True
    # 规则要求"回撤 + (趋势 或 Z)"：另一条不可判定 → neutral，不降级成 risk_off
    assert r["regime"] == "neutral"


def test_slow_regime_all_windows_short_of_the_series_is_neutral_and_keeps_evidence():
    n = 5
    ds = qd.dates(n)
    r = detect_regime_slow(ds, [100.0] * n, [0.1] * n, as_of=ds[-1])
    assert r["regime"] == "neutral"
    assert r["triggered_count"] == 0
    for name in ("trend", "volatility_z"):
        assert r["criteria"][name]["status"] == "insufficient_data", name
    assert r["warmup"]["trend"] == 199  # 仍是声明的位数，不因缺口改成 0
    assert r["evidence"]


def test_slow_regime_missing_inside_window_is_insufficient_and_keeps_other_criteria():
    """窗内含缺失 → 该判据 insufficient_data（不跳过缺失位继续算），其余判据不受污染。"""
    n = 260
    ds = qd.dates(n)
    px = _trend_down_with_deep_drawdown()
    vol = _calm_then_vol_spike(n)
    vol[n - 5] = None  # Z 窗内缺失；回撤窗无缺失
    r = detect_regime_slow(ds, px, vol, as_of=ds[-1])
    assert r["criteria"]["volatility_z"]["status"] == "insufficient_data"
    assert r["criteria"]["drawdown"]["status"] == "ok"
    assert r["criteria"]["trend"]["status"] == "ok"


def test_slow_regime_constant_volatility_has_undefined_z():
    """波动率恒定 → Z 的分母为 0 → 判据报 insufficient_data，不产 inf/NaN。"""
    n = 60
    ds = qd.dates(n)
    r = detect_regime_slow(ds, [100.0] * n, [0.2] * n, as_of=ds[-1], z_window=20)
    c = r["criteria"]["volatility_z"]
    assert c["status"] == "insufficient_data"
    assert c["value"] is None and c["triggered"] is False
    json.dumps(r, allow_nan=False)  # 返回里不能有 inf/NaN


def test_slow_regime_all_null_prices_reports_insufficient_data():
    """全缺失输入：逐判据 insufficient_data + evidence 的 all_null 事实，不抛异常。"""
    n = 60
    ds = qd.dates(n)
    r = detect_regime_slow(ds, [None] * n, [None] * n, as_of=ds[-1])
    for name in ("trend", "drawdown", "volatility_z"):
        assert r["criteria"][name]["status"] == "insufficient_data", name
    assert r["regime"] == "neutral"
    assert _evidence_value(r, f"{SLOW}.prices_all_null") is True


def test_slow_regime_single_observation_does_not_crash():
    ds = qd.dates(1)
    r = detect_regime_slow(ds, [100.0], [0.1], as_of=ds[0])
    assert r["regime"] == "neutral"
    assert r["criteria"]["drawdown"]["value"] == 0.0
    assert r["input_length"] == 1


def test_slow_regime_non_positive_price_level_yields_no_drawdown_number():
    """非正水位（峰值 ≤ 0）→ 回撤比值无定义 → insufficient_data（不发明一个数）。"""
    ds = qd.dates(3)
    r = detect_regime_slow(ds, [-1.0, -2.0, -3.0], [0.1] * 3, as_of=ds[-1], z_window=2)
    assert r["criteria"]["drawdown"]["status"] == "insufficient_data"
    assert r["criteria"]["drawdown"]["value"] is None


def test_slow_regime_empty_series_is_refused_naming_prices():
    with pytest.raises(base.ComputeError, match="prices is empty"):
        detect_regime_slow([], [], [], as_of="2025-01-01")


def test_slow_regime_drawdown_comparison_is_strict_at_the_threshold():
    """边界：回撤恰等于阈值**不算**触发（严格 >），略超一位即触发。"""
    ds = qd.dates(2)
    at = detect_regime_slow(ds, [100.0, 90.0], [0.1, 0.2], as_of=ds[-1],
                            trend_window=1, trend_ma_window=1, z_window=2)
    assert at["criteria"]["drawdown"]["value"] == 0.10  # 恰好 10%
    assert at["criteria"]["drawdown"]["triggered"] is False

    over = detect_regime_slow(ds, [100.0, 89.9], [0.1, 0.2], as_of=ds[-1],
                              trend_window=1, trend_ma_window=1, z_window=2)
    assert over["criteria"]["drawdown"]["triggered"] is True


def test_slow_regime_z_comparison_is_strict_at_the_threshold():
    """边界：Z 恰等于阈值不算触发（严格 >）；阈值略降即触发。

    构造：[1.0, 2.0] 的窗口（z_window=2）→ 总体 Z 恰为 1.0。
    """
    ds = qd.dates(2)
    px = [100.0, 101.0]
    at = detect_regime_slow(ds, px, [1.0, 2.0], as_of=ds[-1],
                            trend_window=1, trend_ma_window=1, z_window=2, z_threshold=1.0)
    assert at["criteria"]["volatility_z"]["value"] == 1.0
    assert at["criteria"]["volatility_z"]["triggered"] is False

    over = detect_regime_slow(ds, px, [1.0, 2.0], as_of=ds[-1],
                              trend_window=1, trend_ma_window=1, z_window=2,
                              z_threshold=0.99)
    assert over["criteria"]["volatility_z"]["triggered"] is True


def test_slow_regime_trend_comparison_is_strict_at_zero():
    """边界：趋势值恰为 0（短均线 = 长均线）不算"向下"（严格 < 0）。"""
    n = 220
    ds = qd.dates(n)
    r = detect_regime_slow(ds, [100.0] * n, _strictly_falling_vol(n), as_of=ds[-1])
    assert r["criteria"]["trend"]["value"] == 0.0
    assert r["criteria"]["trend"]["triggered"] is False


def test_slow_regime_flat_series_is_neutral():
    """平坦序列 → neutral：risk_on 要求趋势**为正**，恰好 0 不满足。"""
    n = 220
    ds = qd.dates(n)
    r = detect_regime_slow(ds, [100.0] * n, _strictly_falling_vol(n), as_of=ds[-1])
    assert r["regime"] == "neutral"


def test_shock_fast_short_series_is_insufficient_not_a_crash():
    """序列短于冲击窗 / 波动率窗 → 两个信号各报 insufficient_data，不抛异常。"""
    n = 3
    ds = qd.dates(n)
    r = detect_shock_fast(ds, [100.0, 99.0, 98.0], [0.1, 0.1, 0.1], as_of=ds[-1])
    assert r["shock_return"] is None and r["shock_triggered"] is False
    assert r["volatility_jump_ratio"] is None
    assert r["volatility_jump_triggered"] is False
    assert r["status"] == "insufficient_data"
    assert r["triggered"] is False
    assert r["warmup"] == {"shock_return": 5, "volatility_jump_ratio": 60}


def test_shock_fast_comparisons_are_strict_at_the_threshold():
    """边界：N 日收益恰等于阈值 / 跳变比恰等于阈值都不触发（严格 <、>）。"""
    ds = qd.dates(3)
    at = detect_shock_fast(ds, [100.0, 100.0, 75.0], [0.2] * 3, as_of=ds[-1],
                           shock_window=1, shock_threshold=0.25, vol_window=1,
                           vol_jump_ratio=1.5)
    assert at["shock_return"] == -0.25  # 恰 -25%
    assert at["shock_triggered"] is False
    assert at["volatility_jump_ratio"] == 1.0  # 波动率恒定 → 恰 1.0
    assert at["volatility_jump_triggered"] is False
    assert at["triggered"] is False

    over = detect_shock_fast(ds, [100.0, 100.0, 74.9], [0.2] * 3, as_of=ds[-1],
                             shock_window=1, shock_threshold=0.25, vol_window=1,
                             vol_jump_ratio=1.5)
    assert over["shock_triggered"] is True

    jump_at = detect_shock_fast(ds, [100.0] * 3, [0.25, 0.25, 0.5], as_of=ds[-1],
                                shock_window=1, shock_threshold=0.25, vol_window=1,
                                vol_jump_ratio=2.0)
    assert jump_at["volatility_jump_ratio"] == 2.0  # 恰 2.0（0.5/0.25 二进制精确）
    assert jump_at["volatility_jump_triggered"] is False


def test_shock_fast_null_inside_shock_window_is_insufficient_not_skipped():
    """冲击窗内含缺失 → insufficient_data（不跨过缺失位算一个更长的收益）。"""
    n = 6
    ds = qd.dates(n)
    px = [100.0, 100.0, 100.0, 100.0, None, 90.0]
    r = detect_shock_fast(ds, px, [0.2] * n, as_of=ds[-1], shock_window=4, vol_window=2)
    assert r["shock_return"] is None
    assert r["shock_triggered"] is False


def test_shock_fast_non_positive_baseline_is_insufficient():
    """波动率基线非正 → 跳变比无定义 → insufficient_data（不返回负/巨正比值）。"""
    n = 10
    ds = qd.dates(n)
    r = detect_shock_fast(ds, [100.0] * n, [0.0] * n, as_of=ds[-1], vol_window=3)
    assert r["volatility_jump_ratio"] is None
    assert r["status"] == "insufficient_data"


# ── 拒绝：as_of 口径 ─────────────────────────────────────────────


@pytest.mark.parametrize("fn,extra", [
    (detect_regime_slow, {"trend_window": -1}),      # 另一个参数也是坏的
    (detect_shock_fast, {"shock_window": -1}),
    (document_regime_defaults, {}),
])
def test_as_of_is_required_validated_first_and_without_default(fn, extra):
    """`as_of` 必填、无默认值，且是**第一个**校验动作（缺它时报的不是别的错）。"""
    ds = qd.dates(5)
    assert inspect.signature(fn).parameters["as_of"].default is inspect.Parameter.empty
    kw = {"dates": ds}
    if fn is not document_regime_defaults:
        kw.update(prices=[100.0] * 5, volatility=[0.1] * 5)
    with pytest.raises(base.ComputeError, match="as_of"):
        fn(as_of=None, **kw, **extra)


@pytest.mark.parametrize("fn", [detect_regime_slow, detect_shock_fast,
                                document_regime_defaults])
def test_invalid_as_of_is_refused(fn):
    ds = qd.dates(5)
    kw = {"dates": ds}
    if fn is not document_regime_defaults:
        kw.update(prices=[100.0] * 5, volatility=[0.1] * 5)
    with pytest.raises(base.ComputeError, match="as_of"):
        fn(as_of="2025/01/01", **kw)


def test_detectors_refuse_future_data_and_name_the_first_date():
    """口径纪律：晚于 as_of 的数据点一律拒绝并点名**首个**越界日期（含波动率序列）。"""
    n = 40
    ds = qd.dates(n)
    px = [100.0] * n
    vol = [0.1] * n
    as_of = ds[30]
    for call in (detect_regime_slow, detect_shock_fast):
        with pytest.raises(base.ComputeError) as ei:
            call(ds, px, vol, as_of=as_of)
        assert ds[31] in str(ei.value), call.__name__
        assert "after as_of" in str(ei.value)


@pytest.mark.parametrize("call", [detect_regime_slow, detect_shock_fast])
def test_volatility_series_goes_through_check_series(call):
    """波动率与价格共用日期轴：越界由该轴点名首个日期；波动率自身的值形状同样被校验。"""
    n = 40
    ds = qd.dates(n)
    px = [100.0] * n
    # 越界：共用日期轴 → 点名首个越界日期（价格先校验，日期相同）
    with pytest.raises(base.ComputeError) as ei:
        call(ds, px, [0.1] * n, as_of=ds[30])
    assert ds[31] in str(ei.value)
    # 波动率自身的非法值：必须由 check_series 点名 volatility（不是静默转 0/NaN）
    bad = [0.1] * (n - 1) + ["not-a-number"]
    with pytest.raises(base.ComputeError) as ei2:
        call(ds, px, bad, as_of=ds[-1])
    assert "volatility" in str(ei2.value)
    assert "not numeric" in str(ei2.value)


def test_volatility_length_mismatch_is_refused_naming_volatility():
    ds = qd.dates(40)
    with pytest.raises(base.ComputeError) as ei:
        detect_regime_slow(ds, [100.0] * 40, [0.1] * 39, as_of=ds[-1])
    assert "volatility" in str(ei.value)
    assert "length mismatch" in str(ei.value)


def test_fast_detector_refuses_bad_parameters_naming_them():
    ds = qd.dates(10)
    px, vol = [100.0] * 10, [0.1] * 10
    with pytest.raises(base.ComputeError, match="shock_window"):
        detect_shock_fast(ds, px, vol, as_of=ds[-1], shock_window=0)
    with pytest.raises(base.ComputeError, match="cooldown_days"):
        detect_shock_fast(ds, px, vol, as_of=ds[-1], cooldown_days=0)
    with pytest.raises(base.ComputeError, match="de_risk_fraction"):
        detect_shock_fast(ds, px, vol, as_of=ds[-1], de_risk_fraction=1.5)
    with pytest.raises(base.ComputeError, match="shock_threshold"):
        detect_shock_fast(ds, px, vol, as_of=ds[-1], shock_threshold=-0.1)


def test_slow_detector_refuses_bad_parameters_naming_them():
    ds = qd.dates(10)
    px, vol = [100.0] * 10, [0.1] * 10
    with pytest.raises(base.ComputeError, match="drawdown_threshold"):
        detect_regime_slow(ds, px, vol, as_of=ds[-1], drawdown_threshold=1.5)
    with pytest.raises(base.ComputeError, match="z_threshold"):
        detect_regime_slow(ds, px, vol, as_of=ds[-1], z_threshold=-1.0)
    with pytest.raises(base.ComputeError, match="trend_ma_window"):
        detect_regime_slow(ds, px, vol, as_of=ds[-1], trend_ma_window=0)
    with pytest.raises(base.ComputeError, match="z_window"):
        detect_regime_slow(ds, px, vol, as_of=ds[-1], z_window=0)
    with pytest.raises(base.ComputeError, match="trend_window"):
        detect_regime_slow(ds, px, vol, as_of=ds[-1], trend_window=0)


# ── 形状 / 确定性 / 注册 ─────────────────────────────────────────


def test_returns_are_json_safe_and_carry_params_warmup_and_evidence():
    px = _trend_down_with_deep_drawdown()  # 260 点
    ds = qd.dates(len(px))
    vol = _calm_then_vol_spike(len(px))
    for r in (
        detect_regime_slow(ds, px, vol, as_of=ds[-1]),
        detect_shock_fast(ds, px, vol, as_of=ds[-1]),
        document_regime_defaults(as_of=ds[-1], dates=ds),
    ):
        json.dumps(r, allow_nan=False)
        assert r["as_of"] == ds[-1]
        assert isinstance(r["params"], dict) and isinstance(r["warmup"], dict)
        assert r["evidence"]
        assert all(e["source"].startswith("fd-quant:") for e in r["evidence"])


def test_evidence_source_carries_tool_name_and_parameters():
    n = 260
    ds = qd.dates(n)
    r = detect_regime_slow(ds, _trend_down_with_deep_drawdown(), _calm_then_vol_spike(n),
                           as_of=ds[-1])
    src = r["evidence"][0]["source"]
    assert SLOW in src
    assert "trend_window=60" in src and "z_window=60" in src


@pytest.mark.parametrize("tool", [SLOW, FAST, DOC])
def test_every_tool_is_bitwise_deterministic(tool):
    """确定性：同输入两次调用逐位一致（IEEE 754 精确比较，含证据与参数回显）。"""
    n = 260
    ds = qd.dates(n)
    px = _trend_down_with_deep_drawdown()
    vol = _calm_then_vol_spike(n)
    calls = {
        SLOW: lambda: detect_regime_slow(ds, px, vol, as_of=ds[-1]),
        FAST: lambda: detect_shock_fast(ds, px, vol, as_of=ds[-1]),
        DOC: lambda: document_regime_defaults(as_of=ds[-1], dates=ds),
    }
    base.assert_bitequal(calls[tool](), calls[tool](), what=tool)


def test_regime_tools_are_registered_under_the_group():
    by_group = tools_by_group()
    names = {t.name for t in by_group["regime_detection"]}
    assert names == {SLOW, FAST, DOC}
    for t in by_group["regime_detection"]:
        assert t.group == "regime_detection"
        assert t.effective_level == "commercial"
