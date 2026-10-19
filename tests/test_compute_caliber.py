"""计算面三条口径兑现（quant-compute-plane 组 4）。

三条此前都是"契约写了、实现没兑现"，且都**不报错**——数字照常返回，只是错的：

1. **年化频率**（4.1）：默认 252 是**交易日**口径，但从不看日期轴。自然日序列
   （含周末长假）按 252 年化会系统性低估——偏差静默，下游对比才发现对不上。
2. **蒙特卡洛负尾部**（4.2）：正态增长路径可以让终值为负，而"负的企业价值"在
   多数估值语境下无意义。此前照常返回负值带，下游可能把 -11 读成"估值 11"。
3. **KDJ / OBV 的 NaN 纪律**（4.3）：模块头声明"递推类自首个 NaN 起其后全为
   null"，但两者在缺失后都**恢复**输出——照文档写的假设会被悄悄违背。
"""
from __future__ import annotations

import datetime
import sys

import numpy as np
import pytest

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "tests"))

from fd_quant.compute.portfolio import (  # noqa: E402
    summarize_calmar, summarize_information_ratio, summarize_return,
    summarize_sharpe, summarize_sortino, summarize_volatility,
)
from fd_quant.compute.technical import compute_kdj, compute_obv  # noqa: E402
from fd_quant.compute.valuation import estimate_monte_carlo  # noqa: E402
from fd_quant.compute.base import ComputeError  # noqa: E402


def _dates(n: int, start: str = "2025-01-01") -> list[str]:
    d0 = datetime.date.fromisoformat(start)
    return [(d0 + datetime.timedelta(days=i)).isoformat() for i in range(n)]


# ── 4.1 年化频率声明 ────────────────────────────────────────────

def test_annualization_marks_frequency_unverified_by_default():
    """未给 frequency 时必须显式标注"未经验证"。

    这条标注是唯一能阻止"自然日序列被按 252 年化"的东西——偏差本身不报错。
    """
    dates = _dates(10)
    vals = [100.0 * 1.001 ** i for i in range(10)]
    r = summarize_return(dates, vals, dates[-1])
    assert r["frequency_verified"] is False
    assert "未经验证" in r["frequency_note"]
    # 提示应指出后果（低估）而非只说"没验"
    assert "365" in r["frequency_note"]


def test_annualization_hints_the_inferred_frequency():
    dates = _dates(40)
    vals = [100.0 * 1.001 ** i for i in range(40)]
    r = summarize_return(dates, vals, dates[-1])
    assert "daily" in r["frequency_note"], "应给出由日期轴推断出的频率供参考"
    assert "中位间隔" in r["frequency_note"]


def test_declared_frequency_matching_periods_per_year_is_verified():
    dates = _dates(10)
    vals = [100.0 * 1.001 ** i for i in range(10)]
    r = summarize_return(dates, vals, dates[-1], periods_per_year=252, frequency="daily")
    assert r["frequency_verified"] is True
    assert r["frequency"] == "daily"
    assert r["frequency_note"] is None


def test_declared_frequency_contradicting_periods_per_year_is_rejected():
    """声明与传参矛盾时拒绝并点名两个数——静默取一个会让人不知道用的是哪个。"""
    dates = _dates(10)
    vals = [100.0 * 1.001 ** i for i in range(10)]
    with pytest.raises(ComputeError) as exc:
        summarize_return(dates, vals, dates[-1], periods_per_year=252, frequency="monthly")
    assert "252" in str(exc.value) and "12" in str(exc.value)


def test_unknown_frequency_is_rejected_naming_the_valid_set():
    dates = _dates(10)
    vals = [100.0 * 1.001 ** i for i in range(10)]
    with pytest.raises(ComputeError) as exc:
        summarize_return(dates, vals, dates[-1], frequency="fortnightly")
    assert "daily" in str(exc.value), "拒绝时应给出合法取值"


@pytest.mark.parametrize("fn,extra", [
    (summarize_return, {}),
    (summarize_volatility, {}),
])
def test_frequency_fields_present_on_annualizing_tools(fn, extra):
    dates = _dates(10)
    vals = [100.0 * 1.001 ** i for i in range(10)]
    r = fn(dates, vals, dates[-1], **extra)
    assert "frequency_verified" in r and "frequency_note" in r


def test_sharpe_sortino_calmar_ir_also_carry_the_annotation():
    """六个年化类工具都必须有，漏一个就有一个静默偏差的口子。"""
    dates = _dates(30)
    vals = [100.0 * (1.001 ** i) for i in range(30)]
    rets = [0.001] * 30
    r1 = summarize_sharpe(dates, rets, dates[-1], risk_free=0.0)
    r2 = summarize_sortino(dates, rets, dates[-1], risk_free=0.0, target=0.0)
    r3 = summarize_calmar(dates, vals, dates[-1])
    r4 = summarize_information_ratio(dates, rets, rets, dates[-1])
    for r in (r1, r2, r3, r4):
        assert "frequency_verified" in r, f"{r.get('tool')} 缺频率标注"


# ── 4.2 蒙特卡洛负值尾部 ───────────────────────────────────────

def test_monte_carlo_flags_a_negative_tail():
    """mean=-0.5 / std=0.6 的夹具必然产出负值终值——必须标出且给出占比。"""
    r = estimate_monte_carlo(
        base_value=100.0, as_of="2025-01-01", growth_mean=-0.5, growth_std=0.6,
        discount_rate=0.1, n_simulations=1000, seed=42, horizon=5)
    assert r["tail_negative"] is True
    assert r["tail_negative_share"] > 0
    assert r["tail_negative_note"], "标注必须带说明，不能只有一个布尔值"
    assert "负" in r["tail_negative_note"]
    # 最左分位本身为负——这是"负"的直接证据
    assert r["p5"] < 0


def test_monte_carlo_leaves_a_positive_distribution_unflagged():
    r = estimate_monte_carlo(
        base_value=100.0, as_of="2025-01-01", growth_mean=0.1, growth_std=0.05,
        discount_rate=0.1, n_simulations=1000, seed=42, horizon=5)
    assert r["tail_negative"] is False
    assert r["tail_negative_share"] == 0.0
    assert r["tail_negative_note"] is None


def test_negative_tail_is_annotated_not_clipped():
    """标注而非裁剪：把负值裁掉等于替调用方决定假设可接受。

    负终值是**正态增长假设**的后果，不是计算错误；如实呈现才能让调用方判断
    该假设是否该换成对数正态。
    """
    r = estimate_monte_carlo(
        base_value=100.0, as_of="2025-01-01", growth_mean=-0.5, growth_std=0.6,
        discount_rate=0.1, n_simulations=1000, seed=42, horizon=5)
    assert min(r["percentile_values"].values()) < 0, "负值应保留在分位带里"
    assert r["p5"] == r["percentile_values"]["p5"], "顶层分位键与映射应一致"


# ── 4.3 KDJ / OBV 的 NaN 纪律 ─────────────────────────────────

def _ohlc(n=20, seed=13):
    import quantdata as qd
    return qd.ohlcv(n, seed=seed)


def test_kdj_clean_series_still_produces_values():
    """回归防线：修 NaN 纪律不能把正常序列也变成全 null。

    第一版实现在这里栽过——warmup 区的 null（前 n_period-1 位，设计如此）被
    当成了污染，导致干净序列也全为 null。
    """
    d = _ohlc(20)
    r = compute_kdj(d["dates"], d["high"], d["low"], d["close"], as_of=max(d["dates"]))
    vals = [v for v in r["k"] if v is not None]
    assert vals, "干净序列必须有输出"
    assert all(0.0 <= v <= 100.0 for v in vals)


def test_kdj_stops_at_the_first_gap():
    """递推指标自首个缺失起其后全为 null——与模块头声明一致。"""
    d = _ohlc(20)
    close = list(d["close"])
    close[10] = None
    r = compute_kdj(d["dates"], d["high"], d["low"], close, as_of=max(d["dates"]))
    assert r["k"][9] is not None, "缺失之前应照常输出"
    assert all(v is None for v in r["k"][10:]), "缺失之后不得恢复"
    assert all(v is None for v in r["j"][10:])


def test_obv_stop_at_the_first_gap():
    """OBV 是累加指标：断过之后无法在不发明数据的前提下续上。"""
    d = _ohlc(20)
    close, vol = list(d["close"]), list(d["volume"])
    close[10] = None
    vol[10] = None
    r = compute_obv(d["dates"], close, vol, as_of=max(d["dates"]))
    assert r["value"][9] is not None
    assert all(v is None for v in r["value"][10:])


def test_obv_clean_series_still_accumulates():
    d = _ohlc(20)
    r = compute_obv(d["dates"], d["close"], d["volume"], as_of=max(d["dates"]))
    assert r["value"][0] == 0.0
    assert len([v for v in r["value"] if v is not None]) == 20


def test_nan_discipline_matches_the_module_declaration():
    """KDJ 与 OBV 都属递推/累加类，行为须与 EMA/RSI 同族一致。

    这条是"文档与实现一致"的机器化：模块头声明递推类自首个 NaN 起全为 null，
    两个工具此前是例外，现在必须合流。
    """
    d = _ohlc(20)
    close = list(d["close"])
    close[10] = None
    for fn in (lambda: compute_kdj(d["dates"], d["high"], d["low"], close,
                                   as_of=max(d["dates"]))["d"],
               lambda: compute_obv(d["dates"], close, d["volume"],
                                   as_of=max(d["dates"]))["value"]):
        vals = fn()
        first_null = next(i for i, v in enumerate(vals) if v is None and i >= 10)
        assert all(v is None for v in vals[first_null:]), "缺失之后出现恢复值"
