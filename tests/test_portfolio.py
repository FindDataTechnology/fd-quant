"""组合统计族测试：逐 spec Scenario 一个用例 + 定义对拍 + 边界 + 拒绝 + 确定性。

"独立对拍"在各用例内就地写一份朴素实现（不调用被测代码的任何内部函数）：
这是唯一能发现"实现与定义一起写错"的方式。spec 措辞（`spec.md` 的
`#### Scenario:` 与 `tasks.md` §3 的验证句）逐条引用在用例 docstring 里。
"""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

import quantdata as qd
from fd_quant.compute import all_tools
from fd_quant.compute.base import ComputeError, assert_bitequal
from fd_quant.compute.portfolio import (
    TOOLS,
    summarize_beta,
    summarize_calmar,
    summarize_concentration,
    summarize_drawdown,
    summarize_information_ratio,
    summarize_return,
    summarize_sharpe,
    summarize_sortino,
    summarize_turnover,
    summarize_volatility,
)

PPY = 252


# ── 独立朴素实现（只依赖定义，不碰被测代码） ─────────────────────


def naive_returns(vals):
    """简单收益 r_i = x_i/x_{i-1} - 1，独立于被测代码。"""
    return [vals[i] / vals[i - 1] - 1.0 for i in range(1, len(vals))]


def naive_mean(xs):
    total = 0.0
    for x in xs:
        total += x
    return total / len(xs)


def naive_stdev(xs):
    m = naive_mean(xs)
    acc = 0.0
    for x in xs:
        acc += (x - m) ** 2
    return math.sqrt(acc / (len(xs) - 1))


def naive_drawdown(vals):
    """最大回撤 + 峰值/谷底索引：逐点扫历史峰值，独立实现。"""
    peak_val = vals[0]
    peak_idx = 0
    dd = []
    for i, x in enumerate(vals):
        if x > peak_val:
            peak_val, peak_idx = x, i
        dd.append(x / peak_val - 1.0)
    trough_idx = min(range(len(dd)), key=lambda i: (dd[i], i))
    best_peak = max(range(trough_idx + 1), key=lambda i: (vals[i], -i))
    recovery = None
    for j in range(trough_idx + 1, len(vals)):
        if vals[j] >= vals[best_peak]:
            recovery = j
            break
    return dd, dd[trough_idx], best_peak, trough_idx, recovery


def naive_cov(a, b):
    """协方差（ddof=1），独立实现。"""
    ma, mb = naive_mean(a), naive_mean(b)
    acc = 0.0
    for x, y in zip(a, b):
        acc += (x - ma) * (y - mb)
    return acc / (len(a) - 1)


def naive_var(a):
    return naive_cov(a, a)


def _no_inf_nan(obj):
    """递归断言：没有任何 inf/NaN 混在返回里（JSON 会把它写成非法字面量）。"""
    json.dumps(obj, allow_nan=False)  # allow_nan=False 时 inf/NaN 直接抛
    if isinstance(obj, float):
        assert math.isfinite(obj), f"non-finite float in payload: {obj!r}"
    elif isinstance(obj, dict):
        for v in obj.values():
            _no_inf_nan(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _no_inf_nan(v)
    return True


# ── Scenario: 最大回撤含起止日期 ─────────────────────────────────


def test_drawdown_updown_matches_hand_computation():
    """spec `最大回撤含起止日期`：对先涨后跌的权益序列，返回回撤幅度与其峰值日、谷底日。

    夹具 `equity_updown(30, 20)`：从 100.0 起每日 +2% 共 30 日、其后每日 -2.5% 共 20 日。
    手算：峰值 = 100 × 1.02^29（最后一个上涨日，索引 29）；谷底 = 峰值 × 0.975^20（末日，
    索引 49）；回撤幅度 = 0.975^20 - 1。
    """
    ds, vals = qd.equity_updown(up=30, down=20)
    assert len(ds) == 50
    r = summarize_drawdown(ds, vals, as_of=qd.of_dates_max(ds))

    peak_i, trough_i = 29, 49
    hand_peak = 100.0 * 1.02**peak_i
    hand_trough = hand_peak * 0.975**20
    assert r["max_drawdown"] == pytest.approx(0.975**20 - 1.0, rel=1e-12)
    assert r["max_drawdown"] < 0.0
    assert r["peak_date"] == ds[peak_i]
    assert r["trough_date"] == ds[trough_i]
    assert r["recovery_date"] is None  # 谷底后无数据，未修复
    assert r["duration_days"] == 20
    assert r["peak_value"] == pytest.approx(hand_peak, rel=1e-12)
    assert r["trough_value"] == pytest.approx(hand_trough, rel=1e-12)
    # 回撤序列与输入等长、逐位 <= 0、首位为 0
    dd = r["drawdown_series"]
    assert len(dd) == 50
    assert dd[0] == 0.0
    assert all(v is not None and v <= 0.0 for v in dd)
    assert dd[trough_i] == pytest.approx(r["max_drawdown"], rel=1e-12)
    assert _no_inf_nan(r)


def test_drawdown_matches_independent_naive_implementation():
    """对拍：回撤序列/幅度/峰值/谷底/修复日与就地朴素实现逐点一致。"""
    ds, vals = qd.equity_updown(up=17, down=9)
    r = summarize_drawdown(ds, vals, as_of=qd.of_dates_max(ds))
    dd, mdd, peak, trough, recovery = naive_drawdown(vals)
    for got, exp in zip(r["drawdown_series"], dd):
        assert got == pytest.approx(exp, abs=1e-15)
    assert r["max_drawdown"] == pytest.approx(mdd, abs=1e-15)
    assert r["peak_date"] == ds[peak]
    assert r["trough_date"] == ds[trough]
    assert r["recovery_date"] == (ds[recovery] if recovery is not None else None)


def test_drawdown_recovers_reports_recovery_date():
    """修复路径：回到峰值之上时 recovery_date 非 null，且不晚于谷底。"""
    ds = qd.dates(6)
    vals = [100.0, 120.0, 90.0, 95.0, 130.0, 125.0]
    r = summarize_drawdown(ds, vals, as_of=qd.of_dates_max(ds))
    assert r["trough_date"] == ds[2]
    assert r["peak_date"] == ds[1]
    assert r["recovery_date"] == ds[4]  # 130 >= 120，首个修复日
    assert r["max_drawdown"] == pytest.approx(90.0 / 120.0 - 1.0, rel=1e-12)


def test_drawdown_monotone_rise_is_zero_not_missing():
    """从未回撤：max_drawdown = 0.0（退化事件，不是 null），duration = 0。"""
    ds = qd.dates(10)
    vals = [100.0 * 1.01**i for i in range(10)]
    r = summarize_drawdown(ds, vals, as_of=qd.of_dates_max(ds))
    assert r["max_drawdown"] == 0.0
    assert r["peak_date"] == ds[0]
    assert r["trough_date"] == ds[0]
    assert r["duration_days"] == 0
    assert r["drawdown_series"] == [0.0] * 10
    assert r["reason"] is None
    assert _no_inf_nan(r)


# ── Scenario: 无风险利率显式 ─────────────────────────────────────


def test_sharpe_default_risk_free_is_echoed_as_zero():
    """spec `无风险利率显式`：未传无风险利率 → 使用默认 0 并在返回中显式标注 risk_free = 0。"""
    ds, vals = qd.equity_updown(up=30, down=20)
    a = qd.of_dates_max(ds)
    r = summarize_sharpe(ds, vals, as_of=a)
    assert r["risk_free"] == 0.0
    assert r["params"]["risk_free"] == 0.0
    assert r["periods_per_year"] == PPY
    assert r["params"]["periods_per_year"] == PPY
    assert r["sharpe"] is not None
    # 无风险利率也出现在 evidence 里（口径可追溯）
    assert any(e["label"] == "risk_free" and e["value"] == 0.0 for e in r["evidence"])


def test_sharpe_explicit_risk_free_changes_value_and_is_echoed():
    """spec `无风险利率显式`：传入 0.02 时按 0.02 计算且标注。"""
    ds, vals = qd.equity_updown(up=30, down=20)
    a = qd.of_dates_max(ds)
    base = summarize_sharpe(ds, vals, as_of=a)
    rf = summarize_sharpe(ds, vals, as_of=a, risk_free=0.02)
    assert rf["risk_free"] == 0.02
    assert rf["params"]["risk_free"] == 0.02
    assert rf["sharpe"] != base["sharpe"]
    assert rf["sharpe"] < base["sharpe"]  # 更高的无风险利率 → 更低的 Sharpe
    # 对拍：年化超额口径
    rets = naive_returns(vals)
    excess = [x - 0.02 / PPY for x in rets]
    expect = naive_mean(excess) / naive_stdev(excess) * math.sqrt(PPY)
    assert rf["sharpe"] == pytest.approx(expect, rel=1e-12)


def test_sharpe_matches_independent_naive_implementation():
    """对拍：Sharpe 与 naive 均值/样本标准差（ddof=1）逐位接近到 1e-12。"""
    ds, vals = qd.equity_from_returns(qd.returns(60, seed=31))
    a = qd.of_dates_max(ds)
    r = summarize_sharpe(ds, vals, as_of=a, risk_free=0.015)
    excess = [x - 0.015 / PPY for x in naive_returns(vals)]
    expect = naive_mean(excess) / naive_stdev(excess) * math.sqrt(PPY)
    assert r["sharpe"] == pytest.approx(expect, rel=1e-12)
    assert r["mean_excess_return"] == pytest.approx(naive_mean(excess), rel=1e-12)
    assert r["excess_volatility"] == pytest.approx(naive_stdev(excess), rel=1e-12)
    assert r["annualization_factor"] == pytest.approx(math.sqrt(PPY), rel=1e-15)


def test_sortino_target_and_risk_free_are_echoed_and_match_naive():
    """Sortino：下行偏差只计低于 target 的期（分母取全部期数），target 与 risk_free 均回显。"""
    ds, vals = qd.equity_from_returns(qd.returns(80, seed=37))
    a = qd.of_dates_max(ds)
    r = summarize_sortino(ds, vals, as_of=a, risk_free=0.02, target=0.0005)
    assert r["risk_free"] == 0.02
    assert r["target"] == 0.0005
    assert r["params"] == {"periods_per_year": PPY, "risk_free": 0.02, "target": 0.0005}
    excess = [x - 0.02 / PPY for x in naive_returns(vals)]
    sq = [min(x - 0.0005, 0.0) ** 2 for x in excess]
    dd_dev = math.sqrt(sum(sq) / len(sq))
    expect = naive_mean(excess) / dd_dev * math.sqrt(PPY)
    assert r["downside_deviation"] == pytest.approx(dd_dev, rel=1e-12)
    assert r["sortino"] == pytest.approx(expect, rel=1e-12)
    assert r["downside_observations"] == sum(1 for x in excess if x - 0.0005 < 0.0)


def test_sortino_no_downside_is_null_with_reason():
    """全期都在 target 之上 → 下行偏差为 0 → null + 原因（不是 inf）。"""
    ds = qd.dates(10)
    vals = [100.0 * 1.01**i for i in range(10)]
    r = summarize_sortino(ds, vals, as_of=qd.of_dates_max(ds), target=-1.0)
    assert r["sortino"] is None
    assert r["downside_deviation"] == 0.0
    assert "downside" in r["reason"].lower()
    assert _no_inf_nan(r)


# ── Scenario（spec 3.2）：IR / TE / Beta 对基准 ─────────────────


def test_ir_and_te_are_zero_when_portfolio_equals_benchmark():
    """spec tasks 3.2：对构造的"组合=基准"序列 IR 与 TE 为 0（容差 1e-9）。"""
    ds, vals = qd.equity_updown(up=30, down=20)
    r = summarize_information_ratio(ds, vals, vals, as_of=qd.of_dates_max(ds))
    assert abs(r["information_ratio"] - 0.0) <= 1e-9
    assert abs(r["tracking_error"] - 0.0) <= 1e-9
    assert r["mean_active_return"] == pytest.approx(0.0, abs=1e-15)
    assert r["active_return_volatility"] == pytest.approx(0.0, abs=1e-15)
    assert r["periods_per_year"] == PPY
    assert _no_inf_nan(r)


def test_ir_matches_independent_naive_implementation():
    """对拍：IR/TE 与 naive 主动收益均值/样本标准差（ddof=1）一致。"""
    rets = qd.returns(90, seed=41)
    bench = qd.returns(90, seed=43)
    ds, vals = qd.equity_from_returns(rets)
    bds, bvals = qd.equity_from_returns(bench)
    assert ds == bds
    r = summarize_information_ratio(ds, vals, bvals, as_of=qd.of_dates_max(ds))
    active = [a - b for a, b in zip(naive_returns(vals), naive_returns(bvals))]
    vol = naive_stdev(active)
    assert r["information_ratio"] == pytest.approx(
        naive_mean(active) / vol * math.sqrt(PPY), rel=1e-12
    )
    assert r["tracking_error"] == pytest.approx(vol * math.sqrt(PPY), rel=1e-12)
    assert r["active_return_volatility"] == pytest.approx(vol, rel=1e-12)


def test_beta_is_one_when_portfolio_equals_benchmark():
    """spec tasks 3.2：对构造的"组合=基准"序列 Beta 为 1（容差 1e-9）。"""
    ds, vals = qd.equity_updown(up=30, down=20)
    r = summarize_beta(ds, vals, vals, as_of=qd.of_dates_max(ds))
    assert abs(r["beta"] - 1.0) <= 1e-9
    assert abs(r["correlation"] - 1.0) <= 1e-9
    assert r["alpha"] == pytest.approx(0.0, abs=1e-15)
    assert r["observations"] == len(vals) - 1
    assert _no_inf_nan(r)


def test_beta_and_alpha_match_independent_naive_implementation():
    """对拍：beta = cov/var、alpha = 均值收益差 - beta×基准均值（显式循环口径）。"""
    rets = qd.returns(70, seed=47)
    bench = qd.returns(70, seed=53)
    ds, vals = qd.equity_from_returns(rets)
    bds, bvals = qd.equity_from_returns(bench)
    assert ds == bds
    r = summarize_beta(ds, vals, bvals, as_of=qd.of_dates_max(ds))
    rp, rb = naive_returns(vals), naive_returns(bvals)
    var_b = naive_var(rb)
    cov = naive_cov(rp, rb)
    beta = cov / var_b
    assert r["beta"] == pytest.approx(beta, rel=1e-12)
    assert r["alpha"] == pytest.approx(naive_mean(rp) - beta * naive_mean(rb), rel=1e-12)
    assert r["correlation"] == pytest.approx(
        cov / (naive_stdev(rp) * naive_stdev(rb)), rel=1e-12
    )
    assert r["covariance"] == pytest.approx(cov, rel=1e-12)
    assert r["benchmark_variance"] == pytest.approx(var_b, rel=1e-12)
    assert -1.0 - 1e-12 <= r["correlation"] <= 1.0 + 1e-12


def test_leveraged_portfolio_beta_is_two():
    """定义对拍：组合收益 = 2 × 基准收益 ⇒ beta = 2.0（含确定性构造）。"""
    bench = qd.returns(50, seed=59)
    ds, bvals = qd.equity_from_returns(bench)
    _, vals = qd.equity_from_returns([2.0 * b for b in bench])
    r = summarize_beta(ds, vals, bvals, as_of=qd.of_dates_max(ds))
    assert r["beta"] == pytest.approx(2.0, rel=1e-12)
    assert r["correlation"] == pytest.approx(1.0, rel=1e-12)


# ── Scenario: 集中度对边界权重 ───────────────────────────────────


def test_concentration_single_weight_one():
    """spec `集中度对边界权重`：传入单一权重 1.0 → HHI = 1.0、前 1 权重占比 = 1.0。"""
    ds = [qd.as_of(1)]
    r = summarize_concentration(ds, [1.0], as_of=qd.of_dates_max(ds))
    assert r["hhi"] == pytest.approx(1.0, abs=1e-12)
    assert r["top_n_share"] == pytest.approx(1.0, abs=1e-12)
    assert r["effective_n"] == pytest.approx(1.0, abs=1e-12)
    assert r["top_n_used"] == 1
    assert r["weight_count"] == 1
    assert r["observation_date"] == ds[0]
    assert r["gross_exposure"] == pytest.approx(1.0, abs=1e-12)
    assert _no_inf_nan(r)


def test_concentration_matches_independent_naive_implementation():
    """对拍：HHI = Σw²、前 N 占比 = 最大 N 个权重 / Σw（独立排序实现）。"""
    ds = qd.dates(6)
    w = [0.05, 0.30, 0.10, 0.25, 0.20, 0.10]
    r = summarize_concentration(ds, w, as_of=qd.of_dates_max(ds), top_n=3)
    assert r["hhi"] == pytest.approx(sum(x * x for x in w), rel=1e-12)
    assert r["top_n_share"] == pytest.approx(
        sum(sorted(w, reverse=True)[:3]) / sum(w), rel=1e-12
    )
    assert r["effective_n"] == pytest.approx(1.0 / sum(x * x for x in w), rel=1e-12)
    assert r["top_n_used"] == 3


def test_concentration_equal_weights_gives_hhi_one_over_n():
    """等权 N 只：HHI = 1/N、有效持仓数 = N、前 N 占比 = 1.0。"""
    ds = qd.dates(4)
    r = summarize_concentration(ds, [0.25] * 4, as_of=qd.of_dates_max(ds), top_n=4)
    assert r["hhi"] == pytest.approx(0.25, rel=1e-12)
    assert r["effective_n"] == pytest.approx(4.0, rel=1e-12)
    assert r["top_n_share"] == pytest.approx(1.0, rel=1e-12)


def test_concentration_top_n_larger_than_holdings_is_capped():
    """top_n 超过持仓数 → 按持仓数取（top_n_used 回显实际取值）。"""
    ds = qd.dates(3)
    r = summarize_concentration(ds, [0.5, 0.3, 0.2], as_of=qd.of_dates_max(ds), top_n=99)
    assert r["top_n_used"] == 3
    assert r["top_n_share"] == pytest.approx(1.0, rel=1e-12)
    assert r["params"]["top_n"] == 99  # 参数回显实际传入值


def test_concentration_single_date_accepts_one_date_with_many_weights():
    """截面形态：单个观察日 + 多只权重（长度不等时视为单一截面）。"""
    ds = [qd.dates(10)[-1]]
    r = summarize_concentration(ds, [0.6, 0.4], as_of=ds[0], top_n=1)
    assert r["weight_count"] == 2
    assert r["observation_date"] == ds[0]
    assert r["top_n_used"] == 1
    assert r["top_n_share"] == pytest.approx(0.6, rel=1e-12)
    assert r["hhi"] == pytest.approx(0.36 + 0.16, rel=1e-12)
    assert r["cash_weight"] == pytest.approx(0.0, abs=1e-12)


# ── Scenario（spec 3.3）：换手率 ─────────────────────────────────


def test_turnover_two_period_swap_is_one():
    """spec tasks 3.3：两期权重序列互换时换手率按定义计算正确（[1,0] → [0,1] 为 1.0）。"""
    ds = [qd.as_of(1)]
    r = summarize_turnover(ds, [1.0, 0.0], as_of=qd.of_dates_max(ds), weights_prior=[0.0, 1.0])
    assert r["turnover"] == pytest.approx(1.0, abs=1e-12)
    assert r["baseline"] == "explicit"
    assert r["position_count"] == 2
    assert _no_inf_nan(r)


def test_turnover_identical_vectors_is_zero():
    """同一组权重 → 换手率 0.0。"""
    ds = [qd.as_of(1)]
    r = summarize_turnover(ds, [0.4, 0.35, 0.25], as_of=qd.of_dates_max(ds),
                           weights_prior=[0.4, 0.35, 0.25])
    assert r["turnover"] == 0.0
    assert r["baseline"] == "explicit"


def test_turnover_matches_independent_naive_implementation():
    """对拍：换手率 = 0.5 × Σ|w_t - w_{t-1}|。"""
    ds = qd.dates(5)
    cur = [0.20, 0.15, 0.30, 0.25, 0.10]
    prior = [0.30, 0.10, 0.25, 0.25, 0.10]
    r = summarize_turnover(ds, cur, as_of=qd.of_dates_max(ds), weights_prior=prior)
    expect = 0.5 * sum(abs(a - b) for a, b in zip(cur, prior))
    assert r["turnover"] == pytest.approx(expect, rel=1e-12)
    # 手算：0.10+0.05+0.05+0.0+0.0 = 0.20，取半 = 0.10
    assert expect == pytest.approx(0.10, rel=1e-12)


def test_turnover_partial_rotation_counts_names_traded():
    """给定两期 symbols 时输出 names_traded（逐位不等的标的数）。"""
    ds = [qd.as_of(1)]
    r = summarize_turnover(
        ds, [0.5, 0.3, 0.2], as_of=qd.of_dates_max(ds),
        weights_prior=[0.5, 0.2, 0.3], symbols=["A", "B", "C"], symbols_prior=["A", "B", "C"],
    )
    assert r["turnover"] == pytest.approx(0.5 * (0.0 + 0.1 + 0.1), rel=1e-12)
    assert r["names_traded"] == 2  # B 与 C 变了，A 没变


def test_turnover_omitted_prior_is_zero_baseline_initial_build():
    """缺上期权重 → 按全零（现金）基线，即初始建仓换手，并在 baseline 键回显。"""
    ds = [qd.as_of(1)]
    r = summarize_turnover(ds, [0.6, 0.2], as_of=qd.of_dates_max(ds))
    assert r["baseline"] == "zero"
    assert r["turnover"] == pytest.approx(0.5 * 0.8, rel=1e-12)
    assert "names_traded" not in r  # 未给 symbols → 不含该键（不是 null）


def test_turnover_aligns_by_symbol_not_by_position():
    """同一组合换个顺序传入 → 没发生轮动，换手必须为 0。

    逐位相减会把「B,A」与「A,B」当成两次全量买卖，报出 1.0 的换手——一个
    从未交易过的组合因此看起来像做了一次完整往返。标的身份才是口径。
    """
    ds = [qd.as_of(1)]
    r = summarize_turnover(
        ds, [0.4, 0.6], as_of=qd.of_dates_max(ds),
        weights_prior=[0.6, 0.4],
        symbols=["B", "A"], symbols_prior=["A", "B"],
    )
    assert r["turnover"] == 0.0
    assert r["names_traded"] == 0
    assert r["turnover_basis"] == "aligned"


def test_turnover_detects_real_rotation_after_reordering():
    """顺序打乱后仍能识别真实轮动：A 0.6→0.4、B 0.4→0.6，换手 0.2。"""
    ds = [qd.as_of(1)]
    r = summarize_turnover(
        ds, [0.6, 0.4], as_of=qd.of_dates_max(ds),
        weights_prior=[0.6, 0.4],
        symbols=["B", "A"], symbols_prior=["A", "B"],
    )
    assert r["turnover"] == pytest.approx(0.2, rel=1e-12)
    assert r["names_traded"] == 2


def test_turnover_counts_exit_and_entry_as_full_weight():
    """一进一出：B 清仓、C 新进，各按全额权重计入换手（0.5 × (0.4+0.4)）。"""
    ds = [qd.as_of(1)]
    r = summarize_turnover(
        ds, [0.6, 0.4], as_of=qd.of_dates_max(ds),
        weights_prior=[0.6, 0.4],
        symbols=["A", "C"], symbols_prior=["A", "B"],
    )
    assert r["turnover"] == pytest.approx(0.4, rel=1e-12)
    assert r["names_traded"] == 2


def test_turnover_without_symbols_declares_positional_basis():
    """缺标的代码时退回逐位口径，并在返回中声明该口径。"""
    ds = [qd.as_of(1)]
    r = summarize_turnover(
        ds, [0.6, 0.4], as_of=qd.of_dates_max(ds), weights_prior=[0.4, 0.6],
    )
    assert r["turnover_basis"] == "positional"
    assert r["params"]["turnover_basis"] == "positional"
    assert "names_traded" not in r


def test_turnover_alignment_is_deterministic():
    """同一输入两次调用逐位一致（标的后排序归约）。"""
    ds = [qd.as_of(1)]
    kw = dict(
        weights_prior=[0.6, 0.4],
        symbols=["B", "A"], symbols_prior=["A", "B"],
    )
    a = summarize_turnover(ds, [0.6, 0.4], as_of=qd.of_dates_max(ds), **kw)
    b = summarize_turnover(ds, [0.6, 0.4], as_of=qd.of_dates_max(ds), **kw)
    assert a["turnover"] == b["turnover"]
    assert a["names_traded"] == b["names_traded"]


# ── 年化收益 / 年化波动 ─────────────────────────────────────────


def test_return_annualizes_from_total_over_span():
    """年化口径：(末值/首值)^(periods_per_year/期数) - 1，periods_per_year 回显。"""
    n = 101
    ds = qd.dates(n)
    vals = [100.0 * 1.001**i for i in range(n)]
    r = summarize_return(ds, vals, as_of=qd.of_dates_max(ds))
    assert r["total_return"] == pytest.approx(vals[-1] / vals[0] - 1.0, rel=1e-12)
    assert r["annualized_return"] == pytest.approx(
        (vals[-1] / vals[0]) ** (PPY / 100) - 1.0, rel=1e-12
    )
    assert r["periods_per_year"] == PPY
    assert r["intervals"] == 100
    assert r["params"] == {"periods_per_year": PPY}
    assert r["warmup"] == {"total_return": 0, "annualized_return": 0}


def test_return_periods_per_year_is_configurable_and_echoed():
    """年化因子可配（默认 252）：传 12 时按 12 算且回显。"""
    n = 13
    ds = qd.dates(n)
    vals = [100.0 * 1.01**i for i in range(n)]
    r = summarize_return(ds, vals, as_of=qd.of_dates_max(ds), periods_per_year=12)
    assert r["periods_per_year"] == 12
    assert r["params"]["periods_per_year"] == 12
    assert r["annualized_return"] == pytest.approx((vals[-1] / vals[0]) ** (12 / 12) - 1.0, rel=1e-12)


def test_volatility_annualizes_sample_stdev_and_echoes_factor():
    """年化波动 = 简单收益样本标准差 (ddof=1) × sqrt(periods_per_year)，因子回显。"""
    ds, vals = qd.equity_from_returns(qd.returns(120, seed=61))
    r = summarize_volatility(ds, vals, as_of=qd.of_dates_max(ds))
    per = naive_stdev(naive_returns(vals))
    assert r["per_period_volatility"] == pytest.approx(per, rel=1e-12)
    assert r["volatility"] == pytest.approx(per * math.sqrt(PPY), rel=1e-12)
    assert r["annualization_factor"] == pytest.approx(math.sqrt(PPY), rel=1e-15)
    assert r["periods_per_year"] == PPY
    assert r["observations"] == len(vals) - 1


def test_calmar_is_annualized_return_over_abs_drawdown():
    """Calmar = 年化收益 / |最大回撤|，与独立算的两个分量一致。"""
    ds, vals = qd.equity_updown(up=30, down=20)
    a = qd.of_dates_max(ds)
    r = summarize_calmar(ds, vals, as_of=a)
    ann = summarize_return(ds, vals, as_of=a)["annualized_return"]
    mdd = summarize_drawdown(ds, vals, as_of=a)["max_drawdown"]
    assert r["annualized_return"] == pytest.approx(ann, rel=1e-12)
    assert r["max_drawdown"] == pytest.approx(mdd, rel=1e-12)
    assert r["calmar"] == pytest.approx(ann / abs(mdd), rel=1e-12)
    assert _no_inf_nan(r)


def test_calmar_zero_drawdown_is_null_with_note():
    """Calmar 零回撤：分母为 0 → null + 说明，绝不返回 inf。"""
    ds = qd.dates(20)
    vals = [100.0 * 1.01**i for i in range(20)]
    r = summarize_calmar(ds, vals, as_of=qd.of_dates_max(ds))
    assert r["calmar"] is None
    assert r["max_drawdown"] == 0.0
    assert r["annualized_return"] is not None  # 分子照给，只有比率被拒绝
    assert "0" in r["reason"] and "not infinite" in r["reason"]
    assert _no_inf_nan(r)


# ── 零方差 / 退化统计量：拒绝而非 inf/NaN ───────────────────────


def test_zero_variance_equity_refuses_instead_of_inf():
    """常数权益（零波动）：Sharpe 与 Sortino 给 null + 明确原因，JSON 里无 inf/NaN。"""
    ds = qd.dates(30)
    vals = [100.0] * 30
    a = qd.of_dates_max(ds)

    sh = summarize_sharpe(ds, vals, as_of=a)
    assert sh["sharpe"] is None
    assert sh["excess_volatility"] == 0.0
    assert "standard deviation is 0" in sh["reason"]
    assert "not infinite" in sh["reason"]
    assert _no_inf_nan(sh)

    so = summarize_sortino(ds, vals, as_of=a)
    assert so["sortino"] is None
    assert "downside" in so["reason"].lower()
    assert _no_inf_nan(so)

    # 零波动本身是良定义的 0（不是拒绝）
    vol = summarize_volatility(ds, vals, as_of=a)
    assert vol["volatility"] == 0.0
    assert vol["reason"] is None

    # 回撤为 0、Calmar 被拒绝
    dd = summarize_drawdown(ds, vals, as_of=a)
    assert dd["max_drawdown"] == 0.0
    assert summarize_calmar(ds, vals, as_of=a)["calmar"] is None


def test_zero_variance_benchmark_refuses_beta_instead_of_inf():
    """常数基准（基准方差为 0）→ beta 为 null + 原因，不是 inf。"""
    ds = qd.dates(20)
    port = [100.0 * 1.01**i for i in range(20)]
    flat = [50.0] * 20
    r = summarize_beta(ds, port, flat, as_of=qd.of_dates_max(ds))
    assert r["beta"] is None
    assert r["benchmark_variance"] == 0.0
    assert "variance is 0" in r["reason"]
    assert _no_inf_nan(r)


def test_constant_active_return_with_nonzero_mean_refuses_ir():
    """主动收益零方差但均值非 0 → IR 为 null + 原因（不是 inf）。

    用二进制精确可表示的构造：基准收益 [1.0, -0.5]、组合收益 [2.5, 1.0]，
    主动收益恒为 1.5（逐位精确），故样本标准差恰为 0.0 而均值非 0。
    """
    ds = qd.dates(3)
    bvals = [100.0, 200.0, 100.0]   # 收益 [1.0, -0.5]，均精确
    vals = [100.0, 350.0, 700.0]    # 收益 [2.5, 1.0]，均精确
    r = summarize_information_ratio(ds, vals, bvals, as_of=qd.of_dates_max(ds))
    assert r["active_return_volatility"] == 0.0
    assert r["mean_active_return"] == pytest.approx(1.5, abs=1e-15)
    assert r["information_ratio"] is None
    assert r["tracking_error"] == pytest.approx(0.0, abs=1e-12)
    assert "standard deviation" in r["reason"]
    assert _no_inf_nan(r)


# ── as_of 与越界拒绝 ────────────────────────────────────────────


@pytest.mark.parametrize("spec", TOOLS, ids=[t.name for t in TOOLS])
def test_missing_as_of_is_refused_first(spec):
    """as_of 是第一个校验动作：None 时错误文本含 as_of（不会先被别的参数错误遮住）。"""
    import inspect

    d = qd.ohlcv(10, seed=71)
    kw = {"dates": d["dates"]}
    for p in inspect.signature(spec.fn).parameters:
        if p in ("dates", "as_of"):
            continue
        if p in ("values", "close"):
            kw[p] = d["close"]
        elif p in ("benchmark", "benchmark_returns"):
            kw[p] = d["close"]
        elif p == "weights":
            kw[p] = [0.5, 0.5]
        elif p == "weights_prior":
            kw[p] = [0.4, 0.6]
    with pytest.raises(ComputeError, match="as_of"):
        spec.fn(as_of=None, **kw)


@pytest.mark.parametrize("spec", TOOLS, ids=[t.name for t in TOOLS])
def test_future_data_is_refused_naming_first_date(spec):
    """越界拒绝且点名**首个**越界日期（不静默截断）。"""
    import inspect

    d = qd.ohlcv(30, seed=73)
    kw = {"dates": d["dates"], "as_of": d["dates"][5]}
    for p in inspect.signature(spec.fn).parameters:
        if p in ("dates", "as_of"):
            continue
        if p in ("values", "close"):
            kw[p] = d["close"]
        elif p in ("benchmark", "benchmark_returns"):
            kw[p] = d["close"]
        elif p == "weights":
            kw[p] = [0.5, 0.5]
        elif p == "weights_prior":
            kw[p] = [0.4, 0.6]
    first_future = d["dates"][6]
    with pytest.raises(ComputeError) as ei:
        spec.fn(**kw)
    assert first_future in str(ei.value)
    # 错误还必须指明是哪一列越界（调用方据此知道该剪谁）
    assert "values" in str(ei.value) or "weights" in str(ei.value)
    assert "as_of" in str(ei.value)


_NAV_TOOLS = [t for t in TOOLS if t.name != "summarize_concentration" and t.name != "summarize_turnover"]


@pytest.mark.parametrize("spec", _NAV_TOOLS, ids=[t.name for t in _NAV_TOOLS])
def test_non_positive_nav_is_refused_naming_the_date(spec):
    """非正 NAV：报错点名该日期与值（几何收益/回撤需要严格正水位）。"""
    import inspect

    ds = qd.dates(10)
    bad = [100.0, 101.0, 0.0, 99.0, 98.0, 97.0, 96.0, 95.0, 94.0, 93.0]
    kw = {"dates": ds, "as_of": qd.of_dates_max(ds), "values": bad}
    if "benchmark" in inspect.signature(spec.fn).parameters:
        kw["benchmark"] = [100.0 * 1.001**i for i in range(10)]
    with pytest.raises(ComputeError) as ei:
        spec.fn(**kw)
    assert ds[2] in str(ei.value)
    assert "non-positive" in str(ei.value)
    assert "0.0" in str(ei.value)


def test_non_positive_nav_names_the_date_in_return_error():
    """非正 NAV 的错误文本点名日期（spec 要求"clear error naming the date"）。"""
    ds = qd.dates(5)
    vals = [100.0, -5.0, 100.0, 101.0, 102.0]
    with pytest.raises(ComputeError) as ei:
        summarize_return(ds, vals, as_of=qd.of_dates_max(ds))
    assert ds[1] in str(ei.value)
    assert "non-positive" in str(ei.value)


# ── 边界：空 / 过短 / 全缺失 / 极端值 ───────────────────────────


def test_empty_and_too_short_inputs_are_refused_with_field_names():
    """空序列与长度不足：错误文本点名字段与所需点数。"""
    with pytest.raises(ComputeError, match="values is empty"):
        summarize_return([], [], as_of=qd.as_of(1))
    with pytest.raises(ComputeError) as ei:
        summarize_return([qd.dates(1)[0]], [100.0], as_of=qd.as_of(1))
    assert "at least 2 points" in str(ei.value)
    with pytest.raises(ComputeError) as ei:
        summarize_sharpe(qd.dates(2), [100.0, 101.0], as_of=qd.as_of(2))
    assert "at least 3 points" in str(ei.value)
    with pytest.raises(ComputeError, match="length mismatch"):
        summarize_return(qd.dates(5), [100.0, 101.0], as_of=qd.as_of(5))


def test_length_mismatch_between_two_series_names_the_offending_column():
    """两列不等长：错误点名是哪一列（benchmark / weights_prior）。"""
    ds = qd.dates(10)
    vals = [100.0 * 1.001**i for i in range(10)]
    with pytest.raises(ComputeError) as ei:
        summarize_beta(ds, vals, [100.0] * 5, as_of=qd.of_dates_max(ds))
    assert "benchmark" in str(ei.value)
    with pytest.raises(ComputeError) as ei:
        summarize_turnover(ds, [0.5, 0.5], as_of=qd.of_dates_max(ds), weights_prior=[0.3, 0.3, 0.4])
    assert "weights_prior" in str(ei.value)


def test_all_missing_series_yields_null_with_reason():
    """全缺失序列：每个统计量都给 null + reason，不抛异常也不出 NaN。"""
    ds = qd.dates(10)
    vals = [None] * 10
    a = qd.of_dates_max(ds)
    for r, key in (
        (summarize_return(ds, vals, as_of=a), "annualized_return"),
        (summarize_volatility(ds, vals, as_of=a), "volatility"),
        (summarize_drawdown(ds, vals, as_of=a), "max_drawdown"),
        (summarize_sharpe(ds, vals, as_of=a), "sharpe"),
        (summarize_sortino(ds, vals, as_of=a), "sortino"),
        (summarize_calmar(ds, vals, as_of=a), "calmar"),
        (summarize_beta(ds, vals, [100.0 * 1.01**i for i in range(10)], as_of=a), "beta"),
    ):
        assert r[key] is None, key
        assert "missing" in r["reason"], key
        assert _no_inf_nan(r)
    dd = summarize_drawdown(ds, vals, as_of=a)
    assert dd["drawdown_series"] == [None] * 10
    assert dd["peak_date"] is None and dd["trough_date"] is None


def test_nan_inside_series_is_reported_with_date_and_never_inf():
    """序列中间有 NaN：数值键为 null 且 reason 点名该日期；JSON 里无 NaN/inf。"""
    ds = qd.dates(6)
    vals = [100.0, 101.0, None, 102.0, 103.0, 99.0]
    a = qd.of_dates_max(ds)
    r = summarize_drawdown(ds, vals, as_of=a)
    assert r["max_drawdown"] is None
    assert ds[2] in r["reason"]
    assert _no_inf_nan(r)
    vol = summarize_volatility(ds, vals, as_of=a)
    assert vol["volatility"] is None and "missing" in vol["reason"]


def test_two_point_series_shapes_are_still_json_safe():
    """最短可用长度（2 点）：可算的不抛、不可算的给 null，且返回恒可 JSON 化。"""
    ds = qd.dates(2)
    vals = [100.0, 110.0]
    a = qd.of_dates_max(ds)
    r = summarize_return(ds, vals, as_of=a)
    assert r["total_return"] == pytest.approx(0.1, rel=1e-12)
    assert len(r["evidence"]) >= 1
    assert _no_inf_nan(r)
    dd = summarize_drawdown(ds, vals, as_of=a)
    assert dd["drawdown_series"] == [0.0, 0.0]
    assert _no_inf_nan(dd)


def test_extreme_return_overflow_is_refused_not_inf():
    """极端值：年化放大到溢出 float64 → null + 原因，绝不返回 inf。

    ratio = 1e290/1e-10 = 1e300 本身有限（故 total_return 有值），但再取
    (252/1) 次幂即溢出——溢出必须表现为 null + 原因，不是 inf。
    """
    ds = qd.dates(2)
    vals = [1e-10, 1e290]
    r = summarize_return(ds, vals, as_of=qd.of_dates_max(ds))
    assert r["annualized_return"] is None
    assert "overflow" in r["reason"] or "not finite" in r["reason"]
    assert r["total_return"] is not None
    assert r["intervals"] == 1
    assert _no_inf_nan(r)


def test_ratio_itself_overflowing_still_yields_json_safe_null():
    """比值的构造本身就溢出（1e300/1e-300）→ 两个收益均为 null + 原因，无 inf。"""
    ds = qd.dates(2)
    vals = [1e-300, 1e300]
    r = summarize_return(ds, vals, as_of=qd.of_dates_max(ds))
    assert r["total_return"] is None
    assert r["annualized_return"] is None
    assert r["reason"] is not None
    assert _no_inf_nan(r)


def test_zero_weight_vector_is_refused_with_field_name():
    """全零权重的集中度无从定义 → 报错点名 weights。"""
    ds = qd.dates(3)
    with pytest.raises(ComputeError) as ei:
        summarize_concentration(ds, [0.0, 0.0, 0.0], as_of=qd.of_dates_max(ds))
    assert "weights" in str(ei.value)


def test_negative_weight_concentration_reports_refusal_not_a_number():
    """带符号权重向量：集中度定义在多头的权重单纯形上 → null + 原因（不出数）。"""
    ds = qd.dates(3)
    r = summarize_concentration(ds, [0.6, -0.2, 0.6], as_of=qd.of_dates_max(ds))
    assert r["hhi"] is None and r["top_n_share"] is None
    assert r["weights_long_only"] is False
    assert "negative" in r["reason"]
    assert _no_inf_nan(r)


def test_invalid_scalar_params_are_refused_by_name():
    """非法标量参数：报错点名参数名（不静默取默认值）。"""
    ds, vals = qd.equity_updown(up=10, down=5)
    a = qd.of_dates_max(ds)
    with pytest.raises(ComputeError, match="periods_per_year"):
        summarize_return(ds, vals, as_of=a, periods_per_year=0)
    with pytest.raises(ComputeError, match="periods_per_year"):
        summarize_sharpe(ds, vals, as_of=a, periods_per_year=2.5)
    with pytest.raises(ComputeError, match="risk_free"):
        summarize_sharpe(ds, vals, as_of=a, risk_free=float("nan"))
    with pytest.raises(ComputeError, match="target"):
        summarize_sortino(ds, vals, as_of=a, target=float("inf"))
    with pytest.raises(ComputeError, match="top_n"):
        summarize_concentration(qd.dates(3), [0.5, 0.5], as_of=qd.as_of(3), top_n=0)


def test_symbols_without_symbols_prior_is_refused_by_name():
    """只给半组 symbols：报错点名缺的那个参数。"""
    ds = [qd.as_of(1)]
    with pytest.raises(ComputeError) as ei:
        summarize_turnover(ds, [1.0], as_of=ds[0], symbols=["A"])
    assert "symbols_prior" in str(ei.value) and "together" in str(ei.value)


# ── 确定性（逐位一致）──────────────────────────────────────────


@pytest.mark.parametrize("spec", TOOLS, ids=[t.name for t in TOOLS])
def test_bitwise_determinism_every_tool(spec):
    """spec `同输入重跑逐位一致`：任一工具同输入两次逐位相等（IEEE 754 精确比较）。"""
    import inspect

    d = qd.ohlcv(80, seed=79)
    kw = {"dates": d["dates"], "as_of": d["dates"][-1]}
    for p in inspect.signature(spec.fn).parameters:
        if p in ("dates", "as_of"):
            continue
        if p in ("values", "close"):
            kw[p] = d["close"]
        elif p in ("benchmark", "benchmark_returns"):
            kw[p] = d["high"]
        elif p == "weights":
            kw[p] = [0.4, 0.35, 0.25]
        elif p == "weights_prior":
            kw[p] = [0.3, 0.4, 0.3]
    assert_bitequal(spec.fn(**kw), spec.fn(**kw), what=spec.name)


@pytest.mark.parametrize("spec", TOOLS, ids=[t.name for t in TOOLS])
def test_every_return_is_json_serialisable_without_nan(spec):
    """返回必须能 `json.dumps(..., allow_nan=False)`——inf/NaN 会被写成非法字面量。"""
    import inspect

    d = qd.ohlcv(80, seed=83)
    kw = {"dates": d["dates"], "as_of": d["dates"][-1]}
    for p in inspect.signature(spec.fn).parameters:
        if p in ("dates", "as_of"):
            continue
        if p in ("values", "close"):
            kw[p] = d["close"]
        elif p in ("benchmark", "benchmark_returns"):
            kw[p] = d["high"]
        elif p == "weights":
            kw[p] = [0.4, 0.35, 0.25]
        elif p == "weights_prior":
            kw[p] = [0.3, 0.4, 0.3]
    r = spec.fn(**kw)
    json.dumps(r, allow_nan=False)
    assert r["params"] is not None and "warmup" in r
    assert isinstance(r["warmup"], dict)
    # warmup 对序列型输出是 0（这些统计量不消耗窗口）
    assert all(isinstance(v, int) for v in r["warmup"].values())


def test_all_tools_registered_under_portfolio_group():
    """族注册：10 个工具全部在 portfolio_statistics 组，名字唯一且带正确动词前缀。"""
    assert len(TOOLS) == 10
    assert {t.name for t in TOOLS} == {
        "summarize_return", "summarize_volatility", "summarize_drawdown", "summarize_sharpe",
        "summarize_sortino", "summarize_calmar", "summarize_information_ratio",
        "summarize_beta", "summarize_concentration", "summarize_turnover",
    }
    for t in TOOLS:
        assert t.group == "portfolio_statistics"
        assert t.name.startswith("summarize_")
    registry = {t.name for t in all_tools()}
    assert {t.name for t in TOOLS} <= registry
