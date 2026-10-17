"""估值算子族测试：DCF（WACC 分步 + 两法终值）、DDM、可比倍数、Monte Carlo、
敏感性网格、以及"扣值不扣结论"合成。

对拍纪律与 `test_technical.py` 一致：关键定义在用例内**就地**写一份朴素实现
（纯 python 循环 / 显式折现因子），不调用被测代码的任何内部函数——那是唯一能
发现"实现与定义一起写错"的方式。

随机性纪律：Monte Carlo 的"同种子逐位可复现"用 `assert_bitequal` 断言；此外用
**另一个**随机流（本文件自带的 LCG + Box–Muller，与 `quantdata.py` 同一构造）独立
重跑同一模型，把百分位带与实现值做统计对拍——换一条流仍落在同一分布上，才说明
实现算的是"这个模型"，而不是"某条位流"。
"""
from __future__ import annotations

import math

import pytest

import quantdata as qd
from fd_quant.compute.base import ComputeError, assert_bitequal
from fd_quant.compute.valuation import (
    build_valuation_synthesis,
    compute_comparable_multiples,
    compute_sensitivity_grid,
    estimate_dcf,
    estimate_ddm,
    estimate_monte_carlo,
)
from fd_quant.contracts.models import Evidence, Producer, Rating

AS_OF = "2026-06-30"


# ── 本文件私有的朴素实现对拍工具 ─────────────────────────────────


def _pv(rate: float, flows) -> float:
    """独立折现：逐期 `flow_t / (1+r)^t`，纯 python 循环（不用被测代码的因子表）。"""
    total = 0.0
    df = 1.0
    for f in flows:
        df = df * (1.0 + rate)
        total += f / df
    return total


def _df(rate: float, n: int) -> float:
    """(1+r)^n 的顺序乘法。"""
    acc = 1.0
    for _ in range(n):
        acc = acc * (1.0 + rate)
    return acc


class _LCG:
    """最小 LCG + Box–Muller（与 `tests/quantdata.py` 同一构造，跨版本逐位一致）。

    存在的意义：给 Monte Carlo 一个**独立于 numpy 位流**的对照组。
    """

    _MASK = (1 << 31) - 1

    def __init__(self, seed: int) -> None:
        self.state = seed & self._MASK

    def u01(self) -> float:
        self.state = (1103515245 * self.state + 12345) & self._MASK
        return self.state / self._MASK

    def normal(self) -> float:
        u1 = max(self.u01(), 1e-12)
        u2 = self.u01()
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


def _independent_mc(base, mu, sigma, rate, horizon, n, seed) -> list[float]:
    """独立重算 Monte Carlo 的终值分布：value = base * Π(1+g_t) / (1+r)^h。"""
    rng = _LCG(seed)
    out = [0.0] * n
    for i in range(n):
        path = 1.0
        for _ in range(horizon):
            path *= 1.0 + mu + sigma * rng.normal()
        out[i] = base * path / _df(rate, horizon)
    out.sort()
    return out


def _quantile_sorted(sorted_values: list[float], q: float) -> float:
    """排序表 + 分数下标的线性插值（numpy 默认 method="linear"），测试侧独立实现。"""
    n = len(sorted_values)
    if n == 1:
        return sorted_values[0]
    idx = q * (n - 1)
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return sorted_values[lo]
    frac = idx - lo
    return sorted_values[lo] + frac * (sorted_values[hi] - sorted_values[lo])


# ── DCF：两法终值（spec Scenario: DCF 两法终值） ──────────────────


def test_dcf_perpetuity_matches_independent_discounting():
    """spec Scenario「DCF 两法终值」：永续增长法结果可与独立折现逐项对上。

    手算：PV = Σ FCF_t/(1+r)^t，终值 = FCF_T*(1+g)/(r-g)，再折回 T 期，
    最后减净债、除股数。
    """
    f = qd.dcf_inputs()
    rate, g = f["discount_rate"], f["terminal_growth"]
    n = len(f["free_cash_flows"])
    r = estimate_dcf(as_of=AS_OF, **f)

    pv_explicit = _pv(rate, f["free_cash_flows"])
    terminal = f["free_cash_flows"][-1] * (1.0 + g) / (rate - g)
    pv_terminal = terminal / _df(rate, n)
    ev = pv_explicit + pv_terminal

    assert r["terminal_method"] == "perpetuity"
    assert r["terminal_value"] == pytest.approx(terminal, rel=1e-15)
    assert r["pv_explicit_fcf"] == pytest.approx(pv_explicit, rel=1e-12, abs=1e-9)
    assert r["pv_terminal_value"] == pytest.approx(pv_terminal, rel=1e-12)
    assert r["enterprise_value"] == pytest.approx(ev, rel=1e-12)
    assert r["equity_value"] == pytest.approx(ev - f["net_debt"], rel=1e-12)
    assert r["value_per_share"] == pytest.approx((ev - f["net_debt"]) / f["shares_outstanding"],
                                                rel=1e-12)
    # 终值占比自洽：pv_terminal / EV
    assert r["terminal_value_share"] == pytest.approx(pv_terminal / ev, rel=1e-12)
    assert r["pv_explicit_fcf"] + r["pv_terminal_value"] == r["enterprise_value"]
    # 假设可见（spec: 假设参数在返回中可见）
    assert r["assumptions"]["terminal_inputs_used"] == {"terminal_growth": g}
    assert r["assumptions"]["terminal_growth"] == g
    assert r["assumptions"]["discount_rate"] == r["discount_rate"]


def test_dcf_exit_multiple_terminal_is_multiple_times_metric():
    """spec Scenario「DCF 两法终值」：退出倍数法终值 = 倍数 × 指标，且假设可见。"""
    f = qd.dcf_inputs()
    r = estimate_dcf(as_of=AS_OF, terminal_method="exit_multiple", **f)

    assert r["terminal_method"] == "exit_multiple"
    assert r["terminal_value"] == f["exit_multiple"] * f["exit_metric"]
    pv_terminal = r["terminal_value"] / _df(f["discount_rate"], len(f["free_cash_flows"]))
    assert r["pv_terminal_value"] == pytest.approx(pv_terminal, rel=1e-12)
    assert r["assumptions"]["terminal_inputs_used"] == {
        "exit_multiple": f["exit_multiple"], "exit_metric": f["exit_metric"]
    }
    # 价格由该终值推出：EV - net_debt 再除股数
    assert r["value_per_share"] == pytest.approx(
        (r["enterprise_value"] - f["net_debt"]) / f["shares_outstanding"], rel=1e-12
    )


def test_dcf_two_terminal_methods_share_pv_explicit_but_differ_in_value():
    """两法只差终值：显式期现值相同，终值与每股价值不同（两个数都要给出）。"""
    f = qd.dcf_inputs()
    p = estimate_dcf(as_of=AS_OF, **f)
    e = estimate_dcf(as_of=AS_OF, terminal_method="exit_multiple", **f)
    assert p["pv_explicit_fcf"] == e["pv_explicit_fcf"]
    assert p["terminal_value"] != e["terminal_value"]
    assert p["value_per_share"] != e["value_per_share"]


def test_dcf_wacc_components_are_stepped_and_auditable():
    """DCF 的 WACC 必须分步可审：CAPM 股权成本、税后债务成本、权重加权逐项可核。"""
    comps = {
        "risk_free": 0.03,
        "equity_beta": 1.2,
        "equity_risk_premium": 0.05,
        "cost_of_debt": 0.06,
        "tax_rate": 0.25,
        "debt_weight": 0.4,
        "equity_weight": 0.6,
    }
    f = {k: v for k, v in qd.dcf_inputs().items() if k != "discount_rate"}
    r = estimate_dcf(as_of=AS_OF, wacc_components=comps, **f)

    assert r["wacc_basis"] == "components"
    bd = r["wacc_breakdown"]
    # 独立手算（同一算式，逐位可比）
    cost_of_equity = 0.03 + 1.2 * 0.05
    after_tax = 0.06 * (1.0 - 0.25)
    wacc = 0.6 * cost_of_equity + 0.4 * after_tax
    assert bd["cost_of_equity"] == cost_of_equity
    assert bd["after_tax_cost_of_debt"] == after_tax
    assert bd["equity_weighted"] == 0.6 * cost_of_equity
    assert bd["debt_weighted"] == 0.4 * after_tax
    assert bd["wacc"] == wacc
    assert r["discount_rate"] == wacc
    # 分步算出的 WACC 与直接给同一个 rate 必须给出相同的估值（可审计性）：
    # 只有 wacc_basis / wacc_breakdown 按来源规则不同，数值结果逐位相同。
    direct = estimate_dcf(as_of=AS_OF, discount_rate=wacc, **f)
    assert direct["wacc_basis"] == "given" and direct["wacc_breakdown"] is None
    assert r["value_per_share"] == direct["value_per_share"]
    assert r["enterprise_value"] == direct["enterprise_value"]
    assert r["equity_value"] == direct["equity_value"]
    assert r["pv_explicit_fcf"] == direct["pv_explicit_fcf"]
    assert r["pv_terminal_value"] == direct["pv_terminal_value"]


def test_dcf_wacc_components_rejection_semantics():
    """WACC 成分的拒绝语义：两条来源互斥、成分缺失点名、权重和不为 1 被拒。"""
    f = {k: v for k, v in qd.dcf_inputs().items() if k != "discount_rate"}
    good = {
        "risk_free": 0.03, "equity_beta": 1.2, "equity_risk_premium": 0.05,
        "cost_of_debt": 0.06, "tax_rate": 0.25, "debt_weight": 0.4, "equity_weight": 0.6,
    }
    with pytest.raises(ComputeError, match="not both"):
        estimate_dcf(as_of=AS_OF, discount_rate=0.09, wacc_components=good, **f)
    with pytest.raises(ComputeError, match="equity_risk_premium"):
        estimate_dcf(as_of=AS_OF,
                     wacc_components={k: v for k, v in good.items() if k != "equity_risk_premium"},
                     **f)
    with pytest.raises(ComputeError, match="must sum to 1"):
        estimate_dcf(as_of=AS_OF,
                     wacc_components={**good, "debt_weight": 0.5, "equity_weight": 0.6}, **f)
    with pytest.raises(ComputeError, match="discount_rate is required"):
        estimate_dcf(as_of=AS_OF, **f)


def test_dcf_rejects_terminal_growth_at_or_above_discount_rate_naming_both():
    """永续法发散边界：g >= r 时点名两个数（不做静默截断或改算法）。"""
    f = qd.dcf_inputs()
    with pytest.raises(ComputeError) as ei:
        estimate_dcf(as_of=AS_OF, **{**f, "terminal_growth": f["discount_rate"]})
    msg = str(ei.value)
    assert "terminal_growth" in msg and "discount_rate" in msg
    assert "0.09" in msg


def test_dcf_rejects_non_positive_shares_and_future_as_of():
    """股数非正点名该字段；as_of 非法/越界口径在第一个动作被拒。"""
    f = qd.dcf_inputs()
    with pytest.raises(ComputeError, match="shares_outstanding"):
        estimate_dcf(as_of=AS_OF, **{**f, "shares_outstanding": 0.0})
    with pytest.raises(ComputeError, match="as_of"):
        estimate_dcf(as_of="2026-13-01", **f)
    with pytest.raises(ComputeError, match="as_of"):
        estimate_dcf(as_of=None, **f)


# ── DDM ──────────────────────────────────────────────────────────


def test_ddm_gordon_matches_closed_form():
    """Gordon：V = DPS*(1+g)/(r-g)，与手算逐位一致。"""
    r = estimate_ddm(dividend_per_share=2.0, growth_rate=0.04, required_return=0.09,
                     as_of=AS_OF)
    assert r["value_per_share"] == 2.0 * (1.0 + 0.04) / (0.09 - 0.04)
    assert r["value_per_share"] == pytest.approx(41.6, rel=1e-12)
    assert r["params"]["stage"] == "gordon"
    assert r["dividend_schedule"] is None
    assert r["assumptions"]["dividend_per_share"] == 2.0


def test_ddm_explicit_schedule_sums_pv_then_gordon_terminal():
    """显式股息表：V = Σ DPS_t/(1+r)^t + [DPS_T*(1+g)/(r-g)]/(1+r)^T。"""
    schedule = {"2027": 2.0, "2028": 2.2}
    r = estimate_ddm(dividend_schedule=schedule, growth_rate=0.03, required_return=0.10,
                     as_of=AS_OF)
    pv_explicit = _pv(0.10, [2.0, 2.2])
    terminal = 2.2 * (1.0 + 0.03) / (0.10 - 0.03)
    pv_terminal = terminal / _df(0.10, 2)
    assert r["params"]["stage"] == "multi_period"
    assert r["pv_explicit_dividends"] == pytest.approx(pv_explicit, rel=1e-12)
    assert r["pv_terminal_value"] == pytest.approx(pv_terminal, rel=1e-12)
    assert r["value_per_share"] == pytest.approx(pv_explicit + pv_terminal, rel=1e-12)
    assert r["value_per_share"] == r["pv_explicit_dividends"] + r["pv_terminal_value"]
    assert r["dividend_per_share"] is None
    assert r["dividend_schedule"] == schedule


def test_ddm_rejects_divergence_and_ambiguous_dividend_source():
    """g >= r 点名两个数；两条股息来源互斥；都没有时点名缺失的输入。"""
    with pytest.raises(ComputeError) as ei:
        estimate_ddm(dividend_per_share=2.0, growth_rate=0.09, required_return=0.09,
                     as_of=AS_OF)
    msg = str(ei.value)
    assert "growth_rate" in msg and "required_return" in msg
    with pytest.raises(ComputeError, match="not both"):
        estimate_ddm(dividend_per_share=2.0, dividend_schedule={"2027": 2.0},
                     growth_rate=0.03, required_return=0.10, as_of=AS_OF)
    with pytest.raises(ComputeError, match="dividend_per_share is required"):
        estimate_ddm(growth_rate=0.03, required_return=0.10, as_of=AS_OF)


# ── 可比公司倍数（winsorize 去极值） ──────────────────────────────


def test_comps_winsorize_clips_outliers_and_moves_the_statistics():
    """winsorize 确实去极值：极端倍数被夹到分位边界，统计量与隐含估值随之改变。

    手算（winsorize=0.05，n=6，线性插值）：
      下界 = q(0.05) → idx=0.25 → 9 + 0.25*(10-9) = 9.25（9 被抬到 9.25）
      上界 = q(0.95) → idx=4.75 → 13 + 0.75*(40-13) = 33.25（40 被压到 33.25）
    即 clipped_count = 2，mean 下降而 median 不变。
    """
    multiples = [9.0, 10.0, 11.0, 12.0, 13.0, 40.0]
    r = compute_comparable_multiples(multiples, metric_value=200.0, winsorize=0.05,
                                     as_of=AS_OF)
    assert r["lower_bound"] == pytest.approx(9.25, rel=1e-12)
    assert r["upper_bound"] == pytest.approx(33.25, rel=1e-12)
    assert r["clipped_count"] == 2
    assert r["winsorized"] == [9.25, 10.0, 11.0, 12.0, 13.0, 33.25]
    assert r["original"] == multiples  # 原表原样回显，未被就地改写
    # 去极值前后统计量的差异（median 在这组数上不变，mean 明显下降）
    assert r["mean_original"] == pytest.approx(95.0 / 6.0, rel=1e-12)
    assert r["mean"] < r["mean_original"]
    assert r["median"] == r["median_original"] == 11.5
    assert r["implied_valuation"] == pytest.approx(11.5 * 200.0, rel=1e-12)

    # winsorize=0 关掉裁剪：一个点都不改
    off = compute_comparable_multiples(multiples, metric_value=200.0, winsorize=0.0,
                                       as_of=AS_OF)
    assert off["clipped_count"] == 0
    assert off["winsorized"] == multiples
    assert off["lower_bound"] == 9.0 and off["upper_bound"] == 40.0
    assert off["mean"] == off["mean_original"]


def test_comps_winsorize_median_equals_the_hand_computed_winsorized_median():
    """spec 5.1（tasks 5.1 验证）：winsorize 对含极端值的倍数表改变中位数至**预期**。

    "预期"由独立的朴素实现给出（就地排序 + 分数下标线性插值，不调用被测模块的
    任何内部函数）：本表手算
      下界 = q(0.30) → idx = 1.5 → 8 + 0.5*(9-8) = 8.5      （7 被抬到 8.5）
      上界 = q(0.70) → idx = 3.5 → 10 + 0.5*(11-10) = 10.5  （300 被压到 10.5）
      clipped = [9, 8.5, 10.5, 10, 10.5, 8.5]（改写 4 项）→ 中位数 9.5
    断言返回的 clipped 表、median、mean、p75 与手算逐位一致。

    注：对称双侧裁剪下"预期中位数"恰等于原始中位数 9.5——这不是实现偷懒，而是
    数学结论（type-7 分位对单调非减变换等变，clip 是单调非减的，且 w≤0.5 时
    L ≤ median ≤ U，故 median(clip(x)) = clip(median(x)) = median(x)）；穷举证书：
    n≤9 的整数表 × 7 档 w 的精确有理数计算里 0 例中位数移动。去极值真正改变的
    统计量是 mean 与 p75，两者本节都断言到具体数值。
    """
    table = [9.0, 7.0, 300.0, 10.0, 11.0, 8.0]
    r = compute_comparable_multiples(table, metric_value=100.0, winsorize=0.30,
                                     as_of=AS_OF)
    s = sorted(table)
    lower = _quantile_sorted(s, 0.30)
    upper = _quantile_sorted(s, 0.70)
    expected = [min(max(v, lower), upper) for v in table]
    expected_clipped = sorted(expected)
    expected_median = _quantile_sorted(expected_clipped, 0.5)

    assert lower == 8.5 and upper == 10.5
    assert r["lower_bound"] == lower and r["upper_bound"] == upper
    assert r["winsorized"] == expected
    assert r["clipped_count"] == 4
    assert r["median"] == expected_median == 9.5           # spec：中位数至预期
    assert r["median"] == r["median_original"]             # 预期值 == 原始值（见上）
    assert r["mean"] == pytest.approx(sum(expected_clipped) / 6.0, rel=1e-15)
    assert r["mean"] == pytest.approx(9.5, rel=1e-15)
    assert r["mean"] < r["mean_original"]                   # 上尾极端值被剪掉
    assert r["p75"] == pytest.approx(10.375, rel=1e-15)
    assert r["p75"] < r["p75_original"] == 10.75
    assert r["implied_valuation"] == r["median"] * 100.0
    # 原表未被动过（去极值只改本工具的副本，不就地改写调用方输入）
    assert r["original"] == table and table == [9.0, 7.0, 300.0, 10.0, 11.0, 8.0]


def test_comps_median_is_invariant_under_symmetric_winsorize():
    """spec 5.1 措辞的边界（已向主线报告）：对称双侧裁剪**不可能**移动中位数。

    分位函数对单调非减变换是不变的，草算如下：设 w < 0.5、L = q(w)、U = q(1-w)，
    对偶数 n 有 (L+U)/2 = median（两个分位下标对称落在中位区间内），奇数 n 的中位
    秩样本必然落在 [L, U] 内。故 `median` 与 `median_original` 恒等——本工具的
    "去极值确实改变结果"落在 mean / p25 / p75 / 裁剪表上，中位数是稳健量、按定义
    不受极端值影响。返回里成对给出两套统计量，正是为了让调用方看到"哪条动了"。

    这条用例把该不变量钉住（同时用穷举证书佐证：n≤9 的整数表 × 7 档 w 的精确有理
    数计算里，0 例中位数移动；浮点求值下可能出现末位扰动，故留 1e-12 相对容差）。
    若将来实现改成单侧或非对称裁剪，这条会红。
    """
    tables = [
        [9.0, 10.0, 11.0, 12.0, 13.0, 40.0],
        [1.0, 9.14, 209.81, 454.96],
        [2.0, 3.0, 100.0, 200.0, 400.0],
        [0.5, 1.0, 200.0, 900.0, 1200.0, 5000.0, 9000.0],
    ]
    for table in tables:
        for w in (0.05, 0.10, 0.20, 0.25, 0.30, 0.40, 0.45):
            r = compute_comparable_multiples(table, metric_value=10.0, winsorize=w,
                                             as_of=AS_OF)
            assert r["median"] == pytest.approx(r["median_original"], rel=1e-12), (table, w)
            assert r["median"] == pytest.approx(_quantile_sorted(sorted(table), 0.5),
                                                rel=1e-12)
            # 对照：p25/p75 在裁剪后**会**动（稳健量与尾部分位的行为差异）
            assert r["p25"] == _quantile_sorted(sorted(r["winsorized"]), 0.25)
            assert r["p75"] == _quantile_sorted(sorted(r["winsorized"]), 0.75)


def test_comps_peer_mapping_names_the_clipped_peers():
    """给了 peer 名时点名被裁剪的 peer（调用方要知道是哪一家被夹了）。"""
    r = compute_comparable_multiples({"a": 9.0, "b": 10.0, "c": 11.0, "d": 12.0,
                                      "e": 13.0, "outlier": 40.0},
                                     metric_value=200.0, winsorize=0.05, as_of=AS_OF)
    assert r["peer_names"] == ["a", "b", "c", "d", "e", "outlier"]
    assert set(r["clipped_peers"]) == {"a", "outlier"}
    assert r["winsorized"][-1] == pytest.approx(33.25, rel=1e-12)


def test_comps_rejects_out_of_range_winsorize_and_empty_table():
    """winsorize 越界（>=0.5 会把所有点夹到中位数）与空表都点名拒绝。"""
    with pytest.raises(ComputeError, match="winsorize"):
        compute_comparable_multiples([10.0, 20.0], metric_value=1.0, winsorize=0.5,
                                     as_of=AS_OF)
    with pytest.raises(ComputeError, match="winsorize"):
        compute_comparable_multiples([10.0, 20.0], metric_value=1.0, winsorize=-0.1,
                                     as_of=AS_OF)
    with pytest.raises(ComputeError, match="must not be empty"):
        compute_comparable_multiples([], metric_value=1.0, as_of=AS_OF)


# ── Monte Carlo（spec Scenario: Monte Carlo 可复现） ──────────────


def test_monte_carlo_same_seed_is_bitwise_reproducible():
    """spec Scenario「Monte Carlo 可复现」：同一 seed 两次调用百分位带逐位一致。"""
    kw = dict(base_value=1400.0, growth_mean=0.06, growth_std=0.12, discount_rate=0.09,
              n_simulations=2000, horizon=5, seed=2026, as_of=AS_OF)
    a = estimate_monte_carlo(**kw)
    b = estimate_monte_carlo(**kw)
    assert_bitequal(a, b, what="estimate_monte_carlo(same seed)")
    for label in ("p5", "p25", "p50", "p75", "p95"):
        assert a[label] == b[label]
    assert a["p5"] < a["p25"] < a["p50"] < a["p75"] < a["p95"]
    assert a["seed"] == 2026
    assert a["percentile_levels"] == {"p5": 0.05, "p25": 0.25, "p50": 0.50,
                                      "p75": 0.75, "p95": 0.95}


def test_monte_carlo_seed_actually_drives_the_stream():
    """换 seed 结果必须变（否则 seed 只是摆设，可复现性无从谈起）。"""
    kw = dict(base_value=1400.0, growth_mean=0.06, growth_std=0.12, discount_rate=0.09,
              n_simulations=1000, horizon=5, as_of=AS_OF)
    a = estimate_monte_carlo(seed=1, **kw)
    b = estimate_monte_carlo(seed=2, **kw)
    assert a["p50"] != b["p50"]


def test_monte_carlo_band_matches_independent_model_and_analytic_mean():
    """独立对拍：换一条随机流（LCG+Box–Muller）重跑同一模型，分位带应落在同一分布上；
    均值还要贴近解析期望 base*(1+mu)^h/(1+r)^h（中位数则因对数正态右偏低于均值）。
    """
    base, mu, sigma, rate, h, n = 1400.0, 0.06, 0.12, 0.09, 5, 20000
    r = estimate_monte_carlo(base_value=base, growth_mean=mu, growth_std=sigma,
                             discount_rate=rate, n_simulations=n, horizon=h, seed=42,
                             as_of=AS_OF)
    # E[value] = base * E[Π(1+g)] / (1+r)^h = base*(1+mu)^h/(1+r)^h
    analytic_mean = base * _df(mu, h) / _df(rate, h)
    assert r["mean"] == pytest.approx(analytic_mean, rel=0.02)

    # 对数正态右偏：中位数必须低于均值（分布形状的方向性检查）
    assert r["p50"] < r["mean"]
    assert r["p50"] == pytest.approx(analytic_mean, rel=0.06)

    ref = _independent_mc(base, mu, sigma, rate, h, n, seed=999)
    # 容差按 n=20000 的抽样误差给（约为实测偏差的 4~5 倍）：换一条随机流仍须落在
    # 同一分布上。模型级错误（漏折现 / 少一次复利）会偏出 20% 以上，必被抓住。
    assert r["p5"] == pytest.approx(_quantile_sorted(ref, 0.05), rel=0.08)
    assert r["p25"] == pytest.approx(_quantile_sorted(ref, 0.25), rel=0.05)
    assert r["p50"] == pytest.approx(_quantile_sorted(ref, 0.50), rel=0.04)
    assert r["p75"] == pytest.approx(_quantile_sorted(ref, 0.75), rel=0.05)
    assert r["p95"] == pytest.approx(_quantile_sorted(ref, 0.95), rel=0.08)
    # 分布离散度也要对：独立流的 std 与实现值同量级
    ref_mean = sum(ref) / n
    ref_var = sum((x - ref_mean) ** 2 for x in ref) / n
    assert r["std"] == pytest.approx(math.sqrt(ref_var), rel=0.06)


def test_monte_carlo_stream_is_pinned_to_golden_values():
    """位流钉死：给定 seed 的百分位带落在写死的一组数上。

    这是"随机种子可复现"的最强形式——不只"两次一样"，而是"每一次都是这一组数"。
    若某次 numpy 升级换了 `default_rng(seed).standard_normal` 的位流（该序列被
    numpy 承诺跨版本稳定，改了就是破坏性变更），本用例会红，而不是让历史结果
    在无提示的情况下改变。golden 值取自本实现首次落地时的运行结果。
    """
    r = estimate_monte_carlo(base_value=1000.0, growth_mean=0.05, growth_std=0.10,
                             discount_rate=0.08, n_simulations=1000, horizon=3,
                             seed=12345, as_of=AS_OF)
    assert r["p5"] == 683.826373688597
    assert r["p25"] == 805.877177427436
    assert r["p50"] == 921.2428640934643
    assert r["p75"] == 1029.374510441639
    assert r["p95"] == 1180.99593372752
    assert r["mean"] == 923.1492156833922
    assert r["std"] == 151.89856768900108


def test_monte_carlo_custom_percentiles_and_rejections():
    """自定义百分位带；保留键冲突、非法水平、非法 seed 都点名拒绝。"""
    r = estimate_monte_carlo(base_value=100.0, growth_mean=0.05, growth_std=0.10,
                             discount_rate=0.08, n_simulations=500, horizon=3, seed=7,
                             percentiles={"lo": 0.10, "hi": 0.90}, as_of=AS_OF)
    assert set(r["percentile_values"]) == {"lo", "hi"}
    assert r["lo"] < r["hi"]
    assert "p5" not in r

    kw = dict(base_value=100.0, growth_mean=0.05, growth_std=0.10, discount_rate=0.08,
              n_simulations=100, horizon=3, as_of=AS_OF)
    with pytest.raises(ComputeError, match="reserved"):
        estimate_monte_carlo(percentiles={"mean": 0.5}, seed=1, **kw)
    with pytest.raises(ComputeError, match=r"percentiles\['bad'\]"):
        estimate_monte_carlo(percentiles={"bad": 1.5}, seed=1, **kw)
    with pytest.raises(ComputeError, match="seed"):
        estimate_monte_carlo(seed=1.5, **kw)
    with pytest.raises(ComputeError, match="seed"):
        estimate_monte_carlo(seed=-1, **kw)
    with pytest.raises(ComputeError, match="not both"):
        estimate_monte_carlo(free_cash_flows=[1.0], seed=1, **kw)


def test_monte_carlo_10000_simulations_finish_within_two_seconds(capsys):
    """spec 5.2 验证条款：10,000 次模拟在 2 秒内完成（**记录实测**）。

    先热身一次（排除首次调用的一次性导入/分配开销），再对同一参数计时；实测值
    用 `capsys.disabled()` 直接打印到测试输出，供报告与回归对比。
    """
    import time

    kw = dict(base_value=1400.0, growth_mean=0.06, growth_std=0.12,
              discount_rate=0.09, n_simulations=10000, horizon=5, seed=42,
              as_of=AS_OF)
    estimate_monte_carlo(**kw)  # warm-up
    start = time.perf_counter()
    r = estimate_monte_carlo(**kw)
    elapsed = time.perf_counter() - start
    with capsys.disabled():
        print(f"\n[measured] estimate_monte_carlo 10,000 sims (horizon 5): "
              f"{elapsed:.4f}s (limit 2.0s)")
    assert r["n_simulations"] == 10000
    assert len(r["percentile_values"]) == 5
    assert elapsed < 2.0, f"10,000 simulations took {elapsed:.4f}s (limit 2.0s)"


def test_monte_carlo_pv_base_from_free_cash_flows():
    """base_value 可由 free_cash_flows 的现值给出（此时 base_value_source 回显来源）。"""
    f = qd.dcf_inputs()
    r = estimate_monte_carlo(free_cash_flows=f["free_cash_flows"],
                             growth_mean=0.05, growth_std=0.10,
                             discount_rate=f["discount_rate"], n_simulations=500,
                             horizon=3, seed=11, as_of=AS_OF)
    assert r["base_value_source"] == "free_cash_flows_pv"
    assert r["base_value"] == pytest.approx(_pv(f["discount_rate"], f["free_cash_flows"]),
                                            rel=1e-12)
    assert r["p50"] > 0.0


# ── 敏感性网格（双变量） ─────────────────────────────────────────


def test_sensitivity_grid_recomputes_dcf_per_cell_in_order():
    """双变量网格：每个格点重算 DCF，行序 = 先 x 后 y 跟随给定值序。"""
    f = qd.dcf_inputs()
    r = compute_sensitivity_grid(
        base_inputs=f,
        variable_x={"name": "discount_rate", "values": [0.08, 0.10]},
        variable_y={"name": "terminal_growth", "values": [0.02, 0.03]},
        as_of=AS_OF,
    )
    assert r["x_name"] == "discount_rate" and r["y_name"] == "terminal_growth"
    assert r["value_metric"] == "value_per_share"
    assert len(r["grid"]) == 4
    assert [(row["x_value"], row["y_value"]) for row in r["grid"]] == [
        (0.08, 0.02), (0.08, 0.03), (0.10, 0.02), (0.10, 0.03),
    ]
    for row in r["grid"]:
        expected = estimate_dcf(as_of=AS_OF, **{**f,
                                                "discount_rate": row["x_value"],
                                                "terminal_growth": row["y_value"]})
        assert row["value"] == expected["value_per_share"]
    # 基线格 = base_inputs 原样
    assert r["base_case_value"] == estimate_dcf(as_of=AS_OF, **f)["value_per_share"]
    # 单调性：折现率越高价值越低、永续增速越高价值越高（方向性检查）
    by_key = {(row["x_value"], row["y_value"]): row["value"] for row in r["grid"]}
    assert by_key[(0.08, 0.03)] > by_key[(0.10, 0.03)] > 0.0
    assert by_key[(0.08, 0.03)] > by_key[(0.08, 0.02)]


def test_sensitivity_grid_rejects_illegal_axis_and_names_the_cell():
    """非法轴名 / 两轴同名 / 基线里夹带 as_of / 格点参数非法（点名该格点取值）。"""
    f = qd.dcf_inputs()
    with pytest.raises(ComputeError, match="variable_x\\['name'\\]"):
        compute_sensitivity_grid(base_inputs=f,
                                 variable_x={"name": "beta", "values": [1.0]},
                                 variable_y={"name": "terminal_growth", "values": [0.02]},
                                 as_of=AS_OF)
    with pytest.raises(ComputeError, match="must differ"):
        compute_sensitivity_grid(base_inputs=f,
                                 variable_x={"name": "terminal_growth", "values": [0.02]},
                                 variable_y={"name": "terminal_growth", "values": [0.03]},
                                 as_of=AS_OF)
    with pytest.raises(ComputeError, match="as_of"):
        compute_sensitivity_grid(base_inputs={**f, "as_of": AS_OF},
                                 variable_x={"name": "discount_rate", "values": [0.08]},
                                 variable_y={"name": "terminal_growth", "values": [0.02]},
                                 as_of=AS_OF)
    # 0.09 的永续增速撞上 0.09 的折现率 → 该格点被拒，错误里带上 x/y 取值
    with pytest.raises(ComputeError) as ei:
        compute_sensitivity_grid(base_inputs=f,
                                 variable_x={"name": "discount_rate", "values": [0.09]},
                                 variable_y={"name": "terminal_growth", "values": [0.09]},
                                 as_of=AS_OF)
    msg = str(ei.value)
    assert "grid cell" in msg and "discount_rate=0.09" in msg and "terminal_growth=0.09" in msg


# ── 估值合成：扣值不扣结论（spec Scenario 两条） ─────────────────


def test_synthesis_withholds_price_target_when_methods_disagree_beyond_threshold():
    """spec Scenario「方法分歧时扣值不扣结论」：阈值 0.2、两法相差 35% 时
    price_target 为 null，并附两方法各自结果与偏离幅度的分歧说明，不给均值。
    """
    r = build_valuation_synthesis(
        methods=[{"name": "dcf", "value": 100.0}, {"name": "comparables", "value": 135.0}],
        disagreement_threshold=0.2,
        as_of=AS_OF,
    )
    assert r["price_target"] is None
    assert r["weights_used"] is None
    assert r["weighting_basis"] == "withheld"
    assert "mean" not in r and "weighted_mean" not in r  # 不静默取平均
    assert r["method_count"] == 2
    assert r["method_values"] == [{"name": "dcf", "value": 100.0},
                                  {"name": "comparables", "value": 135.0}]
    assert r["min"] == 100.0 and r["max"] == 135.0 and r["median"] == 117.5
    assert r["disagreement"] == pytest.approx(35.0 / 117.5, rel=1e-12)
    assert r["disagreement"] > r["disagreement_threshold"]

    note = r["disagreement_note"]
    assert note and r["note"] == note
    assert "扣值不扣结论" in note
    # 分歧说明含两方法各自结果与偏离幅度
    for name, value in (("dcf", "100.0"), ("comparables", "135.0")):
        assert name in note and value in note
    # outlier 清单：两个极值方法各一条，偏离符号正确（对中位数的有符号偏离）
    outliers = {o["name"]: o for o in r["outliers"]}
    assert set(outliers) == {"dcf", "comparables"}
    assert outliers["dcf"]["deviation"] == pytest.approx((100.0 - 117.5) / 117.5, rel=1e-12)
    assert outliers["comparables"]["deviation"] == pytest.approx((135.0 - 117.5) / 117.5,
                                                                rel=1e-12)
    assert outliers["dcf"]["deviation"] < 0.0 < outliers["comparables"]["deviation"]


def test_synthesis_gives_point_estimate_when_disagreement_within_threshold():
    """spec Scenario「分歧在阈值内给点位」：相差 8%、阈值 20% → 给出具体点位与权重说明。"""
    r = build_valuation_synthesis(
        methods=[{"name": "dcf", "value": 100.0}, {"name": "comparables", "value": 108.0}],
        disagreement_threshold=0.2,
        as_of=AS_OF,
    )
    assert r["price_target"] == pytest.approx(104.0, rel=1e-12)  # 等权均值
    assert r["weighting_basis"] == "equal"
    assert r["weights_used"] == {"dcf": 0.5, "comparables": 0.5}
    assert r["disagreement"] == pytest.approx(8.0 / 104.0, rel=1e-12)
    assert r["disagreement"] < r["disagreement_threshold"]
    assert r["disagreement_note"] is None
    assert "weighted mean" in r["note"] and "equal weights" in r["note"]

    # 自定义权重（内部归一化）：0.6/0.4 → 0.6*100 + 0.4*108
    w = build_valuation_synthesis(
        methods=[{"name": "dcf", "value": 100.0}, {"name": "comparables", "value": 108.0}],
        disagreement_threshold=0.2,
        weights={"dcf": 0.6, "comparables": 0.4},
        as_of=AS_OF,
    )
    assert w["price_target"] == pytest.approx(103.2, rel=1e-12)
    assert w["weighting_basis"] == "custom"
    assert w["weights_used"] == {"dcf": 0.6, "comparables": 0.4}


def test_synthesis_single_method_always_returns_a_number():
    """单方法：分歧恒为 0，无论阈值多少都必须给点位（不能因为"没有方法可合"而扣值）。"""
    for threshold in (0.0, 0.20, 1.0):
        r = build_valuation_synthesis(
            methods=[{"name": "only", "value": 88.0}],
            disagreement_threshold=threshold, as_of=AS_OF,
        )
        assert r["price_target"] == 88.0, threshold
        assert r["disagreement"] == 0.0
        assert r["weights_used"] == {"only": 1.0}
        assert r["outliers"] == []
        assert r["method_count"] == 1


def test_synthesis_three_methods_flags_only_extremes_as_outliers():
    """三方法：只有极值被列为 outlier（中间那个不是）。"""
    r = build_valuation_synthesis(
        methods=[{"name": "a", "value": 90.0}, {"name": "b", "value": 100.0},
                 {"name": "c", "value": 140.0}],
        disagreement_threshold=0.2, as_of=AS_OF,
    )
    assert r["price_target"] is None
    assert {o["name"] for o in r["outliers"]} == {"a", "c"}
    assert all(o["name"] != "b" for o in r["outliers"])


def test_synthesis_rejection_semantics():
    """方法名重复、权重缺项、中位数非正、空清单都点名拒绝。"""
    with pytest.raises(ComputeError, match="duplicate method name"):
        build_valuation_synthesis(
            methods=[{"name": "a", "value": 10.0}, {"name": "a", "value": 11.0}],
            as_of=AS_OF)
    with pytest.raises(ComputeError, match="weights is missing"):
        build_valuation_synthesis(
            methods=[{"name": "a", "value": 10.0}, {"name": "b", "value": 11.0}],
            weights={"a": 1.0}, as_of=AS_OF)
    with pytest.raises(ComputeError, match="median"):
        build_valuation_synthesis(
            methods=[{"name": "a", "value": -10.0}, {"name": "b", "value": -4.0}],
            as_of=AS_OF)
    with pytest.raises(ComputeError, match="must not be empty"):
        build_valuation_synthesis(methods=[], as_of=AS_OF)


# ── 跨 change 集成：合成输出 → 批 0.2 `Rating` 契约（tasks 5.5） ──
#
# 只**消费**契约（`fd_quant.contracts.models` + `schemas/rating.schema.json`），
# 不改契约、不改 schema。组装规则本身就是被断言的对象：
#   - 超阈（扣值）：side="review" 且 price_target/conviction 均为 null；
#   - 阈值内：真实 side（如 buy）+ 真实 price_target + target_evidence_ids 可解。


def _assemble_rating(synth: dict, *, side: str, subject: str = "600519.SS") -> dict:
    """把 `build_valuation_synthesis` 的输出组装成 Rating 实例（字典形态）。"""
    price_target = synth["price_target"]
    target_ids: list[str] = []
    if price_target is not None:
        target_ids = [e["evidence_id"] for e in synth["evidence"]
                      if e["label"] == "price target"]
        assert target_ids, "非 null 点位必须能找到支撑它的 evidence 引用"
    return Rating(
        as_of=synth["as_of"],
        subject=subject,
        side=side,
        basis=synth["note"],
        producer=Producer(pack="pack-equity-research", skill="equity-research-thesis"),
        evidence=[Evidence.from_dict(e) for e in synth["evidence"]],
        conviction=None if side == "review" else 0.6,
        price_target=price_target,
        target_evidence_ids=target_ids,
        time_horizon="6-12 个月",
    ).to_dict()


def test_synth_disagreement_assembles_into_a_review_rating():
    """跨 change（tasks 5.5）：分歧案例 → side="review"、price_target=None 的
    Rating 实例，经 `fd_quant.contracts.models.validate` 通过。"""
    import json as _json

    import jsonschema

    from fd_quant.contracts.models import load_schema, validate

    synth = build_valuation_synthesis(
        methods=[{"name": "dcf", "value": 100.0},
                 {"name": "comparables", "value": 135.0}],
        disagreement_threshold=0.2, as_of=AS_OF,
    )
    assert synth["price_target"] is None  # 前置：确实扣了值

    payload = _assemble_rating(synth, side="review")
    assert payload["side"] == "review"
    assert payload["price_target"] is None and payload["conviction"] is None

    check = validate(payload)
    assert check["valid"], check["errors"]
    assert validate(payload, kind="rating")["errors"] == []
    jsonschema.validate(payload, load_schema("rating"))
    assert _json.dumps(payload, allow_nan=False)
    # 分歧说明与每个方法值都进了 basis / evidence（判定可审计、数字有出处）
    assert "扣值不扣结论" in payload["basis"]
    labels = [e["label"] for e in payload["evidence"]]
    assert "method dcf" in labels and "method comparables" in labels
    assert "decision" in labels


def test_synth_point_estimate_assembles_into_a_buy_rating():
    """跨 change（tasks 5.5）：阈值内案例 → 真实 side 与真实 price_target 的 Rating
    实例（target_evidence_ids 可解），经 `validate` 通过。"""
    import json as _json

    import jsonschema

    from fd_quant.contracts.models import load_schema, validate

    synth = build_valuation_synthesis(
        methods=[{"name": "dcf", "value": 100.0},
                 {"name": "comparables", "value": 108.0}],
        disagreement_threshold=0.2, as_of=AS_OF,
    )
    assert synth["price_target"] is not None  # 前置：确实给了点位

    payload = _assemble_rating(synth, side="buy")
    assert payload["side"] == "buy"
    assert payload["price_target"] == pytest.approx(104.0, rel=1e-12)
    assert payload["target_evidence_ids"]
    ids = {e["evidence_id"] for e in payload["evidence"]}
    assert set(payload["target_evidence_ids"]) <= ids
    assert not (set(payload["target_evidence_ids"]) - ids)

    check = validate(payload, kind="rating")
    assert check["valid"], check["errors"]
    jsonschema.validate(payload, load_schema("rating"))
    assert _json.dumps(payload, allow_nan=False)
    # evidence 携带方法值（每个数字可追溯到出处）
    values = {e["label"]: e["value"] for e in payload["evidence"]}
    assert values["method dcf"] == 100.0
    assert values["method comparables"] == 108.0
    assert values["price target"] == pytest.approx(104.0, rel=1e-12)


def test_a_review_rating_may_not_carry_a_target():
    """契约侧反向确认：review 态带点位会被 `validate` 拒绝。

    这条钉住的是**契约的**语义（review 是显式态、不可降级），因此不依赖本族工具；
    若哪天本族真的把 review 与点位一起产出，组装处会先在这条上红。
    """
    from fd_quant.contracts.models import validate

    synth = build_valuation_synthesis(
        methods=[{"name": "dcf", "value": 100.0},
                 {"name": "comparables", "value": 135.0}],
        disagreement_threshold=0.2, as_of=AS_OF,
    )
    payload = _assemble_rating(synth, side="review")
    payload["side"] = "review"
    payload["price_target"] = 104.0       # 人为违规：review 不得带点位
    payload["target_evidence_ids"] = []
    check = validate(payload, kind="rating")
    assert not check["valid"]
    assert any(e["path"] == "price_target" for e in check["errors"])


# ── 族级通用纪律（as_of 必填 / 确定性 / 可序列化） ────────────────


@pytest.mark.parametrize("tool, kw", [
    (estimate_dcf, qd.dcf_inputs()),
    (estimate_ddm, {"dividend_per_share": 2.0, "growth_rate": 0.04, "required_return": 0.09}),
    (compute_comparable_multiples, {"multiples": [9.0, 10.0, 11.0], "metric_value": 100.0}),
    (estimate_monte_carlo, {"base_value": 100.0, "growth_mean": 0.05, "growth_std": 0.1,
                            "discount_rate": 0.08, "n_simulations": 200, "seed": 5}),
    (compute_sensitivity_grid, {"base_inputs": qd.dcf_inputs(),
                                "variable_x": {"name": "discount_rate", "values": [0.08]},
                                "variable_y": {"name": "terminal_growth", "values": [0.02]}}),
    (build_valuation_synthesis, {"methods": [{"name": "a", "value": 10.0}]}),
])
def test_all_tools_require_as_of_and_are_bitwise_repeatable(tool, kw):
    """as_of 是必填且第一个被校验；同输入两次调用逐位一致；返回 JSON 可序列化。"""
    import json

    with pytest.raises(ComputeError, match="as_of"):
        tool(as_of=None, **kw)
    with pytest.raises(ComputeError, match="as_of"):
        tool(as_of="not-a-date", **kw)
    a = tool(as_of=AS_OF, **kw)
    b = tool(as_of=AS_OF, **kw)
    assert_bitequal(a, b, what=tool.__name__)
    assert a["as_of"] == AS_OF
    assert json.dumps(a, allow_nan=False)
    assert a["evidence"]


def test_snapshot_envelope_reports_null_date_axis():
    """快照型工具：输入无逐点日期轴 → 外壳起止日期与长度为 null（口径与 evidence 仍在）。"""
    r = estimate_dcf(as_of=AS_OF, **qd.dcf_inputs())
    assert r["input_start"] is None and r["input_end"] is None and r["input_length"] is None
    assert r["warmup"] == {}
    assert all(e["as_of"] == AS_OF and e["source"].startswith("fd-quant:") for e in r["evidence"])
