"""基本面比率族测试：手算对拍、缺字段点名、DuPont 恒等、红旗触发矩阵、逐位确定。

每个 spec Scenario 至少一个用例，用例 docstring 引用 spec 措辞。所有期望值都在本文件
内用**简单算术**重新算一遍（不调用被测代码的内部函数）——这是唯一能发现"实现与定义
一起写错"的方式。夹具用 `quantdata.three_statements()`（自洽三表，单位为亿元）。
"""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

import quantdata as qd
from fd_quant.compute import base
from fd_quant.compute.base import ComputeError
from fd_quant.compute.fundamental import (
    DEFAULT_THRESHOLDS,
    GROUP,
    compute_cash_flow_quality,
    compute_dupont,
    compute_efficiency,
    compute_growth,
    compute_interest_coverage,
    compute_leverage_and_liquidity,
    compute_profitability,
    detect_red_flags,
)

AS_OF = "2026-06-30"
F = qd.three_statements


# ── 夹具（就地构造；不改 quantdata） ─────────────────────────────


def _base_fundamentals(**overrides) -> dict:
    """红旗检测用的完整快照：营收 +10%、应收 +60%、存货 +20%，其余自洽。"""
    f = dict(F())
    f.update({
        "revenue": 1100.0,
        "revenue_prior": 1000.0,
        "accounts_receivable": 160.0,
        "accounts_receivable_prior": 100.0,
        "inventory": 168.0,
        "inventory_prior": 140.0,
    })
    f.update(overrides)
    return f


def _flat_growth(**overrides) -> dict:
    """增速持平的快照（应收与营收同增 10%）——spec Scenario 的"不触发"侧。"""
    return _base_fundamentals(
        **{"accounts_receivable": 110.0, "accounts_receivable_prior": 100.0, **overrides}
    )


# ── 盈利比率 ─────────────────────────────────────────────────────


def test_profitability_hand_computed():
    """spec 4.1：对一组构造三表输入逐类断言数值与手算一致（盈利类）。"""
    f = F()
    r = compute_profitability(f, as_of=AS_OF)
    x = r["ratios"]
    rev, gp, ni = 1000.0, 400.0, 130.0
    eq, ta, ebit = 1200.0, 2000.0, 180.0
    # 投入资本：无 invested_capital / total_debt，退到总资产 − 流动负债
    invested = ta - 350.0
    assert x["gross_margin"] == pytest.approx(gp / rev, abs=1e-15)
    assert x["net_margin"] == pytest.approx(ni / rev, abs=1e-15)
    assert x["roe"] == pytest.approx(ni / eq, abs=1e-15)
    assert x["roa"] == pytest.approx(ni / ta, abs=1e-15)
    assert x["roic"] == pytest.approx(ebit * (1 - 0.25) / invested, abs=1e-15)
    assert r["sources"] == {
        "gross_profit": "gross_profit",
        "invested_capital": "total_assets_minus_current_liabilities",
    }
    assert r["params"] == {"tax_rate": 0.25}
    assert r["warmup"] == {}
    # 标量族：无日期轴 → 外壳的起止与长度为 null
    assert r["input_start"] is None and r["input_end"] is None
    assert r["input_length"] is None
    assert r["as_of"] == AS_OF
    assert len(r["evidence"]) >= 5


def test_profitability_invested_capital_precedence_and_tax_echo():
    """投入资本三级口径固定顺序；tax_rate 生效值回显且真的进 NOPAT。"""
    f = F()
    direct = compute_profitability({**f, "invested_capital": 1000.0}, as_of=AS_OF, tax_rate=0.0)
    assert direct["sources"]["invested_capital"] == "invested_capital"
    assert direct["ratios"]["roic"] == pytest.approx(180.0 / 1000.0, abs=1e-15)
    assert direct["params"] == {"tax_rate": 0.0}

    with_debt = compute_profitability({**f, "total_debt": 300.0}, as_of=AS_OF)
    assert with_debt["sources"]["invested_capital"] == "total_debt_plus_total_equity"
    assert with_debt["ratios"]["roic"] == pytest.approx(180.0 * 0.75 / 1500.0, abs=1e-15)

    # 毛利分子也可推导（缺 gross_profit 时用 revenue − cogs，口径回显）
    derived = compute_profitability(
        {k: v for k, v in f.items() if k != "gross_profit"},
        as_of=AS_OF,
    )
    assert derived["sources"]["gross_profit"] == "revenue_minus_cost_of_goods_sold"
    assert derived["ratios"]["gross_margin"] == pytest.approx((1000.0 - 600.0) / 1000.0, abs=1e-15)

    with pytest.raises(ComputeError, match="tax_rate"):
        compute_profitability(f, as_of=AS_OF, tax_rate=1.5)


@pytest.mark.parametrize(
    "field",
    ["revenue", "net_income", "total_equity", "total_assets", "ebit"],
)
def test_profitability_missing_field_is_named(field):
    """缺字段 → 报错点名该字段，不返回部分结果。"""
    f = {k: v for k, v in F().items() if k != field}
    with pytest.raises(ComputeError) as ei:
        compute_profitability(f, as_of=AS_OF)
    assert field in str(ei.value)


def test_profitability_missing_invested_capital_is_named():
    """三个投入资本口径都不可得 → 点名 invested_capital（不猜一个分母）。"""
    f = {
        k: v for k, v in F().items()
        if k not in ("invested_capital", "total_debt", "current_liabilities")
    }
    with pytest.raises(ComputeError) as ei:
        compute_profitability(f, as_of=AS_OF)
    assert "invested_capital" in str(ei.value)


def test_profitability_refuses_non_positive_denominator_naming_field():
    """分母非正 → 点名该分母字段，不返回 inf/NaN。"""
    with pytest.raises(ComputeError) as ei:
        compute_profitability({**F(), "total_equity": 0.0}, as_of=AS_OF)
    assert "total_equity" in str(ei.value)


# ── 成长比率 ─────────────────────────────────────────────────────


def test_growth_hand_computed():
    """spec 4.1（成长类）：营收/净利同比与手算一致。"""
    r = compute_growth(F(), as_of=AS_OF)
    assert r["ratios"]["revenue_yoy"] == pytest.approx(1000.0 / 900.0 - 1.0, abs=1e-15)
    assert r["ratios"]["net_income_yoy"] == pytest.approx(130.0 / 110.0 - 1.0, abs=1e-15)
    assert r["params"] == {} and r["warmup"] == {}


@pytest.mark.parametrize("field", ["revenue", "revenue_prior", "net_income", "net_income_prior"])
def test_growth_missing_prior_is_named(field):
    """缺基期 → 报错点名该基期字段（缺一个也说不了增速）。"""
    f = {k: v for k, v in F().items() if k != field}
    with pytest.raises(ComputeError) as ei:
        compute_growth(f, as_of=AS_OF)
    assert field in str(ei.value)


def test_growth_refuses_non_positive_prior_naming_field():
    """非正基期上的增速会发明一个符号 → 拒绝并点名该基期字段。"""
    with pytest.raises(ComputeError) as ei:
        compute_growth({**F(), "revenue_prior": 0.0}, as_of=AS_OF)
    assert "revenue_prior" in str(ei.value)


# ── 偿债与流动性 ─────────────────────────────────────────────────


def test_leverage_and_liquidity_hand_computed():
    """spec 4.1（偿债类）：四个比率与手算一致。"""
    r = compute_leverage_and_liquidity(F(), as_of=AS_OF)
    x = r["ratios"]
    assert x["current_ratio"] == pytest.approx(700.0 / 350.0, abs=1e-15)
    assert x["quick_ratio"] == pytest.approx((700.0 - 150.0) / 350.0, abs=1e-15)
    assert x["debt_to_asset_ratio"] == pytest.approx(800.0 / 2000.0, abs=1e-15)
    assert x["interest_coverage"] == pytest.approx(180.0 / 20.0, abs=1e-15)
    assert r["params"] == {} and r["warmup"] == {}


def test_leverage_fails_atomically_when_interest_expense_missing():
    """spec Scenario 缺输入字段时的明确报错：缺利息费用 → 报错指明字段，不返回部分结果。"""
    f = {k: v for k, v in F().items() if k != "interest_expense"}
    with pytest.raises(ComputeError) as ei:
        compute_leverage_and_liquidity(f, as_of=AS_OF)
    msg = str(ei.value)
    assert "interest_expense" in msg
    # 原子语义：整个调用失败，不留任何部分比率
    assert "ratios" not in msg


def test_interest_coverage_tool_names_missing_field():
    """spec Scenario 同上（独立工具形态）：缺 interest_expense 报错点名，不返回部分结果。"""
    f = {k: v for k, v in F().items() if k != "interest_expense"}
    with pytest.raises(ComputeError) as ei:
        compute_interest_coverage(f, as_of=AS_OF)
    assert "interest_expense" in str(ei.value)

    ok = compute_interest_coverage(F(), as_of=AS_OF)
    assert ok["interest_coverage"] == pytest.approx(180.0 / 20.0, abs=1e-15)

    # interest_expense = 0：无利息负担时该倍数无定义 → 拒绝，不给 inf
    with pytest.raises(ComputeError) as ei2:
        compute_interest_coverage({**F(), "interest_expense": 0.0}, as_of=AS_OF)
    assert "interest_expense" in str(ei2.value)


# ── 效率比率 ─────────────────────────────────────────────────────


def test_efficiency_hand_computed_period_end_and_mixed_basis():
    """spec 4.1（效率类）：平均余额与期末余额两种口径各自手算一致，且口径回显。"""
    r = compute_efficiency(F(), as_of=AS_OF)
    x, by = r["ratios"], r["basis_by_ratio"]
    # 有三个比率：应收/存货有期初 → 平均余额；总资产无期初 → 期末
    assert by == {
        "receivable_turnover": "average",
        "inventory_turnover": "average",
        "total_asset_turnover": "ending",
    }
    assert r["basis"] == "mixed"
    assert x["receivable_turnover"] == pytest.approx(1000.0 / ((120.0 + 75.0) / 2.0), abs=1e-15)
    assert x["inventory_turnover"] == pytest.approx(600.0 / ((150.0 + 140.0) / 2.0), abs=1e-15)
    assert x["total_asset_turnover"] == pytest.approx(1000.0 / 2000.0, abs=1e-15)
    assert r["params"] == {"inventory_turnover_numerator": "cost_of_goods_sold"}


def test_efficiency_all_average_and_all_ending():
    """全期初 → 一律 average；无期初 → 一律 ending 且回显。"""
    all_avg = compute_efficiency({**F(), "total_assets_prior": 1800.0}, as_of=AS_OF)
    assert all_avg["basis"] == "average"
    assert all_avg["ratios"]["total_asset_turnover"] == pytest.approx(
        1000.0 / ((2000.0 + 1800.0) / 2.0), abs=1e-15
    )

    ending = compute_efficiency(
        {k: v for k, v in F().items()
         if k not in ("accounts_receivable_prior", "inventory_prior", "cost_of_goods_sold")},
        as_of=AS_OF,
    )
    assert ending["basis"] == "ending"
    assert ending["basis_by_ratio"] == {
        "receivable_turnover": "ending",
        "inventory_turnover": "ending",
        "total_asset_turnover": "ending",
    }
    assert ending["ratios"]["receivable_turnover"] == pytest.approx(1000.0 / 120.0, abs=1e-15)
    # 无 cogs 时存货周转退回 revenue 并在 params 里回显（两种口径含义不同，不静默）
    assert ending["ratios"]["inventory_turnover"] == pytest.approx(1000.0 / 150.0, abs=1e-15)
    assert ending["params"] == {"inventory_turnover_numerator": "revenue"}


def test_efficiency_refuses_non_positive_balance_naming_field():
    """余额非正 → 点名该字段（周转率分母不能是 0）。期末口径的零余额直接拒绝。"""
    ending_zero = {k: v for k, v in F().items() if k != "inventory_prior"}
    ending_zero["inventory"] = 0.0
    with pytest.raises(ComputeError) as ei:
        compute_efficiency(ending_zero, as_of=AS_OF)
    assert "inventory" in str(ei.value)

    # 平均口径为 0（期末与期初都非正）同样点名
    with pytest.raises(ComputeError) as ei2:
        compute_efficiency({**F(), "accounts_receivable": -120.0,
                            "accounts_receivable_prior": 120.0}, as_of=AS_OF)
    assert "accounts_receivable" in str(ei2.value)


# ── 现金流质量 ───────────────────────────────────────────────────


def test_cash_flow_quality_hand_computed():
    """spec 4.1（现金流类）：CFO/净利 与自由现金流与手算一致。"""
    r = compute_cash_flow_quality(F(), as_of=AS_OF)
    assert r["cfo_to_net_income"] == pytest.approx(210.0 / 130.0, abs=1e-15)
    assert r["free_cash_flow"] == pytest.approx(210.0 - 80.0, abs=1e-15)
    assert r["note"] is None
    assert r["params"] == {} and r["warmup"] == {}


def test_cash_flow_quality_negative_earnings_is_not_flipped():
    """净利为负而 CFO 为正：比值照实为负并加 note 说明读法失效，绝不取绝对值蒙混。"""
    f = {**F(), "net_income": -50.0, "cash_from_operations": 100.0}
    r = compute_cash_flow_quality(f, as_of=AS_OF)
    assert r["cfo_to_net_income"] == pytest.approx(100.0 / -50.0, abs=1e-15)
    assert r["cfo_to_net_income"] < 0.0
    assert r["free_cash_flow"] == pytest.approx(100.0 - 80.0, abs=1e-15)
    assert isinstance(r["note"], str) and "negative" in r["note"]

    # 净利为 0：比值无定义 → 拒绝并点名，不返回 inf
    with pytest.raises(ComputeError) as ei:
        compute_cash_flow_quality({**F(), "net_income": 0.0}, as_of=AS_OF)
    assert "net_income" in str(ei.value)


# ── DuPont ───────────────────────────────────────────────────────


def test_dupont_three_factor_product_matches_direct_roe():
    """spec Scenario DuPont 三因子乘积还原：三因子乘积与直接计算的 ROE 在 1e-9 内一致。"""
    r = compute_dupont(F(), as_of=AS_OF)
    fac = r["factors"]
    assert fac["net_margin"] == pytest.approx(130.0 / 1000.0, abs=1e-15)
    assert fac["asset_turnover"] == pytest.approx(1000.0 / 2000.0, abs=1e-15)
    assert fac["equity_multiplier"] == pytest.approx(2000.0 / 1200.0, abs=1e-15)
    # 独立手算：三个因子按定义逐个相乘（不调用被测代码的组合逻辑）
    expect = (130.0 / 1000.0) * (1000.0 / 2000.0) * (2000.0 / 1200.0)
    assert r["product"] == pytest.approx(expect, abs=1e-15)
    assert r["roe"] == pytest.approx(130.0 / 1200.0, abs=1e-15)
    assert abs(r["product"] - r["roe"]) <= 1e-9
    assert abs(r["residual"]) <= 1e-9
    assert r["residual"] == r["product"] - r["roe"]
    assert r["params"] == {} and r["warmup"] == {}


def test_dupont_identity_over_many_inputs():
    """恒等对随机量级稳健：product − roe 恒在 1e-9 内，且 residual 就是该差。"""
    for ni, rev, ta, eq in (
        (7.3, 101.7, 233.9, 88.1),
        (-12.5, 400.0, 900.0, 250.0),
        (0.0, 55.5, 77.7, 33.3),
    ):
        f = {"net_income": ni, "revenue": rev, "total_assets": ta, "total_equity": eq}
        r = compute_dupont(f, as_of=AS_OF)
        assert abs(r["product"] - r["roe"]) <= 1e-9
        assert r["residual"] == r["product"] - r["roe"]


# ── 红旗检测 ─────────────────────────────────────────────────────


def test_red_flags_scenario_receivables_60_vs_revenue_10():
    """spec Scenario 红旗规则的触发与不触发：应收增速 60% vs 营收增速 10% →
    返回含 `应收账款增速显著高于营收` 规则条目（名称、触发值、依据）。"""
    r = detect_red_flags(_base_fundamentals(), as_of=AS_OF)
    assert "应收账款增速显著高于营收" in r["triggered_rules"]
    entry = next(e for e in r["flags"] if e["rule"] == "应收账款增速显著高于营收")
    assert entry["triggered"] is True
    # 触发值 = 应收增速 − 营收增速（独立手算）
    expect = (160.0 / 100.0 - 1.0) - (1100.0 / 1000.0 - 1.0)
    assert entry["value"] == pytest.approx(expect, abs=1e-15)
    assert entry["threshold"] == 0.20
    assert "应收账款增速" in entry["basis"] and "营收增速" in entry["basis"]
    # 四条规则全部给出，各带 triggered 布尔
    assert [e["rule"] for e in r["flags"]] == [
        "应计利润异常",
        "应收账款增速显著高于营收",
        "存货增速显著高于营收",
        "毛利率与同行背离",
    ]
    assert all(isinstance(e["triggered"], bool) for e in r["flags"])
    assert all({"rule", "triggered", "value", "threshold", "basis"} == set(e) for e in r["flags"])


def test_red_flags_equal_growth_does_not_trigger():
    """spec Scenario 不触发侧：对增速持平的输入该规则不出现（triggered_rules 不含它）。"""
    r = detect_red_flags(_flat_growth(), as_of=AS_OF)
    assert "应收账款增速显著高于营收" not in r["triggered_rules"]
    entry = next(e for e in r["flags"] if e["rule"] == "应收账款增速显著高于营收")
    assert entry["triggered"] is False
    assert entry["value"] == pytest.approx(0.0, abs=1e-15)  # 同增 10% → 差为 0


def test_red_flags_custom_threshold_changes_outcome_and_is_echoed():
    """spec 4.3：阈值可配——0.20 → 不触发，0.05 → 触发；生效阈值逐条回显。"""
    # 应收 +20% vs 营收 +10% → 差 ≈ 0.10：高于 0.05、低于 0.20
    f = _flat_growth(accounts_receivable=120.0)
    loose = detect_red_flags(f, as_of=AS_OF)
    assert loose["thresholds"]["receivables_vs_revenue_growth"] == 0.20
    assert "应收账款增速显著高于营收" not in loose["triggered_rules"]

    tight = detect_red_flags(f, as_of=AS_OF, thresholds={"receivables_vs_revenue_growth": 0.05})
    assert tight["thresholds"]["receivables_vs_revenue_growth"] == 0.05
    assert "应收账款增速显著高于营收" in tight["triggered_rules"]
    # 生效阈值既在顶层 thresholds 也在 params 里
    assert tight["params"]["thresholds"] == tight["thresholds"]
    # 未覆盖的键保持默认值
    assert tight["thresholds"]["accruals_ratio"] == DEFAULT_THRESHOLDS["accruals_ratio"]


def test_red_flags_defaults_are_documented_and_echoed():
    """四个规则名与默认阈值逐条回显（调用方据此看见哪些默认值生效）。"""
    r = detect_red_flags(_base_fundamentals(), as_of=AS_OF)
    assert r["thresholds"] == {
        "accruals_ratio": 0.10,
        "receivables_vs_revenue_growth": 0.20,
        "inventory_vs_revenue_growth": 0.20,
        "gross_margin_vs_reference": 0.10,
    }
    assert r["params"]["peer_gross_margin"] is None
    assert r["warmup"] == {}


def test_red_flags_accruals_rule():
    """应计利润异常：|净利 − CFO| / 总资产 超阈（绝对值口径）才触发。"""
    f = {**F(), "net_income": 400.0, "cash_from_operations": 0.0, "total_assets": 1000.0}
    r = detect_red_flags(f, as_of=AS_OF)
    entry = next(e for e in r["flags"] if e["rule"] == "应计利润异常")
    assert entry["value"] == pytest.approx(0.4, abs=1e-15)
    assert entry["triggered"] is True

    clean = detect_red_flags(F(), as_of=AS_OF)  # |130−210|/2000 = 0.04 < 0.10
    entry2 = next(e for e in clean["flags"] if e["rule"] == "应计利润异常")
    assert entry2["value"] == pytest.approx(0.04, abs=1e-15)
    assert entry2["triggered"] is False


def test_red_flags_inventory_rule():
    """存货增速显著高于营收：同一形状、默认阈值 0.20。"""
    f = _flat_growth(inventory=168.0, inventory_prior=100.0)  # 存货 +68% vs 营收 +10%
    r = detect_red_flags(f, as_of=AS_OF)
    entry = next(e for e in r["flags"] if e["rule"] == "存货增速显著高于营收")
    assert entry["triggered"] is True
    assert entry["value"] == pytest.approx(0.68 - 0.10, abs=1e-15)
    assert "存货增速显著高于营收" in r["triggered_rules"]


def test_red_flags_gross_margin_vs_peer_benchmark():
    """毛利率与同行背离：以可比公司毛利率为对照（同行基准用例）+ 退回期初口径。"""
    # 本公司毛利率 0.50 vs 同行 0.30 → 差 0.20 > 0.10 触发
    f = {**F(), "revenue": 1000.0, "gross_profit": 500.0}
    hit = detect_red_flags(f, as_of=AS_OF, peer_gross_margin=0.30)
    entry = next(e for e in hit["flags"] if e["rule"] == "毛利率与同行背离")
    assert entry["triggered"] is True
    assert entry["value"] == pytest.approx(0.20, abs=1e-15)
    assert "毛利率与同行背离" in hit["triggered_rules"]
    assert hit["params"]["peer_gross_margin"] == 0.30

    # 同行 0.45 → 差 0.05，不触发
    near = detect_red_flags(f, as_of=AS_OF, peer_gross_margin=0.45)
    assert "毛利率与同行背离" not in near["triggered_rules"]

    # 未传 peer 时退回 fundamentals['gross_margin_prior']（0.36）：|0.4−0.36| = 0.04
    fallback = detect_red_flags(F(), as_of=AS_OF)
    entry2 = next(e for e in fallback["flags"] if e["rule"] == "毛利率与同行背离")
    assert entry2["value"] == pytest.approx(0.04, abs=1e-15)
    assert entry2["triggered"] is False


def test_red_flags_missing_reference_and_unknown_threshold_are_named():
    """缺毛利率对照 / 未知阈值键 / 负阈值 → 报错点名，不静默忽略配置。"""
    f = {k: v for k, v in F().items() if k != "gross_margin_prior"}
    with pytest.raises(ComputeError) as ei:
        detect_red_flags(f, as_of=AS_OF)
    assert "peer_gross_margin" in str(ei.value) and "gross_margin_prior" in str(ei.value)

    with pytest.raises(ComputeError) as ei2:
        detect_red_flags(F(), as_of=AS_OF, thresholds={"no_such_rule": 0.1})
    assert "no_such_rule" in str(ei2.value)

    with pytest.raises(ComputeError) as ei3:
        detect_red_flags(F(), as_of=AS_OF, thresholds={"accruals_ratio": -0.1})
    assert "accruals_ratio" in str(ei3.value)


@pytest.mark.parametrize(
    "field",
    ["revenue", "revenue_prior", "accounts_receivable", "accounts_receivable_prior",
     "inventory", "inventory_prior", "net_income", "cash_from_operations", "total_assets"],
)
def test_red_flags_missing_field_is_named(field):
    """四条规则全部求值 → 规则所需的任一字段缺失都报错点名，不返回部分规则。"""
    with pytest.raises(ComputeError) as ei:
        detect_red_flags(_base_fundamentals(**{field: None}), as_of=AS_OF)
    assert field in str(ei.value)


# ── 口径、形状与确定性（跨工具） ─────────────────────────────────

_CALLS = {
    "compute_profitability": lambda f: compute_profitability(f, as_of=AS_OF),
    "compute_growth": lambda f: compute_growth(f, as_of=AS_OF),
    "compute_leverage_and_liquidity": lambda f: compute_leverage_and_liquidity(f, as_of=AS_OF),
    "compute_interest_coverage": lambda f: compute_interest_coverage(f, as_of=AS_OF),
    "compute_efficiency": lambda f: compute_efficiency(f, as_of=AS_OF),
    "compute_cash_flow_quality": lambda f: compute_cash_flow_quality(f, as_of=AS_OF),
    "compute_dupont": lambda f: compute_dupont(f, as_of=AS_OF),
    "detect_red_flags": lambda f: detect_red_flags(f, as_of=AS_OF),
}


#: 每个工具真正读取（且缺了必须报错点名）的一个必填字段——非数值/非有限的探针位置。
_PROBE_FIELD = {
    "compute_profitability": "revenue",
    "compute_growth": "revenue",
    "compute_leverage_and_liquidity": "current_assets",
    "compute_interest_coverage": "ebit",
    "compute_efficiency": "accounts_receivable",
    "compute_cash_flow_quality": "cash_from_operations",
    "compute_dupont": "net_income",
    "detect_red_flags": "accounts_receivable",
}


def test_group_is_fundamental_ratios():
    from fd_quant.compute.fundamental import SAMPLES, TOOLS, _SAMPLE_SNAPSHOT

    assert GROUP == "fundamental_ratios"
    assert len(TOOLS) == 8
    assert all(spec.group == GROUP for spec in TOOLS)
    # 声明的快照样例与测试夹具同源（共享键必须逐位一致）——防止两份快照各自漂移
    fixture = F()
    for key, value in _SAMPLE_SNAPSHOT.items():
        assert key in fixture, f"sample key {key!r} not in three_statements()"
        assert value == fixture[key], f"sample key {key!r}: {value} != fixture {fixture[key]}"
    # 每个工具都声明了一组可直接调用的样例 kwargs（闸门据此接入快照型工具）
    assert set(SAMPLES) == {spec.name for spec in TOOLS}
    assert all(sample["as_of"] == "2026-06-30" for sample in SAMPLES.values())


@pytest.mark.parametrize("name", sorted(_CALLS), ids=sorted(_CALLS))
def test_as_of_is_validated_first_and_required(name):
    """spec 硬约束 1：as_of 必填且是第一个校验动作——缺 as_of 时错误只说 as_of。"""
    with pytest.raises(ComputeError) as ei:
        _call_with_as_of(name, None)
    assert "as_of" in str(ei.value)

    with pytest.raises(ComputeError) as ei2:
        _call_with_as_of(name, "2026/06/30")  # 非 ISO
    assert "as_of" in str(ei2.value)

    import inspect
    from fd_quant.compute.fundamental import TOOLS

    fn = next(s.fn for s in TOOLS if s.name == name)
    assert inspect.signature(fn).parameters["as_of"].default is inspect.Parameter.empty


def _call_with_as_of(name: str, as_of) -> dict:
    """以给定的 as_of 调一次（其余输入完整），用于校验"as_of 第一个被检查"。"""
    fn = {
        "compute_profitability": compute_profitability,
        "compute_growth": compute_growth,
        "compute_leverage_and_liquidity": compute_leverage_and_liquidity,
        "compute_interest_coverage": compute_interest_coverage,
        "compute_efficiency": compute_efficiency,
        "compute_cash_flow_quality": compute_cash_flow_quality,
        "compute_dupont": compute_dupont,
        "detect_red_flags": detect_red_flags,
    }[name]
    return fn(F(), as_of=as_of)


@pytest.mark.parametrize("name", sorted(_CALLS), ids=sorted(_CALLS))
def test_as_of_error_wins_over_missing_fundamentals(name):
    """顺序证明：as_of 非法而 fundamentals 也不全时，报错必须是 as_of（不是字段缺失）。"""
    fn = {
        "compute_profitability": compute_profitability,
        "compute_growth": compute_growth,
        "compute_leverage_and_liquidity": compute_leverage_and_liquidity,
        "compute_interest_coverage": compute_interest_coverage,
        "compute_efficiency": compute_efficiency,
        "compute_cash_flow_quality": compute_cash_flow_quality,
        "compute_dupont": compute_dupont,
        "detect_red_flags": detect_red_flags,
    }[name]
    with pytest.raises(ComputeError) as ei:
        fn(as_of=None)  # fundamentals 也没传
    assert "as_of" in str(ei.value)


@pytest.mark.parametrize("name", sorted(_CALLS), ids=sorted(_CALLS))
def test_missing_fundamentals_is_refused_not_silently_accepted(name):
    """fundamentals 缺省为 None 只为兼容统一调用形状：真的缺了必须报错点名。"""
    import inspect
    from fd_quant.compute.fundamental import TOOLS

    fn = next(s.fn for s in TOOLS if s.name == name)
    with pytest.raises(ComputeError) as ei:
        fn(as_of=AS_OF)
    assert "fundamentals" in str(ei.value)
    # dates 是"接受但本族不用"的可选关键字：不传也能调通
    assert inspect.signature(fn).parameters["dates"].default is None


@pytest.mark.parametrize("name", sorted(_CALLS), ids=sorted(_CALLS))
def test_dates_is_accepted_and_does_not_change_the_result(name):
    """基本面快照没有日期轴：传 dates 不改变任何数值（否则它就成了一份隐形输入）。"""
    d = qd.ohlcv(80, seed=101)
    plain = _CALLS[name](F())
    with_dates = {
        "compute_profitability": lambda: compute_profitability(F(), as_of=AS_OF, dates=d["dates"]),
        "compute_growth": lambda: compute_growth(F(), as_of=AS_OF, dates=d["dates"]),
        "compute_leverage_and_liquidity": lambda: compute_leverage_and_liquidity(
            F(), as_of=AS_OF, dates=d["dates"]),
        "compute_interest_coverage": lambda: compute_interest_coverage(
            F(), as_of=AS_OF, dates=d["dates"]),
        "compute_efficiency": lambda: compute_efficiency(F(), as_of=AS_OF, dates=d["dates"]),
        "compute_cash_flow_quality": lambda: compute_cash_flow_quality(
            F(), as_of=AS_OF, dates=d["dates"]),
        "compute_dupont": lambda: compute_dupont(F(), as_of=AS_OF, dates=d["dates"]),
        "detect_red_flags": lambda: detect_red_flags(F(), as_of=AS_OF, dates=d["dates"]),
    }[name]()
    base.assert_bitequal(plain, with_dates, what=name)


@pytest.mark.parametrize("name", sorted(_CALLS), ids=sorted(_CALLS))
def test_bitwise_determinism(name):
    """spec Scenario 同输入重跑逐位一致。"""
    base.assert_bitequal(_CALLS[name](F()), _CALLS[name](F()), what=name)


def _assert_no_numpy_leaves(value) -> None:
    if isinstance(value, dict):
        for k, v in value.items():
            assert isinstance(k, str), f"non-str key {k!r}"
            _assert_no_numpy_leaves(v)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _assert_no_numpy_leaves(item)
    else:
        assert not isinstance(value, np.generic), f"numpy scalar leaked: {type(value)}"
        assert not isinstance(value, np.ndarray), "ndarray leaked"


@pytest.mark.parametrize("name", sorted(_CALLS), ids=sorted(_CALLS))
def test_json_serialisable_without_nan_and_no_numpy_scalars(name):
    """spec 硬约束 8：allow_nan=False 可序列化，且不泄漏 numpy 标量/数组。"""
    r = _CALLS[name](F())
    _assert_no_numpy_leaves(r)
    dumped = json.dumps(r, allow_nan=False, sort_keys=True)
    for token in ("NaN", "Infinity"):
        assert token not in dumped, f"{name}: {token} in payload"
    assert r["evidence"] and all(e["source"].startswith("fd-quant:") for e in r["evidence"])
    assert all(name.split("_", 1)[1] in e["source"] for e in r["evidence"])


@pytest.mark.parametrize("name", sorted(_CALLS), ids=sorted(_CALLS))
def test_every_tool_declares_params_and_evidence_source(name):
    """spec 硬约束 10/11：params 回显 + evidence 非空且 source 含工具名。"""
    r = _CALLS[name](F())
    assert isinstance(r["params"], dict)
    assert r["warmup"] == {}
    ids = [e["evidence_id"] for e in r["evidence"]]
    assert len(ids) == len(set(ids))
    for e in r["evidence"]:
        assert set(e) == {"evidence_id", "label", "value", "unit", "as_of", "source"}
        assert e["as_of"] == AS_OF


@pytest.mark.parametrize("name", sorted(_CALLS), ids=sorted(_CALLS))
def test_non_numeric_and_non_finite_fields_are_refused(name):
    """字段存在但非数值 / 非有限 → 点名该字段（不把 NaN 静静带进结果）。"""
    field = _PROBE_FIELD[name]  # 每个工具真正读取的一个必填字段
    for bad in (float("nan"), float("inf"), "1000"):
        with pytest.raises(ComputeError) as ei:
            _CALLS[name]({**F(), field: bad})
        assert field in str(ei.value), f"{name}: bad {field} not named in {ei.value}"


def test_no_pandas_or_scipy_import_in_family_source():
    """依赖纪律：本族只 numpy + 标准库。"""
    from pathlib import Path

    src = Path(__file__).resolve().parents[1] / "fd_quant" / "compute" / "fundamental.py"
    text = src.read_text()
    assert "pandas" not in text
    assert "scipy" not in text
