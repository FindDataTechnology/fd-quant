"""基本面比率族：盈利 / 成长 / 偿债与流动性 / 效率 / 现金流质量 / DuPont / 红旗检测。

与序列族（技术指标、组合统计）不同，本族的主输入是一份**基本面快照**
（`fundamentals: dict[field, number]`）而不是逐点日期序列：因此统一外壳里的
`input_start / input_end / input_length` 一律为 `null`（契约允许：标量型工具向
`result_envelope` 传 `series=None`）。`as_of` 仍是必填 kwarg，且是每个函数的
**第一个校验动作**——它既是"这份快照的口径日"，也是 evidence 的 `as_of`。

单位由调用方决定（本族不假设币种，也不做单位换算）：比率一律无量纲，自由现金流
与输入同单位。金额字段请全程使用同一单位。

`dates` 参数为可选且**本族不使用**：它只为与序列族保持统一的可调用形状而接受——
基本面快照没有逐点日期轴，传入即被忽略（这是显式取舍：不假装有日期轴，
也不因为多了个参数就拒绝调用）。

缺字段纪律：所需字段缺失时抛 `ComputeError` 并**点名该字段**，绝不返回部分结果；
分母非正时点名该分母字段，绝不把 inf/NaN 放进返回。返回值保证
`json.dumps(..., allow_nan=False)` 可序列化（只出 python 标量，不出 numpy 标量）。
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Optional, Sequence

import numpy as np

from . import ToolSpec
from .base import (
    ComputeError,
    EvidenceBuilder,
    check_as_of,
    check_finite,
    deterministic_reduction,
    result_envelope,
)

GROUP = "fundamental_ratios"

#: 红旗规则的中文名（与 spec 逐字一致）。规则顺序即返回 `flags` 的顺序，固定不变。
_RULE_ACCRUALS = "应计利润异常"
_RULE_RECEIVABLES = "应收账款增速显著高于营收"
_RULE_INVENTORY = "存货增速显著高于营收"
_RULE_GROSS_MARGIN = "毛利率与同行背离"

#: 红旗规则的 ASCII 键：既用于 `thresholds` 配置覆盖，也用于 evidence label。
_RULE_ACC_KEY = "accruals_ratio"
_RULE_AR_KEY = "receivables_vs_revenue_growth"
_RULE_INV_KEY = "inventory_vs_revenue_growth"
_RULE_GM_KEY = "gross_margin_vs_reference"

#: 红旗阈值默认值（可被 `thresholds` 覆盖；覆盖值必须有限且非负）。触发判据一律
#: 为**严格大于**阈值——刚好等于阈值不算触发（边界不放大噪音）。
DEFAULT_THRESHOLDS: dict[str, float] = {
    _RULE_ACC_KEY: 0.10,
    _RULE_AR_KEY: 0.20,
    _RULE_INV_KEY: 0.20,
    _RULE_GM_KEY: 0.10,
}


# ── 内部工具（本文件私有；不从其他族 import 私有函数） ──────────


def _envelope(tool: str, as_of: str, b: EvidenceBuilder, payload: dict) -> dict:
    """标量外壳：本族无逐点日期轴，`series=None` → 起止日期与长度为 null。"""
    return result_envelope(tool, as_of, None, b, payload)


def _fundamentals(payload: Any, *, tool: str) -> Mapping[str, Any]:
    """主输入校验：必须是 field → number 的映射；缺失时报错点名 `fundamentals`。"""
    if payload is None:
        raise ComputeError(
            f"{tool}: fundamentals is required (a mapping of field -> number, e.g. "
            f"{{'revenue': 1000.0, 'net_income': 130.0}}); this family has no date "
            f"axis, so there is no series from which to infer it"
        )
    if not isinstance(payload, Mapping):
        raise ComputeError(
            f"{tool}: fundamentals must be a mapping of field -> number, "
            f"got {type(payload).__name__}"
        )
    return payload


def _missing(f: Mapping[str, Any], fields: Sequence[str], *, tool: str) -> None:
    """所需字段一次性点名（缺一个或多个都在同一条错误里列全）。"""
    missing = [name for name in fields if f.get(name) is None]
    if missing:
        raise ComputeError(
            f"{tool}: fundamentals is missing required field(s): {', '.join(missing)}"
        )


def _num(f: Mapping[str, Any], field: str, *, tool: str) -> float:
    """取一个有限数值字段：非数值 / 非有限 → 报错点名该字段（str 不静默转型）。"""
    v = f[field]
    if v is None or isinstance(v, bool) or isinstance(v, (str, bytes)):
        raise ComputeError(
            f"{tool}: fundamentals[{field!r}] must be a number, got {v!r}"
        )
    try:
        out = float(v)
    except (TypeError, ValueError):
        raise ComputeError(
            f"{tool}: fundamentals[{field!r}] must be a number, got {v!r}"
        ) from None
    if not math.isfinite(out):
        raise ComputeError(
            f"{tool}: fundamentals[{field!r}] must be finite, got {v!r}"
        )
    return out


def _positive(value: float, field: str, *, tool: str, ratio: str) -> float:
    """分母守卫：非正分母拒绝并点名该字段（不返回 inf/NaN）。"""
    if value <= 0.0:
        raise ComputeError(
            f"{tool}: {field} must be positive to compute {ratio}, got {value!r} "
            f"(a non-positive denominator is refused, not reported as inf/NaN)"
        )
    return value


def _gross_profit(f: Mapping[str, Any], revenue: float, *, tool: str) -> tuple[float, str]:
    """毛利率的分子：优先直接字段 `gross_profit`，缺失时用 `revenue - cost_of_goods_sold`。

    规则只有一条（先直接、后推导），取值口径在返回的 `sources` 里回显——调用方因此
    知道分子是财报直接披露的还是本工具推导的。两者都缺 → 报错点名两者。
    """
    if f.get("gross_profit") is not None:
        return _num(f, "gross_profit", tool=tool), "gross_profit"
    if f.get("cost_of_goods_sold") is not None:
        cogs = _num(f, "cost_of_goods_sold", tool=tool)
        return revenue - cogs, "revenue_minus_cost_of_goods_sold"
    raise ComputeError(
        f"{tool}: fundamentals is missing required field 'gross_profit' "
        f"(or, as an alternative, 'cost_of_goods_sold')"
    )


def _invested_capital(f: Mapping[str, Any], *, tool: str) -> tuple[float, str, str]:
    """ROIC 的投入资本：按固定顺序取第一个可得的口径，口径 token 原样回显。

    1. 直接字段 `invested_capital`（最精确，调用方自己算好）→ `"invested_capital"`；
    2. `total_debt + total_equity`（带息负债 + 股东权益，自下而上）→
       `"total_debt_plus_total_equity"`；
    3. `total_assets − current_liabilities`（"资本占用/资本运用"的常用自上而下近似，
       即总资产扣掉流动负债）→ `"total_assets_minus_current_liabilities"`。

    三级次序是文档化的定义，不是启发式猜测：每一级的 token 都进 `sources`，调用方
    因此知道 ROIC 的分母到底是谁。三级都不可得 → 报错点名 `invested_capital`。
    """
    if f.get("invested_capital") is not None:
        return (
            _num(f, "invested_capital", tool=tool),
            "invested_capital",
            "invested_capital",
        )
    if f.get("total_debt") is not None:
        debt = _num(f, "total_debt", tool=tool)
        equity = _num(f, "total_equity", tool=tool)
        return (
            debt + equity,
            "total_debt_plus_total_equity",
            "invested_capital (total_debt + total_equity)",
        )
    if f.get("current_liabilities") is not None:
        assets = _num(f, "total_assets", tool=tool)
        liabilities = _num(f, "current_liabilities", tool=tool)
        return (
            assets - liabilities,
            "total_assets_minus_current_liabilities",
            "invested_capital (total_assets - current_liabilities)",
        )
    raise ComputeError(
        f"{tool}: fundamentals is missing required field 'invested_capital' for roic "
        f"(alternatives: 'total_debt' + 'total_equity', or 'total_assets' + "
        f"'current_liabilities')"
    )


def _balance_basis(
    f: Mapping[str, Any], current: str, prior: str, *, tool: str, ratio: str
) -> tuple[float, str]:
    """周转率分母的余额口径：有期初（prior）→ (期末 + 期初)/2（"average"），否则期末（"ending"）。

    加法顺序固定为"期末 + 期初"再除以 2，使同一输入逐位可复现。混合口径不做静默
    平均：每个比率各自回显 basis，哪些用了平均余额一望可知。
    """
    end = _num(f, current, tool=tool)
    if f.get(prior) is not None:
        begin = _num(f, prior, tool=tool)
        avg = (end + begin) / 2.0
        _positive(avg, f"{current}/{prior} (average balance)", tool=tool, ratio=ratio)
        return avg, "average"
    _positive(end, current, tool=tool, ratio=ratio)
    return end, "ending"


# ── 盈利比率 ─────────────────────────────────────────────────────


def compute_profitability(
    fundamentals: Optional[Mapping[str, Any]] = None,
    *,
    as_of: str,
    dates: Optional[Sequence[str]] = None,
    tax_rate: float = 0.25,
) -> dict:
    """盈利比率：毛利率 / 净利率 / ROE / ROA / ROIC（含 NOPAT）。

    Input: fundamentals (dict, 必填字段: revenue, gross_profit 或 cost_of_goods_sold,
      net_income, total_equity, total_assets, ebit, 以及 ROIC 的投入资本, 依次取第一个
      可得者: invested_capital → total_debt+total_equity → total_assets-current_liabilities;
      金额单位由调用方统一), as_of (ISO 日期, 必填, 第一个被校验的参数), dates (可选,
      本族无日期轴 → 被忽略), tax_rate (默认 0.25, 仅用于 ROIC 的
      NOPAT = ebit × (1 − tax_rate), 必须在 [0,1] 内并原值回显)。
    Returns: ratios (五个无量纲比率: gross_margin = 毛利/营收, net_margin = 净利/营收,
      roe = 净利/股东权益, roa = 净利/总资产, roic = NOPAT/投入资本), sources (口径回显:
      gross_profit 取自 'gross_profit' 还是 'revenue_minus_cost_of_goods_sold',
      invested_capital 取自哪一级及其 token), params ({"tax_rate": 实际生效值}),
      warmup (空字典, 本族无序列故无 warmup 概念); 另加统一外壳 as_of /
      input_start=null / input_end=null / input_length=null / evidence (每个比率与
      NOPAT 各一条, source 含工具名与 tax_rate)。缺任一必填字段或分母
      (revenue/total_equity/total_assets/投入资本) 非正 → ComputeError 点名该字段,
      不返回部分结果。
    """
    tool = "compute_profitability"
    a = check_as_of(as_of)
    tax = check_finite(tax_rate, name="tax_rate")
    if not 0.0 <= tax <= 1.0:
        raise ComputeError(f"{tool}: tax_rate must be within [0, 1], got {tax!r}")

    f = _fundamentals(fundamentals, tool=tool)
    _missing(f, ("revenue", "net_income", "total_equity", "total_assets", "ebit"), tool=tool)

    revenue = _positive(
        _num(f, "revenue", tool=tool), "revenue", tool=tool, ratio="gross_margin/net_margin"
    )
    equity = _positive(_num(f, "total_equity", tool=tool), "total_equity", tool=tool, ratio="roe")
    assets = _positive(_num(f, "total_assets", tool=tool), "total_assets", tool=tool, ratio="roa")
    net_income = _num(f, "net_income", tool=tool)
    ebit = _num(f, "ebit", tool=tool)

    gross_profit, gross_src = _gross_profit(f, revenue, tool=tool)
    invested, invested_src, invested_label = _invested_capital(f, tool=tool)
    invested = _positive(invested, invested_label, tool=tool, ratio="roic")

    nopat = ebit * (1.0 - tax)
    ratios = {
        "gross_margin": gross_profit / revenue,
        "net_margin": net_income / revenue,
        "roe": net_income / equity,
        "roa": net_income / assets,
        "roic": nopat / invested,
    }

    ev_params = {"tax_rate": tax}
    b = EvidenceBuilder(tool, a)
    for name, value in ratios.items():
        b.add(name, value, unit="ratio", params=ev_params)
    b.add("nopat", nopat, unit="currency", params=ev_params)
    b.add("gross_profit", gross_profit, unit="currency", params=ev_params)
    b.add("invested_capital", invested, unit="currency", params=ev_params)

    return _envelope(tool, a, b, {
        "ratios": ratios,
        "sources": {"gross_profit": gross_src, "invested_capital": invested_src},
        "params": {"tax_rate": tax},
        "warmup": {},
    })


# ── 成长比率 ─────────────────────────────────────────────────────


def compute_growth(
    fundamentals: Optional[Mapping[str, Any]] = None,
    *,
    as_of: str,
    dates: Optional[Sequence[str]] = None,
) -> dict:
    """成长比率：营收同比 / 净利同比（YoY，小数口径而非百分数）。

    Input: fundamentals (dict, 必填字段: revenue, revenue_prior, net_income,
      net_income_prior; 金额单位由调用方统一), as_of (ISO 日期, 必填, 第一个被校验的
      参数), dates (可选, 本族无日期轴 → 被忽略)。
    Returns: ratios (revenue_yoy / net_income_yoy, 无量纲小数: 0.1 表示 +10%),
      params (空字典, 本工具无参数), warmup (空字典); 另加统一外壳 as_of /
      input_start=null / input_end=null / input_length=null / evidence (每个比率一条)。
      基期 (revenue_prior / net_income_prior) 必须为正: 非正基期上的增速会发明一个
      符号, 故拒绝并点名该基期字段; 缺字段同样点名拒绝, 不返回部分结果。
    """
    tool = "compute_growth"
    a = check_as_of(as_of)
    f = _fundamentals(fundamentals, tool=tool)
    _missing(f, ("revenue", "revenue_prior", "net_income", "net_income_prior"), tool=tool)

    revenue = _num(f, "revenue", tool=tool)
    revenue_prior = _positive(
        _num(f, "revenue_prior", tool=tool),
        "revenue_prior", tool=tool, ratio="revenue_yoy",
    )
    net_income = _num(f, "net_income", tool=tool)
    net_income_prior = _positive(
        _num(f, "net_income_prior", tool=tool),
        "net_income_prior", tool=tool, ratio="net_income_yoy",
    )

    ratios = {
        "revenue_yoy": revenue / revenue_prior - 1.0,
        "net_income_yoy": net_income / net_income_prior - 1.0,
    }
    b = EvidenceBuilder(tool, a)
    for name, value in ratios.items():
        b.add(name, value, unit="ratio", params=None)

    return _envelope(tool, a, b, {"ratios": ratios, "params": {}, "warmup": {}})


# ── 偿债与流动性 ─────────────────────────────────────────────────


def compute_leverage_and_liquidity(
    fundamentals: Optional[Mapping[str, Any]] = None,
    *,
    as_of: str,
    dates: Optional[Sequence[str]] = None,
) -> dict:
    """偿债与流动性比率：流动比率 / 速动比率 / 资产负债率 / 利息保障倍数。

    Input: fundamentals (dict, 必填字段: current_assets, current_liabilities, inventory,
      total_liabilities, total_assets, ebit, interest_expense; 金额单位由调用方统一),
      as_of (ISO 日期, 必填, 第一个被校验的参数), dates (可选, 本族无日期轴 → 被忽略)。
    Returns: ratios (current_ratio = 流动资产/流动负债, quick_ratio = (流动资产−存货)/
      流动负债, debt_to_asset_ratio = 总负债/总资产, interest_coverage = EBIT/利息费用;
      前三个无量纲, 利息保障倍数为倍数), params (空字典), warmup (空字典); 另加统一外壳
      as_of / input_start=null / input_end=null / input_length=null / evidence。
      **原子语义**: 四个比率同一次计算——任一字段缺失或分母 (current_liabilities /
      total_assets / interest_expense) 非正 → 整次调用报错并点名该字段, 不返回部分比率。
      只要利息保障倍数请用 compute_interest_coverage。
    """
    tool = "compute_leverage_and_liquidity"
    a = check_as_of(as_of)
    f = _fundamentals(fundamentals, tool=tool)
    _missing(
        f,
        ("current_assets", "current_liabilities", "inventory", "total_liabilities",
         "total_assets", "ebit", "interest_expense"),
        tool=tool,
    )

    current_assets = _num(f, "current_assets", tool=tool)
    current_liabilities = _positive(
        _num(f, "current_liabilities", tool=tool),
        "current_liabilities", tool=tool, ratio="current_ratio/quick_ratio",
    )
    inventory = _num(f, "inventory", tool=tool)
    total_liabilities = _num(f, "total_liabilities", tool=tool)
    total_assets = _positive(
        _num(f, "total_assets", tool=tool), "total_assets", tool=tool, ratio="debt_to_asset_ratio"
    )
    ebit = _num(f, "ebit", tool=tool)
    interest_expense = _positive(
        _num(f, "interest_expense", tool=tool),
        "interest_expense", tool=tool, ratio="interest_coverage",
    )

    ratios = {
        "current_ratio": current_assets / current_liabilities,
        "quick_ratio": (current_assets - inventory) / current_liabilities,
        "debt_to_asset_ratio": total_liabilities / total_assets,
        "interest_coverage": ebit / interest_expense,
    }
    b = EvidenceBuilder(tool, a)
    for name, value in ratios.items():
        b.add(name, value, unit="ratio", params=None)

    return _envelope(tool, a, b, {"ratios": ratios, "params": {}, "warmup": {}})


def compute_interest_coverage(
    fundamentals: Optional[Mapping[str, Any]] = None,
    *,
    as_of: str,
    dates: Optional[Sequence[str]] = None,
) -> dict:
    """利息保障倍数 = EBIT / 利息费用（独立工具，只要这一个比率时的最小调用）。

    Input: fundamentals (dict, 必填字段: ebit, interest_expense; 金额单位由调用方统一),
      as_of (ISO 日期, 必填, 第一个被校验的参数), dates (可选, 本族无日期轴 → 被忽略)。
    Returns: interest_coverage (倍数, 无量纲), params (空字典), warmup (空字典);
      另加统一外壳 as_of / input_start=null / input_end=null / input_length=null / evidence。
      缺 interest_expense → ComputeError 点名该字段, 不返回部分结果; interest_expense = 0
      (无利息负担) 时该倍数无定义, 同样拒绝并点名——不返回 inf/巨大哨兵值。
    """
    tool = "compute_interest_coverage"
    a = check_as_of(as_of)
    f = _fundamentals(fundamentals, tool=tool)
    _missing(f, ("ebit", "interest_expense"), tool=tool)

    ebit = _num(f, "ebit", tool=tool)
    interest_expense = _positive(
        _num(f, "interest_expense", tool=tool),
        "interest_expense", tool=tool, ratio="interest_coverage",
    )
    value = ebit / interest_expense

    b = EvidenceBuilder(tool, a)
    b.add("interest_coverage", value, unit="ratio", params=None)
    b.add("ebit", ebit, unit="currency", params=None)
    b.add("interest_expense", interest_expense, unit="currency", params=None)

    return _envelope(tool, a, b, {
        "interest_coverage": value,
        "params": {},
        "warmup": {},
    })


# ── 效率比率 ─────────────────────────────────────────────────────


def compute_efficiency(
    fundamentals: Optional[Mapping[str, Any]] = None,
    *,
    as_of: str,
    dates: Optional[Sequence[str]] = None,
) -> dict:
    """效率比率：应收周转 / 存货周转 / 总资产周转（次/期，无量纲）。

    Input: fundamentals (dict, 必填字段: revenue, accounts_receivable, inventory,
      total_assets; 可选: 期初余额 accounts_receivable_prior / inventory_prior /
      total_assets_prior, 以及 cost_of_goods_sold), as_of (ISO 日期, 必填, 第一个被校验
      的参数), dates (可选, 本族无日期轴 → 被忽略)。
    Returns: ratios (receivable_turnover = revenue/应收余额, inventory_turnover =
      存货周转分子/存货余额, total_asset_turnover = revenue/总资产余额), basis (单一口径
      token: 三个比率全部用平均余额 → "average", 全部用期末 → "ending", 混合 → "mixed"),
      basis_by_ratio (逐比率回显: 提供了对应期初值 → "average" = (期末+期初)/2, 否则
      "ending" = 期末), params ({"inventory_turnover_numerator":
      "cost_of_goods_sold" | "revenue"} —— 存货周转的分子优先用 cost_of_goods_sold,
      缺失时退回 revenue 并回显, 两者经济含义不同, 调用方据此解读), warmup (空字典);
      另加统一外壳 as_of / input_start=null / input_end=null / input_length=null /
      evidence。缺字段或余额非正 → 报错点名该字段。
    """
    tool = "compute_efficiency"
    a = check_as_of(as_of)
    f = _fundamentals(fundamentals, tool=tool)
    _missing(
        f,
        ("revenue", "accounts_receivable", "inventory", "total_assets"),
        tool=tool,
    )

    revenue = _positive(
        _num(f, "revenue", tool=tool),
        "revenue", tool=tool, ratio="receivable_turnover/total_asset_turnover",
    )
    ar_avg, ar_basis = _balance_basis(
        f, "accounts_receivable", "accounts_receivable_prior",
        tool=tool, ratio="receivable_turnover",
    )
    inv_avg, inv_basis = _balance_basis(
        f, "inventory", "inventory_prior", tool=tool, ratio="inventory_turnover"
    )
    ta_avg, ta_basis = _balance_basis(
        f, "total_assets", "total_assets_prior", tool=tool, ratio="total_asset_turnover"
    )
    if f.get("cost_of_goods_sold") is not None:
        inv_numerator = _num(f, "cost_of_goods_sold", tool=tool)
        inv_num_src = "cost_of_goods_sold"
    else:
        inv_numerator = revenue
        inv_num_src = "revenue"

    ratios = {
        "receivable_turnover": revenue / ar_avg,
        "inventory_turnover": inv_numerator / inv_avg,
        "total_asset_turnover": revenue / ta_avg,
    }
    basis_by_ratio = {
        "receivable_turnover": ar_basis,
        "inventory_turnover": inv_basis,
        "total_asset_turnover": ta_basis,
    }
    distinct = sorted(set(basis_by_ratio.values()))
    overall_basis = distinct[0] if len(distinct) == 1 else "mixed"
    params = {"inventory_turnover_numerator": inv_num_src}
    b = EvidenceBuilder(tool, a)
    for name, value in ratios.items():
        b.add(name, value, unit="ratio", params=params)
    b.add("inventory_turnover_numerator_value", inv_numerator, unit="currency", params=params)

    return _envelope(tool, a, b, {
        "ratios": ratios,
        "basis": overall_basis,
        "basis_by_ratio": basis_by_ratio,
        "params": params,
        "warmup": {},
    })


# ── 现金流质量 ───────────────────────────────────────────────────


def compute_cash_flow_quality(
    fundamentals: Optional[Mapping[str, Any]] = None,
    *,
    as_of: str,
    dates: Optional[Sequence[str]] = None,
) -> dict:
    """现金流质量：经营现金流 / 净利 与自由现金流 (= 经营现金流 − 资本开支)。

    Input: fundamentals (dict, 必填字段: cash_from_operations, net_income,
      capital_expenditure; 金额单位由调用方统一。capital_expenditure 以**正数**表示
      流出, 即 FCF = cash_from_operations − capital_expenditure; 若你的来源用负数表示
      流出, 请先取绝对值——本工具不做符号推断), as_of (ISO 日期, 必填, 第一个被校验的
      参数), dates (可选, 本族无日期轴 → 被忽略)。
    Returns: cfo_to_net_income (无量纲倍数, 照实报出——净利为负时该值为负, 绝不取绝对值),
      free_cash_flow (与输入同单位的原始金额), note (null 或字符串: net_income < 0 时
      说明"比值 > 1 即质量好"的读法失效, 比值仍如实给出), params (空字典), warmup (空字典);
      另加统一外壳 as_of / input_start=null / input_end=null / input_length=null / evidence。
      net_income = 0 时比值无定义 → 报错点名 net_income; 缺字段同样点名拒绝, 不返回部分结果。
    """
    tool = "compute_cash_flow_quality"
    a = check_as_of(as_of)
    f = _fundamentals(fundamentals, tool=tool)
    _missing(
        f, ("cash_from_operations", "net_income", "capital_expenditure"), tool=tool
    )

    cfo = _num(f, "cash_from_operations", tool=tool)
    net_income = _num(f, "net_income", tool=tool)
    capex = _num(f, "capital_expenditure", tool=tool)
    if net_income == 0.0:
        raise ComputeError(
            f"{tool}: net_income must be non-zero to compute cfo_to_net_income "
            f"(cash_from_operations / net_income), got {net_income!r} "
            f"(division by zero is refused, not reported as inf/NaN)"
        )

    ratio = cfo / net_income
    fcf = cfo - capex
    note: Optional[str] = None
    if net_income < 0.0:
        note = (
            f"net_income is negative ({net_income!r}): the '>1 means high cash quality' "
            f"reading of cfo_to_net_income breaks down because the denominator's sign "
            f"flips the ratio; the ratio is reported as-is (no abs()) and free_cash_flow "
            f"is unaffected"
        )

    b = EvidenceBuilder(tool, a)
    b.add("cfo_to_net_income", ratio, unit="ratio", params=None)
    b.add("free_cash_flow", fcf, unit="currency", params=None)
    b.add("net_income_negative", net_income < 0.0, params=None)

    return _envelope(tool, a, b, {
        "cfo_to_net_income": ratio,
        "free_cash_flow": fcf,
        "note": note,
        "params": {},
        "warmup": {},
    })


# ── DuPont 三因子分解 ────────────────────────────────────────────


def compute_dupont(
    fundamentals: Optional[Mapping[str, Any]] = None,
    *,
    as_of: str,
    dates: Optional[Sequence[str]] = None,
) -> dict:
    """DuPont 三因子分解：净利率 × 总资产周转 × 权益乘数 = ROE。

    Input: fundamentals (dict, 必填字段: net_income, revenue, total_assets, total_equity;
      金额单位由调用方统一), as_of (ISO 日期, 必填, 第一个被校验的参数), dates (可选,
      本族无日期轴 → 被忽略)。
    Returns: factors (net_margin = net_income/revenue, asset_turnover = revenue/total_assets,
      equity_multiplier = total_assets/total_equity), product (三因子按固定从左到右顺序相乘
      的乘积), roe (直接计算的 net_income/total_equity), residual (product − roe, 用
      deterministic_reduction 的 Kahan 减法得到; 恒等式使 |residual| ≤ 1e-9),
      params (空字典), warmup (空字典); 另加统一外壳 as_of / input_start=null /
      input_end=null / input_length=null / evidence (三因子 + 乘积 + ROE + residual)。
      分母 (revenue / total_assets / total_equity) 非正 → 报错点名该字段; 缺字段同样点名。
    """
    tool = "compute_dupont"
    a = check_as_of(as_of)
    f = _fundamentals(fundamentals, tool=tool)
    _missing(f, ("net_income", "revenue", "total_assets", "total_equity"), tool=tool)

    net_income = _num(f, "net_income", tool=tool)
    revenue = _positive(_num(f, "revenue", tool=tool), "revenue", tool=tool, ratio="net_margin")
    total_assets = _positive(
        _num(f, "total_assets", tool=tool), "total_assets", tool=tool, ratio="asset_turnover"
    )
    total_equity = _positive(
        _num(f, "total_equity", tool=tool), "total_equity", tool=tool, ratio="equity_multiplier"
    )

    net_margin = net_income / revenue
    asset_turnover = revenue / total_assets
    equity_multiplier = total_assets / total_equity  # 也是直接 ROE 的分母

    # 三因子乘积：归约顺序固定为从左到右（净利率 → 周转 → 乘数）。乘法链条本身逐位
    # 确定（不像 np.sum 有分块问题），但顺序即定义：不排序、不重排、不换结合律。
    product = net_margin * asset_turnover
    product = product * equity_multiplier
    roe = net_income / total_equity
    # residual 用契约的 Kahan 减法（deterministic_reduction）——product 与 roe 的末位
    # 比较在同一归约纪律下进行，同输入逐位可复现。
    residual = deterministic_reduction(np.array([product, -roe], dtype=np.float64))

    b = EvidenceBuilder(tool, a)
    for name, value in (
        ("net_margin", net_margin),
        ("asset_turnover", asset_turnover),
        ("equity_multiplier", equity_multiplier),
        ("product", product),
        ("roe", roe),
        ("residual", residual),
    ):
        b.add(name, value, unit="ratio", params=None)

    return _envelope(tool, a, b, {
        "factors": {
            "net_margin": net_margin,
            "asset_turnover": asset_turnover,
            "equity_multiplier": equity_multiplier,
        },
        "product": product,
        "roe": roe,
        "residual": residual,
        "params": {},
        "warmup": {},
    })


# ── 红旗检测 ─────────────────────────────────────────────────────


def detect_red_flags(
    fundamentals: Optional[Mapping[str, Any]] = None,
    *,
    as_of: str,
    dates: Optional[Sequence[str]] = None,
    thresholds: Optional[Mapping[str, Any]] = None,
    peer_gross_margin: Optional[float] = None,
) -> dict:
    """红旗检测：应计异常 / 应收异动 / 存货异动 / 毛利率背离（阈值可配，默认值回显）。

    Input: fundamentals (dict, 必填字段: revenue, revenue_prior, accounts_receivable,
      accounts_receivable_prior, inventory, inventory_prior, net_income,
      cash_from_operations, total_assets, 以及 gross_profit 或 cost_of_goods_sold,
      以及毛利率对照之一), as_of (ISO 日期, 必填, 第一个被校验的参数), dates (可选, 本族
      无日期轴 → 被忽略), thresholds (可选 dict, 键为 accruals_ratio /
      receivables_vs_revenue_growth / inventory_vs_revenue_growth /
      gross_margin_vs_reference, 缺省见 DEFAULT_THRESHOLDS = 0.10/0.20/0.20/0.10),
      peer_gross_margin (可选, 同行毛利率小数; 缺省退回 fundamentals['gross_margin_prior'])。
    Returns: flags (四条规则**逐条**给出: rule 为中文规则名, triggered 为 bool, value 为
      触发值同阈值口径, threshold 为生效阈值, basis 为人类可读判定依据含公式与代入数值),
      triggered_rules (仅被触发规则的中文名, 顺序固定), thresholds (生效阈值回显,
      调用方据此看见哪些默认值被覆盖), params ({"thresholds": 生效阈值,
      "peer_gross_margin": 传入值或 null}), warmup (空字典); 另加统一外壳 as_of /
      input_start=null / input_end=null / input_length=null / evidence (逐规则一条)。
      四条规则全部求值, 故规则所需字段缺一即报错并点名该字段, 不返回部分结果;
      阈值必须有限且非负, 未知阈值键同样点名拒绝(避免配置被静默忽略)。
    """
    tool = "detect_red_flags"
    a = check_as_of(as_of)
    eff = _effective_thresholds(thresholds, tool=tool)
    peer: Optional[float] = None
    if peer_gross_margin is not None:
        peer = check_finite(peer_gross_margin, name="peer_gross_margin")

    f = _fundamentals(fundamentals, tool=tool)
    _missing(
        f,
        ("revenue", "revenue_prior", "accounts_receivable", "accounts_receivable_prior",
         "inventory", "inventory_prior", "net_income", "cash_from_operations",
         "total_assets"),
        tool=tool,
    )

    revenue = _positive(
        _num(f, "revenue", tool=tool), "revenue", tool=tool, ratio="gross_margin/revenue_growth"
    )
    revenue_prior = _positive(
        _num(f, "revenue_prior", tool=tool),
        "revenue_prior", tool=tool, ratio="revenue_growth",
    )
    ar = _num(f, "accounts_receivable", tool=tool)
    ar_prior = _positive(
        _num(f, "accounts_receivable_prior", tool=tool),
        "accounts_receivable_prior", tool=tool, ratio="accounts_receivable_growth",
    )
    inventory = _num(f, "inventory", tool=tool)
    inventory_prior = _positive(
        _num(f, "inventory_prior", tool=tool),
        "inventory_prior", tool=tool, ratio="inventory_growth",
    )
    total_assets = _positive(
        _num(f, "total_assets", tool=tool), "total_assets", tool=tool, ratio="accruals_ratio"
    )
    net_income = _num(f, "net_income", tool=tool)
    cfo = _num(f, "cash_from_operations", tool=tool)
    gross_profit, gross_src = _gross_profit(f, revenue, tool=tool)
    if peer is not None:
        ref, ref_src = peer, "peer_gross_margin"
    elif f.get("gross_margin_prior") is not None:
        ref, ref_src = _num(f, "gross_margin_prior", tool=tool), "gross_margin_prior"
    else:
        raise ComputeError(
            f"{tool}: missing gross-margin reference for '{_RULE_GROSS_MARGIN}': pass "
            f"peer_gross_margin=... as an argument or fundamentals['gross_margin_prior']"
        )

    revenue_growth = revenue / revenue_prior - 1.0
    ar_growth = ar / ar_prior - 1.0
    inventory_growth = inventory / inventory_prior - 1.0
    accruals_ratio = abs((net_income - cfo) / total_assets)
    gross_margin = gross_profit / revenue
    margin_gap = abs(gross_margin - ref)

    th_acc = eff[_RULE_ACC_KEY]
    th_ar = eff[_RULE_AR_KEY]
    th_inv = eff[_RULE_INV_KEY]
    th_gm = eff[_RULE_GM_KEY]
    growth_gap = ar_growth - revenue_growth
    inventory_gap = inventory_growth - revenue_growth

    flags = [
        {
            "rule": _RULE_ACCRUALS,
            "triggered": accruals_ratio > th_acc,
            "value": accruals_ratio,
            "threshold": th_acc,
            "basis": (
                f"accruals_ratio = |net_income - cash_from_operations| / total_assets "
                f"= |{net_income} - {cfo}| / {total_assets} = {accruals_ratio}; "
                f"阈值 {th_acc}（绝对值口径, thresholds['{_RULE_ACC_KEY}'], 严格大于才触发）"
            ),
        },
        {
            "rule": _RULE_RECEIVABLES,
            "triggered": growth_gap > th_ar,
            "value": growth_gap,
            "threshold": th_ar,
            "basis": (
                f"应收账款增速 - 营收增速 = ({ar} / {ar_prior} - 1) - ({revenue} / "
                f"{revenue_prior} - 1) = {ar_growth} - {revenue_growth} = {growth_gap}; "
                f"阈值 {th_ar}（单侧, thresholds['{_RULE_AR_KEY}'], 严格大于才触发）"
            ),
        },
        {
            "rule": _RULE_INVENTORY,
            "triggered": inventory_gap > th_inv,
            "value": inventory_gap,
            "threshold": th_inv,
            "basis": (
                f"存货增速 - 营收增速 = ({inventory} / {inventory_prior} - 1) - ({revenue} / "
                f"{revenue_prior} - 1) = {inventory_growth} - {revenue_growth} = "
                f"{inventory_gap}; 阈值 {th_inv}（单侧, thresholds['{_RULE_INV_KEY}'], "
                f"严格大于才触发）"
            ),
        },
        {
            "rule": _RULE_GROSS_MARGIN,
            "triggered": margin_gap > th_gm,
            "value": margin_gap,
            "threshold": th_gm,
            "basis": (
                f"|当前毛利率 - 对照毛利率| = |{gross_margin} - {ref}| = {margin_gap}; "
                f"当前毛利率 = {gross_src} / revenue, 对照取自 {ref_src}; "
                f"阈值 {th_gm}（绝对值口径, thresholds['{_RULE_GM_KEY}'], 严格大于才触发）"
            ),
        },
    ]
    rule_keys = (_RULE_ACC_KEY, _RULE_AR_KEY, _RULE_INV_KEY, _RULE_GM_KEY)
    triggered = [entry["rule"] for entry in flags if entry["triggered"]]

    ev_params = {"peer_gross_margin": peer}
    b = EvidenceBuilder(tool, a)
    for key, entry in zip(rule_keys, flags):
        b.add(f"rule:{key}", entry["value"], unit="ratio", params=ev_params)
    b.add("triggered_rules", triggered, unit=None, params=ev_params)
    b.add("triggered_count", len(triggered), unit=None, params=ev_params)

    return _envelope(tool, a, b, {
        "flags": flags,
        "triggered_rules": triggered,
        "thresholds": dict(eff),
        "params": {"thresholds": dict(eff), "peer_gross_margin": peer},
        "warmup": {},
    })


def _effective_thresholds(
    thresholds: Optional[Mapping[str, Any]], *, tool: str
) -> dict[str, float]:
    """红旗阈值的生效值：默认值 + 覆盖值。

    未知键直接报错点名（配置被静默忽略是最难查的一类错）；覆盖值必须有限且非负
    （负阈值会把规则反转成"某方向越小越危险"，不是本工具要表达的意思）。
    """
    eff = dict(DEFAULT_THRESHOLDS)
    if thresholds is None:
        return eff
    if not isinstance(thresholds, Mapping):
        raise ComputeError(
            f"{tool}: thresholds must be a mapping of key -> number, "
            f"got {type(thresholds).__name__}"
        )
    unknown = [k for k in thresholds if k not in eff]
    if unknown:
        raise ComputeError(
            f"{tool}: unknown threshold key(s): {', '.join(sorted(unknown))}; "
            f"valid keys are {', '.join(DEFAULT_THRESHOLDS)}"
        )
    for key in DEFAULT_THRESHOLDS:  # 固定顺序扫描，与默认表顺序一致
        if key not in thresholds:
            continue
        value = thresholds[key]
        if value is None or isinstance(value, bool) or isinstance(value, (str, bytes)):
            raise ComputeError(
                f"{tool}: thresholds[{key!r}] must be a number, got {value!r}"
            )
        number = check_finite(value, name=f"thresholds[{key!r}]")
        if number < 0.0:
            raise ComputeError(
                f"{tool}: thresholds[{key!r}] must be non-negative, got {value!r}"
            )
        eff[key] = number
    return eff


TOOLS: list[ToolSpec] = [
    ToolSpec("compute_profitability", GROUP, compute_profitability),
    ToolSpec("compute_growth", GROUP, compute_growth),
    ToolSpec("compute_leverage_and_liquidity", GROUP, compute_leverage_and_liquidity),
    ToolSpec("compute_interest_coverage", GROUP, compute_interest_coverage),
    ToolSpec("compute_efficiency", GROUP, compute_efficiency),
    ToolSpec("compute_cash_flow_quality", GROUP, compute_cash_flow_quality),
    ToolSpec("compute_dupont", GROUP, compute_dupont),
    ToolSpec("detect_red_flags", GROUP, detect_red_flags),
]

#: 快照型工具没有日期轴，跨族合规闸门无法按参数名猜出一组合法输入——这里为每个
#: 工具声明一份完整、确定性的样例调用参数（工具名 → kwargs），供闸门与自检复用。
#: 与 `tests/quantdata.three_statements()` 是同一组自洽三表数量级（亿元），此处就
#: 写成字面量：计算层不依赖测试夹具，快照与夹具的漂移由测试自己断言。
_SAMPLE_SNAPSHOT: dict[str, Any] = {
    "revenue": 1000.0,
    "cost_of_goods_sold": 600.0,
    "gross_profit": 400.0,
    "operating_income": 180.0,
    "net_income": 130.0,
    "interest_expense": 20.0,
    "income_tax": 30.0,
    "total_assets": 2000.0,
    "total_equity": 1200.0,
    "total_liabilities": 800.0,
    "current_assets": 700.0,
    "current_liabilities": 350.0,
    "inventory": 150.0,
    "accounts_receivable": 120.0,
    "accounts_payable": 90.0,
    "cash_from_operations": 210.0,
    "capital_expenditure": 80.0,
    "ebit": 180.0,
    "revenue_prior": 900.0,
    "net_income_prior": 110.0,
    "accounts_receivable_prior": 75.0,
    "inventory_prior": 140.0,
    "gross_margin_prior": 0.36,
}

SAMPLE_AS_OF = "2026-06-30"

SAMPLES: dict[str, dict] = {
    name: {"fundamentals": dict(_SAMPLE_SNAPSHOT), "as_of": SAMPLE_AS_OF}
    for name in (
        "compute_profitability",
        "compute_growth",
        "compute_leverage_and_liquidity",
        "compute_interest_coverage",
        "compute_efficiency",
        "compute_cash_flow_quality",
        "compute_dupont",
        "detect_red_flags",
    )
}
