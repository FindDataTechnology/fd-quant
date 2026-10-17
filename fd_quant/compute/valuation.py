"""估值算子族：DCF（WACC 分步 + 两种终值）/ DDM / 可比倍数（winsorize）/ Monte Carlo /
敏感性网格 / 估值合成（扣值不扣结论）。

与序列族（技术指标、组合统计）不同，本族的主输入是**假设快照**——一组标量假设、
一张倍数表、一份方法清单——而不是逐点日期序列：因此统一外壳里的
`input_start / input_end / input_length` 一律为 `null`（`result_envelope` 收
`series=None`，契约允许标量型工具这样做）。`as_of` 仍是必填 kwarg，且是每个函数的
**第一个校验动作**——它既是"这组假设的口径日"，也是 evidence 的 `as_of`。

**单位纪律**（本族不做任何单位换算，调用方自己统一）：

- 金额列（`free_cash_flows` / `enterprise_value` / `equity_value` / `terminal_value`）
  与 `shares_outstanding` 用调用方自己的货币单位（元/亿元/美元皆可，全程一致）；
- 每股价值（`value_per_share` / `price_target`）单位是同一种货币的"每股"；
- 比率一律是小数（`0.09` = 9%，不是 9）；
- 倍数（倍数表、`exit_multiple`）无量纲；`implied_valuation` 与 `metric_value` 同单位。

**确定性**：归约只用 `deterministic_reduction`（Kahan 从左到右）或 `np.cumsum` /
`np.cumprod` 顺序扫描。Monte Carlo 只用 `np.random.default_rng(seed)`（PCG64，跨版本
稳定的位流），`seed` 有固定默认值 42；禁止全局随机接口与 `random` 模块。
分位数一律用**显式**线性插值（`_quantile`：排序表 + 分数下标，等价 numpy 默认的
`method="linear"`）——定义写在本文件里，不借第三方默认值。

**拒绝而非编造**：缺字段、非有限数、分母非正、`terminal_growth >= discount_rate`
等情况一律抛 `ComputeError` 并点名该字段/两个数；返回恒可
`json.dumps(..., allow_nan=False)`（不出 NaN/inf/np 标量）。
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, Optional, Union

import numpy as np

from . import ToolSpec
from .base import (
    ComputeError,
    EvidenceBuilder,
    check_as_of,
    check_finite,
    check_positive_window,
    deterministic_reduction,
    result_envelope,
    to_none_list,
)

GROUP = "valuation_operators"

#: 终值法的两个合法取值（spec：「永续增长与退出倍数两种终值」）。
TERMINAL_PERPETUITY = "perpetuity"
TERMINAL_EXIT_MULTIPLE = "exit_multiple"
TERMINAL_METHODS = (TERMINAL_PERPETUITY, TERMINAL_EXIT_MULTIPLE)

#: WACC 分步法需要的成分（缺一个就点名一份，不做默认假设）。
WACC_FIELDS = (
    "risk_free",
    "equity_beta",
    "equity_risk_premium",
    "cost_of_debt",
    "tax_rate",
    "debt_weight",
    "equity_weight",
)

#: Monte Carlo 默认百分位带（label → 分位水平；水平是小数）。
DEFAULT_PERCENTILES: tuple[tuple[str, float], ...] = (
    ("p5", 0.05),
    ("p25", 0.25),
    ("p50", 0.50),
    ("p75", 0.75),
    ("p95", 0.95),
)

#: 敏感性网格允许的两个自变量名（都是 DCF 的标量假设；取值必须为有限数）。
GRID_VARIABLES = (
    "discount_rate",
    "terminal_growth",
    "exit_multiple",
    "exit_metric",
    "shares_outstanding",
    "net_debt",
)

#: Monte Carlo 每次调用允许的最大模拟数（10^7 是 OOM 护栏，不是精度建议）。
_MAX_SIMULATIONS = 10_000_000


# ── 内部工具（本文件私有；不从其他族 import 私有函数） ──────────


def _envelope(tool: str, as_of: str, b: EvidenceBuilder, payload: dict) -> dict:
    """标量外壳：本族无逐点日期轴，`series=None` → 起止日期与长度为 null。"""
    return result_envelope(tool, as_of, None, b, payload)


def _validate_method(method: Any, *, tool: str) -> str:
    """终值法取值校验：不在两个合法取值里 → 点名该字段与合法值。"""
    if method not in TERMINAL_METHODS:
        raise ComputeError(
            f"{tool}: terminal_method must be one of {list(TERMINAL_METHODS)}, "
            f"got {method!r}"
        )
    return str(method)


def _optional_number(value: Any, *, name: str, tool: str) -> Optional[float]:
    """可选标量：None 原样返回（表未提供），否则 check_finite 并点名。"""
    if value is None:
        return None
    return check_finite(value, name=f"{tool}: {name}")


def _number(value: Any, *, name: str, tool: str) -> float:
    """必填标量：None → 点名缺失；非有限 → 点名该参数。"""
    if value is None:
        raise ComputeError(f"{tool}: {name} is required (got None)")
    return check_finite(value, name=f"{tool}: {name}")


def _num_array(values: Any, *, name: str, tool: str) -> np.ndarray:
    """数值序列校验：非空、逐位有限；违规点名**该下标**（不静默丢掉那一项）。

    接受任何可 `len()` 的序列（含 ndarray），拒绝标量与字符串；`Mapping` 不是序列。
    """
    if values is None or isinstance(values, (str, bytes)) or isinstance(values, Mapping):
        raise ComputeError(
            f"{tool}: {name} must be a non-empty sequence of numbers, got {values!r}"
        )
    try:
        n = len(values)
    except TypeError:
        raise ComputeError(
            f"{tool}: {name} must be a non-empty sequence of numbers, got {values!r}"
        ) from None
    if n == 0:
        raise ComputeError(f"{tool}: {name} must not be empty")
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        v = values[i]
        if v is None or isinstance(v, bool) or isinstance(v, (str, bytes)):
            raise ComputeError(f"{tool}: {name}[{i}] must be a number, got {v!r}")
        try:
            f = float(v)
        except (TypeError, ValueError):
            raise ComputeError(f"{tool}: {name}[{i}] must be a number, got {v!r}") from None
        if not math.isfinite(f):
            raise ComputeError(f"{tool}: {name}[{i}] must be finite, got {v!r}")
        out[i] = f
    return out


def _quantile(sorted_values: Sequence[float], q: float) -> float:
    """分位数：排序表 + 分数下标的**线性插值**（numpy 默认 `method="linear"`,
    Hyndman–Fan type 7，「closest ranks 之间线性插值」）。

    定义写在这里而不是调 np.percentile，是为了让"中位数/分位点到底怎么插的"
    留在本文件里可读：`idx = q*(n-1)`，取 `floor(idx)` 与 `ceil(idx)` 两个秩，
    按 `idx - floor(idx)` 线性插值；n=1 直接返回该点。单调、与输入顺序无关。
    """
    n = len(sorted_values)
    if n == 1:
        return float(sorted_values[0])
    idx = q * (n - 1)
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return float(sorted_values[lo])
    frac = idx - lo
    a = float(sorted_values[lo])
    b = float(sorted_values[hi])
    return a + frac * (b - a)


def _kahan_mean(arr: np.ndarray) -> float:
    """固定顺序均值（Kahan 求和 ÷ n）——不借 np.mean 的成对归约。"""
    n = len(arr)
    if n == 0:
        raise ComputeError("mean of an empty array is undefined")
    return deterministic_reduction(arr) / n


def _n(x: Any) -> str:
    """数值 → 文本：用 `str(float)`（如 100.0 → '100.0'），说明文字里的数不打折扣。"""
    return str(float(x))


def _json_plain(value: Any) -> Any:
    """回显调用方结构时的 JSON 安全化：numpy 标量 → python 标量（不出 np.float64）。"""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Mapping):
        return {k: _json_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_plain(v) for v in value]
    return value


def _resolve_discount_rate(
    discount_rate: Any, wacc_components: Any, *, tool: str
) -> tuple[str, float, Optional[dict]]:
    """折现率的两条来源（互斥）：直接给定，或由 WACC 成分分步算出。

    分步法（CAPM + 税后债务成本）：

    - `cost_of_equity = risk_free + equity_beta * equity_risk_premium`
    - `after_tax_cost_of_debt = cost_of_debt * (1 - tax_rate)`
    - `wacc = equity_weight * cost_of_equity + debt_weight * after_tax_cost_of_debt`

    三个中间值随返回一起给出（`wacc_breakdown`），使算术可审计。两条来源同时给出
    → 拒绝并点名两者（不猜哪个优先）；都未给出 → 点名两者。
    """
    if discount_rate is not None and wacc_components is not None:
        raise ComputeError(
            f"{tool}: pass either discount_rate or wacc_components, not both "
            f"(discount_rate={discount_rate!r}, wacc_components=<{len(wacc_components)} keys>) "
            f"— two sources for one rate is ambiguous"
        )
    if discount_rate is not None:
        rate = _number(discount_rate, name="discount_rate", tool=tool)
        if rate <= 0.0:
            raise ComputeError(
                f"{tool}: discount_rate must be > 0 (a non-positive discount rate is "
                f"refused, not used), got {discount_rate!r}"
            )
        return "given", rate, None
    if wacc_components is None:
        raise ComputeError(
            f"{tool}: discount_rate is required unless wacc_components is supplied "
            f"(wacc_components keys: {', '.join(WACC_FIELDS)})"
        )
    if not isinstance(wacc_components, Mapping):
        raise ComputeError(
            f"{tool}: wacc_components must be a mapping of "
            f"{', '.join(WACC_FIELDS)}, got {type(wacc_components).__name__}"
        )
    missing = [k for k in WACC_FIELDS if wacc_components.get(k) is None]
    if missing:
        raise ComputeError(
            f"{tool}: wacc_components is missing required key(s): {', '.join(missing)}"
        )
    c = {k: check_finite(wacc_components[k], name=f"{tool}: wacc_components[{k!r}]")
         for k in WACC_FIELDS}
    if c["debt_weight"] < 0.0 or c["equity_weight"] < 0.0:
        raise ComputeError(
            f"{tool}: wacc_components['debt_weight'] and ['equity_weight'] must be "
            f"non-negative, got debt_weight={c['debt_weight']!r}, "
            f"equity_weight={c['equity_weight']!r}"
        )
    weight_sum = c["debt_weight"] + c["equity_weight"]
    if abs(weight_sum - 1.0) > 1e-6:
        raise ComputeError(
            f"{tool}: wacc_components debt_weight ({_n(c['debt_weight'])}) + "
            f"equity_weight ({_n(c['equity_weight'])}) must sum to 1, got {_n(weight_sum)}"
        )
    if c["equity_beta"] < 0.0:
        raise ComputeError(
            f"{tool}: wacc_components['equity_beta'] must be non-negative, "
            f"got {c['equity_beta']!r}"
        )
    cost_of_equity = c["risk_free"] + c["equity_beta"] * c["equity_risk_premium"]
    after_tax_cost_of_debt = c["cost_of_debt"] * (1.0 - c["tax_rate"])
    equity_weighted = c["equity_weight"] * cost_of_equity
    debt_weighted = c["debt_weight"] * after_tax_cost_of_debt
    wacc = equity_weighted + debt_weighted
    if not math.isfinite(wacc) or wacc <= 0.0:
        raise ComputeError(
            f"{tool}: wacc_components produced a non-positive WACC ({wacc!r}); "
            f"check cost_of_debt/tax_rate/equity_risk_premium"
        )
    breakdown = {
        "risk_free": c["risk_free"],
        "equity_beta": c["equity_beta"],
        "equity_risk_premium": c["equity_risk_premium"],
        "cost_of_equity": cost_of_equity,
        "cost_of_debt": c["cost_of_debt"],
        "tax_rate": c["tax_rate"],
        "after_tax_cost_of_debt": after_tax_cost_of_debt,
        "equity_weight": c["equity_weight"],
        "debt_weight": c["debt_weight"],
        "equity_weighted": equity_weighted,
        "debt_weighted": debt_weighted,
        "wacc": wacc,
    }
    return "components", wacc, breakdown


def _pv_factors(rate: float, n: int) -> list[float]:
    """逐期折现因子 `1/(1+r)^t`（t=1..n），顺序相乘（不用幂运算的库实现）。"""
    out: list[float] = []
    df = 1.0
    for _ in range(n):
        df = df * (1.0 + rate)
        out.append(1.0 / df)
    return out


# ── DCF ──────────────────────────────────────────────────────────


def estimate_dcf(
    free_cash_flows: Any,
    *,
    as_of: str,
    discount_rate: Any = None,
    terminal_growth: Any = None,
    exit_multiple: Any = None,
    exit_metric: Any = None,
    shares_outstanding: Any = None,
    net_debt: Any = 0.0,
    wacc_components: Optional[Mapping[str, float]] = None,
    terminal_method: Any = TERMINAL_PERPETUITY,
) -> dict:
    """DCF 估值：显式期自由现金流折现 + 两种终值法之一（永续增长 / 退出倍数）。

    Input: free_cash_flows (非空数值序列, 单位=调用方货币, 逐期 FCF; 允许负值),
      as_of (ISO 日期, 必填, 第一个被校验的参数),
      discount_rate (小数, WACC; 与 wacc_components 二选一),
      terminal_growth (小数, 永续法用; 必须 < discount_rate),
      exit_multiple (倍数) + exit_metric (与 free_cash_flows 同货币, 退出倍数法用),
      shares_outstanding (>0, 股数), net_debt (默认 0.0, 同货币; 负值=净现金),
      wacc_components (可选 mapping: risk_free / equity_beta / equity_risk_premium /
      cost_of_debt / tax_rate / debt_weight / equity_weight → 分步算 WACC),
      terminal_method (默认 "perpetuity"; 另一取值 "exit_multiple")。
    Returns: enterprise_value (货币), equity_value = EV - net_debt (货币), value_per_share
      (货币/股), terminal_method, terminal_value (未折现终值), pv_terminal_value,
      terminal_value_share (= 终值现值/EV, 小数), pv_explicit_fcf (显式期现值),
      discount_rate (实际使用值), wacc_basis ("given"|"components"),
      wacc_breakdown (分步中间值; given 时为 null), assumptions (每个输入的回显 +
      terminal_inputs_used 点名本法真正用到的键), params, warmup (空字典, 本族无 warmup);
      另加统一外壳 as_of / input_start=null / input_end=null / input_length=null /
      evidence。**拒绝语义**: terminal_growth >= discount_rate、shares_outstanding <= 0、
      free_cash_flows 为空或有非有限值、EV <= 0 → ComputeError 点名该字段或两个数。
    """
    tool = "estimate_dcf"
    a = check_as_of(as_of)
    method = _validate_method(terminal_method, tool=tool)

    wacc_basis, rate, breakdown = _resolve_discount_rate(
        discount_rate, wacc_components, tool=tool
    )

    shares = _number(shares_outstanding, name="shares_outstanding", tool=tool)
    if shares <= 0.0:
        raise ComputeError(
            f"{tool}: shares_outstanding must be positive to compute value_per_share, "
            f"got {shares_outstanding!r}"
        )
    debt = 0.0 if net_debt is None else check_finite(
        net_debt, name=f"{tool}: net_debt"
    )

    fcf = _num_array(free_cash_flows, name="free_cash_flows", tool=tool)
    n = len(fcf)

    factors = _pv_factors(rate, n)
    pv_terms = np.empty(n, dtype=np.float64)
    for i in range(n):
        pv_terms[i] = float(fcf[i]) * factors[i]
    pv_explicit = deterministic_reduction(pv_terms)
    df_terminal = 1.0 / factors[-1]  # = (1+r)^n，与逐期因子同一条乘法链

    g = _optional_number(terminal_growth, name="terminal_growth", tool=tool)
    em = _optional_number(exit_multiple, name="exit_multiple", tool=tool)
    metric = _optional_number(exit_metric, name="exit_metric", tool=tool)

    if method == TERMINAL_PERPETUITY:
        if g is None:
            raise ComputeError(
                f"{tool}: terminal_growth is required for terminal_method='perpetuity' "
                f"(or pass terminal_method='exit_multiple' with exit_multiple + exit_metric)"
            )
        if g <= -1.0:
            raise ComputeError(
                f"{tool}: terminal_growth must be > -1 (a growth rate cannot shrink the "
                f"cash flow to zero or below), got {_n(g)}"
            )
        if not g < rate:
            raise ComputeError(
                f"{tool}: terminal_growth ({_n(g)}) must be < discount_rate ({_n(rate)}) "
                f"for the perpetuity terminal value; a growth rate at or above the "
                f"discount rate makes the terminal value diverge"
            )
        last = float(fcf[n - 1])
        terminal_value = last * (1.0 + g) / (rate - g)
        used = {"terminal_growth": g}
    else:
        if em is None or metric is None:
            raise ComputeError(
                f"{tool}: exit_multiple and exit_metric are both required for "
                f"terminal_method='exit_multiple' "
                f"(got exit_multiple={exit_multiple!r}, exit_metric={exit_metric!r})"
            )
        if em <= 0.0:
            raise ComputeError(
                f"{tool}: exit_multiple must be positive, got {_n(em)}"
            )
        if metric <= 0.0:
            raise ComputeError(
                f"{tool}: exit_metric must be positive, got {_n(metric)}"
            )
        terminal_value = em * metric
        used = {"exit_multiple": em, "exit_metric": metric}

    pv_terminal = terminal_value / df_terminal
    enterprise_value = pv_explicit + pv_terminal
    if enterprise_value <= 0.0:
        raise ComputeError(
            f"{tool}: enterprise_value is non-positive ({_n(enterprise_value)}); a "
            f"non-positive EV cannot yield a per-share value — check free_cash_flows / "
            f"{'terminal_growth' if method == TERMINAL_PERPETUITY else 'exit_multiple'}"
        )
    equity_value = enterprise_value - debt
    value_per_share = equity_value / shares
    terminal_share = pv_terminal / enterprise_value

    assumptions = {
        "free_cash_flows": to_none_list(fcf),
        "discount_rate": rate,
        "wacc_basis": wacc_basis,
        "terminal_method": method,
        "terminal_growth": g,
        "exit_multiple": em,
        "exit_metric": metric,
        "shares_outstanding": shares,
        "net_debt": debt,
        "terminal_inputs_used": used,
    }
    params = {
        "terminal_method": method,
        "forecast_years": n,
        "net_debt": debt,
        "wacc_basis": wacc_basis,
    }

    b = EvidenceBuilder(tool, a)
    src = {"terminal_method": method, "discount_rate": rate}
    b.add("enterprise value", enterprise_value, unit="currency", params=src)
    b.add("equity value", equity_value, unit="currency", params=src)
    b.add("value per share", value_per_share, unit="currency/share", params=src)
    b.add("terminal value", terminal_value, unit="currency", params=src)
    b.add("pv terminal value", pv_terminal, unit="currency", params=src)
    b.add("terminal value share", terminal_share, unit="ratio", params=src)
    b.add("pv explicit fcf", pv_explicit, unit="currency", params=src)
    b.add("discount rate used", rate, unit="ratio", params=src)
    if breakdown is not None:
        b.add("cost of equity", breakdown["cost_of_equity"], unit="ratio", params=src)
        b.add("after-tax cost of debt", breakdown["after_tax_cost_of_debt"], unit="ratio",
              params=src)

    return _envelope(tool, a, b, {
        "enterprise_value": enterprise_value,
        "equity_value": equity_value,
        "value_per_share": value_per_share,
        "terminal_method": method,
        "terminal_value": terminal_value,
        "pv_terminal_value": pv_terminal,
        "terminal_value_share": terminal_share,
        "pv_explicit_fcf": pv_explicit,
        "discount_rate": rate,
        "wacc_basis": wacc_basis,
        "wacc_breakdown": breakdown,
        "assumptions": assumptions,
        "params": params,
        "warmup": {},
    })


# ── DDM ──────────────────────────────────────────────────────────


def estimate_ddm(
    dividend_per_share: Any = None,
    *,
    as_of: str,
    growth_rate: Any = None,
    required_return: Any = None,
    dividend_schedule: Optional[Mapping[str, float]] = None,
) -> dict:
    """股息折现模型（DDM）：单期 Gordon 永续增长，或显式期股息表 + Gordon 终值。

    Input: dividend_per_share (>0, 货币/股, 当期已付股息; 与 dividend_schedule 互斥),
      as_of (ISO 日期, 必填, 第一个被校验的参数),
      growth_rate (小数, 股息永续增速; 必须 < required_return),
      required_return (小数, 股权要求回报率; 必须 > 0),
      dividend_schedule (可选 mapping: 期别标签 → 该期每股股息, 按插入顺序为第 1..T 期;
      给定时近端股息全部来自该表, 终值仍用 growth_rate).
    Returns: value_per_share (货币/股; Gordon = DPS*(1+g)/(r-g), 显式表 =
      Σ DPS_t/(1+r)^t + [DPS_T*(1+g)/(r-g)]/(1+r)^T),
      dividend_per_share (单期法的当期股息; 显式表法为 null), growth_rate,
      required_return, dividend_schedule (回显; 未用时为 null), assumptions, params,
      warmup (空字典); 另加统一外壳 as_of / input_start=null / input_end=null /
      input_length=null / evidence。**拒绝语义**: growth_rate >= required_return
      （点名两个数）、required_return <= 0、股息非正与非有限值 → ComputeError。
    """
    tool = "estimate_ddm"
    a = check_as_of(as_of)

    if dividend_per_share is not None and dividend_schedule is not None:
        raise ComputeError(
            f"{tool}: pass either dividend_per_share or dividend_schedule, not both "
            f"— two sources for the dividend stream is ambiguous"
        )
    if dividend_per_share is None and dividend_schedule is None:
        raise ComputeError(
            f"{tool}: dividend_per_share is required unless dividend_schedule is "
            f"supplied (a mapping of period label -> dividend per share)"
        )

    g = _number(growth_rate, name="growth_rate", tool=tool)
    r = _number(required_return, name="required_return", tool=tool)
    if r <= 0.0:
        raise ComputeError(
            f"{tool}: required_return must be > 0 (a non-positive required return is "
            f"refused, not used), got {required_return!r}"
        )
    if g <= -1.0:
        raise ComputeError(
            f"{tool}: growth_rate must be > -1, got {_n(g)}"
        )
    if not g < r:
        raise ComputeError(
            f"{tool}: growth_rate ({_n(g)}) must be < required_return ({_n(r)}) "
            f"— at or above it the Gordon series diverges (no finite value exists)"
        )

    b = EvidenceBuilder(tool, a)
    src = {"growth_rate": g, "required_return": r}

    if dividend_schedule is None:
        dps = _number(dividend_per_share, name="dividend_per_share", tool=tool)
        if dps <= 0.0:
            raise ComputeError(
                f"{tool}: dividend_per_share must be positive, got {_n(dps)}"
            )
        value = dps * (1.0 + g) / (r - g)
        payload_extra: dict = {
            "dividend_per_share": dps,
            "dividend_schedule": None,
            "pv_explicit_dividends": None,
            "pv_terminal_value": None,
        }
        b.add("dividend per share", dps, unit="currency/share", params=src)
        b.add("value per share", value, unit="currency/share", params=src)
    else:
        if not isinstance(dividend_schedule, Mapping):
            raise ComputeError(
                f"{tool}: dividend_schedule must be a mapping of period label -> "
                f"dividend per share, got {type(dividend_schedule).__name__}"
            )
        labels = list(dividend_schedule.keys())
        if not labels:
            raise ComputeError(f"{tool}: dividend_schedule must not be empty")
        terms = np.empty(len(labels), dtype=np.float64)
        schedule_echo: dict[str, float] = {}
        factors = _pv_factors(r, len(labels))
        for i, label in enumerate(labels):
            v = dividend_schedule[label]
            if v is None or isinstance(v, bool) or isinstance(v, (str, bytes)):
                raise ComputeError(
                    f"{tool}: dividend_schedule[{label!r}] must be a number, got {v!r}"
                )
            d = check_finite(v, name=f"{tool}: dividend_schedule[{label!r}]")
            if d <= 0.0:
                raise ComputeError(
                    f"{tool}: dividend_schedule[{label!r}] must be positive, got {_n(d)}"
                )
            terms[i] = d * factors[i]
            schedule_echo[str(label)] = d
        pv_explicit = deterministic_reduction(terms)
        last = float(terms[-1] / factors[-1])
        df_terminal = 1.0 / factors[-1]
        pv_terminal = (last * (1.0 + g) / (r - g)) / df_terminal
        value = pv_explicit + pv_terminal
        payload_extra = {
            "dividend_per_share": None,
            "dividend_schedule": schedule_echo,
            "pv_explicit_dividends": pv_explicit,
            "pv_terminal_value": pv_terminal,
        }
        b.add("dividend periods", len(labels), unit="count", params=src)
        b.add("pv explicit dividends", pv_explicit, unit="currency/share", params=src)
        b.add("pv terminal value", pv_terminal, unit="currency/share", params=src)
        b.add("value per share", value, unit="currency/share", params=src)

    assumptions = {
        "dividend_per_share": payload_extra["dividend_per_share"],
        "growth_rate": g,
        "required_return": r,
        "dividend_schedule": payload_extra["dividend_schedule"],
    }
    payload = {
        "value_per_share": value,
        "growth_rate": g,
        "required_return": r,
        "assumptions": assumptions,
        "params": {"stage": "multi_period" if dividend_schedule is not None else "gordon"},
        "warmup": {},
    }
    payload.update(payload_extra)
    return _envelope(tool, a, b, payload)


# ── 可比公司倍数 ─────────────────────────────────────────────────


def _multiples_input(multiples: Any, *, tool: str) -> tuple[Optional[list[str]], np.ndarray]:
    """倍数表输入：`{peer: multiple}` 映射（peer 名回显）或等长数值序列。"""
    if isinstance(multiples, Mapping):
        if not multiples:
            raise ComputeError(f"{tool}: multiples mapping must not be empty")
        names = [str(k) for k in multiples.keys()]
        values = [multiples[k] for k in multiples.keys()]
        return names, _num_array(values, name="multiples", tool=tool)
    return None, _num_array(multiples, name="multiples", tool=tool)


def compute_comparable_multiples(
    multiples: Union[Mapping[str, float], Sequence[float]],
    metric_value: Any = None,
    *,
    as_of: str,
    winsorize: Any = 0.05,
) -> dict:
    """可比公司倍数：去极值（winsorize 可配）后的中位/均值/四分位与隐含估值。

    Input: multiples (非空数值序列, 单位=倍数; 或映射 {peer 名 → 倍数}, 名字随返回回显),
      metric_value (标的公司指标值, 单位=调用方货币; 如净利润/EBITDA; 可为负),
      as_of (ISO 日期, 必填, 第一个被校验的参数),
      winsorize (默认 0.05, 取值范围 [0, 0.5); 双侧分位裁剪比例, 0 = 不裁剪).
    Returns: original (原表, 输入顺序), winsorized (裁剪后, 同一顺序), clipped_count
      (被改写的个数), clipped_peers (给了 peer 名时点名被裁剪的 peer; 否则 null),
      median / mean / p25 / p75 (均为**裁剪后**表的统计量; mean 用固定顺序 Kahan 求和),
      median_original / mean_original / p25_original / p75_original (原表对照),
      lower_bound / upper_bound (实际裁剪边界), implied_valuation
      (= median * metric_value, 与 metric_value 同货币), peer_names (给了映射时非 null),
      assumptions, params, warmup (空字典); 另加统一外壳 as_of / input_start=null /
      input_end=null / input_length=null / evidence。**分位定义**: 排序表 + 分数下标的
      线性插值（numpy 默认 method="linear"），见 `_quantile`。**拒绝语义**:
      winsorize 越界、公式 inputs 非有限、倍数表为空 → ComputeError 点名该字段；
      裁剪只改写极端值到边界，不删除任何一项。
    """
    tool = "compute_comparable_multiples"
    a = check_as_of(as_of)

    w = check_finite(winsorize, name=f"{tool}: winsorize")
    if w < 0.0 or w >= 0.5:
        raise ComputeError(
            f"{tool}: winsorize must be in [0, 0.5), got {_n(w)} "
            f"(0 disables clipping; 0.5 would clip every point to the median)"
        )
    metric = _number(metric_value, name="metric_value", tool=tool)

    names, values = _multiples_input(multiples, tool=tool)
    original = [float(v) for v in values]
    ordered = sorted(original)

    if w == 0.0:
        lower, upper = ordered[0], ordered[-1]
    else:
        lower = _quantile(ordered, w)
        upper = _quantile(ordered, 1.0 - w)
        if lower > upper:  # 仅当极端退化（全部相等）时可能；保持边界有序
            lower, upper = upper, lower
    clipped = [min(max(v, lower), upper) for v in original]
    clipped_count = sum(1 for i in range(len(original)) if clipped[i] != original[i])
    clipped_peers = (
        [names[i] for i in range(len(names)) if clipped[i] != original[i]]
        if names is not None else None
    )

    wins_arr = np.asarray(clipped, dtype=np.float64)
    orig_arr = np.asarray(original, dtype=np.float64)
    sorted_w = sorted(clipped)
    sorted_o = sorted(original)
    stats = {
        "median": _quantile(sorted_w, 0.5),
        "mean": _kahan_mean(wins_arr),
        "p25": _quantile(sorted_w, 0.25),
        "p75": _quantile(sorted_w, 0.75),
    }
    stats_original = {
        "median_original": _quantile(sorted_o, 0.5),
        "mean_original": _kahan_mean(orig_arr),
        "p25_original": _quantile(sorted_o, 0.25),
        "p75_original": _quantile(sorted_o, 0.75),
    }
    implied = stats["median"] * metric

    params = {"winsorize": w, "peer_count": len(original)}
    assumptions = {
        "multiples": original,
        "metric_value": metric,
        "winsorize": w,
    }
    b = EvidenceBuilder(tool, a)
    src = {"winsorize": w}
    b.add("multiples count", len(original), unit="count", params=src)
    b.add("lower bound", lower, unit="multiple", params=src)
    b.add("upper bound", upper, unit="multiple", params=src)
    b.add("clipped count", clipped_count, unit="count", params=src)
    b.add("median", stats["median"], unit="multiple", params=src)
    b.add("median original", stats_original["median_original"], unit="multiple", params=src)
    b.add("p25", stats["p25"], unit="multiple", params=src)
    b.add("p75", stats["p75"], unit="multiple", params=src)
    b.add("mean", stats["mean"], unit="multiple", params=src)
    b.add("implied valuation", implied, unit="currency", params=src)

    payload = {
        "original": original,
        "winsorized": clipped,
        "clipped_count": clipped_count,
        "clipped_peers": clipped_peers,
        "peer_names": names,
        "lower_bound": lower,
        "upper_bound": upper,
        "implied_valuation": implied,
        "assumptions": assumptions,
        "params": params,
        "warmup": {},
    }
    payload.update(stats)
    payload.update(stats_original)
    return _envelope(tool, a, b, payload)


# ── Monte Carlo ──────────────────────────────────────────────────


def estimate_monte_carlo(
    base_value: Any = None,
    *,
    as_of: str,
    free_cash_flows: Any = None,
    growth_mean: Any = None,
    growth_std: Any = None,
    discount_rate: Any = None,
    n_simulations: Any = 10000,
    seed: Any = 42,
    horizon: Any = 5,
    percentiles: Optional[Mapping[str, float]] = None,
) -> dict:
    """Monte Carlo 估值：按正态增长路径抽样的终值分布（P5/P25/P50/P75/P95 百分位带）。

    Input: base_value (>0, 货币或货币/股, 与 free_cash_flows 二选一, 现有的内在价值),
      free_cash_flows (可选非空数值序列: 未给 base_value 时用它在 discount_rate 下的
      现值作 base_value; 两者都给 → 拒绝并点名两者),
      as_of (ISO 日期, 必填, 第一个被校验的参数),
      growth_mean / growth_std (小数: 每期增速的均值与标准差; std >= 0),
      discount_rate (小数, >0: 每期折现率), n_simulations (默认 10000, 正整数, 上限 1e7),
      seed (默认 42: 随机种子, 同一 seed 逐位可复现), horizon (默认 5: 期数, 正整数),
      percentiles (可选 mapping: 标签 → 分位水平(0,1); 默认 p5/p25/p50/p75/p95).
    Returns: 每个百分位标签一个键 (默认 p5/p25/p50/p75/p95, 同 base_value 单位),
      percentile_values (同一批数的映射), percentile_levels (标签 → 水平回显),
      mean (固定顺序 Kahan), std (总体标准差 ddof=0), n_simulations, seed, horizon,
      base_value 与 base_value_source ("given"|"free_cash_flows_pv"), assumptions,
      params, warmup (空字典); 另加统一外壳 as_of / input_start=null / input_end=null /
      input_length=null / evidence。**模型**: 每期增速独立 N(growth_mean, growth_std),
      value_i = base_value * Π(1+g_it) / (1+discount_rate)^horizon（numpy 向量化,
      按行 cumprod 顺序扫描, 无逐次模拟的 Python 循环）。**分位定义**: 排序 + 线性插值
      (np.percentile 的 method="linear"), 一次调用算出整条带。
    """
    tool = "estimate_monte_carlo"
    a = check_as_of(as_of)

    n_sims = check_positive_window(n_simulations, name=f"{tool}: n_simulations")
    if n_sims > _MAX_SIMULATIONS:
        raise ComputeError(
            f"{tool}: n_simulations must be <= {_MAX_SIMULATIONS}, got {n_sims} "
            f"(a larger run is an OOM, not a better estimate)"
        )
    periods = check_positive_window(horizon, name=f"{tool}: horizon")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ComputeError(f"{tool}: seed must be an integer, got {seed!r}")
    if seed < 0:
        raise ComputeError(
            f"{tool}: seed must be a non-negative integer (np.random.default_rng), "
            f"got {seed!r}"
        )

    mean_g = _number(growth_mean, name="growth_mean", tool=tool)
    std_g = _number(growth_std, name="growth_std", tool=tool)
    if std_g < 0.0:
        raise ComputeError(f"{tool}: growth_std must be >= 0, got {_n(std_g)}")
    rate = _number(discount_rate, name="discount_rate", tool=tool)
    if rate <= 0.0:
        raise ComputeError(
            f"{tool}: discount_rate must be > 0 (a non-positive discount rate is "
            f"refused, not used), got {discount_rate!r}"
        )

    if base_value is not None and free_cash_flows is not None:
        raise ComputeError(
            f"{tool}: pass either base_value or free_cash_flows, not both "
            f"— two sources for the base value is ambiguous"
        )
    if base_value is None and free_cash_flows is None:
        raise ComputeError(
            f"{tool}: base_value is required unless free_cash_flows is supplied "
            f"(the present value of those flows is then used as the base)"
        )
    if base_value is not None:
        base = _number(base_value, name="base_value", tool=tool)
        base_source = "given"
        flows: Optional[list] = None
    else:
        fcf = _num_array(free_cash_flows, name="free_cash_flows", tool=tool)
        factors = _pv_factors(rate, len(fcf))
        terms = np.empty(len(fcf), dtype=np.float64)
        for i in range(len(fcf)):
            terms[i] = float(fcf[i]) * factors[i]
        base = deterministic_reduction(terms)
        base_source = "free_cash_flows_pv"
        flows = to_none_list(fcf)
    if base <= 0.0:
        raise ComputeError(
            f"{tool}: base_value must be positive to describe a value distribution, "
            f"got {_n(base)}" + ("" if base_source == "given" else " (PV of free_cash_flows)")
        )

    labels, levels = _percentile_spec(percentiles, tool=tool)

    rng = np.random.default_rng(seed)  # PCG64：默认位流跨版本稳定
    draws = rng.standard_normal((n_sims, periods))
    factors_path = 1.0 + (mean_g + std_g * draws)
    compounded = np.cumprod(factors_path, axis=1)[:, -1]  # 每行顺序扫描（非成对归约）
    values = base * compounded / ((1.0 + rate) ** periods)

    band = np.percentile(values, [level * 100.0 for level in levels], method="linear")
    percentile_values = {labels[i]: float(band[i]) for i in range(len(labels))}
    mean_value = _kahan_mean(values)
    variance_terms = (values - mean_value) ** 2
    std_value = math.sqrt(deterministic_reduction(variance_terms) / n_sims)

    assumptions = {
        "base_value": base,
        "base_value_source": base_source,
        "free_cash_flows": flows,
        "growth_mean": mean_g,
        "growth_std": std_g,
        "discount_rate": rate,
        "n_simulations": n_sims,
        "seed": seed,
        "horizon": periods,
    }
    params = {
        "n_simulations": n_sims,
        "seed": seed,
        "horizon": periods,
        "percentile_levels": {label: level for label, level in zip(labels, levels)},
    }

    b = EvidenceBuilder(tool, a)
    src = {"seed": seed, "n_simulations": n_sims, "horizon": periods}
    b.add("base value", base, unit="currency", params=src)
    for label in labels:
        b.add(label, percentile_values[label], unit="currency", params=src)
    b.add("mean", mean_value, unit="currency", params=src)
    b.add("std", std_value, unit="currency", params=src)

    payload = {
        "percentile_values": percentile_values,
        "percentile_levels": {label: level for label, level in zip(labels, levels)},
        "mean": mean_value,
        "std": std_value,
        "n_simulations": n_sims,
        "seed": seed,
        "horizon": periods,
        "base_value": base,
        "base_value_source": base_source,
        "assumptions": assumptions,
        "params": params,
        "warmup": {},
    }
    payload.update(percentile_values)
    return _envelope(tool, a, b, payload)


def _percentile_spec(
    percentiles: Any, *, tool: str
) -> tuple[list[str], list[float]]:
    """百分位规格：默认 p5/p25/p50/p75/p95；自定义时逐项校验标签与水平。"""
    if percentiles is None:
        return [label for label, _ in DEFAULT_PERCENTILES], [
            level for _, level in DEFAULT_PERCENTILES
        ]
    if not isinstance(percentiles, Mapping):
        raise ComputeError(
            f"{tool}: percentiles must be a mapping of label -> level in (0, 1), "
            f"got {type(percentiles).__name__}"
        )
    if not percentiles:
        raise ComputeError(f"{tool}: percentiles must not be empty")
    reserved = {
        "as_of", "input_start", "input_end", "input_length", "evidence", "params",
        "warmup", "assumptions", "mean", "std", "seed", "horizon", "n_simulations",
        "base_value", "base_value_source", "percentile_values", "percentile_levels",
    }
    labels: list[str] = []
    levels: list[float] = []
    for label, level in percentiles.items():
        if not isinstance(label, str) or not label:
            raise ComputeError(
                f"{tool}: percentiles keys must be non-empty strings, got {label!r}"
            )
        if label in reserved:
            raise ComputeError(
                f"{tool}: percentiles label {label!r} collides with a reserved return "
                f"key; choose another label"
            )
        lv = check_finite(level, name=f"{tool}: percentiles[{label!r}]")
        if not 0.0 < lv < 1.0:
            raise ComputeError(
                f"{tool}: percentiles[{label!r}] must be in (0, 1), got {_n(lv)}"
            )
        labels.append(label)
        levels.append(lv)
    return labels, levels


# ── 双变量敏感性网格 ─────────────────────────────────────────────


def compute_sensitivity_grid(
    base_inputs: Mapping[str, Any],
    variable_x: Mapping[str, Any],
    variable_y: Mapping[str, Any],
    *,
    as_of: str,
    terminal_method: Any = TERMINAL_PERPETUITY,
) -> dict:
    """双变量敏感性网格：对每个 (x, y) 组合重算 DCF，给出每股价值网格。

    Input: base_inputs (mapping: 一整套 DCF 假设, 键同 estimate_dcf 的 kwargs, 不含
      as_of 与 terminal_method——口径与方法由本次调用显式给出),
      variable_x / variable_y (各是一个 mapping: {"name": <DCF 标量假设名>, "values":
      [非空有限数值列表]}; 合法名 = discount_rate / terminal_growth / exit_multiple /
      exit_metric / shares_outstanding / net_debt; 两者 name 必须不同),
      as_of (ISO 日期, 必填, 第一个被校验的参数),
      terminal_method (默认 "perpetuity"; 传给每个格点).
    Returns: grid (list, 行 = {"x_value", "y_value", "value"}; 先 x 后 y、按给定值序,
      value = 该格点的 value_per_share, 货币/股), x_name / y_name, x_values / y_values,
      value_metric ("value_per_share"), base_case_value (用 base_inputs 原样算出的每股
      价值), assumptions (回显本次生效的全部 DCF 假设, 与 estimate_dcf 同名键,
      便于消费方跨工具统一读取), params, warmup (空字典); 另加统一外壳 as_of /
      input_start=null / input_end=null / input_length=null / evidence。**拒绝语义**:
      variable name 不在合法集合 / 两个 name 相同 / values 为空或有非有限值 /
      base_inputs 里带 as_of 或 terminal_method / 任一格点参数非法（如
      terminal_growth >= discount_rate）→ ComputeError；格点错误会点名该格点的
      x/y 取值（不在网格里留洞）。
    """
    tool = "compute_sensitivity_grid"
    a = check_as_of(as_of)
    method = _validate_method(terminal_method, tool=tool)

    if not isinstance(base_inputs, Mapping):
        raise ComputeError(
            f"{tool}: base_inputs must be a mapping of DCF assumptions, got "
            f"{type(base_inputs).__name__}"
        )
    if not base_inputs:
        raise ComputeError(f"{tool}: base_inputs must not be empty")
    for forbidden in ("as_of", "terminal_method"):
        if forbidden in base_inputs:
            raise ComputeError(
                f"{tool}: base_inputs must not carry {forbidden!r} — it is passed "
                f"explicitly to this tool (one source for the {forbidden} of a grid)"
            )

    specs: list[tuple[str, str, list[float]]] = []
    for axis, variable in (("variable_x", variable_x), ("variable_y", variable_y)):
        if not isinstance(variable, Mapping):
            raise ComputeError(
                f"{tool}: {axis} must be a mapping with 'name' and 'values', got "
                f"{type(variable).__name__}"
            )
        name = variable.get("name")
        if name not in GRID_VARIABLES:
            raise ComputeError(
                f"{tool}: {axis}['name'] must be one of {', '.join(GRID_VARIABLES)}, "
                f"got {name!r}"
            )
        if variable.get("values") is None:
            raise ComputeError(
                f"{tool}: {axis}['values'] is required (a non-empty list of numbers)"
            )
        arr = _num_array(variable["values"], name=f"{axis}['values']", tool=tool)
        specs.append((axis, str(name), [float(v) for v in arr]))
    if specs[0][1] == specs[1][1]:
        raise ComputeError(
            f"{tool}: variable_x['name'] and variable_y['name'] must differ, both are "
            f"{specs[0][1]!r} (a grid needs two axes)"
        )
    x_name, x_values = specs[0][1], specs[0][2]
    y_name, y_values = specs[1][1], specs[1][2]

    def _cell(**overrides: Any) -> float:
        kw = dict(base_inputs)
        kw.update(overrides)
        try:
            return float(estimate_dcf(terminal_method=method, as_of=a, **kw)["value_per_share"])
        except ComputeError as e:
            point = ", ".join(f"{k}={_n(v)}" for k, v in overrides.items())
            raise ComputeError(f"{tool}: grid cell ({point}) refused: {e}") from None

    grid: list[dict] = []
    for xv in x_values:
        for yv in y_values:
            grid.append({
                "x_value": xv,
                "y_value": yv,
                "value": _cell(**{x_name: xv, y_name: yv}),
            })
    base_case = _cell()
    values = np.asarray([row["value"] for row in grid], dtype=np.float64)

    params = {
        "terminal_method": method,
        "x_name": x_name,
        "y_name": y_name,
        "cells": len(grid),
    }
    b = EvidenceBuilder(tool, a)
    src = {"terminal_method": method, "x_name": x_name, "y_name": y_name}
    b.add("grid cells", len(grid), unit="count", params=src)
    b.add("base case value", base_case, unit="currency/share", params=src)
    b.add("value min", float(np.min(values)), unit="currency/share", params=src)
    b.add("value max", float(np.max(values)), unit="currency/share", params=src)

    return _envelope(tool, a, b, {
        "grid": grid,
        "x_name": x_name,
        "y_name": y_name,
        "x_values": x_values,
        "y_values": y_values,
        "value_metric": "value_per_share",
        "base_case_value": base_case,
        "assumptions": _json_plain(dict(base_inputs)),
        "params": params,
        "warmup": {},
    })


# ── 估值合成（扣值不扣结论） ─────────────────────────────────────


def build_valuation_synthesis(
    methods: Sequence[Mapping[str, Any]],
    *,
    as_of: str,
    disagreement_threshold: Any = 0.20,
    weights: Optional[Mapping[str, float]] = None,
) -> dict:
    """多方法估值合成：分歧超阈值 → 扣值（price_target=null）不扣结论，绝不静默取均值。

    Input: methods (非空序列, 每项 {"name": 非空字符串, "value": 有限数值, 单位按约定
      = 每股价格/调用方货币; 名字必须互不相同}),
      as_of (ISO 日期, 必填, 第一个被校验的参数),
      disagreement_threshold (默认 0.20, 小数, >= 0; 分歧比 = (max-min)/median 与之比),
      weights (可选 mapping: 方法名 → 权重, 必须覆盖全部方法名; 非负、和为正, 内部
      归一化后用于加权).
    Returns: price_target (加权点位; 分歧超阈时为 null), weights_used (实际权重;
      超阈时为 null), weighting_basis ("equal"|"custom"|"withheld"), disagreement
      (实测分歧比, 小数), disagreement_threshold (回显), method_values (每法原始结果),
      outliers (极值方法对中位数的偏离, 有符号小数; 无极值时为空表), note (判定说明),
      disagreement_note (超阈时的分歧说明; 否则 null), median / min / max (方法值),
      method_count, params, warmup (空字典); 另加统一外壳 as_of / input_start=null /
      input_end=null / input_length=null / evidence (每个方法值一条 + 判定本身)。
      **超阈时不返回任何均值键**: 只给分歧说明与极值清单。**拒绝语义**: methods 为空 /
      缺 name 或 value / 方法名重复 / 权重覆盖不全或为负 / 中位数 <= 0（分歧比无定义）
      → ComputeError 点名该字段。返回恒可 json.dumps(..., allow_nan=False)。
    """
    tool = "build_valuation_synthesis"
    a = check_as_of(as_of)

    threshold = check_finite(disagreement_threshold, name=f"{tool}: disagreement_threshold")
    if threshold < 0.0:
        raise ComputeError(
            f"{tool}: disagreement_threshold must be >= 0 (a ratio), got {_n(threshold)}"
        )
    if methods is None or isinstance(methods, (str, bytes)) or isinstance(methods, Mapping):
        raise ComputeError(
            f"{tool}: methods must be a non-empty sequence of "
            f"{{'name': str, 'value': number}}, got {methods!r}"
        )
    try:
        method_count = len(methods)
    except TypeError:
        raise ComputeError(
            f"{tool}: methods must be a non-empty sequence of "
            f"{{'name': str, 'value': number}}, got {methods!r}"
        ) from None
    if method_count == 0:
        raise ComputeError(f"{tool}: methods must not be empty")

    names: list[str] = []
    values: list[float] = []
    for i in range(method_count):
        m = methods[i]
        if not isinstance(m, Mapping):
            raise ComputeError(
                f"{tool}: methods[{i}] must be a mapping with 'name' and 'value', "
                f"got {type(m).__name__}"
            )
        name = m.get("name")
        if not isinstance(name, str) or not name:
            raise ComputeError(
                f"{tool}: methods[{i}]['name'] must be a non-empty string, got {name!r}"
            )
        if m.get("value") is None:
            raise ComputeError(f"{tool}: methods[{i}]['value'] is required (got None)")
        value = check_finite(m["value"], name=f"{tool}: methods[{i}]['value']")
        if name in names:
            raise ComputeError(f"{tool}: duplicate method name {name!r} in methods")
        names.append(name)
        values.append(value)

    if weights is not None:
        if not isinstance(weights, Mapping):
            raise ComputeError(
                f"{tool}: weights must be a mapping of method name -> weight, got "
                f"{type(weights).__name__}"
            )
        unknown = [k for k in weights if k not in names]
        if unknown:
            raise ComputeError(
                f"{tool}: weights names no such method: {', '.join(sorted(map(str, unknown)))} "
                f"(method names are {', '.join(names)})"
            )
        missing_w = [n for n in names if weights.get(n) is None]
        if missing_w:
            raise ComputeError(
                f"{tool}: weights is missing a weight for method(s): "
                f"{', '.join(missing_w)}"
            )
        raw = [check_finite(weights[n], name=f"{tool}: weights[{n!r}]") for n in names]
        if any(w < 0.0 for w in raw):
            raise ComputeError(
                f"{tool}: weights must be non-negative, got "
                f"{', '.join(f'{n}={_n(w)}' for n, w in zip(names, raw))}"
            )
        weight_total = deterministic_reduction(np.asarray(raw, dtype=np.float64))
        if weight_total <= 0.0:
            raise ComputeError(
                f"{tool}: weights must sum to a positive number, got {_n(weight_total)}"
            )
        effective_w = [w / weight_total for w in raw]
        basis = "custom"
    else:
        effective_w = [1.0 / method_count] * method_count
        weight_total = 1.0
        basis = "equal"

    ordered = sorted(values)
    median = _quantile(ordered, 0.5)
    if median <= 0.0:
        raise ComputeError(
            f"{tool}: the median of the method values is {_n(median)} — the disagreement "
            f"ratio (max-min)/median is undefined for a non-positive median "
            f"(values: {', '.join(_n(v) for v in values)})"
        )
    lo, hi = ordered[0], ordered[-1]
    lo_i = min(range(method_count), key=lambda i: values[i])
    hi_i = min(range(method_count), key=lambda i: -values[i])
    spread = hi - lo
    disagreement = spread / median
    disagreement_note: Optional[str] = None
    price_target: Optional[float] = None
    weights_used: Optional[dict] = None

    if disagreement > threshold:
        # 扣值不扣结论：保留判定所需的分歧事实，绝不给一个（加权）均值。
        weighting_basis = "withheld"
        spread_over_low = spread / abs(lo)
        deviation_of = {
            i: (values[i] - median) / median for i in range(method_count)
        }
        extremes = sorted(
            (i for i in range(method_count) if values[i] in (lo, hi)),
            key=lambda i: values[i],
        )
        rendered = ", ".join(
            f"'{names[i]}'={_n(values[i])} (deviation {_n(deviation_of[i])})"
            for i in extremes
        )
        note = (
            f"method values disagree beyond the threshold: min '{names[lo_i]}'="
            f"{_n(lo)} and max '{names[hi_i]}'={_n(hi)} span {_n(spread)} = "
            f"{_n(disagreement)} of the median {_n(median)} (={_n(spread_over_low * 100.0)}% "
            f"of the lower method) > disagreement_threshold {_n(threshold)}; "
            f"price_target withheld as null — 扣值不扣结论: the judgement is kept, the "
            f"point estimate is not, and no mean is returned. Outliers: {rendered}."
        )
        disagreement_note = note
    else:
        weighted_terms = np.asarray(
            [effective_w[i] * values[i] for i in range(method_count)], dtype=np.float64
        )
        price_target = deterministic_reduction(weighted_terms)
        weights_used = {names[i]: effective_w[i] for i in range(method_count)}
        weighting_basis = basis
        basis_text = (
            "equal weights (1/n per method)"
            if basis == "equal"
            else "caller-supplied weights, normalised to sum to 1 "
                 f"({', '.join(f'{n}={_n(w)}' for n, w in weights_used.items())}; "
                 f"input total {_n(weight_total)})"
        )
        note = (
            f"method values agree within the threshold: min '{names[lo_i]}'="
            f"{_n(lo)} and max '{names[hi_i]}'={_n(hi)} span {_n(spread)} = "
            f"{_n(disagreement)} of the median {_n(median)} <= disagreement_threshold "
            f"{_n(threshold)}; price_target {_n(price_target)} is the weighted mean of "
            f"{method_count} method value(s) with {basis_text}"
        )

    outliers: list[dict] = []
    if spread > 0.0:
        for name, value in sorted(zip(names, values), key=lambda t: t[1]):
            if value in (lo, hi):
                outliers.append({
                    "name": name,
                    "value": value,
                    "deviation": (value - median) / median,
                })

    method_values = [{"name": n, "value": v} for n, v in zip(names, values)]

    params = {
        "disagreement_threshold": threshold,
        "method_count": method_count,
        "weights_provided": weights is not None,
    }
    b = EvidenceBuilder(tool, a)
    src = {"disagreement_threshold": threshold}
    for name, value in zip(names, values):
        b.add(f"method {name}", value, unit="currency/share", params=src)
    b.add("disagreement", disagreement, unit="ratio", params=src)
    b.add("disagreement threshold", threshold, unit="ratio", params=src)
    b.add("weighting basis", weighting_basis, params=src)
    b.add("decision", "withheld" if price_target is None else "point_estimate", params=src)
    b.add("price target", price_target, unit="currency/share", params=src)

    return _envelope(tool, a, b, {
        "price_target": price_target,
        "weights_used": weights_used,
        "weighting_basis": weighting_basis,
        "disagreement": disagreement,
        "disagreement_threshold": threshold,
        "method_values": method_values,
        "outliers": outliers,
        "note": note,
        "disagreement_note": disagreement_note,
        "median": median,
        "min": lo,
        "max": hi,
        "method_count": method_count,
        "assumptions": {
            "disagreement_threshold": threshold,
            "weights_provided": weights is not None,
            "method_count": method_count,
        },
        "params": params,
        "warmup": {},
    })


TOOLS: list[ToolSpec] = [
    ToolSpec("estimate_dcf", GROUP, estimate_dcf),
    ToolSpec("estimate_ddm", GROUP, estimate_ddm),
    ToolSpec("compute_comparable_multiples", GROUP, compute_comparable_multiples),
    ToolSpec("estimate_monte_carlo", GROUP, estimate_monte_carlo),
    ToolSpec("compute_sensitivity_grid", GROUP, compute_sensitivity_grid),
    ToolSpec("build_valuation_synthesis", GROUP, build_valuation_synthesis),
]

#: 快照型工具没有日期轴，跨族合规闸门无法按参数名猜出一组合法输入——这里为每个
#: 工具声明一份完整、确定性的样例调用参数（工具名 → kwargs），供闸门与自检复用。
#: 量级与 `tests/quantdata.dcf_inputs()` 一致（此处就写成字面量：计算层不依赖测试
#: 夹具，快照与夹具的漂移由测试自己断言）。
SAMPLE_AS_OF = "2026-06-30"

_SAMPLE_DCF_INPUTS: dict[str, Any] = {
    "free_cash_flows": [100.0, 110.0, 121.0, 133.1, 146.4],
    "discount_rate": 0.09,
    "terminal_growth": 0.03,
    "shares_outstanding": 50.0,
    "net_debt": 200.0,
    "exit_multiple": 12.0,
    "exit_metric": 200.0,
}

SAMPLES: dict[str, dict] = {
    "estimate_dcf": {
        **_SAMPLE_DCF_INPUTS,
        "terminal_method": TERMINAL_PERPETUITY,
        "as_of": SAMPLE_AS_OF,
    },
    "estimate_ddm": {
        "dividend_per_share": 2.0,
        "growth_rate": 0.04,
        "required_return": 0.09,
        "as_of": SAMPLE_AS_OF,
    },
    "compute_comparable_multiples": {
        "multiples": [9.0, 10.0, 11.0, 12.0, 13.0, 40.0],
        "metric_value": 200.0,
        "winsorize": 0.05,
        "as_of": SAMPLE_AS_OF,
    },
    "estimate_monte_carlo": {
        "base_value": 1400.0,
        "growth_mean": 0.06,
        "growth_std": 0.12,
        "discount_rate": 0.09,
        "n_simulations": 10000,
        "seed": 42,
        "horizon": 5,
        "as_of": SAMPLE_AS_OF,
    },
    "compute_sensitivity_grid": {
        "base_inputs": dict(_SAMPLE_DCF_INPUTS),
        "variable_x": {"name": "discount_rate", "values": [0.08, 0.10]},
        "variable_y": {"name": "terminal_growth", "values": [0.02, 0.03]},
        "as_of": SAMPLE_AS_OF,
    },
    "build_valuation_synthesis": {
        "methods": [{"name": "dcf", "value": 100.0}, {"name": "comparables", "value": 108.0}],
        "disagreement_threshold": 0.20,
        "weights": {"dcf": 0.6, "comparables": 0.4},
        "as_of": SAMPLE_AS_OF,
    },
}
