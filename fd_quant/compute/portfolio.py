"""组合统计族：年化收益 / 年化波动 / 最大回撤 / Sharpe / Sortino / Calmar / IR+TE / Beta / 集中度 / 换手率。

口径与纪律（与 `technical.py` 同一套，读一份即可跨族推理）：

- **as_of 先行**：每个函数第一步都是 `check_as_of`，随后序列经 `check_series` 逐位配对
  校验；晚于 `as_of` 的数据点一律拒绝并点名**首个**越界日期，绝不静默截断。
- **参数一律回显**：`periods_per_year`（每年期数，默认 252，A 股按策略层调参）与
  `risk_free`（无风险利率，年化小数，默认 0.0）既进 `params` 也在顶层同名键上出现；
  Sortino 的 `target`（每期门槛，小数）同办。绝不内置一个调用方看不见的假设值。
- **收益率口径**：序列入参是**水平序列**（权益/净值；`benchmark` 是与 `values` 同一比较
  口径的基准水平序列）。内部简单收益 `r_i = x_i / x_{i-1} - 1`，一律为小数（0.01 = 1%）。
- **缺失（null/NaN）传染整条统计量**：这些工具的输出是**一个**数（或一条等长序列），
  序列里任何一位缺失都会污染它；故任一序列入参含 null/NaN（或无法差分的零水位）→
  数值输出为 `null` 并附 `reason`（点名首个出问题的日期），而不是跳过缺失位继续算
  （跳过等于替调用方发明一个数据点）。
- **退化统计量只拒绝、不产出 inf/NaN**：零超额波动（常数权益）、零下行偏差、零基准
  方差、零回撤（Calmar 分母）、零超额波动下的非零均值 → 输出 `null` + `reason`，
  绝不返回 inf；返回恒可 `json.dumps(..., allow_nan=False)`。
- **确定性**：均值/方差/协方差只用显式循环或 `deterministic_reduction`（Kahan 从左到右），
  不使用 `np.mean`/`np.std`/`np.cov`/`@` 等成对或 BLAS 归约——那是唯一随线程数变化的路径。

**权重形态**（`summarize_concentration` / `summarize_turnover`）：`dates` 与权重等长时按
逐位配对走 `check_series`；长度不等时权重视为**单一截面**（例如把一段观察窗口的日期与
一份截面权重一起传入），此时逐位只校验日期（ISO、严格递增、不晚于 `as_of`），观察日取
`dates[-1]`，截面大小由 `weight_count` 给出。两种形态下越界日期同样点名拒绝。
"""
from __future__ import annotations

import math
from datetime import date as _date
from typing import Any, Optional, Sequence

import numpy as np

from . import ToolSpec
from .base import (
    ComputeError,
    EvidenceBuilder,
    Series,
    check_as_of,
    check_finite,
    check_positive_window,
    check_series,
    deterministic_reduction,
    result_envelope,
    to_none_list,
)

GROUP = "portfolio_statistics"


# ── 内部工具 ─────────────────────────────────────────────────────


def _num(x: Any) -> Optional[float]:
    """标量 → JSON 安全浮点（None / NaN / ±inf → None，np 标量 → python float）。"""
    if x is None:
        return None
    f = float(x)
    return f if math.isfinite(f) else None


def _count(x: Any) -> int:
    return int(x)


def _aligned_turnover(
    weights: np.ndarray,
    symbols: Sequence[str],
    weights_prior: np.ndarray,
    symbols_prior: Sequence[str],
) -> dict:
    """Per-instrument weight deltas across two periods.

    The book is keyed by instrument code rather than by list position: the same
    holdings passed in a different order are the same portfolio, and subtracting
    by position would report a full round trip for a book that never traded.
    Instruments present on only one side are matched against weight 0, so a new
    entry or a full exit contributes its full weight — that is a real trade, not
    a mismatch.

    Reduction is over instruments sorted by code so the sum is order-independent
    and reproducible (the compute plane's determinism rule).
    """
    cur: dict[str, float] = {}
    for code, w in zip(symbols, weights):
        cur[str(code)] = cur.get(str(code), 0.0) + float(w)
    prev: dict[str, float] = {}
    for code, w in zip(symbols_prior, weights_prior):
        prev[str(code)] = prev.get(str(code), 0.0) + float(w)

    names = sorted(set(cur) | set(prev))
    deltas = np.array(
        [abs(cur.get(n, 0.0) - prev.get(n, 0.0)) for n in names],
        dtype=np.float64,
    )
    traded = sum(1 for n in names if cur.get(n, 0.0) != prev.get(n, 0.0))
    return {"deltas": deltas, "names_traded": traded}


def _first_nonfinite(arr: np.ndarray) -> Optional[int]:
    for i in range(len(arr)):
        if not math.isfinite(float(arr[i])):
            return i
    return None


def _missing_reason(
    arr: np.ndarray, dates: Optional[Sequence[str]], name: str
) -> Optional[str]:
    """缺失诊断：点名首个非有限值（不是"跳过缺失位继续算"——那等于发明数据点）。

    `dates` 与 `arr` 不等长时（截面权重那种"日期是观察窗口而非逐位配对"的形态）
    只报索引——报一个配不上的日期比不报更容易误导。
    """
    bad = _first_nonfinite(arr)
    if bad is None:
        return None
    cnt = sum(1 for v in arr if not math.isfinite(float(v)))
    where = f" at {dates[bad]}" if dates is not None and len(dates) == len(arr) else ""
    return (
        f"{name} has {cnt} missing value(s) (first{where}, index {bad}); "
        f"a missing point contaminates the whole statistic, so it is not computed"
    )


def _mean(arr: np.ndarray) -> float:
    """固定顺序均值（Kahan 从左到右）。"""
    return deterministic_reduction(arr) / len(arr)


def _stdev(arr: np.ndarray) -> float:
    """样本标准差（ddof=1），固定顺序；调用方保证 len >= 2 且无缺失。"""
    m = _mean(arr)
    sq = (arr - m) ** 2
    return math.sqrt(deterministic_reduction(sq) / (len(arr) - 1))


def _simple_returns(values: np.ndarray) -> np.ndarray:
    """简单收益 r_i = x_i / x_{i-1} - 1（长度 = 入参长度 - 1）。

    前值 ≤ 0（非正水位）或任一端缺失 → 该期收益为 NaN（缺失），由调用方决定拒绝
    还是出 null；这里绝不填空值。
    """
    n = len(values)
    out = np.full(max(n - 1, 0), np.nan)
    for i in range(1, n):
        prev = float(values[i - 1])
        cur = float(values[i])
        if math.isfinite(prev) and math.isfinite(cur) and prev > 0.0:
            out[i - 1] = cur / prev - 1.0
    return out


def _benchmark_returns_or_reason(
    benchmark: Series,
) -> tuple[np.ndarray, Optional[str]]:
    """基准差分 + 可操作诊断。

    基准与 `values` 同一比较口径（**水平**序列），故同样逐位差分。只有水位为
    正数的那些期才能差分；首个不可差分的水位就被点名。这不是"组合缺数"，而是
    **输入口径可疑**——若调用方传的是逐期收益序列（其"水位"会取负值），差分结果
    无意义，诊断里直接说明并给两条出路，绝不猜测口径（猜错等于静默换定义）。
    """
    rets = _simple_returns(benchmark.values)
    vals = benchmark.values
    for i in range(len(vals) - 1):
        v = float(vals[i])
        if not math.isfinite(v):
            return rets, (
                f"benchmark level at {benchmark.dates[i]} (index {i}) is missing; the return "
                f"across it is undefined, so the statistic is not computed"
            )
        if v <= 0.0:
            return rets, (
                f"benchmark level {v!r} at {benchmark.dates[i]} (index {i}) is not a positive "
                f"NAV, so the return on {benchmark.dates[i + 1]} is undefined; pass a cumulative "
                f"NAV / index-level series (levels are positive by construction) — a per-period "
                f"return series cannot be differenced this way"
            )
    return rets, None


def _leading_nulls(arr: np.ndarray) -> int:
    """首个非 NaN 位置之前的 null 个数（全 NaN → len）。"""
    for i in range(len(arr)):
        if np.isfinite(arr[i]):
            return i
    return len(arr)


def _warmup(lines: dict) -> dict[str, int]:
    """逐输出线声明 warmup：首个非 null 位置之前的 null 个数（标量线按长度 1 计）。

    组合统计不消耗窗口（不取滚动窗），故序列线一律 0；只有整条统计量因缺失被置
    null 时才非 0——调用方据此知道"这个键为什么没有数"。
    """
    out: dict[str, int] = {}
    for name, v in lines.items():
        if isinstance(v, np.ndarray):
            out[name] = _leading_nulls(v)
        elif isinstance(v, (list, tuple)):
            out[name] = _leading_nulls(np.asarray(v, dtype=np.float64)) if v else 0
        else:
            out[name] = 0 if _num(v) is not None else 1
    return out


def _emit_line(
    b: EvidenceBuilder, name: str, arr: np.ndarray, params: dict, unit: Optional[str] = None
) -> None:
    """序列证据：长度 + 首值 + 末值（点数不是信息，首末才是）。"""
    b.add(f"{name} length", int(len(arr)), unit="points", params=params)
    first: Optional[float] = None
    last: Optional[float] = None
    for i in range(len(arr)):
        if np.isfinite(arr[i]):
            first = float(arr[i])
            break
    for i in range(len(arr) - 1, -1, -1):
        if np.isfinite(arr[i]):
            last = float(arr[i])
            break
    b.add(f"{name} first", first, unit=unit, params=params)
    b.add(f"{name} last", last, unit=unit, params=params)


def _prep_positive_series(
    dates: Sequence[Any], values: Sequence[Any], as_of: str, *, name: str
) -> Series:
    """权益/净值水平序列校验：非正水位拒绝（点名该日期），缺失留给调用方出 null。

    缺失先于正值判定：含缺失的序列交给 `_missing_reason` 出 null + 原因，而不是报
    "非正值"（此时正值检查没有意义）。
    """
    s = check_series(dates, values, as_of=as_of, name=name)
    if _first_nonfinite(s.values) is not None:
        return s
    for i in range(len(s)):
        v = float(s.values[i])
        if v <= 0.0:
            raise ComputeError(
                f"{name}: non-positive NAV {v!r} at {s.dates[i]} (index {i}); "
                f"geometric return / drawdown need strictly positive values"
            )
    return s



#: 频率 → 每年期数的标准映射。用于**校验**调用方给的 periods_per_year 是否与
#: 日期轴一致，不用于静默推断（见 _frequency_check）。
_STANDARD_PERIODS_PER_YEAR = {"daily": 252, "weekly": 52, "monthly": 12, "quarterly": 4, "annual": 1}


def _infer_frequency(dates: Sequence[str]) -> tuple[Optional[str], Optional[float]]:
    """从日期轴的中位间隔猜频率；样本不足（<3 点）时返回 (None, None)。

    **只用于给出提示，不用于改写调用方的 periods_per_year**：猜错频率会把年化
    结果悄悄改掉，那比"没验证"更坏——调用方以为自己算的是日频年化，实际拿到
    月频的。
    """
    import datetime as _dt

    if len(dates) < 3:
        return None, None
    parsed = []
    for d in dates:
        try:
            parsed.append(_dt.date.fromisoformat(d))
        except (TypeError, ValueError):
            return None, None
    gaps = [(parsed[i + 1] - parsed[i]).days for i in range(len(parsed) - 1)]
    gaps = [g for g in gaps if g > 0]
    if not gaps:
        return None, None
    gaps.sort()
    med = gaps[len(gaps) // 2]
    if med <= 4:
        return "daily", med
    if med <= 10:
        return "weekly", med
    if med <= 45:
        return "monthly", med
    if med <= 135:
        return "quarterly", med
    return "annual", med


def _frequency_check(
    dates: Sequence[str], ppy: int, declared: Optional[str], tool: str
) -> dict:
    """年化频率声明（quant-compute-plane）：未给 frequency 时标注"未经验证"。

    **为什么必须标注**：默认的 252 是**交易日**口径。若调用方传的是自然日序列
    （含周末与长假），真实每年期数接近 365，按 252 年化会系统性低估年化收益与
    波动——偏差不报错、不留痕，只有下游对比时才发现数字对不上。

    `declared` 非空时按标准映射**校验**（不符即报 ComputeError 点名两个数），
    空则标注 unverified 并附推断出的频率供参考。
    """
    if declared is not None:
        key = str(declared).strip().lower()
        if key not in _STANDARD_PERIODS_PER_YEAR:
            raise ComputeError(
                f"{tool}: frequency must be one of "
                f"{sorted(_STANDARD_PERIODS_PER_YEAR)}, got {declared!r}")
        expected = _STANDARD_PERIODS_PER_YEAR[key]
        if expected != int(ppy):
            raise ComputeError(
                f"{tool}: periods_per_year={ppy} contradicts frequency={declared!r} "
                f"(expected {expected}); either fix periods_per_year or drop "
                f"frequency to accept the unverified default")
        return {
            "frequency": key,
            "frequency_verified": True,
            "frequency_note": None,
        }
    guess, med_gap = _infer_frequency(dates)
    note = (
        "频率未经验证：periods_per_year 取调用方传入值（默认 252 = 交易日口径），"
        "本层未校验它与日期轴一致。若序列是自然日（含周末与长假），真实每年期数"
        "接近 365，按 252 年化会系统性低估年化收益与波动。传 frequency="
        "(daily|weekly|monthly|quarterly|annual) 可启用校验。"
    )
    if guess is not None:
        note += f" 由日期轴中位间隔 {med_gap:g} 天推断，序列看起来像 {guess} 频。"
    return {"frequency": None, "frequency_verified": False, "frequency_note": note}

def _annualize(start: float, end: float, intervals: int, ppy: int) -> tuple[float, Optional[float], Optional[str]]:
    """总收益与几何年化收益：`(end/start) ** (ppy/intervals) - 1`。

    基数正（由 `_prep_positive_series` 保证）故指数为实数；溢出 float64 时返回
    None + 原因，绝不返回 inf。
    """
    ratio = end / start
    total = ratio - 1.0
    try:
        ann = math.pow(ratio, ppy / intervals) - 1.0
    except OverflowError:
        return total, None, (
            f"annualized return overflows float64: {end!r}/{start!r} ** ({ppy}/{intervals}); "
            f"check periods_per_year against the span"
        )
    if not math.isfinite(ann):
        return total, None, (
            f"annualized return is not finite: {end!r}/{start!r} ** ({ppy}/{intervals})"
        )
    return total, ann, None


def _drawdown_event(values: np.ndarray) -> dict:
    """最大回撤事件（全族唯一定义，drawdown 与 calmar 共用）。

    回撤 = `x_i / max(x_0..x_i) - 1` ≤ 0；峰值取谷底之前**首个**最高点，谷底取
    **首个**最低回撤点（并列取先出现者，保证逐位确定）。序列从未低于起始水位时
    回撤全为 0，谷底 = 峰值 = 首位——这是"无回撤"的退化事件（不是缺失），
    recovery 即峰值日本身、duration 为 0。
    """
    n = len(values)
    dd = np.empty(n, dtype=np.float64)
    cur = float(values[0])
    for i in range(n):
        x = float(values[i])
        if x > cur:
            cur = x
        dd[i] = x / cur - 1.0

    trough = 0
    for i in range(1, n):
        if dd[i] < dd[trough]:
            trough = i
    peak = 0
    for i in range(1, trough + 1):
        if values[i] > values[peak]:
            peak = i

    if trough == peak:
        recovery: Optional[int] = peak
    else:
        recovery = None
        for j in range(trough + 1, n):
            if float(values[j]) >= float(values[peak]):
                recovery = j
                break
    return {
        "drawdown": dd,
        "max_drawdown": float(dd[trough]),
        "peak_index": peak,
        "trough_index": trough,
        "recovery_index": recovery,
        "peak_value": float(values[peak]),
        "trough_value": float(values[trough]),
    }


def _to_float_array(values: Sequence[Any], name: str) -> np.ndarray:
    """权重列表 → float64（None → NaN）：与 `check_series` 同一取值口径。

    契约层与本层各自校验同一形状（见族作者契约 §2.9），故这里不 import 它的私有函数。
    """
    out = np.empty(len(values), dtype=np.float64)
    for i, v in enumerate(values):
        if v is None:
            out[i] = np.nan
            continue
        try:
            out[i] = float(v)
        except (TypeError, ValueError):
            raise ComputeError(f"{name}[{i}] is not numeric: {v!r}") from None
    return out


def _prep_weight_vector(
    dates: Sequence[Any], weights: Sequence[Any], as_of: str, *, name: str
) -> tuple[Series, str]:
    """权重向量 + 观察日（族文档 §权重形态 的两种输入形态）。

    - 逐位配对（`len(dates) == len(weights)`）：走 `check_series`，日期与权重逐位配对。
    - 截面（长度不等）：权重是单一截面；逐位校验日期（ISO、严格递增、不晚于 as_of，
      越界同样**点名首个**日期），观察日取 `dates[-1]`。
    """
    ds = list(dates)
    if not ds:
        raise ComputeError(f"{name}: dates is empty")
    if not len(weights):
        raise ComputeError(f"{name} is empty")
    if len(ds) == len(weights):
        s = check_series(ds, weights, as_of=as_of, name=name)
        return s, s.dates[-1]

    prev: Optional[str] = None
    for i, d in enumerate(ds):
        try:
            d_iso = check_as_of(d)
        except ComputeError:
            raise ComputeError(f"{name}: dates[{i}] is not an ISO date: {d!r}") from None
        if d_iso > as_of:
            raise ComputeError(
                f"{name}: data point {d_iso} is after as_of={as_of} "
                f"(future data refused; slice the input or raise as_of)"
            )
        if prev is not None and d_iso <= prev:
            raise ComputeError(
                f"{name}: dates must be strictly increasing; {prev} -> {d_iso} at index {i}"
            )
        prev = d_iso
    arr = _to_float_array(weights, name)
    return Series(dates=[str(d) for d in ds], values=arr), str(ds[-1])


def _ratio_guard(value: Optional[float], why: str) -> tuple[Optional[float], Optional[str]]:
    """比率型输出收口：非有限 → (None, 原因)；比例本身的退化在原处已拒绝。"""
    if value is None or not math.isfinite(float(value)):
        return None, why
    return float(value), None


# ── 年化收益 / 年化波动 ──────────────────────────────────────────


def summarize_return(
    dates: Sequence[str],
    values: Sequence[Optional[float]],
    as_of: str,
    periods_per_year: int = 252,
    frequency: Optional[str] = None,
) -> dict:
    """年化收益（几何）：总收益 = 末值/首值 - 1；年化 = (末值/首值)^(periods_per_year/期数) - 1。

    Input: dates (ISO 日期, 严格递增), values (组合权益/NAV 水平序列, 与 dates 等长且逐位 > 0; null 表缺失),
      as_of (ISO 日期, 必填; 晚于它的数据点即拒绝并点名首个越界日期), periods_per_year (每年期数, 默认 252; A 股按策略层调参)。
    Returns: total_return (全期总收益, 小数) / annualized_return (年化收益, 小数) / intervals (收益期数 = 长度 - 1) /
      periods_per_year (回显) / reason (无法计算时给原因, 否则 null) / params / warmup / evidence。
      单位均为小数 (0.12 = 12%)。本统计量不消耗窗口, 序列无缺失时 warmup = 0。
      NaN 语义: values 含缺失 → 两个收益均为 null 并附 reason; 任一位 NAV ≤ 0 → ComputeError 点名该日期。
    """
    a = check_as_of(as_of)
    ppy = check_positive_window(periods_per_year, name="periods_per_year")
    freq = _frequency_check(dates, ppy, frequency, "summarize_return")
    s = _prep_positive_series(dates, values, a, name="values")
    n = len(s)
    if n < 2:
        raise ComputeError(
            f"values needs at least 2 points to annualize (got {n}); "
            f"the annualized return spans values[0] -> values[-1]"
        )
    params = {"periods_per_year": ppy}
    b = EvidenceBuilder("summarize_return", a)
    b.add("observations", n, unit="points", params=params)
    b.add("span_start", s.dates[0], params=params)
    b.add("span_end", s.dates[-1], params=params)

    reason = _missing_reason(s.values, s.dates, "values")
    if reason is not None:
        b.add("annualized_return", None, unit="decimal", params=params)
        return result_envelope("summarize_return", a, s, b, {
            "total_return": None,
            "annualized_return": None,
            "intervals": n - 1,
            "periods_per_year": ppy,
        **freq,
            **freq,
            "reason": reason,
            "params": params,
            "warmup": _warmup({"total_return": None, "annualized_return": None}),
        })

    total, ann, reason = _annualize(float(s.values[0]), float(s.values[-1]), n - 1, ppy)
    b.add("total_return", _num(total), unit="decimal", params=params)
    b.add("annualized_return", _num(ann), unit="decimal", params=params)
    return result_envelope("summarize_return", a, s, b, {
        "total_return": _num(total),
        "annualized_return": _num(ann),
        "intervals": n - 1,
        "periods_per_year": ppy,
        **freq,
        "reason": reason,
        "params": params,
        "warmup": _warmup({"total_return": total, "annualized_return": ann}),
    })


def summarize_volatility(
    dates: Sequence[str],
    values: Sequence[Optional[float]],
    as_of: str,
    periods_per_year: int = 252,
    frequency: Optional[str] = None,
) -> dict:
    """年化波动：权益序列简单收益的样本标准差 (ddof=1) × sqrt(periods_per_year)。

    Input: dates (ISO, 严格递增), values (权益/NAV 水平序列, 与 dates 等长且 > 0; null 表缺失),
      as_of (ISO, 必填; 越界点名首个日期), periods_per_year (每年期数, 默认 252; A 股按策略层调参)。
    Returns: volatility (年化波动, 小数) / per_period_volatility (每期样本标准差, ddof=1) /
      annualization_factor (= sqrt(periods_per_year), 回显) / periods_per_year / observations (收益期数) /
      reason / params / warmup / evidence。无窗口消耗, 无缺失时 warmup = 0。
      values 含缺失 → 数值键为 null 并附 reason; NAV ≤ 0 → ComputeError 点名该日期。
    """
    a = check_as_of(as_of)
    ppy = check_positive_window(periods_per_year, name="periods_per_year")
    freq = _frequency_check(dates, ppy, frequency, "summarize_volatility")
    s = _prep_positive_series(dates, values, a, name="values")
    n = len(s)
    if n < 3:
        raise ComputeError(
            f"values needs at least 3 points (>= 2 returns) for a sample stdev with ddof=1 "
            f"(got {n})"
        )
    factor = math.sqrt(float(ppy))
    params = {"periods_per_year": ppy}
    b = EvidenceBuilder("summarize_volatility", a)
    b.add("observations", n, unit="points", params=params)
    b.add("annualization_factor", factor, unit="ratio", params=params)

    reason = _missing_reason(s.values, s.dates, "values")
    if reason is not None:
        return result_envelope("summarize_volatility", a, s, b, {
            "volatility": None,
            "per_period_volatility": None,
            "annualization_factor": factor,
            "periods_per_year": ppy,
        **freq,
            **freq,
            "observations": n - 1,
            "reason": reason,
            "params": params,
            "warmup": _warmup({"volatility": None, "per_period_volatility": None}),
        })

    rets = _simple_returns(s.values)
    per_period = _stdev(rets)
    vol = per_period * factor
    b.add("volatility", _num(vol), unit="annualised decimal", params=params)
    b.add("per_period_volatility", _num(per_period), unit="decimal", params=params)
    return result_envelope("summarize_volatility", a, s, b, {
        "volatility": _num(vol),
        "per_period_volatility": _num(per_period),
        "annualization_factor": factor,
        "periods_per_year": ppy,
        **freq,
        "observations": n - 1,
        "reason": None,
        "params": params,
        "warmup": _warmup({"volatility": vol, "per_period_volatility": per_period}),
    })


# ── 最大回撤 / Calmar ────────────────────────────────────────────


def summarize_drawdown(
    dates: Sequence[str],
    values: Sequence[Optional[float]],
    as_of: str,
) -> dict:
    """最大回撤：回撤 = 当前值/历史峰值 - 1（≤ 0），取最低点，并给出峰值日/谷底日/修复日。

    Input: dates (ISO, 严格递增), values (权益/NAV 水平序列, 与 dates 等长且 > 0; null 表缺失), as_of (ISO, 必填)。
    Returns: max_drawdown (负值或 0.0, 小数) / peak_date / trough_date / recovery_date (未修复则 null) /
      duration_days (峰值日 → 谷底日的自然日数) / peak_value / trough_value /
      drawdown_series (与输入等长、逐位 ≤ 0 的列表) / reason / params / warmup / evidence。
      无窗口消耗, 故 warmup = 0; 序列从未回撤时 max_drawdown = 0.0 且 peak = trough = recovery = 首日
      (退化事件, 不是缺失)。values 含缺失 → 数值键与整条 drawdown_series 均为 null 并附 reason;
      任一位 NAV ≤ 0 → ComputeError 点名该日期。
    """
    a = check_as_of(as_of)
    s = _prep_positive_series(dates, values, a, name="values")
    n = len(s)
    params: dict = {}
    b = EvidenceBuilder("summarize_drawdown", a)
    b.add("observations", n, unit="points", params=params)

    reason = _missing_reason(s.values, s.dates, "values")
    if reason is not None:
        b.add("max_drawdown", None, unit="decimal", params=params)
        return result_envelope("summarize_drawdown", a, s, b, {
            "max_drawdown": None,
            "peak_date": None,
            "trough_date": None,
            "recovery_date": None,
            "duration_days": None,
            "peak_value": None,
            "trough_value": None,
            "drawdown_series": [None] * n,
            "reason": reason,
            "params": params,
            "warmup": _warmup({"drawdown_series": [None] * n, "max_drawdown": None}),
        })

    ev = _drawdown_event(s.values)
    dd = ev["drawdown"]
    peak_date = s.dates[ev["peak_index"]]
    trough_date = s.dates[ev["trough_index"]]
    recovery_date = s.dates[ev["recovery_index"]] if ev["recovery_index"] is not None else None
    duration = (_date.fromisoformat(trough_date) - _date.fromisoformat(peak_date)).days

    _emit_line(b, "drawdown_series", dd, params, unit="decimal")
    b.add("max_drawdown", _num(ev["max_drawdown"]), unit="decimal", params=params)
    b.add("peak_date", peak_date, params=params)
    b.add("trough_date", trough_date, params=params)
    b.add("recovery_date", recovery_date, params=params)
    return result_envelope("summarize_drawdown", a, s, b, {
        "max_drawdown": _num(ev["max_drawdown"]),
        "peak_date": peak_date,
        "trough_date": trough_date,
        "recovery_date": recovery_date,
        "duration_days": _count(duration),
        "peak_value": _num(ev["peak_value"]),
        "trough_value": _num(ev["trough_value"]),
        "drawdown_series": to_none_list(dd),
        "reason": None,
        "params": params,
        "warmup": _warmup({"drawdown_series": dd, "max_drawdown": ev["max_drawdown"]}),
    })


def summarize_calmar(
    dates: Sequence[str],
    values: Sequence[Optional[float]],
    as_of: str,
    periods_per_year: int = 252,
    frequency: Optional[str] = None,
) -> dict:
    """Calmar 比率 = 年化收益 / |最大回撤|（回撤为 0 时输出 null + 原因，绝不返回 inf）。

    Input: dates (ISO, 严格递增), values (权益/NAV 水平序列, 与 dates 等长且 > 0; null 表缺失),
      as_of (ISO, 必填), periods_per_year (每年期数, 默认 252; A 股按策略层调参)。
    Returns: calmar (倍, 年化收益 / |回撤|) / annualized_return (小数) / max_drawdown (负值或 0.0) /
      periods_per_year (回显) / reason (无法计算时给原因, 否则 null) / params / warmup / evidence。
      无窗口消耗, 无缺失时 warmup = 0。序列从未回撤 → calmar 为 null 并说明分母为 0;
      values 含缺失 → 数值键为 null 并附 reason; NAV ≤ 0 → ComputeError 点名该日期。
    """
    a = check_as_of(as_of)
    ppy = check_positive_window(periods_per_year, name="periods_per_year")
    freq = _frequency_check(dates, ppy, frequency, "summarize_calmar")
    s = _prep_positive_series(dates, values, a, name="values")
    n = len(s)
    if n < 2:
        raise ComputeError(
            f"values needs at least 2 points for a drawdown-adjusted ratio (got {n})"
        )
    params = {"periods_per_year": ppy}
    b = EvidenceBuilder("summarize_calmar", a)
    b.add("observations", n, unit="points", params=params)

    reason = _missing_reason(s.values, s.dates, "values")
    if reason is not None:
        b.add("calmar", None, unit="ratio", params=params)
        return result_envelope("summarize_calmar", a, s, b, {
            "calmar": None,
            "annualized_return": None,
            "max_drawdown": None,
            "periods_per_year": ppy,
        **freq,
            **freq,
            "reason": reason,
            "params": params,
            "warmup": _warmup({"calmar": None, "annualized_return": None}),
        })

    ev = _drawdown_event(s.values)
    _total, ann, overflow = _annualize(float(s.values[0]), float(s.values[-1]), n - 1, ppy)
    max_dd = ev["max_drawdown"]
    calmar: Optional[float] = None
    if overflow is not None:
        reason = overflow
    elif ann is None:
        reason = "annualized return is not computable (see above)"
    elif max_dd == 0.0:
        reason = (
            "max drawdown is 0.0 (the series never fell below its running peak) so the "
            "Calmar denominator is 0; Calmar is undefined here, not infinite"
        )
    else:
        calmar, why = _ratio_guard(ann / abs(max_dd), "Calmar is not finite for this input")
        reason = why

    b.add("max_drawdown", _num(max_dd), unit="decimal", params=params)
    b.add("annualized_return", _num(ann), unit="decimal", params=params)
    b.add("calmar", _num(calmar), unit="ratio", params=params)
    return result_envelope("summarize_calmar", a, s, b, {
        "calmar": _num(calmar),
        "annualized_return": _num(ann),
        "max_drawdown": _num(max_dd),
        "periods_per_year": ppy,
        **freq,
        "reason": reason,
        "params": params,
        "warmup": _warmup({"calmar": calmar, "annualized_return": ann, "max_drawdown": max_dd}),
    })


# ── Sharpe / Sortino ─────────────────────────────────────────────


def summarize_sharpe(
    dates: Sequence[str],
    values: Sequence[Optional[float]],
    as_of: str,
    periods_per_year: int = 252,
    frequency: Optional[str] = None,
    risk_free: float = 0.0,
) -> dict:
    """Sharpe 比率：每期超额收益均值 / 每期超额收益样本标准差 × sqrt(periods_per_year)。

    超额 = 简单收益 - risk_free/periods_per_year（risk_free 为**年化**小数）。
    Input: dates (ISO, 严格递增), values (权益/NAV 水平序列, 与 dates 等长且 > 0; null 表缺失),
      as_of (ISO, 必填), periods_per_year (每年期数, 默认 252),
      risk_free (年化无风险利率, 小数, 默认 0.0 = 显式假定为零, 没有任何隐藏假设值)。
    Returns: sharpe (倍) / risk_free (回显, 含默认 0.0) / periods_per_year / mean_excess_return (每期, 小数) /
      excess_volatility (每期样本标准差, ddof=1) / annualization_factor (= sqrt(periods_per_year)) /
      observations / reason / params / warmup / evidence。无窗口消耗, 无缺失时 warmup = 0。
      超额收益标准差为 0（常数权益等）→ sharpe 为 null 并报告该事实, 不返回 inf。
    """
    a = check_as_of(as_of)
    ppy = check_positive_window(periods_per_year, name="periods_per_year")
    freq = _frequency_check(dates, ppy, frequency, "summarize_sharpe")
    rf = check_finite(risk_free, name="risk_free")
    s = _prep_positive_series(dates, values, a, name="values")
    n = len(s)
    if n < 3:
        raise ComputeError(
            f"values needs at least 3 points (>= 2 returns) for a Sharpe ratio (got {n})"
        )
    factor = math.sqrt(float(ppy))
    params = {"periods_per_year": ppy, "risk_free": rf}
    b = EvidenceBuilder("summarize_sharpe", a)
    b.add("risk_free", rf, unit="annualised decimal", params=params)
    b.add("observations", n, unit="points", params=params)
    b.add("annualization_factor", factor, unit="ratio", params=params)

    reason = _missing_reason(s.values, s.dates, "values")
    if reason is not None:
        return result_envelope("summarize_sharpe", a, s, b, {
            "sharpe": None,
            "risk_free": rf,
            "periods_per_year": ppy,
        **freq,
            **freq,
            "annualization_factor": factor,
            "mean_excess_return": None,
            "excess_volatility": None,
            "observations": n - 1,
            "reason": reason,
            "params": params,
            "warmup": _warmup({"sharpe": None, "mean_excess_return": None}),
        })

    rets = _simple_returns(s.values)
    excess = rets - rf / ppy
    mean_excess = _mean(excess)
    vol = _stdev(excess)
    sharpe: Optional[float] = None
    if vol == 0.0:
        reason = (
            f"excess return standard deviation is 0 (constant excess return "
            f"{mean_excess!r} per period) so the Sharpe denominator is 0; the ratio is "
            f"undefined here, not infinite"
        )
    else:
        sharpe, reason = _ratio_guard(
            mean_excess / vol * factor, "Sharpe ratio is not finite for this input"
        )

    b.add("sharpe", _num(sharpe), unit="ratio", params=params)
    b.add("mean_excess_return", _num(mean_excess), unit="decimal", params=params)
    b.add("excess_volatility", _num(vol), unit="decimal", params=params)
    return result_envelope("summarize_sharpe", a, s, b, {
        "sharpe": _num(sharpe),
        "risk_free": rf,
        "periods_per_year": ppy,
        **freq,
        "annualization_factor": factor,
        "mean_excess_return": _num(mean_excess),
        "excess_volatility": _num(vol),
        "observations": n - 1,
        "reason": reason,
        "params": params,
        "warmup": _warmup({"sharpe": sharpe, "mean_excess_return": mean_excess}),
    })


def summarize_sortino(
    dates: Sequence[str],
    values: Sequence[Optional[float]],
    as_of: str,
    periods_per_year: int = 252,
    frequency: Optional[str] = None,
    risk_free: float = 0.0,
    target: float = 0.0,
) -> dict:
    """Sortino 比率：每期超额收益均值 / 下行偏差 × sqrt(periods_per_year)。

    超额 = 简单收益 - risk_free/periods_per_year（risk_free 为**年化**小数）。
    下行偏差 = sqrt(Σ min(超额 - target, 0)² / 全部期数)（target 为**每期**门槛，小数）：
    分母取**全部期数**（通行 target-downside-deviation 口径），不是只数低于门槛的那些期。
    Input: dates (ISO, 严格递增), values (权益/NAV 水平序列, 与 dates 等长且 > 0; null 表缺失),
      as_of (ISO, 必填), periods_per_year (每年期数, 默认 252), risk_free (年化, 默认 0.0),
      target (每期门槛, 默认 0.0 = 要求每期都有正超额)。
    Returns: sortino (倍) / risk_free 与 target (均回显, 含默认值) / periods_per_year / mean_excess_return (每期) /
      downside_deviation (每期) / downside_observations (低于门槛的期数) / annualization_factor /
      observations / reason / params / warmup / evidence。无窗口消耗, 无缺失时 warmup = 0。
      没有任何一期低于 target（下行偏差为 0）→ sortino 为 null 并报告该事实, 不返回 inf。
    """
    a = check_as_of(as_of)
    ppy = check_positive_window(periods_per_year, name="periods_per_year")
    freq = _frequency_check(dates, ppy, frequency, "summarize_sortino")
    rf = check_finite(risk_free, name="risk_free")
    tgt = check_finite(target, name="target")
    s = _prep_positive_series(dates, values, a, name="values")
    n = len(s)
    if n < 3:
        raise ComputeError(
            f"values needs at least 3 points (>= 2 returns) for a Sortino ratio (got {n})"
        )
    factor = math.sqrt(float(ppy))
    params = {"periods_per_year": ppy, "risk_free": rf, "target": tgt}
    b = EvidenceBuilder("summarize_sortino", a)
    b.add("risk_free", rf, unit="annualised decimal", params=params)
    b.add("target", tgt, unit="per-period decimal", params=params)
    b.add("observations", n, unit="points", params=params)

    reason = _missing_reason(s.values, s.dates, "values")
    if reason is not None:
        return result_envelope("summarize_sortino", a, s, b, {
            "sortino": None,
            "risk_free": rf,
            "target": tgt,
            "periods_per_year": ppy,
        **freq,
            **freq,
            "annualization_factor": factor,
            "mean_excess_return": None,
            "downside_deviation": None,
            "downside_observations": None,
            "observations": n - 1,
            "reason": reason,
            "params": params,
            "warmup": _warmup({"sortino": None, "mean_excess_return": None}),
        })

    rets = _simple_returns(s.values)
    excess = rets - rf / ppy
    shortfall = excess - tgt
    squared = np.empty(len(shortfall), dtype=np.float64)
    below = 0
    for i in range(len(shortfall)):
        d = float(shortfall[i])
        if d < 0.0:
            squared[i] = d * d
            below += 1
        else:
            squared[i] = 0.0
    downside_dev = math.sqrt(deterministic_reduction(squared) / len(shortfall))
    mean_excess = _mean(excess)

    sortino: Optional[float] = None
    if downside_dev == 0.0:
        reason = (
            f"no period falls below target={tgt!r} (downside deviation is 0) so the Sortino "
            f"denominator is 0; the ratio is undefined here, not infinite"
        )
    else:
        sortino, reason = _ratio_guard(
            mean_excess / downside_dev * factor, "Sortino ratio is not finite for this input"
        )

    b.add("sortino", _num(sortino), unit="ratio", params=params)
    b.add("downside_deviation", _num(downside_dev), unit="decimal", params=params)
    b.add("mean_excess_return", _num(mean_excess), unit="decimal", params=params)
    return result_envelope("summarize_sortino", a, s, b, {
        "sortino": _num(sortino),
        "risk_free": rf,
        "target": tgt,
        "periods_per_year": ppy,
        **freq,
        "annualization_factor": factor,
        "mean_excess_return": _num(mean_excess),
        "downside_deviation": _num(downside_dev),
        "downside_observations": _count(below),
        "observations": n - 1,
        "reason": reason,
        "params": params,
        "warmup": _warmup({"sortino": sortino, "mean_excess_return": mean_excess}),
    })


# ── 对基准：信息比 + 跟踪误差 / Beta ─────────────────────────────


def summarize_information_ratio(
    dates: Sequence[str],
    values: Sequence[Optional[float]],
    benchmark: Sequence[Optional[float]],
    as_of: str,
    periods_per_year: int = 252,
    frequency: Optional[str] = None,
) -> dict:
    """信息比 IR = 主动收益均值 / 主动收益标准差 × sqrt(periods_per_year)，同调返回跟踪误差 TE。

    主动收益 = 组合简单收益 - 基准简单收益（两列都按各自水平序列逐位差分）。
    Input: dates (ISO, 严格递增), values (组合权益/NAV 水平序列, 与 dates 等长且 > 0; null 表缺失),
      benchmark (基准**水平**序列, 与 dates 等长, 同一比较口径; 允许零/负水平, 但差分处前值为 0 → 该期缺失),
      as_of (ISO, 必填), periods_per_year (每年期数, 默认 252)。
    Returns: information_ratio (倍) / tracking_error (年化主动收益标准差, 小数) / periods_per_year /
      annualization_factor (= sqrt(periods_per_year)) / mean_active_return (每期) /
      active_return_volatility (每期样本标准差, ddof=1) / observations / reason / params / warmup / evidence。
      组合与基准逐位相同时主动收益恒为 0 → IR = 0.0 且 TE = 0.0（这是确定的 0，不是退化）;
      主动收益标准差为 0 而均值非 0 → IR 为 null + 原因，不返回 inf。无窗口消耗, 无缺失时 warmup = 0。
    """
    a = check_as_of(as_of)
    ppy = check_positive_window(periods_per_year, name="periods_per_year")
    freq = _frequency_check(dates, ppy, frequency, "summarize_information_ratio")
    p = _prep_positive_series(dates, values, a, name="values")
    bm = check_series(dates, benchmark, as_of=a, name="benchmark")
    n = len(p)
    if n < 3:
        raise ComputeError(
            f"values needs at least 3 points (>= 2 returns) for an information ratio (got {n})"
        )
    factor = math.sqrt(float(ppy))
    params = {"periods_per_year": ppy}
    b = EvidenceBuilder("summarize_information_ratio", a)
    b.add("observations", n, unit="points", params=params)
    b.add("annualization_factor", factor, unit="ratio", params=params)

    rets_p = _simple_returns(p.values)
    rets_b, bench_reason = _benchmark_returns_or_reason(bm)
    reasons = [_missing_reason(p.values, p.dates, "values"), bench_reason]
    reason = "; ".join(r for r in reasons if r) or None
    if reason is not None:
        return result_envelope("summarize_information_ratio", a, p, b, {
            "information_ratio": None,
            "tracking_error": None,
            "periods_per_year": ppy,
        **freq,
            **freq,
            "annualization_factor": factor,
            "mean_active_return": None,
            "active_return_volatility": None,
            "observations": n - 1,
            "reason": reason,
            "params": params,
            "warmup": _warmup({"information_ratio": None, "tracking_error": None}),
        })

    active = rets_p - rets_b
    mean_active = _mean(active)
    vol_active = _stdev(active)
    te = vol_active * factor
    if vol_active == 0.0:
        if mean_active == 0.0:
            # 组合 = 基准：主动收益恒 0 → IR 与 TE 都是确定的 0（spec 3.2 的边界断言）。
            ir: Optional[float] = 0.0
            reason = None
        else:
            ir = None
            reason = (
                f"active return has 0 standard deviation but a non-zero mean "
                f"({mean_active!r} per period) so the information ratio denominator is 0; "
                f"the ratio is undefined here, not infinite"
            )
    else:
        ir, reason = _ratio_guard(
            mean_active / vol_active * factor, "information ratio is not finite for this input"
        )

    b.add("information_ratio", _num(ir), unit="ratio", params=params)
    b.add("tracking_error", _num(te), unit="annualised decimal", params=params)
    b.add("mean_active_return", _num(mean_active), unit="decimal", params=params)
    return result_envelope("summarize_information_ratio", a, p, b, {
        "information_ratio": _num(ir),
        "tracking_error": _num(te),
        "periods_per_year": ppy,
        **freq,
        "annualization_factor": factor,
        "mean_active_return": _num(mean_active),
        "active_return_volatility": _num(vol_active),
        "observations": n - 1,
        "reason": reason,
        "params": params,
        "warmup": _warmup({"information_ratio": ir, "tracking_error": te}),
    })


def summarize_beta(
    dates: Sequence[str],
    values: Sequence[Optional[float]],
    benchmark: Sequence[Optional[float]],
    as_of: str,
) -> dict:
    """Beta（对基准）= cov(组合收益, 基准收益) / var(基准收益)，另给每期 alpha 与相关系数。

    协方差/方差用显式循环 + `deterministic_reduction`（Kahan 从左到右），不用矩阵运算。
    Input: dates (ISO, 严格递增), values (组合权益/NAV 水平序列, 与 dates 等长且 > 0; null 表缺失),
      benchmark (基准**水平**序列, 与 dates 等长; 允许零/负水平, 差分处前值为 0 → 该期缺失), as_of (ISO, 必填)。
    Returns: beta (倍; 组合与基准逐位相同 → 1.0) / alpha (每期, 小数 = 组合均值收益 - beta × 基准均值收益) /
      correlation (相关系数, [-1, 1]) / covariance 与 benchmark_variance (每期口径, ddof=1) /
      observations / reason / params / warmup / evidence。无窗口消耗, 无缺失时 warmup = 0。
      基准方差为 0（常数基准）→ beta 为 null + 原因，不返回 inf; 含缺失 → 数值键为 null 并附 reason。
    """
    a = check_as_of(as_of)
    p = _prep_positive_series(dates, values, a, name="values")
    bm = check_series(dates, benchmark, as_of=a, name="benchmark")
    n = len(p)
    if n < 3:
        raise ComputeError(
            f"values needs at least 3 points (>= 2 returns) for a beta regression (got {n})"
        )
    params: dict = {}
    b = EvidenceBuilder("summarize_beta", a)
    b.add("observations", n, unit="points", params=params)

    rets_p = _simple_returns(p.values)
    rets_b, bench_reason = _benchmark_returns_or_reason(bm)
    reasons = [_missing_reason(p.values, p.dates, "values"), bench_reason]
    reason = "; ".join(r for r in reasons if r) or None
    if reason is not None:
        return result_envelope("summarize_beta", a, p, b, {
            "beta": None,
            "alpha": None,
            "correlation": None,
            "covariance": None,
            "benchmark_variance": None,
            "observations": n - 1,
            "reason": reason,
            "params": params,
            "warmup": _warmup({"beta": None, "alpha": None}),
        })

    mean_p = _mean(rets_p)
    mean_b = _mean(rets_b)
    centered_p = rets_p - mean_p
    centered_b = rets_b - mean_b
    m = len(rets_p)
    cov = deterministic_reduction(centered_p * centered_b) / (m - 1)
    var_b = deterministic_reduction(centered_b * centered_b) / (m - 1)
    sd_p = math.sqrt(deterministic_reduction(centered_p * centered_p) / (m - 1))
    sd_b = math.sqrt(var_b)

    beta: Optional[float] = None
    alpha: Optional[float] = None
    if var_b == 0.0:
        reason = (
            "benchmark return variance is 0 (constant benchmark) so the beta denominator is "
            "0; beta is undefined here, not infinite"
        )
    else:
        beta, reason = _ratio_guard(
            cov / var_b, "beta is not finite for this input"
        )
        if beta is not None:
            alpha = mean_p - beta * mean_b
    correlation: Optional[float] = None
    if sd_p == 0.0 or sd_b == 0.0:
        if reason is None:
            reason = (
                "one of the return series has 0 standard deviation so the correlation "
                "denominator is 0; correlation is undefined here, not infinite"
            )
    else:
        correlation, why = _ratio_guard(
            cov / (sd_p * sd_b), "correlation is not finite for this input"
        )
        if reason is None:
            reason = why

    b.add("beta", _num(beta), unit="ratio", params=params)
    b.add("alpha", _num(alpha), unit="per-period decimal", params=params)
    b.add("correlation", _num(correlation), unit="ratio", params=params)
    b.add("benchmark_variance", _num(var_b), unit="decimal", params=params)
    return result_envelope("summarize_beta", a, p, b, {
        "beta": _num(beta),
        "alpha": _num(alpha),
        "correlation": _num(correlation),
        "covariance": _num(cov),
        "benchmark_variance": _num(var_b),
        "observations": m,
        "reason": reason,
        "params": params,
        "warmup": _warmup({"beta": beta, "alpha": alpha, "correlation": correlation}),
    })


# ── 集中度 / 换手率 ──────────────────────────────────────────────


def summarize_concentration(
    dates: Sequence[str],
    weights: Sequence[Optional[float]],
    as_of: str,
    top_n: int = 5,
) -> dict:
    """集中度：HHI = Σ w²、前 top_n 权重占比（按 |w| 取前 top_n，除以总敞口 Σ|w|）与有效持仓数。

    Input: dates (ISO, 严格递增; 与 weights 等长则逐位配对, 长度不等时视为"单一截面 +
      观察窗口", 观察日取 dates[-1]), weights (权重列表, 与 dates 同形; **非负**多头权重,
      null 表缺失), as_of (ISO, 必填; 越界点名首个日期), top_n (取前几名权重, 默认 5,
      超过持仓数时按持仓数)。
    Returns: hhi (Σ w², 全仓单一标的 = 1.0) / effective_n (= 1/HHI, 有效持仓数) /
      top_n_share (前 top_n 权重占比 = Σ 前 top_n / Σw; 单一权重 1.0 → 1.0) / top_n_used /
      weight_count / observation_date / gross_exposure (= Σw, 已投比例) /
      cash_weight (= 1 - Σw, 未投比例; 权重和 > 1 时为负, 是事实标注) /
      weights_long_only (逐位 >= 0 的事实标注) / reason / params / warmup / evidence。
      权重和 <= 0 → ComputeError 点名 (HHI 与占比无从定义); 出现负权重 → 数值键为 null 并附
      reason (集中度定义在多头的权重单纯形上, 带符号向量不出数); 含缺失 → 同样 null + reason。
      无窗口消耗, 无缺失时 warmup = 0。
    """
    a = check_as_of(as_of)
    tn = check_positive_window(top_n, name="top_n")
    s, observation = _prep_weight_vector(dates, weights, a, name="weights")
    w = s.values
    n = len(w)
    params = {"top_n": tn}
    b = EvidenceBuilder("summarize_concentration", a)
    b.add("weight_count", n, unit="positions", params=params)
    b.add("observation_date", observation, params=params)

    reason = _missing_reason(w, s.dates, "weights")
    long_only = True
    if reason is None:
        for i in range(n):
            if float(w[i]) < 0.0:
                long_only = False
                idx = f" at {s.dates[i]}" if len(s.dates) == n else ""
                reason = (
                    f"weights[{i}]{idx} = {float(w[i])!r} is negative; HHI and top-N share are "
                    f"defined on a long-only weight vector (all weights >= 0), so a signed "
                    f"vector yields no number here"
                )
                break

    empty = {
        "hhi": None,
        "effective_n": None,
        "top_n_share": None,
        "top_n_used": None,
        "weight_count": n,
        "observation_date": observation,
        "gross_exposure": None,
        "cash_weight": None,
        "weights_long_only": long_only,
        "reason": reason,
        "params": params,
        "warmup": _warmup({"hhi": None, "top_n_share": None}),
    }
    if reason is not None:
        return result_envelope("summarize_concentration", a, s, b, empty)

    squares = np.empty(n, dtype=np.float64)
    for i in range(n):
        x = float(w[i])
        squares[i] = x * x
    gross = deterministic_reduction(w)
    if gross <= 0.0:
        raise ComputeError(
            f"weights sum to {gross!r} (not > 0); a portfolio with no positive weight has no "
            f"HHI / top-N share to report"
        )
    hhi = deterministic_reduction(squares)

    used = min(tn, n)
    taken = [False] * n
    top_sum = 0.0
    for _ in range(used):
        pick = -1
        for i in range(n):
            if taken[i]:
                continue
            if pick < 0 or float(w[i]) > float(w[pick]):
                pick = i
        taken[pick] = True
        top_sum += float(w[pick])
    top_share = top_sum / gross
    effective_n = 1.0 / hhi

    b.add("hhi", _num(hhi), unit="ratio", params=params)
    b.add("top_n_share", _num(top_share), unit="ratio", params=params)
    b.add("effective_n", _num(effective_n), unit="positions", params=params)
    return result_envelope("summarize_concentration", a, s, b, {
        "hhi": _num(hhi),
        "effective_n": _num(effective_n),
        "top_n_share": _num(top_share),
        "top_n_used": _count(used),
        "weight_count": n,
        "observation_date": observation,
        "gross_exposure": _num(gross),
        "cash_weight": _num(1.0 - gross),
        "weights_long_only": long_only,
        "reason": None,
        "params": params,
        "warmup": _warmup({"hhi": hhi, "top_n_share": top_share, "effective_n": effective_n}),
    })


def summarize_turnover(
    dates: Sequence[str],
    weights: Sequence[Optional[float]],
    as_of: str,
    weights_prior: Optional[Sequence[Optional[float]]] = None,
    symbols: Optional[Sequence[str]] = None,
    symbols_prior: Optional[Sequence[str]] = None,
) -> dict:
    """换手率 = 0.5 × Σ|w_t - w_{t-1}|（两组权重向量逐位之差绝对值和的一半）。

    Input: dates (ISO, 严格递增; 与 weights 等长则逐位配对, 长度不等时视为"单一截面 +
      观察窗口", 观察日取 dates[-1]), weights (本期权重列表; 允许多空/带符号; null 表缺失),
      as_of (ISO, 必填; 越界点名首个日期), weights_prior (上期权重列表, 与 weights 等长;
      省略时按**全零（现金）基线**计算初始建仓换手 = 0.5 × Σ|w_t|, 并在 baseline 键回显该口径),
      symbols / symbols_prior (可选的两期标的代码, 与各自权重等长; 两者都给时才输出 names_traded)。
    Returns: turnover (换手率, 0..1 的倍数, 例如 [1,0] → [0,1] 为 1.0) / baseline ("zero" 或 "explicit") /
      position_count / names_traded (仅当两期 symbols 都给时出现的成交标的数; 不做容差, 逐位不等即计入) /
      reason / params / warmup / evidence。无窗口消耗, 无缺失时 warmup = 0; 含缺失 → turnover 为 null 并附 reason。
    """
    a = check_as_of(as_of)
    cur, observation = _prep_weight_vector(dates, weights, a, name="weights")
    w = cur.values
    n = len(w)
    b = EvidenceBuilder("summarize_turnover", a)
    b.add("position_count", n, unit="positions", params={})
    b.add("observation_date", observation, params={})

    reason = _missing_reason(w, cur.dates, "weights")
    baseline = "zero"
    if weights_prior is None:
        prior = np.zeros(n, dtype=np.float64)
    else:
        baseline = "explicit"
        prior_series, _ = _prep_weight_vector(dates, weights_prior, a, name="weights_prior")
        prior = prior_series.values
        if len(prior) != n:
            raise ComputeError(
                f"weights_prior length {len(prior)} != weights length {n}; "
                f"turnover needs two weight vectors of the same shape"
            )
        if reason is None:
            reason = _missing_reason(prior, prior_series.dates, "weights_prior")

    names_traded: Optional[int] = None
    if (symbols is None) != (symbols_prior is None):
        raise ComputeError(
            "symbols and symbols_prior must be provided together (one of them is missing); "
            "pass both to get names_traded"
        )
    if symbols is not None and symbols_prior is not None:
        if len(symbols) != n:
            raise ComputeError(
                f"symbols length {len(symbols)} != weights length {n}"
            )
        if len(symbols_prior) != len(prior):
            raise ComputeError(
                f"symbols_prior length {len(symbols_prior)} != weights_prior length {len(prior)}"
            )

    if reason is not None:
        return result_envelope("summarize_turnover", a, cur, b, {
            "turnover": None,
            "baseline": baseline,
            "position_count": n,
            "names_traded": None if symbols is None else names_traded,
            "reason": reason,
            "params": {},
            "warmup": _warmup({"turnover": None}),
        })

    if symbols is not None and symbols_prior is not None:
        # Align by instrument, not by position. Two periods holding the same
        # book in a different order are the same portfolio: subtracting position
        # i from position i would report a full round-trip on a book that never
        # traded. Instruments absent from one side count as weight 0 (a new
        # position, or an exit), which is what makes the sum a real turnover.
        aligned = _aligned_turnover(w, symbols, prior, symbols_prior)
        turnover = 0.5 * deterministic_reduction(aligned["deltas"])
        names_traded = aligned["names_traded"]
        turnover_basis = "aligned"
    else:
        deltas = np.empty(n, dtype=np.float64)
        for i in range(n):
            deltas[i] = abs(float(w[i]) - float(prior[i]))
        turnover = 0.5 * deterministic_reduction(deltas)
        if symbols is not None and symbols_prior is not None:
            names_traded = sum(1 for i in range(n) if float(w[i]) != float(prior[i]))
        turnover_basis = "positional"

    payload = {
        "turnover": _num(turnover),
        "baseline": baseline,
        "turnover_basis": turnover_basis,
        "position_count": n,
        "reason": None,
        "params": {"turnover_basis": turnover_basis},
        "warmup": _warmup({"turnover": turnover}),
    }
    if symbols is not None and symbols_prior is not None:
        payload["names_traded"] = _count(names_traded)
    b.add("turnover", _num(turnover), unit="ratio", params={})
    return result_envelope("summarize_turnover", a, cur, b, payload)


TOOLS: list[ToolSpec] = [
    ToolSpec("summarize_return", GROUP, summarize_return),
    ToolSpec("summarize_volatility", GROUP, summarize_volatility),
    ToolSpec("summarize_drawdown", GROUP, summarize_drawdown),
    ToolSpec("summarize_sharpe", GROUP, summarize_sharpe),
    ToolSpec("summarize_sortino", GROUP, summarize_sortino),
    ToolSpec("summarize_calmar", GROUP, summarize_calmar),
    ToolSpec("summarize_information_ratio", GROUP, summarize_information_ratio),
    ToolSpec("summarize_beta", GROUP, summarize_beta),
    ToolSpec("summarize_concentration", GROUP, summarize_concentration),
    ToolSpec("summarize_turnover", GROUP, summarize_turnover),
]
