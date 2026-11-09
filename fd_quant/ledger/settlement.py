"""到期结算的纯函数层：基准映射、序列对齐、覆盖检查、收益与 alpha 算术。

本模块**不碰数据库、不取数、不联网**：标的价格与基准序列全部由调用方传入
（与计算面同一纪律，design.md D2），`store.py` 只负责把结果原子落库。
纯函数的好处是"持有期未满被拒"与"alpha 怎么算"都可以脱离库验证。

三个定义（账本的口径，与批 1.1 组合统计同一收益定义）：

- **持有期按交易日计**：持有窗口 = 对齐后日期轴上从入场日（决策的 `as_of`）
  起的 `holding_days + 1` 个点（入场 + 持有 N 个交易日的收盘 = N 段区间）。
  交易日历不内置——由调用方传入的序列日期集合隐式给出。
- **收益**：`exit_price / entry_price - 1`（简单收益，小数：0.01 = 1%）。
  与 `fd_quant.compute.portfolio.summarize_return` 的 `total_return`
  取同一首末价定义，两者由 tests 对拍钉住（1e-9）。
- **alpha**：`raw_return - benchmark_return`（相对基准，不是年化、不做风险调整）。
"""
from __future__ import annotations

import math
import re
from datetime import date as _date
from typing import Any, Mapping, Optional, Sequence

from ..contracts.models import WEIGHT_SUM_TOLERANCE
from .errors import SettlementRefused

#: 市场后缀 → 基准。可配置（`LedgerStore(benchmark_map=...)`）。
#: `.T→^N225` 沿用 design.md 直接借用的 TradingAgents 映射；`""` 是"无后缀"的
#: 兜底键，也是"映射表里没有这个后缀"时的兜底（无后缀 → SPY）。
DEFAULT_BENCHMARK_MAP: dict[str, str] = {
    ".SS": "000001.SS",   # 上证综指
    ".SZ": "399001.SZ",   # 深证成指
    ".HK": "^HSI",        # 恒生指数
    ".T": "^N225",        # 东证（design.md 的映射表）
    "": "SPY",            # 无市场后缀
}

#: 默认持有期（交易日）。可配置。
DEFAULT_HOLDING_DAYS = 5

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


# ── 校验 ─────────────────────────────────────────────────────────


def is_iso_date(value: Any) -> bool:
    if not isinstance(value, str) or not _DATE_RE.match(value):
        return False
    try:
        _date.fromisoformat(value)
    except ValueError:
        return False
    return True


def check_iso_date(value: Any, *, name: str) -> str:
    if not is_iso_date(value):
        raise SettlementRefused(f"{name} must be an ISO date (YYYY-MM-DD), got {value!r}")
    return str(value)


def check_holding_days(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise SettlementRefused(
            f"holding_days must be a positive integer of trading days, got {value!r}"
        )
    return int(value)


# ── 基准映射 ─────────────────────────────────────────────────────


def resolve_benchmark(
    symbol: str,
    *,
    explicit: Optional[str] = None,
    mapping: Optional[Mapping[str, str]] = None,
) -> dict:
    """基准解析：显式传入永远优先；否则按后缀映射；无后缀（或后缀未登记）→ 默认键。

    返回 `{"benchmark_symbol": ..., "mapping": "explicit" | "suffix:.SS" |
    "default:no-suffix"}`——`mapping` 是"结算结果标注"的来源（可审计：基准是
    调用方点的，还是映射推出来的）。
    """
    if explicit is not None:
        if not isinstance(explicit, str) or not explicit.strip():
            raise SettlementRefused(
                f"benchmark_symbol must be a non-empty string when given, got {explicit!r}"
            )
        return {"benchmark_symbol": explicit.strip(), "mapping": "explicit"}

    table = dict(DEFAULT_BENCHMARK_MAP if mapping is None else mapping)
    text = str(symbol)
    upper = text.upper()
    # 最长后缀优先：`.SZ` 这类更具体的键先于理论上更短的键。
    for key in sorted((k for k in table if k), key=len, reverse=True):
        if upper.endswith(key.upper()):
            return {"benchmark_symbol": table[key], "mapping": f"suffix:{key}"}
    if "" in table:
        return {"benchmark_symbol": table[""], "mapping": "default:no-suffix"}
    raise SettlementRefused(
        f"no benchmark mapping for symbol {symbol!r} (mapping table has no suffix match "
        f"and no default '' entry); pass benchmark_symbol explicitly"
    )


# ── 序列与对齐 ───────────────────────────────────────────────────


def build_series(
    dates: Sequence[Any], values: Sequence[Any], *, name: str
) -> list[tuple[str, Optional[float]]]:
    """日期与价格逐位配对校验（ISO、严格递增、数值或 None）。缺失保留为 None。

    缺失不在这里拒绝：窗口**外**的缺失与结算无关，只有进入算术的点才必须完整
    （见 `settlement_window` 的端点检查）。
    """
    if dates is None or values is None:
        raise SettlementRefused(f"{name}: dates and values are both required")
    dates = list(dates)
    values = list(values)
    if len(dates) != len(values):
        raise SettlementRefused(
            f"{name}: dates and values length mismatch ({len(dates)} vs {len(values)})"
        )
    if not dates:
        raise SettlementRefused(
            f"{name} is empty; the settlement window cannot be covered by zero points"
        )
    out: list[tuple[str, Optional[float]]] = []
    prev: Optional[str] = None
    for i, d in enumerate(dates):
        if not is_iso_date(d):
            raise SettlementRefused(f"{name}: dates[{i}] is not an ISO date: {d!r}")
        d = str(d)
        if prev is not None and d <= prev:
            raise SettlementRefused(
                f"{name}: dates must be strictly increasing; {prev} -> {d} at index {i}"
            )
        prev = d
        v = values[i]
        if v is None:
            out.append((d, None))
            continue
        if isinstance(v, bool):
            raise SettlementRefused(f"{name}: values[{i}] at {d} is not numeric: {v!r}")
        try:
            f = float(v)
        except (TypeError, ValueError):
            raise SettlementRefused(
                f"{name}: values[{i}] at {d} is not numeric: {v!r}"
            ) from None
        if not math.isfinite(f):
            raise SettlementRefused(f"{name}: values[{i}] at {d} is not finite: {v!r}")
        out.append((d, f))
    return out


def _refuse_future(dates_and_values: list[tuple[str, Optional[float]]], *, name: str, as_of: str) -> None:
    """晚于 as_of 的数据点拒绝并点名首个越界日期（不静默截断，与计算面同款）。"""
    for d, _v in dates_and_values:
        if d > as_of:
            raise SettlementRefused(
                f"{name}: data point {d} is after as_of={as_of} "
                f"(future data refused; settlement only sees prices up to as_of)"
            )


def _pick_point(
    series: Mapping[str, Optional[float]], date: str, *, name: str, role: str
) -> float:
    if date not in series:
        raise SettlementRefused(
            f"{name}: no price at {date} ({role}); the settlement needs that exact date "
            f"on the aligned axis"
        )
    v = series[date]
    if v is None:
        raise SettlementRefused(
            f"{name}: missing (null) price at {date} ({role}); a null cannot enter the "
            f"return arithmetic"
        )
    return float(v)


def simple_return(entry_price: Any, exit_price: Any, *, name: str) -> float:
    """`exit/entry - 1`（小数）。非正价格拒绝（点名），非有限结果拒绝。"""
    for label, value in (("entry", entry_price), ("exit", exit_price)):
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise SettlementRefused(f"{name}: {label}_price must be a number, got {value!r}")
        if not math.isfinite(float(value)):
            raise SettlementRefused(f"{name}: {label}_price must be finite, got {value!r}")
        if float(value) <= 0.0:
            raise SettlementRefused(
                f"{name}: {label}_price must be > 0 for a simple return, got {value!r}"
            )
    out = float(exit_price) / float(entry_price) - 1.0
    if not math.isfinite(out):
        raise SettlementRefused(
            f"{name}: return is not finite ({exit_price!r}/{entry_price!r} - 1)"
        )
    return out


def settlement_window(
    *,
    entry_date: str,
    symbol_dates: Sequence[Any],
    symbol_values: Sequence[Any],
    benchmark_dates: Sequence[Any],
    benchmark_values: Sequence[Any],
    as_of: str,
    holding_days: int = DEFAULT_HOLDING_DAYS,
    benchmark_symbol: str,
) -> dict:
    """对齐两序列并算出持有窗口内的原始收益、基准收益与 alpha。

    步骤与拒绝条件（每条都点名缺什么，供调用方在同一轮里修）：

    1. 两条序列各自校验（ISO、严格递增、数值/None），且**不得含晚于 `as_of` 的点**。
    2. 日期轴 = 标的口期 ∩ 基准日期（对齐：只有两边都有价的日子才算交易日）。
    3. 入场日 = 决策的 `as_of`，必须落在对齐轴上（不在 → 拒绝，不静默顺延入场）。
    4. 覆盖检查：从入场日到对齐轴末尾的可用点数 ≥ `holding_days + 1`。
       不足 → 拒绝并报缺口交易日数（`missing_trading_days`）。
    5. 出场日 = 入场日在轴上的第 `holding_days` 个后继；端点价格必须存在且 > 0。

    持有期按**交易日**计：交易日历由调用方序列的日期集合隐式给出。
    """
    a = check_iso_date(as_of, name="as_of")
    ed = check_iso_date(entry_date, name="entry_date")
    hd = check_holding_days(holding_days)
    if not isinstance(benchmark_symbol, str) or not benchmark_symbol.strip():
        raise SettlementRefused("benchmark_symbol must be a non-empty string")

    sym = build_series(symbol_dates, symbol_values, name="symbol_prices")
    ben = build_series(benchmark_dates, benchmark_values, name="benchmark_prices")
    _refuse_future(sym, name="symbol_prices", as_of=a)
    _refuse_future(ben, name="benchmark_prices", as_of=a)

    sym_map = dict(sym)
    ben_map = dict(ben)
    axis = sorted(set(sym_map) & set(ben_map))

    missing_legs = [nm for nm, m in (("symbol_prices", sym_map), ("benchmark_prices", ben_map))
                    if ed not in m]
    if missing_legs:
        later = [d for d in axis if d >= ed]
        raise SettlementRefused(
            f"entry date {ed} (the decision's as_of) is not present in "
            f"{' and '.join(missing_legs)}; the aligned axis runs "
            f"{axis[0] if axis else '<empty>'} .. {axis[-1] if axis else '<empty>'} "
            f"(first aligned date on or after the entry date: "
            f"{later[0] if later else 'none'}). The ledger does not shift the entry date "
            f"silently — supply a series that covers the decision's as_of"
        )

    i0 = axis.index(ed)
    available = len(axis) - i0
    required = hd + 1  # 入场点 + 持有期内的 hd 个交易日
    if available < required:
        last = axis[-1]
        raise SettlementRefused(
            f"holding period not covered: entry {ed} + {hd} trading days needs "
            f"{required} aligned prices (symbol ∩ benchmark), got {available} "
            f"(aligned axis ends {last}); missing {required - available} trading days",
            missing_trading_days=required - available,
        )

    exit_date = axis[i0 + hd]
    entry_price = _pick_point(sym_map, ed, name="symbol_prices", role="entry")
    exit_price = _pick_point(sym_map, exit_date, name="symbol_prices", role="exit")
    bench_entry = _pick_point(ben_map, ed, name="benchmark_prices", role="entry")
    bench_exit = _pick_point(ben_map, exit_date, name="benchmark_prices", role="exit")

    raw_return = simple_return(entry_price, exit_price, name="symbol_prices")
    benchmark_return = simple_return(bench_entry, bench_exit, name="benchmark_prices")
    alpha = raw_return - benchmark_return
    if not math.isfinite(alpha):
        raise SettlementRefused(f"alpha is not finite ({raw_return!r} - {benchmark_return!r})")

    return {
        "entry_date": ed,
        "exit_date": exit_date,
        "holding_days": hd,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "raw_return": raw_return,
        "benchmark_symbol": benchmark_symbol.strip(),
        "benchmark_entry_price": bench_entry,
        "benchmark_exit_price": bench_exit,
        "benchmark_return": benchmark_return,
        "alpha": alpha,
        "aligned_points_available": available,
    }


def portfolio_settlement_window(
    *,
    entry_date: str,
    legs: Mapping[str, tuple[Sequence[Any], Sequence[Any]]],
    weights: Mapping[str, float],
    benchmark_dates: Sequence[Any],
    benchmark_values: Sequence[Any],
    as_of: str,
    holding_days: int = DEFAULT_HOLDING_DAYS,
    benchmark_symbol: str,
) -> dict:
    """组合窗口：`Σ w_i · r_i` 与现金部分，日期轴取**全部标的与基准的交集**。

    ## 为什么轴必须取交集，而不是每条腿各算各的

    若各腿用各自的交易日轴，某只停牌股的出场日就会与其他腿不同——那样算出来的"组合收益"
    是几段不同持有期收益的加权和，它在数学上不是一个组合在某一期的收益。故：轴 = 全部标的
    日期 ∩ 基准日期，入场日必须落在轴上，出场日是轴上入场日的第 `holding_days` 个后继。

    ## 拒绝语义（asm task 4.2：整期原子）

    任一腿在轴上不足以覆盖持有期 → **整次拒绝**，并点名该腿与缺口交易日数。不接受"先用齐的
    腿算个部分收益"——部分结算出的 alpha 与完整组合的 alpha 长得一样，而这个项目最贵的错
    就是"两套口径长得一样"。
    """
    a = check_iso_date(as_of, name="as_of")
    ed = check_iso_date(entry_date, name="entry_date")
    hd = check_holding_days(holding_days)
    if not isinstance(benchmark_symbol, str) or not benchmark_symbol.strip():
        raise SettlementRefused("benchmark_symbol must be a non-empty string")
    if not legs:
        raise SettlementRefused("legs: at least one symbol series is required")

    ben = build_series(benchmark_dates, benchmark_values, name="benchmark_prices")
    _refuse_future(ben, name="benchmark_prices", as_of=a)
    ben_map = dict(ben)

    series_maps: dict[str, dict[str, Optional[float]]] = {}
    for sym, (dates, values) in legs.items():
        if not isinstance(sym, str) or not sym.strip():
            raise SettlementRefused(f"legs: symbol keys must be non-empty strings (got {sym!r})")
        s = build_series(dates, values, name=f"legs[{sym}]")
        _refuse_future(s, name=f"legs[{sym}]", as_of=a)
        series_maps[sym] = dict(s)

    # 权重必须覆盖全部腿，且只覆盖它们：多一条腿或权重对不上都是输入错误，不静默补 0。
    for sym in series_maps:
        if sym not in weights:
            raise SettlementRefused(
                f"weights: missing weight for leg {sym!r}; refusing to assume 0 "
                "(an unweighted leg means the recorded intent did not hold it)"
            )
    for sym in weights:
        if sym not in series_maps:
            raise SettlementRefused(
                f"legs: missing price series for weighted symbol {sym!r}"
            )

    axis_sets = [set(ben_map)] + [set(m) for m in series_maps.values()]
    axis = sorted(set.intersection(*axis_sets))
    if ed not in axis:
        raise SettlementRefused(
            f"entry_date {ed} is not on the aligned trading axis (no price for every leg "
            f"and the benchmark that day); refusing to slide the entry date"
        )
    idx = axis.index(ed)
    if len(axis) - idx - 1 < hd:
        raise SettlementRefused(
            f"insufficient coverage: need {hd + 1} aligned points from {ed}, "
            f"got {len(axis) - idx} (missing_trading_days={hd - (len(axis) - idx - 1)})"
        )
    exit_date = axis[idx + hd]

    leg_returns: dict[str, float] = {}
    for sym, m in series_maps.items():
        entry_px, exit_px = m.get(ed), m.get(exit_date)
        if entry_px is None or entry_px <= 0:
            raise SettlementRefused(f"legs[{sym}]: entry price at {ed} is missing or non-positive")
        if exit_px is None or exit_px <= 0:
            raise SettlementRefused(
                f"legs[{sym}]: exit price at {exit_date} is missing or non-positive"
            )
        leg_returns[sym] = exit_px / entry_px - 1.0

    ben_entry, ben_exit = ben_map[ed], ben_map[exit_date]
    if ben_entry is None or ben_entry <= 0 or ben_exit is None or ben_exit <= 0:
        raise SettlementRefused(
            f"benchmark_prices: endpoints at {ed} / {exit_date} must be positive"
        )
    benchmark_return = ben_exit / ben_entry - 1.0

    invested = sum(float(w) for w in weights.values())
    if invested > 1 + WEIGHT_SUM_TOLERANCE:
        raise SettlementRefused(
            f"weights: Σweights={invested} exceeds 1; rejected, not normalised"
        )
    # 12 位舍入：与权重分配同一把尺，避免 0.19999999999999996 这类尾巴进入报告。
    invested = round(invested, 12)
    cash_weight = round(max(0.0, 1.0 - invested), 12)
    gross = sum(float(weights[sym]) * r for sym, r in leg_returns.items())

    return {
        "entry_date": ed,
        "exit_date": exit_date,
        "holding_days": hd,
        "leg_returns": leg_returns,
        "weights": {k: float(v) for k, v in weights.items()},
        "invested_weight": invested,
        # 现金部分收益记零并单独报出——不摊进标的收益（摊进去就等于给现金编了个收益）。
        "cash_weight": cash_weight,
        "cash_return": 0.0,
        "raw_return": gross,
        "benchmark_symbol": benchmark_symbol,
        "benchmark_entry_price": ben_entry,
        "benchmark_exit_price": ben_exit,
        "benchmark_return": benchmark_return,
        "alpha": gross - benchmark_return,
        "aligned_points": len(axis),
    }
