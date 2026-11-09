"""组合级风控闸（openspec change: add-investment-trading-fleet task 3；change 4 spec portfolio-risk-gate）。

## 为什么只有一份

回测、纸面运行、执行岗硬闸三处都必须调这个模块。判据一旦复制成三份，漂移的那份永远是
没人重读的那份——于是"为什么这笔单被拦"在不同入口会得到不同答案，而报告上看不出来。

## 为什么"不可用"是第三个值，不是 false

四阈值任一未配置 → 返回 `unavailable` 并点名缺失项。把"判据不可用"折叠成 `breached=false`，
系统就会在完全不知道风险偏好的情况下照常下单。这是本仓最贵的一类错：**读不到被当成不存在**。

## 为什么判据不自行处置

判据只回答"触达了什么"，不回答"该怎么办"。回测里触达可能是"下一 bar 调仓"，执行里是
"拒单并升级"，压测里是"记录并继续"。把处置塞进判据 = 把三套语义焊死成一套。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

#: 四个阈值键。缺任一项 → 判据不可用（不是"该项通过"）。
THRESHOLD_KEYS = (
    "per_symbol_max",
    "industry_max",
    "drawdown_max",
    "daily_loss_max",
)

STATUS_OK = "ok"
STATUS_BREACHED = "breached"
STATUS_UNAVAILABLE = "unavailable"

_EPS = 1e-12


class RiskGateError(ValueError):
    """闸的输入本身不合法（不是"触达"，是问不出答案）。"""


@dataclass(frozen=True)
class Thresholds:
    """四个阈值 + 价格新鲜度上限。全部必填——没有"安全的默认值"这回事。"""

    per_symbol_max: float
    industry_max: float
    drawdown_max: float
    daily_loss_max: float
    #: 允许的行情陈旧天数。缺省 None = 不做新鲜度判定（显式选择不判，而不是忘了判）。
    max_staleness_days: Optional[int] = None

    @classmethod
    def from_mapping(cls, raw: Optional[Mapping[str, Any]]) -> "Thresholds | dict":
        """从配置映射构造。缺失项返回 `{"status": unavailable, "missing": [...]}`。"""
        if not raw:
            return {"status": STATUS_UNAVAILABLE, "missing": list(THRESHOLD_KEYS),
                    "reason": "no thresholds configured"}
        missing = [k for k in THRESHOLD_KEYS if raw.get(k) is None]
        if missing:
            return {"status": STATUS_UNAVAILABLE, "missing": missing,
                    "reason": "threshold(s) not configured; defaults are not assumed"}
        vals: dict[str, float] = {}
        for k in THRESHOLD_KEYS:
            v = raw[k]
            if not isinstance(v, (int, float)) or isinstance(v, bool):
                raise RiskGateError(f"{k} must be a number, got {v!r}")
            if not (0 < float(v) <= 1):
                raise RiskGateError(f"{k} must be in (0, 1], got {v!r}")
            vals[k] = float(v)
        stale = raw.get("max_staleness_days")
        if stale is not None and (not isinstance(stale, int) or isinstance(stale, bool) or stale < 0):
            raise RiskGateError(f"max_staleness_days must be a non-negative int, got {stale!r}")
        return cls(max_staleness_days=stale, **vals)


def _item(kind: str, actual: float, limit: float, subject: Optional[str] = None,
          note: Optional[str] = None) -> dict:
    out = {"kind": kind, "actual": actual, "limit": limit}
    if subject is not None:
        out["subject"] = subject
    if note is not None:
        out["note"] = note
    return out


def _undetermined(kind: str, reason: str, subjects: Sequence[str] = ()) -> dict:
    return {"kind": kind, "undetermined": True, "reason": reason, "subjects": list(subjects)}


def _drawdown(equity_series: Sequence[Sequence[Any]]) -> tuple[float, Optional[str]]:
    """峰值到当前的相对回撤（正数）。返回 (回撤, 峰值日)。"""
    peak = None
    peak_day = None
    worst = 0.0
    for row in equity_series:
        day, value = row[0], float(row[1])
        if peak is None or value > peak:
            peak, peak_day = value, str(day)
        if peak and peak > 0:
            dd = (peak - value) / peak
            if dd > worst:
                worst = dd
    return worst, peak_day


def _last_day_loss(equity_series: Sequence[Sequence[Any]]) -> tuple[float, Optional[str]]:
    """最后一个交易日的跌幅（正数）。返回 (跌幅, 该日)。"""
    if len(equity_series) < 2:
        return 0.0, None
    prev = float(equity_series[-2][1])
    cur = float(equity_series[-1][1])
    if prev <= 0:
        return 0.0, None
    return max(0.0, (prev - cur) / prev), str(equity_series[-1][0])


def evaluate_portfolio_gate(
    *,
    weights: Mapping[str, float],
    thresholds: Thresholds | Mapping[str, Any] | None,
    sectors: Optional[Mapping[str, str]] = None,
    equity_series: Optional[Sequence[Sequence[Any]]] = None,
    as_of: Optional[str] = None,
    price_as_of_max: Optional[str] = None,
) -> dict:
    """四阈值判定。纯函数：不取数、不读库、不联网、不调模型。

    Args:
        weights: `{symbol: weight}`，目标权重（非负、Σ ≤ 1）。
        thresholds: 四阈值；``None`` 或残缺 → 返回 ``status="unavailable"`` 并点名缺失项。
        sectors: `{symbol: 行业}`。缺某只标的的归属时，行业集中度报"不可判定"并点名，
            **不**把它归入某个默认行业。
        equity_series: `[(日期, 权益)]`，用于回撤与单日亏损。
        as_of: 本次决策日；与 `price_as_of_max` 一起判定行情陈旧。
        price_as_of_max: 行情水位。两者都给且差距 > `max_staleness_days` → 触达。

    Returns:
        ``{"status": ok|breached|unavailable, "items": [...], "undetermined": [...],
        "inputs": {...}}``。``status`` 是三态的：不可用绝不等同于通过。
    """
    # 1) 权重合法性：拒绝而不归一化（归一化等于替调用方改仓位）
    if not isinstance(weights, Mapping):
        raise RiskGateError("weights must be an object of symbol → weight")
    total = 0.0
    for sym, w in weights.items():
        if not isinstance(w, (int, float)) or isinstance(w, bool):
            raise RiskGateError(f"weights.{sym} must be a number, got {w!r}")
        if float(w) < 0:
            raise RiskGateError(f"weights.{sym} is negative ({w}); rejected, not normalised")
        total += float(w)
    if total > 1 + 1e-9:
        raise RiskGateError(
            f"Σweights={total} exceeds 1; rejected, not normalised"
        )

    # 2) 阈值：不可用即返回，不算"通过"
    if isinstance(thresholds, Thresholds):
        th: Thresholds = thresholds
    else:
        built = Thresholds.from_mapping(thresholds)
        if isinstance(built, dict):
            return {
                "status": STATUS_UNAVAILABLE,
                # breached=None 而不是 False：三态才是本模块的表达力。
                # 折叠成 false 会让"问不出答案"看起来像"问过了没触达"。
                "breached": None,
                "items": [], "undetermined": [],
                "inputs": {"missing": built.get("missing"), "reason": built.get("reason")},
            }
        th = built

    items: list[dict] = []
    undetermined: list[dict] = []

    # 3) 单票集中度
    if weights:
        sym, w = max(weights.items(), key=lambda kv: float(kv[1]))
        if float(w) > th.per_symbol_max + _EPS:
            items.append(_item("per_symbol", float(w), th.per_symbol_max, subject=sym))

    # 4) 行业集中度：缺归属即不可判定，不归并到默认行业
    if sectors is not None and weights:
        agg: dict[str, float] = {}
        unknown: list[str] = []
        for sym, w in weights.items():
            sector = sectors.get(sym)
            if not sector:
                unknown.append(sym)
                continue
            agg[sector] = agg.get(sector, 0.0) + float(w)
        if agg:
            sector, wsum = max(agg.items(), key=lambda kv: kv[1])
            if wsum > th.industry_max + _EPS:
                items.append(_item("industry_concentration", wsum, th.industry_max,
                                   subject=sector))
        if unknown:
            undetermined.append(_undetermined(
                "industry_concentration",
                "sector unknown for these symbols; not folded into any default sector",
                unknown,
            ))
    elif weights:
        undetermined.append(_undetermined(
            "industry_concentration", "no sector map supplied", sorted(weights),
        ))

    # 5) 组合回撤与单日亏损（需要权益序列；缺就不可判定，不当作 0）
    if equity_series:
        dd, peak_day = _drawdown(equity_series)
        if dd > th.drawdown_max + _EPS:
            items.append(_item("drawdown", dd, th.drawdown_max, note=f"peak on {peak_day}"))
        loss, day = _last_day_loss(equity_series)
        if loss > th.daily_loss_max + _EPS:
            items.append(_item("daily_loss", loss, th.daily_loss_max, note=f"on {day}"))
    else:
        undetermined.append(_undetermined(
            "drawdown/daily_loss", "no equity series supplied", [],
        ))

    # 6) 行情陈旧：只有两端日期都给且显式配了上限才判
    if th.max_staleness_days is not None and as_of and price_as_of_max:
        from datetime import date as _d

        try:
            gap = (_d.fromisoformat(as_of) - _d.fromisoformat(price_as_of_max)).days
        except ValueError as exc:
            raise RiskGateError(f"as_of / price_as_of_max must be ISO dates: {exc}") from exc
        if gap > th.max_staleness_days:
            items.append(_item("price_staleness", float(gap), float(th.max_staleness_days),
                               subject=price_as_of_max,
                               note=(f"as_of={as_of} vs price_as_of_max={price_as_of_max}"
                                     f"（差 {gap} 天）")))
    elif th.max_staleness_days is not None:
        undetermined.append(_undetermined(
            "price_staleness", "as_of or price_as_of_max not supplied", [],
        ))

    status = STATUS_BREACHED if items else STATUS_OK
    return {
        "status": status,
        # 与 spec 的 `{breached, items}` 形状对齐；同时保留 status 供三态判定。
        # 两者恒等：breached ⟺ status=="breached"。
        "breached": status == STATUS_BREACHED,
        "items": items,
        "undetermined": undetermined,
        "inputs": {
            "invested_weight": round(total, 12),
            "symbols": sorted(weights),
            "thresholds": {
                "per_symbol_max": th.per_symbol_max,
                "industry_max": th.industry_max,
                "drawdown_max": th.drawdown_max,
                "daily_loss_max": th.daily_loss_max,
                "max_staleness_days": th.max_staleness_days,
            },
        },
    }
