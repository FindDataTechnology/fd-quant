"""技术指标族：SMA / EMA / RSI / MACD / 布林 / ATR / KDJ / OBV / CCI / DMI。

自实现（不依赖 ta-lib 等外部指标库）。定义按业界通行惯例：RSI 与 ATR 用
Wilder 平滑、KDJ 默认 9,3,3、MACD 默认 12,26,9、布林默认 20,2。

**warmup 纪律**：任何指标在样本不足处的值一律为 `null`，绝不用 0 或前值填充。
每个指标在返回里声明 `warmup`（逐输出线：首个非 null 位置之前有多少个 null）
与 `params`（实际生效的参数）——调用方据此知道"从第几位起数字才是数的"。

**NaN 纪律**：输入含 NaN（缺失）时按业界惯例**局部传播**——含 NaN 的窗口
输出 null，窗口外的样本不受污染；递推类指标（EMA/RSI/ATR/DMI/ADX）自首个
NaN 起其后全为 null（不跳过、不填充：跳过等于替调用方发明一个数据点）。

**确定性**：归约只用 `np.cumsum` 顺序扫描或显式 Kahan（`deterministic_reduction`），
不使用 BLAS 矩阵乘法做数值归约——那是唯一随线程数变化的路径。
"""
from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np

from ..contracts.models import Evidence  # noqa: F401  (形状来源，供阅读)
from . import ToolSpec
from .base import (
    EvidenceBuilder,
    Series,
    check_as_of,
    check_positive_window,
    check_series,
    deterministic_reduction,
    result_envelope,
    to_none_list,
)

GROUP = "technical_indicators"


# ── 内部工具 ─────────────────────────────────────────────────────


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    """滚动均值，NaN 局部传播（含 NaN 的窗口 → NaN）。

    用 cumsum 做顺序扫描并扣除窗口前的累积和；`bad` 计数保证"窗口内有 NaN
    就出 NaN，窗口外有 NaN 不影响"。不用 pandas rolling 是为了不引入
    pandas 的版本语义（ddof / min_periods）到定义里。
    """
    n = len(values)
    out = np.full(n, np.nan)
    if window > n:
        return out
    mask = np.isnan(values)
    filled = np.where(mask, 0.0, values)
    cs = np.concatenate(([0.0], np.cumsum(filled)))
    bad = np.concatenate(([0], np.cumsum(mask.astype(np.int64))))
    for i in range(window - 1, n):
        if bad[i + 1] - bad[i + 1 - window] == 0:
            out[i] = (cs[i + 1] - cs[i + 1 - window]) / window
    return out


def _ema(values: np.ndarray, window: int) -> np.ndarray:
    """指数均值：起始处取前 window 个**全有限**样本的简单均值作种子（TA-Lib 惯例），
    其后 `e = a*x + (1-a)*e_prev`，`a = 2/(window+1)`。

    允许前导 NaN：起始位置取"其前 window 个样本全有限"之处，因此
    `_ema(dif, 9)` 这种输入本身带 warmup 洞的调用能正确起步（MACD 的 DEA
    就是这一类）。首个 NaN 之后一律保持 NaN，不跳过——跳过等于替调用方
    发明一个数据点。
    """
    n = len(values)
    out = np.full(n, np.nan)
    start: Optional[int] = None
    for i in range(n):
        if i + 1 >= window and not np.isnan(values[i + 1 - window:i + 1]).any():
            start = i
            break
    if start is None:
        return out
    out[start] = deterministic_reduction(values[start + 1 - window:start + 1]) / window
    alpha = 2.0 / (window + 1.0)
    for i in range(start + 1, n):
        if np.isnan(values[i]):
            break  # 其后保持 NaN（向前传播）
        out[i] = alpha * values[i] + (1.0 - alpha) * out[i - 1]
    return out


def _wilder_avg(values: np.ndarray, window: int) -> np.ndarray:
    """Wilder 递推均值：首个值 = 前 window 个**全部有限**样本的均值，
    其后 `avg = (avg*(window-1) + x)/window`。

    入参前部允许 NaN（如 TR[0] 没有前收），首个可算位置是"其前 window 个样本
    全有限"之处。RSI / ATR / DMI / ADX 共用这一条递推，定义只有一个。
    """
    n = len(values)
    out = np.full(n, np.nan)
    start: Optional[int] = None
    for i in range(n):
        if i + 1 >= window and not np.isnan(values[i + 1 - window:i + 1]).any():
            start = i
            break
    if start is None:
        return out
    out[start] = deterministic_reduction(values[start + 1 - window:start + 1]) / window
    for i in range(start + 1, n):
        if np.isnan(values[i]):
            break
        out[i] = (out[i - 1] * (window - 1) + values[i]) / window
    return out


def _leading_nulls(arr: np.ndarray) -> int:
    """首个非 NaN 位置之前的 null 个数（全 NaN → len）。"""
    finite = np.flatnonzero(~np.isnan(arr))
    return int(finite[0]) if len(finite) else int(len(arr))


def _prep(dates, values, as_of, name="close") -> Series:
    return check_series(dates, values, as_of=as_of, name=name)


def _prep_ohlc(dates, high, low, close, as_of) -> tuple[Series, Series, Series]:
    """HLC 三列对齐校验：日期共用一份（避免同一天被读两次）。

    只校验 high/low/close——本族的 ATR/KDJ/CCI/DMI 都不需要开盘价，
    多收一列会让"输入形状"的文档与实现不一致。
    """
    h = check_series(dates, high, as_of=as_of, name="high")
    l_ = check_series(dates, low, as_of=as_of, name="low")
    c = check_series(dates, close, as_of=as_of, name="close")
    return h, l_, c


def _emit_line(
    b: EvidenceBuilder, name: str, arr: np.ndarray, params: dict, unit: Optional[str] = None
) -> None:
    """序列证据：长度 + 首值 + 末值（不逐点出证据——点数不是信息，首末才是）。"""
    finite = np.flatnonzero(~np.isnan(arr))
    b.add(f"{name} length", int(len(arr)), unit="bars", params=params)
    if len(finite) == 0:
        b.add(f"{name} all_null", True, unit=None, params=params)
        return
    i0, i1 = int(finite[0]), int(finite[-1])
    b.add(f"{name} first", float(arr[i0]), unit=unit, params=params)
    b.add(f"{name} last", float(arr[i1]), unit=unit, params=params)


def _envelope(
    tool: str,
    as_of: str,
    series: Series,
    b: EvidenceBuilder,
    payload: dict,
) -> dict:
    return result_envelope(tool, as_of, series, b, payload)


# ── SMA / EMA ────────────────────────────────────────────────────


def compute_sma(
    dates: Sequence[str],
    values: Sequence[Optional[float]],
    as_of: str,
    window: int = 20,
) -> dict:
    """简单移动平均。

    Input: dates (ISO, 严格递增), values (与 dates 等长; null 表缺失), as_of (ISO, 必填), window (默认 20)。
    Returns: value (与输入等长的列表; 前 window-1 位为 null), plus params, warmup, evidence。
    """
    a = check_as_of(as_of)
    w = check_positive_window(window, name="window")
    s = _prep(dates, values, a)
    arr = _rolling_mean(s.values, w)
    b = EvidenceBuilder("compute_sma", a)
    _emit_line(b, "value", arr, {"window": w})
    return _envelope("compute_sma", a, s, b, {
        "value": to_none_list(arr),
        "params": {"window": w},
        "warmup": {"value": _leading_nulls(arr)},
    })


def compute_ema(
    dates: Sequence[str],
    values: Sequence[Optional[float]],
    as_of: str,
    window: int = 20,
) -> dict:
    """指数移动平均（前 window 个样本的简单均值作种子，其后 a=2/(window+1) 递推）。

    Input: dates, values, as_of, window (默认 20)。
    Returns: value (前 window-1 位 null; 首个非 null 在索引 window-1), params, warmup, evidence。
    """
    a = check_as_of(as_of)
    w = check_positive_window(window, name="window")
    s = _prep(dates, values, a)
    arr = _ema(s.values, w)
    b = EvidenceBuilder("compute_ema", a)
    _emit_line(b, "value", arr, {"window": w})
    return _envelope("compute_ema", a, s, b, {
        "value": to_none_list(arr),
        "params": {"window": w},
        "warmup": {"value": _leading_nulls(arr)},
    })


# ── RSI / MACD ───────────────────────────────────────────────────


def compute_rsi(
    dates: Sequence[str],
    values: Sequence[Optional[float]],
    as_of: str,
    window: int = 14,
) -> dict:
    """相对强弱指标（Wilder 平滑，非简单均值版）。

    Input: dates, values, as_of, window (默认 14)。
    Returns: value (前 window 位 null), params, warmup, evidence。只涨序列趋近 100、只跌趋近 0。
    """
    a = check_as_of(as_of)
    w = check_positive_window(window, name="window")
    s = _prep(dates, values, a)
    v = s.values
    n = len(v)
    gains = np.full(n, np.nan)
    losses = np.full(n, np.nan)
    with np.errstate(invalid="ignore"):
        delta = np.diff(v)
    gains[1:] = np.maximum(delta, 0.0)
    losses[1:] = np.maximum(-delta, 0.0)
    avg_gain = _wilder_avg(gains, w)
    avg_loss = _wilder_avg(losses, w)

    out = np.full(n, np.nan)
    for i in range(n):
        g, l_ = avg_gain[i], avg_loss[i]
        if np.isnan(g) or np.isnan(l_):
            continue
        if l_ == 0.0:
            out[i] = 100.0 if g > 0.0 else 50.0  # 无涨无跌 → 中性
        else:
            rs = g / l_
            out[i] = 100.0 - 100.0 / (1.0 + rs)

    b = EvidenceBuilder("compute_rsi", a)
    _emit_line(b, "value", out, {"window": w})
    return _envelope("compute_rsi", a, s, b, {
        "value": to_none_list(out),
        "params": {"window": w},
        "warmup": {"value": _leading_nulls(out)},
    })


def compute_macd(
    dates: Sequence[str],
    values: Sequence[Optional[float]],
    as_of: str,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    histogram_scale: float = 1.0,
) -> dict:
    """MACD 三线：DIF = EMA(fast) - EMA(slow)，DEA = EMA(signal) of DIF，柱 = (DIF-DEA)*histogram_scale。

    Input: dates, values, as_of, fast (默认 12), slow (默认 26), signal (默认 9),
      histogram_scale (默认 1.0 = 国际惯例；A 股常见软件用 2.0，显式传入即可)。
    Returns: dif / dea / histogram 三条等长列表 + params + warmup (逐线) + evidence。
      EMA 用简单均值种子，故 DIF 自索引 slow-1、DEA 自 slow+signal-2 起非 null。
    """
    a = check_as_of(as_of)
    f = check_positive_window(fast, name="fast")
    sl = check_positive_window(slow, name="slow")
    sg = check_positive_window(signal, name="signal")
    if f >= sl:
        from .base import ComputeError

        raise ComputeError(f"fast ({f}) must be < slow ({sl})")
    scale = float(histogram_scale)
    if not np.isfinite(scale):
        from .base import ComputeError

        raise ComputeError(f"histogram_scale must be finite, got {histogram_scale!r}")

    s = _prep(dates, values, a)
    ema_f = _ema(s.values, f)
    ema_s = _ema(s.values, sl)
    dif = ema_f - ema_s
    dea = _ema(dif, sg)
    hist = (dif - dea) * scale

    params = {"fast": f, "slow": sl, "signal": sg, "histogram_scale": scale}
    b = EvidenceBuilder("compute_macd", a)
    for name, arr in (("dif", dif), ("dea", dea), ("histogram", hist)):
        _emit_line(b, name, arr, params)
    return _envelope("compute_macd", a, s, b, {
        "dif": to_none_list(dif),
        "dea": to_none_list(dea),
        "histogram": to_none_list(hist),
        "params": params,
        "warmup": {
            "dif": _leading_nulls(dif),
            "dea": _leading_nulls(dea),
            "histogram": _leading_nulls(hist),
        },
    })


# ── 布林带 ───────────────────────────────────────────────────────


def compute_bollinger(
    dates: Sequence[str],
    values: Sequence[Optional[float]],
    as_of: str,
    window: int = 20,
    num_std: float = 2.0,
) -> dict:
    """布林带：中轨 = SMA(window)，上下轨 = 中轨 ± num_std × 总体标准差(窗口内, ddof=0)。

    Input: dates, values, as_of, window (默认 20), num_std (默认 2.0)。
    Returns: upper / middle / lower 三条等长列表 + params + warmup (逐线) + evidence。
    """
    a = check_as_of(as_of)
    w = check_positive_window(window, name="window")
    k = float(num_std)
    if not np.isfinite(k) or k < 0:
        from .base import ComputeError

        raise ComputeError(f"num_std must be a finite non-negative number, got {num_std!r}")

    s = _prep(dates, values, a)
    v = s.values
    n = len(v)
    mid = _rolling_mean(v, w)
    sd = np.full(n, np.nan)
    for i in range(w - 1, n):
        win = v[i + 1 - w:i + 1]
        if np.isnan(win).any():
            continue
        m = mid[i]
        # 总体标准差（ddof=0）：除以 w，非 w-1——布林带用总体标准差是通行定义。
        sd[i] = float(np.sqrt(deterministic_reduction((win - m) ** 2) / w))
    upper = mid + k * sd
    lower = mid - k * sd

    params = {"window": w, "num_std": k}
    b = EvidenceBuilder("compute_bollinger", a)
    for name, arr in (("upper", upper), ("middle", mid), ("lower", lower)):
        _emit_line(b, name, arr, params)
    return _envelope("compute_bollinger", a, s, b, {
        "upper": to_none_list(upper),
        "middle": to_none_list(mid),
        "lower": to_none_list(lower),
        "params": params,
        "warmup": {
            "upper": _leading_nulls(upper),
            "middle": _leading_nulls(mid),
            "lower": _leading_nulls(lower),
        },
    })


# ── ATR / KDJ ────────────────────────────────────────────────────


def compute_atr(
    dates: Sequence[str],
    high: Sequence[Optional[float]],
    low: Sequence[Optional[float]],
    close: Sequence[Optional[float]],
    as_of: str,
    window: int = 14,
) -> dict:
    """平均真实波幅（Wilder 平滑）。TR = max(high-low, |high-prev_close|, |low-prev_close|)。

    Input: dates, high, low, close (等长), as_of, window (默认 14)。
    Returns: value (非负; 前 window 位 null) + params + warmup + evidence。
    """
    a = check_as_of(as_of)
    w = check_positive_window(window, name="window")
    h, l_, c = _prep_ohlc(dates, high, low, close, a)
    tr = _true_range(h.values, l_.values, c.values)
    arr = _wilder_avg(tr, w)
    b = EvidenceBuilder("compute_atr", a)
    _emit_line(b, "value", arr, {"window": w})
    return _envelope("compute_atr", a, c, b, {
        "value": to_none_list(arr),
        "params": {"window": w},
        "warmup": {"value": _leading_nulls(arr)},
    })


def _true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    """真实波幅。

    索引 0 用 `high-low` 本身——真实波幅的第一项不需要前收，因此 ATR 的
    warmup 比 RSI 少一位（ATR(window) 出 window-1 个 null，RSI(window) 出
    window 个：后者需要 window 个**变化**，即 window+1 个价格）。不是笔误。
    """
    n = len(high)
    tr = np.full(n, np.nan)
    if n == 0:
        return tr
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1]),
        )
    return tr


def compute_kdj(
    dates: Sequence[str],
    high: Sequence[Optional[float]],
    low: Sequence[Optional[float]],
    close: Sequence[Optional[float]],
    as_of: str,
    n_period: int = 9,
    k_period: int = 3,
    d_period: int = 3,
) -> dict:
    """KDJ（默认 9,3,3）：RSV = (C - LowN)/(HighN - LowN)*100；
    K = ((k_period-1)*K_prev + RSV)/k_period，D 同理于 K；J = 3K - 2D。
    K/D 初值 50（通行惯例），故首个 RSV 处即有 K/D。J 可越出 0..100。

    Input: dates, high, low, close, as_of, n_period (默认 9), k_period (默认 3), d_period (默认 3)。
    Returns: k / d / j 三条等长列表 (前 n_period-1 位 null) + params + warmup + evidence。
    """
    a = check_as_of(as_of)
    npd = check_positive_window(n_period, name="n_period")
    kp = check_positive_window(k_period, name="k_period")
    dp = check_positive_window(d_period, name="d_period")
    h, l_, c = _prep_ohlc(dates, high, low, close, a)

    n = len(c)
    rsv = np.full(n, np.nan)
    for i in range(npd - 1, n):
        hh = h.values[i + 1 - npd:i + 1]
        ll = l_.values[i + 1 - npd:i + 1]
        if np.isnan(hh).any() or np.isnan(ll).any() or np.isnan(c.values[i]):
            continue
        hi, lo = float(np.max(hh)), float(np.min(ll))
        rsv[i] = 50.0 if hi == lo else (float(c.values[i]) - lo) / (hi - lo) * 100.0

    k = np.full(n, np.nan)
    d = np.full(n, np.nan)
    k_prev, d_prev = 50.0, 50.0
    for i in range(n):
        if np.isnan(rsv[i]):
            continue
        k_prev = ((kp - 1) * k_prev + rsv[i]) / kp
        d_prev = ((dp - 1) * d_prev + k_prev) / dp
        k[i], d[i] = k_prev, d_prev
    # NaN 之后的递推：rsv 有洞时 state 停住，其后继续（state 不因缺失被污染）
    j = 3.0 * k - 2.0 * d

    params = {"n_period": npd, "k_period": kp, "d_period": dp}
    b = EvidenceBuilder("compute_kdj", a)
    for name, arr in (("k", k), ("d", d), ("j", j)):
        _emit_line(b, name, arr, params)
    return _envelope("compute_kdj", a, c, b, {
        "k": to_none_list(k),
        "d": to_none_list(d),
        "j": to_none_list(j),
        "params": params,
        "warmup": {"k": _leading_nulls(k), "d": _leading_nulls(d), "j": _leading_nulls(j)},
    })


# ── OBV / CCI / DMI ──────────────────────────────────────────────


def compute_obv(
    dates: Sequence[str],
    close: Sequence[Optional[float]],
    volume: Sequence[Optional[float]],
    as_of: str,
) -> dict:
    """能量潮：收盘涨则 +volume、跌则 -volume、平则不变，从首日 0 起累加。

    Input: dates, close, volume (等长; volume 单位由调用方决定), as_of。
    Returns: value (首值为 0，无 warmup), params, warmup (全 0), evidence。
    """
    a = check_as_of(as_of)
    c = check_series(dates, close, as_of=a, name="close")
    vol = check_series(dates, volume, as_of=a, name="volume")
    n = len(c)
    out = np.full(n, np.nan)
    if n:
        out[0] = 0.0
        running = 0.0
        for i in range(1, n):
            if np.isnan(c.values[i]) or np.isnan(c.values[i - 1]) or np.isnan(vol.values[i]):
                out[i] = np.nan
                continue
            if c.values[i] > c.values[i - 1]:
                running += float(vol.values[i])
            elif c.values[i] < c.values[i - 1]:
                running -= float(vol.values[i])
            out[i] = running

    b = EvidenceBuilder("compute_obv", a)
    _emit_line(b, "value", out, {})
    return _envelope("compute_obv", a, c, b, {
        "value": to_none_list(out),
        "params": {},
        "warmup": {"value": _leading_nulls(out)},
    })


def compute_cci(
    dates: Sequence[str],
    high: Sequence[Optional[float]],
    low: Sequence[Optional[float]],
    close: Sequence[Optional[float]],
    as_of: str,
    window: int = 20,
) -> dict:
    """顺势指标：TP=(H+L+C)/3；CCI = (TP - SMA(TP)) / (0.015 × 窗口内平均绝对偏差)。

    Input: dates, high, low, close, as_of, window (默认 20)。
    Returns: value (前 window-1 位 null) + params + warmup + evidence。
    """
    a = check_as_of(as_of)
    w = check_positive_window(window, name="window")
    h, l_, c = _prep_ohlc(dates, high, low, close, a)
    tp = (h.values + l_.values + c.values) / 3.0
    sma_tp = _rolling_mean(tp, w)
    n = len(tp)
    out = np.full(n, np.nan)
    for i in range(w - 1, n):
        if np.isnan(sma_tp[i]):
            continue
        win = tp[i + 1 - w:i + 1]
        if np.isnan(win).any():
            continue
        mad = deterministic_reduction(np.abs(win - sma_tp[i])) / w
        if mad == 0.0:
            continue  # 无波动 → 定义为 null（不发明一个 0 或 ±∞）
        out[i] = (float(tp[i]) - float(sma_tp[i])) / (0.015 * mad)

    b = EvidenceBuilder("compute_cci", a)
    _emit_line(b, "value", out, {"window": w})
    return _envelope("compute_cci", a, c, b, {
        "value": to_none_list(out),
        "params": {"window": w},
        "warmup": {"value": _leading_nulls(out)},
    })


def compute_dmi(
    dates: Sequence[str],
    high: Sequence[Optional[float]],
    low: Sequence[Optional[float]],
    close: Sequence[Optional[float]],
    as_of: str,
    window: int = 14,
    adx_window: int = 14,
) -> dict:
    """动向指标：+DM/-DM 经 Wilder 平滑得 +DI/-DI，DX = 100×|+DI--DI|/(+DI+-DI)，
    ADX = DX 的 Wilder 平滑。

    Input: dates, high, low, close, as_of, window (默认 14), adx_window (默认 14)。
    Returns: plus_di / minus_di / adx 三条等长列表 + params + warmup (逐线) + evidence。
      +DI/-DI 自索引 window 起非 null；ADX 需再一个窗口，自索引 window+adx_window-1 起。
    """
    a = check_as_of(as_of)
    w = check_positive_window(window, name="window")
    aw = check_positive_window(adx_window, name="adx_window")
    h, l_, c = _prep_ohlc(dates, high, low, close, a)

    n = len(c)
    plus_dm = np.full(n, np.nan)
    minus_dm = np.full(n, np.nan)
    for i in range(1, n):
        up = h.values[i] - h.values[i - 1]
        dn = l_.values[i - 1] - l_.values[i]
        if np.isnan(up) or np.isnan(dn):
            continue
        plus_dm[i] = up if (up > dn and up > 0) else 0.0
        minus_dm[i] = dn if (dn > up and dn > 0) else 0.0

    tr = _true_range(h.values, l_.values, c.values)
    atr = _wilder_avg(tr, w)
    s_plus = _wilder_avg(plus_dm, w)
    s_minus = _wilder_avg(minus_dm, w)

    plus_di = np.full(n, np.nan)
    minus_di = np.full(n, np.nan)
    dx = np.full(n, np.nan)
    for i in range(n):
        if np.isnan(atr[i]) or np.isnan(s_plus[i]) or np.isnan(s_minus[i]) or atr[i] == 0.0:
            continue
        plus_di[i] = 100.0 * s_plus[i] / atr[i]
        minus_di[i] = 100.0 * s_minus[i] / atr[i]
        denom = plus_di[i] + minus_di[i]
        if denom != 0.0:
            dx[i] = 100.0 * abs(plus_di[i] - minus_di[i]) / denom
    adx = _wilder_avg(dx, aw)

    params = {"window": w, "adx_window": aw}
    b = EvidenceBuilder("compute_dmi", a)
    for name, arr in (("plus_di", plus_di), ("minus_di", minus_di), ("adx", adx)):
        _emit_line(b, name, arr, params)
    return _envelope("compute_dmi", a, c, b, {
        "plus_di": to_none_list(plus_di),
        "minus_di": to_none_list(minus_di),
        "adx": to_none_list(adx),
        "params": params,
        "warmup": {
            "plus_di": _leading_nulls(plus_di),
            "minus_di": _leading_nulls(minus_di),
            "adx": _leading_nulls(adx),
        },
    })


TOOLS: list[ToolSpec] = [
    ToolSpec("compute_sma", GROUP, compute_sma),
    ToolSpec("compute_ema", GROUP, compute_ema),
    ToolSpec("compute_rsi", GROUP, compute_rsi),
    ToolSpec("compute_macd", GROUP, compute_macd),
    ToolSpec("compute_bollinger", GROUP, compute_bollinger),
    ToolSpec("compute_atr", GROUP, compute_atr),
    ToolSpec("compute_kdj", GROUP, compute_kdj),
    ToolSpec("compute_obv", GROUP, compute_obv),
    ToolSpec("compute_cci", GROUP, compute_cci),
    ToolSpec("compute_dmi", GROUP, compute_dmi),
]
