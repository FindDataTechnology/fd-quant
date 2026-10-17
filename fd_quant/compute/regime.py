"""regime 检测族：慢速（趋势 + 回撤阈值 + 波动率 Z）与快速（N 日冲击 + 波动率跳变）。

口径与纪律（与 `technical.py` 同一套，读一份即可跨族推理）：

- **as_of 先行**：每个函数第一步都是 `check_as_of`，随后每条序列经 `check_series`
  逐位配对校验；晚于 `as_of` 的数据点一律拒绝并点名**首个**越界日期，绝不静默截断。
- **波动率一律由调用方提供**：本族 SHALL NOT 自行计算、下载或合成波动率序列——指数
  波动率、已实现波动率还是别的口径由调用方决定，本族只消费；缺它即 `ComputeError`
  点名 `volatility`，绝不内置一个空序列当默认值。
- **参数一律回显**：`params` 里是实际生效的值（含默认值）；一个调用方看不见的假设值
  都不内置。
- **判据不发明数字**：数据不足（窗口长于序列、窗口内含缺失、波动率恒为零导致标准差
  为 0、分母 ≤ 0）时该判据 `status = "insufficient_data"`、`value = null`、
  `triggered = false`，绝不产出 NaN/inf，也绝不把 0 当作"没有波动"塞进结论。
- **阈值一律严格比较**（`>` / `<`）：恰好等于阈值**不算**触发——"超阈"在边界上只有
  一个答案，且调用方可以用"把阈值设成想排除的那个值"来表达排除。
- **确定性**：窗口归约只用 `deterministic_reduction`（Kahan 从左到右），不使用
  `np.mean`/`np.std`/`@` 等成对或 BLAS 归约——那是唯一随线程数变化的路径。

**默认值来自美股公开量级，A 股需在策略层调参**（`document_regime_defaults` 把这句话
连同两个检测器的默认值一起回给调用方，见 spec 6.3）。
"""
from __future__ import annotations

import inspect
import math
import re
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
)

GROUP = "regime_detection"

#: spec 6.3 要求的校准注记。工具**描述**与**返回**都必须带上它，文本逐字固定。
CALIBRATION_NOTE = "默认值来自美股公开量级，A 股需在策略层调参"

_ISO_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


# ── 内部工具（技术族私有 helper 不跨族 import，这里各写一份） ─────────


def _is_iso_date(value: Any) -> bool:
    """本地实现：契约层与本层各自独立校验同一形状。"""
    if not isinstance(value, str) or not _ISO_RE.match(value):
        return False
    try:
        _date.fromisoformat(value)
    except ValueError:
        return False
    return True


def _num(x: Any) -> Optional[float]:
    """标量 → JSON 安全浮点（None / NaN / ±inf → None；np 标量 → python float）。"""
    if x is None:
        return None
    f = float(x)
    return f if math.isfinite(f) else None


def _fmt(x: Any) -> str:
    """依据串里的确定性数值渲染（None → "null"；浮点定长小数）。"""
    if x is None:
        return "null"
    if isinstance(x, float):
        return f"{x:.6f}"
    return str(x)


def _emit_line(
    b: EvidenceBuilder, name: str, arr: np.ndarray, params: dict, unit: Optional[str] = None
) -> None:
    """序列证据：长度 + 首值 + 末值（点数不是信息，首末才是）。"""
    finite = np.flatnonzero(~np.isnan(arr))
    b.add(f"{name} length", int(len(arr)), unit="bars", params=params)
    if len(finite) == 0:
        b.add(f"{name} all_null", True, params=params)
        return
    i0, i1 = int(finite[0]), int(finite[-1])
    b.add(f"{name} first", float(arr[i0]), unit=unit, params=params)
    b.add(f"{name} last", float(arr[i1]), unit=unit, params=params)


def _sma_at(arr: np.ndarray, index: int, window: int) -> Optional[float]:
    """`arr[index+1-window : index+1]` 的窗口内均值（Kahan 顺序求和）。

    窗口放不下或窗口内含缺失 → None（缺失局部传播：不跳过、不填充）。
    """
    if index + 1 < window:
        return None
    win = arr[index + 1 - window:index + 1]
    for x in win:
        if math.isnan(float(x)):
            return None
    return deterministic_reduction(win) / window


def _drawdown_depth(values: np.ndarray) -> Optional[float]:
    """从运行峰值回撤的深度：`(峰值 − 末位值) / 峰值`（正数；0.10 = 回撤 10%）。

    峰值取输入区间内的运行最大值。序列含缺失时**整段不可信**（缺失点可能是更高的峰，
    跳过等于替调用方发明一个数据点）→ None；峰值 ≤ 0（非正水位）→ None。
    """
    n = len(values)
    if n == 0:
        return None
    peak: Optional[float] = None
    for i in range(n):
        x = float(values[i])
        if math.isnan(x):
            return None
        if peak is None or x > peak:
            peak = x
    last = float(values[n - 1])
    if peak is None or peak <= 0.0:
        return None
    return (peak - last) / peak


def _volatility_z(values: np.ndarray, window: int) -> Optional[float]:
    """最新波动率相对自身滚动窗口的 Z 分数。

    `z = (v_last − mean(v[−window:])) / stdev(v[−window:])`，窗口**含**最新一位，
    标准差用总体口径（ddof=0，与技术族布林带同一约定）。窗口放不下、窗口内含缺失或
    标准差为 0（波动率恒定）→ None——不发明一个 0 或 ±inf 当结论。
    """
    n = len(values)
    if n < window:
        return None
    win = values[n - window:n]
    for x in win:
        if math.isnan(float(x)):
            return None
    mean = deterministic_reduction(win) / window
    dev = np.empty(window, dtype=np.float64)
    for i in range(window):
        d = float(win[i]) - mean
        dev[i] = d * d
    var = deterministic_reduction(dev) / window
    if var <= 0.0:
        return None
    return (float(values[n - 1]) - mean) / math.sqrt(var)


def _check_date_axis(
    dates: Optional[Sequence[Any]], *, as_of: str, name: str
) -> Optional[list[str]]:
    """仅日期轴（无配对数值）的校验：非空、ISO、严格递增、不晚于 as_of。

    快照型工具为统一调用形状接受一个 `dates` 时用它：**忽略不等于放过**——一个含未来
    数据点的日期轴照样点名首个越界日期拒绝。
    """
    if dates is None:
        return None
    if not dates:
        raise ComputeError(f"{name} is empty")
    parsed: list[str] = []
    for i, d in enumerate(dates):
        if not _is_iso_date(d):
            raise ComputeError(f"{name}[{i}] is not an ISO date: {d!r}")
        parsed.append(str(d))
    for d in parsed:
        if d > as_of:
            raise ComputeError(
                f"{name}: data point {d} is after as_of={as_of} "
                f"(future data refused; slice the input or raise as_of)"
            )
    for i in range(1, len(parsed)):
        if parsed[i] <= parsed[i - 1]:
            raise ComputeError(
                f"{name}: dates must be strictly increasing; "
                f"{parsed[i - 1]} -> {parsed[i]} at index {i}"
            )
    return parsed


def _documented_defaults(fn) -> dict[str, Any]:
    """签名里带**非 None 默认值**的参数 = 校准旋钮（按签名顺序，值原样）。

    从签名读取而非另抄一份常量：默认值只有一处事实源，文档不会漂移。
    """
    out: dict[str, Any] = {}
    for name, p in inspect.signature(fn).parameters.items():
        if p.default is inspect.Parameter.empty or p.default is None:
            continue
        out[name] = p.default
    return out


def _render_defaults(defaults: dict[str, Any]) -> str:
    return ",".join(f"{k}={_fmt(v)}" for k, v in defaults.items())


# ── 慢速检测 ─────────────────────────────────────────────────────


def detect_regime_slow(
    dates: Sequence[str],
    prices: Optional[Sequence[Optional[float]]] = None,
    volatility: Optional[Sequence[Optional[float]]] = None,
    *,
    as_of: str,
    trend_window: int = 60,
    drawdown_threshold: float = 0.10,
    z_threshold: float = 2.0,
    trend_ma_window: int = 200,
    z_window: int = 60,
) -> dict:
    """慢速 regime 判定：趋势 / 回撤 / 波动率 Z 三判据合成 risk_on / neutral / risk_off。

    **波动率必须由调用方提供**（`volatility`，与 `prices` 等长）：本工具不自行计算、
    不下载、也不合成任何波动率序列——指数波动率、已实现波动率还是别的口径由调用方
    决定；未提供即 ComputeError 点名 `volatility`。本工具不取任何市场数据。

    三判据（各报 triggered / value / threshold / basis / status）：
    (a) trend = `sma(trend_window) / sma(trend_ma_window) − 1`，**< 0 触发**（短均线在
        长均线之下 = 趋势向下；经典 20 日短腿传 trend_window=20）；
    (b) drawdown = `(运行峰值 − 最新价) / 运行峰值`（正数），
        **> drawdown_threshold 触发**；
    (c) volatility_z = `(v_last − mean(v[−z_window:])) / stdev(v[−z_window:])`，
        **> z_threshold 触发**（总体标准差 ddof=0，窗口含最新一位）。
    决策规则（唯一出口，确定性）：**risk_off 当且仅当 (b) 触发且 (a)/(c) 至少一个
    触发**；否则 **risk_on 当且仅当 (a) 取值为正且 (c) 未触发**（(a) 不可判定不算"取值为
    正"，(c) 不可判定记为未触发）；其余一律 **neutral**。阈值一律严格比较（> / <）：
    恰好等于阈值不算触发。

    Input: dates (ISO, 严格递增), prices (价格/指数水平序列, 与 dates 等长; null 表缺失),
      volatility (波动率序列, 与 prices 等长, **由调用方提供**; null 表缺失),
      as_of (ISO, 必填, 第一个被校验的参数), trend_window (短均线窗口, 默认 60),
      drawdown_threshold (回撤阈值, 默认 0.10 = 10%, 取值 [0,1]),
      z_threshold (波动率 Z 阈值, 默认 2.0), trend_ma_window (长均线窗口, 默认 200,
      可下调以适配短序列), z_window (Z 的滚动窗口, 默认 60)。
    Returns: regime ("risk_on" / "neutral" / "risk_off"), criteria (键 trend / drawdown /
      volatility_z, 每个含 triggered (bool)、value (无量纲小数或 null)、threshold、
      basis (判据依据串)、status ("ok" / "insufficient_data")), triggered_count (触发判据数),
      volatility_source ("caller_supplied"), params (实际生效参数), warmup (逐判据"首位可算
      之前的 null 个数": trend = max(trend_window, trend_ma_window) − 1, drawdown = 0,
      volatility_z = z_window − 1), plus as_of / input_start / input_end / input_length /
      evidence。数据不足（窗口长于序列、窗口内含缺失、波动率标准差为 0、峰值 ≤ 0）时该判据
      status = "insufficient_data"、value = null、triggered = false——不抛异常、不产 NaN/inf。
    """
    tool = "detect_regime_slow"
    a = check_as_of(as_of)
    tw = check_positive_window(trend_window, name="trend_window")
    tmw = check_positive_window(trend_ma_window, name="trend_ma_window")
    zw = check_positive_window(z_window, name="z_window")
    ddt = check_finite(drawdown_threshold, name="drawdown_threshold")
    if not 0.0 <= ddt <= 1.0:
        raise ComputeError(
            f"{tool}: drawdown_threshold must be a fraction of the running peak within "
            f"[0, 1], got {ddt!r}"
        )
    zth = check_finite(z_threshold, name="z_threshold")
    if zth < 0.0:
        raise ComputeError(f"{tool}: z_threshold must be >= 0, got {zth!r}")

    if prices is None:
        raise ComputeError(
            f"{tool}: prices is required — pass the price/index level series; this tool "
            f"never fetches market data"
        )
    if volatility is None:
        raise ComputeError(
            f"{tool}: volatility is required — pass the caller's volatility series (same "
            f"length as prices, e.g. an index volatility level or realized volatility); "
            f"this tool never computes, fetches or synthesizes volatility, and a silently "
            f"dropped criterion would weaken the decision rule"
        )

    s_px = check_series(dates, prices, as_of=a, name="prices")
    s_vol = check_series(dates, volatility, as_of=a, name="volatility")
    p, v = s_px.values, s_vol.values
    n = len(p)

    params = {
        "trend_window": tw,
        "trend_ma_window": tmw,
        "drawdown_threshold": ddt,
        "z_threshold": zth,
        "z_window": zw,
    }

    # (a) trend：短均线 vs 长均线
    need_ma = max(tw, tmw)
    if n < need_ma:
        trend: dict[str, Any] = {
            "triggered": False,
            "value": None,
            "threshold": 0.0,
            "basis": (
                f"insufficient_data: needs max(trend_window, trend_ma_window) = {need_ma} "
                f"observations, have {n}"
            ),
            "status": "insufficient_data",
        }
    else:
        short = _sma_at(p, n - 1, tw)
        long = _sma_at(p, n - 1, tmw)
        if short is None or long in (None, 0.0):
            trend = {
                "triggered": False,
                "value": None,
                "threshold": 0.0,
                "basis": (
                    f"insufficient_data: sma({tw}) / sma({tmw}) window contains missing "
                    f"values or sma({tmw}) is 0"
                ),
                "status": "insufficient_data",
            }
        else:
            gap = short / long - 1.0
            trend = {
                "triggered": gap < 0.0,
                "value": gap,
                "threshold": 0.0,
                "basis": f"sma({tw}) / sma({tmw}) - 1; < 0 = short MA below long MA",
                "status": "ok",
            }

    # (b) drawdown：当前回撤深度 vs 阈值
    depth = _drawdown_depth(p)
    if depth is None:
        drawdown: dict[str, Any] = {
            "triggered": False,
            "value": None,
            "threshold": ddt,
            "basis": (
                "insufficient_data: prices contain missing values or the running peak is "
                "<= 0"
            ),
            "status": "insufficient_data",
        }
    else:
        drawdown = {
            "triggered": depth > ddt,
            "value": depth,
            "threshold": ddt,
            "basis": "(running peak - last) / running peak; > drawdown_threshold triggers",
            "status": "ok",
        }

    # (c) volatility Z：最新波动率 vs 自身滚动均值
    z = _volatility_z(v, zw)
    if z is None:
        vol_z: dict[str, Any] = {
            "triggered": False,
            "value": None,
            "threshold": zth,
            "basis": (
                f"insufficient_data: needs {zw} volatility observations with non-zero "
                f"stdev and no missing values, have {n}"
            ),
            "status": "insufficient_data",
        }
    else:
        vol_z = {
            "triggered": z > zth,
            "value": z,
            "threshold": zth,
            "basis": (
                f"(v_last - mean(last {zw})) / stdev(last {zw}); > z_threshold triggers"
            ),
            "status": "ok",
        }

    criteria = {"trend": trend, "drawdown": drawdown, "volatility_z": vol_z}
    triggered_count = int(sum(1 for c in criteria.values() if c["triggered"]))

    if drawdown["triggered"] and (trend["triggered"] or vol_z["triggered"]):
        regime = "risk_off"
    elif trend["status"] == "ok" and trend["value"] > 0.0 and not vol_z["triggered"]:
        regime = "risk_on"
    else:
        regime = "neutral"

    b = EvidenceBuilder(tool, a)
    b.add("regime", regime, params=params)
    for key, unit in (("trend", "ratio"), ("drawdown", "ratio"),
                      ("volatility_z", "z-score")):
        crit = criteria[key]
        b.add(f"{key} trigger value", _num(crit["value"]), unit=unit, params=params)
        b.add(f"{key} triggered", bool(crit["triggered"]), params=params)
        b.add(f"{key} status", crit["status"], params=params)
    b.add("triggered count", triggered_count, unit="criteria", params=params)
    _emit_line(b, "prices", p, params)
    _emit_line(b, "volatility", v, params)

    return result_envelope(tool, a, s_px, b, {
        "regime": regime,
        "criteria": criteria,
        "triggered_count": triggered_count,
        "volatility_source": "caller_supplied",
        "params": params,
        "warmup": {
            "trend": need_ma - 1,
            "drawdown": 0,
            "volatility_z": zw - 1,
        },
    })


# ── 快速检测 ─────────────────────────────────────────────────────


def detect_shock_fast(
    dates: Sequence[str],
    prices: Optional[Sequence[Optional[float]]] = None,
    volatility: Optional[Sequence[Optional[float]]] = None,
    *,
    as_of: str,
    shock_window: int = 5,
    shock_threshold: float = 0.08,
    vol_jump_ratio: float = 1.5,
    de_risk_fraction: float = 0.5,
    cooldown_days: int = 5,
    vol_window: int = 60,
) -> dict:
    """快速风险关闭检测：N 日价格冲击 + 波动率跳变，触发时给出强制降仓建议。

    **波动率必须由调用方提供**（`volatility`，与 `prices` 等长）：本工具不自行计算、
    不下载、也不合成波动率；未提供即 ComputeError 点名 `volatility`。

    两个信号（各自给值与阈值）：
    (1) shock = `prices[-1] / prices[-1-shock_window] − 1`，**< −shock_threshold 触发**；
    (2) vol_jump = `volatility[-1] / mean(volatility[−vol_window−1 : −1])`，即最新波动率
        除以其前 `vol_window` 位的均值（分子不含在分母里），**> vol_jump_ratio 触发**。
    触发规则：`triggered = (1) 或 (2)`（快线任一信号即降仓；两个子标志分别回给调用方，
    想要求"两个信号同时成立"的调用方读子标志即可）。阈值一律严格比较：恰好等于阈值
    不算触发。数据不足（`shock_window + 1` 或 `vol_window + 1` 长于序列、窗口内含缺失、
    分母 ≤ 0）时该信号 value = null、triggered = false 并置 status = "insufficient_data"。

    Input: dates (ISO, 严格递增), prices (价格/指数水平序列, 与 dates 等长; null 表缺失),
      volatility (波动率序列, 与 prices 等长, **由调用方提供**; null 表缺失),
      as_of (ISO, 必填, 第一个被校验的参数), shock_window (冲击观测日数 N, 默认 5),
      shock_threshold (N 日跌幅阈值, 默认 0.08 = 8%), vol_jump_ratio (波动率跳变倍数阈值,
      默认 1.5), de_risk_fraction (触发时建议降仓比例, 默认 0.5 = 减半, 取值 [0,1]),
      cooldown_days (建议冷却期, 默认 5, 正整数), vol_window (算波动率均值的回看位数,
      默认 60)。
    Returns: triggered (bool; = shock_triggered 或 volatility_jump_triggered),
      shock_return (N 日收益, 小数, 无量纲; 数据不足为 null), shock_triggered,
      volatility_jump_ratio (最新波动率 / 其前 vol_window 位均值; 数据不足为 null),
      volatility_jump_triggered, status ("ok" / "insufficient_data"), recommendation
      ({action: "reduce_position" / "hold", de_risk_fraction, cooldown_days, basis}——
      两个参数按配置值**原样回显**, 是否适用看 action), volatility_source
      ("caller_supplied"), params (实际生效参数), warmup (逐输出线"首位可算之前的 null
      个数": shock_return = shock_window, volatility_jump_ratio = vol_window),
      plus as_of / input_start / input_end / input_length / evidence。
    """
    tool = "detect_shock_fast"
    a = check_as_of(as_of)
    sw = check_positive_window(shock_window, name="shock_window")
    vw = check_positive_window(vol_window, name="vol_window")
    cd = check_positive_window(cooldown_days, name="cooldown_days")
    sth = check_finite(shock_threshold, name="shock_threshold")
    if sth < 0.0:
        raise ComputeError(f"{tool}: shock_threshold must be >= 0, got {sth!r}")
    vjr = check_finite(vol_jump_ratio, name="vol_jump_ratio")
    if vjr < 0.0:
        raise ComputeError(f"{tool}: vol_jump_ratio must be >= 0, got {vjr!r}")
    df = check_finite(de_risk_fraction, name="de_risk_fraction")
    if not 0.0 <= df <= 1.0:
        raise ComputeError(
            f"{tool}: de_risk_fraction must be a position fraction within [0, 1], got {df!r}"
        )

    if prices is None:
        raise ComputeError(
            f"{tool}: prices is required — pass the price/index level series; this tool "
            f"never fetches market data"
        )
    if volatility is None:
        raise ComputeError(
            f"{tool}: volatility is required — pass the caller's volatility series (same "
            f"length as prices, e.g. an index volatility level or realized volatility); "
            f"this tool never computes, fetches or synthesizes volatility"
        )

    s_px = check_series(dates, prices, as_of=a, name="prices")
    s_vol = check_series(dates, volatility, as_of=a, name="volatility")
    p, v = s_px.values, s_vol.values
    n = len(p)

    params = {
        "shock_window": sw,
        "shock_threshold": sth,
        "vol_jump_ratio": vjr,
        "de_risk_fraction": df,
        "cooldown_days": cd,
        "vol_window": vw,
    }

    # (1) N 日冲击
    if n < sw + 1:
        shock_return: Optional[float] = None
        shock_status = f"insufficient_data: needs shock_window + 1 = {sw + 1} prices, have {n}"
    else:
        base = float(p[n - 1 - sw])
        last = float(p[n - 1])
        # 窗口内含缺失一律不出数（契约 §7：窗口内含 NaN → 该点 null；不跨过缺失位
        # 算一个"看起来更长的收益"）。故检查整段 [n-1-sw, n-1] 而不只是两端。
        hole = math.isnan(base) or math.isnan(last)
        for i in range(n - 1 - sw, n - 1):
            if math.isnan(float(p[i])):
                hole = True
                break
        if hole or base == 0.0:
            shock_return = None
            shock_status = (
                "insufficient_data: the shock window contains missing prices "
                "(or the base price is 0)"
            )
        else:
            shock_return = last / base - 1.0
            shock_status = "ok"
    shock_triggered = bool(shock_return is not None and shock_return < -sth)

    # (2) 波动率跳变：最新除以之前 vol_window 位的均值（分子不含在分母里）
    if n < vw + 1:
        jump: Optional[float] = None
        jump_status = (
            f"insufficient_data: needs vol_window + 1 = {vw + 1} volatility observations, "
            f"have {n}"
        )
    else:
        win = v[n - 1 - vw:n - 1]
        missing = math.isnan(float(v[n - 1]))
        for x in win:
            if math.isnan(float(x)):
                missing = True
                break
        if missing:
            jump = None
            jump_status = "insufficient_data: volatility window contains missing values"
        else:
            mean_vol = deterministic_reduction(win) / vw
            if mean_vol <= 0.0:
                jump = None
                jump_status = (
                    "insufficient_data: trailing mean volatility is <= 0, so the jump "
                    "ratio is undefined"
                )
            else:
                jump = float(v[n - 1]) / mean_vol
                jump_status = "ok"
    jump_triggered = bool(jump is not None and jump > vjr)

    triggered = bool(shock_triggered or jump_triggered)
    status = "ok" if (shock_status == "ok" and jump_status == "ok") else "insufficient_data"

    if triggered:
        hits: list[str] = []
        if shock_triggered:
            hits.append(f"{sw}-day return {_fmt(_num(shock_return))} < -{_fmt(sth)}")
        if jump_triggered:
            hits.append(f"volatility jump ratio {_fmt(_num(jump))} > {_fmt(vjr)}")
        basis = "; ".join(hits) + f" -> cut {_fmt(df)} of the position for {cd} day(s)"
        action = "reduce_position"
    else:
        basis = (
            f"no trigger: {sw}-day return {_fmt(_num(shock_return))} vs -{_fmt(sth)}; "
            f"volatility jump ratio {_fmt(_num(jump))} vs {_fmt(vjr)}"
        )
        action = "hold"

    recommendation = {
        "action": action,
        "de_risk_fraction": df,
        "cooldown_days": cd,
        "basis": basis,
    }

    b = EvidenceBuilder(tool, a)
    b.add("triggered", triggered, params=params)
    b.add("shock return", _num(shock_return), unit="ratio", params=params)
    b.add("shock triggered", shock_triggered, params=params)
    b.add("shock status", shock_status, params=params)
    b.add("volatility jump ratio", _num(jump), unit="ratio", params=params)
    b.add("volatility jump triggered", jump_triggered, params=params)
    b.add("volatility jump status", jump_status, params=params)
    b.add("recommendation action", action, params=params)
    b.add("de_risk_fraction", df, unit="fraction", params=params)
    b.add("cooldown_days", cd, unit="days", params=params)
    _emit_line(b, "prices", p, params)
    _emit_line(b, "volatility", v, params)

    return result_envelope(tool, a, s_px, b, {
        "triggered": triggered,
        "shock_return": _num(shock_return),
        "shock_triggered": shock_triggered,
        "volatility_jump_ratio": _num(jump),
        "volatility_jump_triggered": jump_triggered,
        "status": status,
        "recommendation": recommendation,
        "volatility_source": "caller_supplied",
        "params": params,
        "warmup": {"shock_return": sw, "volatility_jump_ratio": vw},
    })


# ── 默认参数文档化（spec 6.3） ───────────────────────────────────


def document_regime_defaults(
    as_of: str,
    dates: Optional[Sequence[str]] = None,
) -> dict:
    """回显 regime 两个检测器的默认参数，并逐字带上校准注记（spec 6.3）。

    校准注记：**默认值来自美股公开量级，A 股需在策略层调参**——默认值是美股口径的公开
    量级，换了市场就换了量级；A 股的阈值请在策略层显式传参覆盖，不要把这些默认值当真理。

    本工具是快照型（无日期轴上的计算）：`dates` 接受一个可选日期轴只为**统一调用形状**
    （与其余 regime 工具同名同形），它不参与任何计算；但**忽略不等于放过**——提供的日期轴
    照样按同一口径校验（非空、ISO、严格递增、不晚于 `as_of`），含越界日期时点名首个越界
    日期拒绝。默认值从两个检测器的签名读取（唯一事实源，不再抄一份常量），故不会与实现
    漂移。

    Input: as_of (ISO 日期, 必填, 第一个被校验的参数), dates (可选日期轴; 仅校验、不参与
      计算, 省略则返回里 input_start / input_end / input_length 为 null)。
    Returns: tools ({detect_regime_slow: {params, calibration_note}, detect_shock_fast:
      {params, calibration_note}}; params 为该检测器全部带非 None 默认值的参数, 值即默认值
      原样), calibration_note (注记原文, 顶层与每个检测器各一份), params (空字典: 本工具无
      生效参数), warmup (空字典: 本工具无序列输出), plus as_of / input_start / input_end /
      input_length / evidence。
    """
    tool = "document_regime_defaults"
    a = check_as_of(as_of)
    axis = _check_date_axis(dates, as_of=a, name="dates")

    defaults = {
        "detect_regime_slow": _documented_defaults(detect_regime_slow),
        "detect_shock_fast": _documented_defaults(detect_shock_fast),
    }

    b = EvidenceBuilder(tool, a)
    b.add("calibration note", CALIBRATION_NOTE)
    for name, knobs in defaults.items():
        b.add(f"{name} defaults", _render_defaults(knobs))

    # 只为一件事装配 Series：统一外壳要一个带起止日期的输入轴（result_envelope 只读
    # start / end / len）。值位是占位 NaN——本工具没有数值序列，它从不参与计算、也从不出现在
    # 返回的业务键里。
    series = None
    if axis is not None:
        series = Series(dates=list(axis), values=np.full(len(axis), np.nan))

    return result_envelope(tool, a, series, b, {
        "tools": {
            name: {"params": knobs, "calibration_note": CALIBRATION_NOTE}
            for name, knobs in defaults.items()
        },
        "calibration_note": CALIBRATION_NOTE,
        "params": {},
        "warmup": {},
    })


TOOLS: list[ToolSpec] = [
    ToolSpec("detect_regime_slow", GROUP, detect_regime_slow),
    ToolSpec("detect_shock_fast", GROUP, detect_shock_fast),
    ToolSpec("document_regime_defaults", GROUP, document_regime_defaults),
]
