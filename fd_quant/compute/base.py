"""计算面公共地基：as_of 口径、序列校验、确定性纪律、evidence 构造。

六族工具（技术指标 / 组合统计 / 基本面比率 / 估值算子 / regime / 因子统计）
全部经由本模块进出，以保证：

- **口径一致**：每个工具都要求 `as_of`，越界数据点名拒绝（不静默截断）。
- **确定性**：归约顺序固定（`np.sum` 在固定顺序数组上逐位可复现；不使用
  非确定性并行归约），不使用随机源除非显式传 seed。
- **零外部依赖**：本模块只 import numpy / 标准库 + 契约层的 Evidence。
  数据获取、数据库、LLM 一律不在此层出现。

数值约定：序列以 `list[float | None]` 形态出入（None 表示 warmup / 缺失），
内部计算用 float64 ndarray，NaN 表示缺失。
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import date as _date
from typing import Any, Iterable, Optional, Sequence

import numpy as np

from ..contracts.models import Evidence

#: 计算面所有工具的 source 前缀，evidence.source 形如
#: `fd-quant:technical_indicator:atr(window=14)`。
SOURCE_PREFIX = "fd-quant"

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _is_iso_date(value: Any) -> bool:
    """本地实现（不用契约层的私有函数）：契约层与本层各自独立校验同一形状。"""
    if not isinstance(value, str) or not _DATE_RE.match(value):
        return False
    try:
        _date.fromisoformat(value)
    except ValueError:
        return False
    return True


class ComputeError(ValueError):
    """计算面输入违规（越界 / 缺字段 / 形状不符），文本自带可操作信息。"""


# ── as_of 与序列校验 ─────────────────────────────────────────────


@dataclass(frozen=True)
class Series:
    """一段带日期的数值序列（已通过 as_of 校验）。"""

    dates: list[str]
    values: np.ndarray  # float64，NaN 表缺失

    def __len__(self) -> int:
        return len(self.dates)

    @property
    def start(self) -> str:
        return self.dates[0]

    @property
    def end(self) -> str:
        return self.dates[-1]

    def to_list(self) -> list[Optional[float]]:
        """回给调用方的形态：None 表缺失/warmup，不用 NaN（JSON 不合法）。"""
        return [None if (v is None or math.isnan(v)) else float(v) for v in self.values]


def check_as_of(as_of: Any) -> str:
    """as_of 必填且为 ISO 日期。"""
    if as_of is None:
        raise ComputeError("as_of is required (ISO date, e.g. '2026-06-30')")
    if not _is_iso_date(as_of):
        raise ComputeError(f"as_of must be an ISO date (YYYY-MM-DD), got {as_of!r}")
    return str(as_of)


def check_series(
    dates: Sequence[Any],
    values: Sequence[Any],
    *,
    as_of: str,
    name: str = "series",
) -> Series:
    """日期与数值逐位配对校验，并拒绝 as_of 之后的数据点。

    越界时点名**第一个**越界日期（该日期即可操作的事实），不静默截断。
    """
    if len(dates) != len(values):
        raise ComputeError(
            f"{name}: dates and values length mismatch ({len(dates)} vs {len(values)})"
        )
    if not dates:
        raise ComputeError(f"{name} is empty")

    parsed: list[str] = []
    for i, d in enumerate(dates):
        if not _is_iso_date(d):
            raise ComputeError(f"{name}: dates[{i}] is not an ISO date: {d!r}")
        parsed.append(str(d))

    # 越界检查先于排序检查：调用方看到的是"数据晚于口径"，那才是要修的事。
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

    out = np.empty(len(values), dtype=np.float64)
    for i, v in enumerate(values):
        if v is None:
            out[i] = np.nan
            continue
        try:
            out[i] = float(v)
        except (TypeError, ValueError):
            raise ComputeError(f"{name}: values[{i}] is not numeric: {v!r}") from None
    return Series(dates=parsed, values=out)


def check_positive_window(window: Any, *, name: str = "window") -> int:
    if not isinstance(window, int) or isinstance(window, bool) or window < 1:
        raise ComputeError(f"{name} must be a positive integer, got {window!r}")
    return window


def check_finite(value: Any, *, name: str) -> float:
    """标量参数校验：必须是有限数（拒 NaN / inf）。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ComputeError(f"{name} must be a number, got {value!r}")
    f = float(value)
    if not math.isfinite(f):
        raise ComputeError(f"{name} must be finite, got {value!r}")
    return f


# ── evidence 构造 ────────────────────────────────────────────────


class EvidenceBuilder:
    """单次响应内 evidence_id 唯一性的构造器（对齐契约层六字段形状）。

    id 用 `<tool>.<label>` 的稳定 slug——同一次调用里 label 重复时追加序号，
    使"id 在单次响应内唯一"由构造器保证而非各工具各自小心。
    """

    def __init__(self, tool: str, as_of: str) -> None:
        self.tool = tool
        self.as_of = as_of
        self._items: list[Evidence] = []
        self._seen: dict[str, int] = {}

    def source(self, params: Optional[dict] = None) -> str:
        """`fd-quant:technical_indicator:atr(window=14)`——工具 + 参数可追溯。"""
        if not params:
            return f"{SOURCE_PREFIX}:{self.tool}"
        rendered = ",".join(f"{k}={_render_param(v)}" for k, v in params.items())
        return f"{SOURCE_PREFIX}:{self.tool}({rendered})"

    def add(
        self,
        label: str,
        value: Any,
        *,
        unit: Optional[str] = None,
        params: Optional[dict] = None,
    ) -> Evidence:
        slug = _slug(label)
        n = self._seen.get(slug, 0)
        self._seen[slug] = n + 1
        evidence_id = f"{self.tool}.{slug}" if n == 0 else f"{self.tool}.{slug}.{n}"
        ev = Evidence(
            evidence_id=evidence_id,
            label=label,
            value=_jsonable(value),
            unit=unit,
            as_of=self.as_of,
            source=self.source(params),
        )
        self._items.append(ev)
        return ev

    def extend(self, items: Iterable[Evidence]) -> None:
        for ev in items:
            self.add(ev.label, ev.value, unit=ev.unit)

    @property
    def items(self) -> list[Evidence]:
        return list(self._items)

    def to_dicts(self) -> list[dict]:
        return [ev.to_dict() for ev in self._items]


def _slug(text: str) -> str:
    s = re.sub(r"[^0-9a-zA-Z]+", "_", text.strip().lower()).strip("_")
    return s or "value"


def _render_param(v: Any) -> str:
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)


def _jsonable(value: Any) -> Any:
    """evidence.value 的 JSON 安全化：NaN/None → None，numpy 标量 → python 标量。"""
    if value is None:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        f = float(value)
        return None if math.isnan(f) else f
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.ndarray, list, tuple)):
        return [_jsonable(x) for x in value]
    return value


# ── 统一返回外壳 ─────────────────────────────────────────────────


def result_envelope(
    tool: str,
    as_of: str,
    series: Optional[Series],
    builder: EvidenceBuilder,
    payload: dict,
) -> dict:
    """所有工具的统一返回：口径回带 + 起止日期 + evidence + 载荷。

    `as_of` 与输入起止日期是"这个数是在什么口径下算的"的最小事实，
    因此放在外壳而非各 payload 里——调用方不必知道每个工具的键名。
    """
    out: dict[str, Any] = {
        "as_of": as_of,
        "input_start": series.start if series else None,
        "input_end": series.end if series else None,
        "input_length": len(series) if series else None,
        "evidence": builder.to_dicts(),
    }
    out.update(payload)
    return out


def to_none_list(arr: np.ndarray) -> list[Optional[float]]:
    """ndarray → JSON 安全列表（NaN → None）。"""
    return [None if (v is None or (isinstance(v, float) and math.isnan(v))) else float(v)
            for v in arr]


# ── 确定性基线 ───────────────────────────────────────────────────


def deterministic_reduction(arr: np.ndarray) -> float:
    """固定顺序的求和归约（Kahan 补偿）。

    存在的原因是 `np.sum` 的成对求和顺序随数组长度/分块策略变化，跨版本或
    跨线程数可能给出不同末位。本函数用严格的从左到右 Kahan 求和，使"同输入
    逐位一致"成为实现保证而非环境运气。
    """
    total = 0.0
    c = 0.0
    for x in arr.ravel():
        y = float(x) - c
        t = total + y
        c = (t - total) - y
        total = t
    return total


def check_determinism(fn, *args, **kwargs) -> None:
    """自检辅助：同输入两次调用逐位一致（IEEE 754 精确比较）。"""
    a = fn(*args, **kwargs)
    b = fn(*args, **kwargs)
    assert_bitequal(a, b, what=getattr(fn, "__name__", "call"))


def assert_bitequal(a: Any, b: Any, *, what: str = "result") -> None:
    """两值逐位相等断言（NaN 视为不一致——两次都 NaN 说明该报的错没报）。

    单独成函数是因为"同输入两次逐位一致"既要在同一进程内比（`check_determinism`），
    也要跨进程比（重跑落盘的证据），后者拿到的是两个已算好的值。
    """
    if not _deep_bitequal(a, b):
        raise ComputeError(f"non-deterministic result for {what}: two calls differ bitwise")


def _deep_bitequal(a: Any, b: Any) -> bool:
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        if a.shape != b.shape:
            return False
        return bool(np.array_equal(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_deep_bitequal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_deep_bitequal(x, y) for x, y in zip(a, b))
    if isinstance(a, float) and isinstance(b, float):
        return a == b  # NaN != NaN，两次都 NaN 视为不一致（该报的错）
    return a == b
