"""校验族：walk-forward 切分 / 留出切分 / 无未来函数护栏 / 观测充分性闸门。

这四件是全部分析/交易 pack 的共同护栏——**任何回测结论的可信度都取决于
"训练窗没有偷看测试窗、as_of 时刻没有用未来数据、样本够不够撑起这个结论"**。
做成独立工具而不是塞进回测引擎，是为了让调用方能在自己的流程里随时复核，而不是
只能相信引擎（设计 D7：验证器只做检查，**不改数据**——一旦"顺手修正"就失去了验证意义）。

**为什么检查器是"越界即报错"而不是"报告一份越界清单"**：未来函数是静默发生的
——一段序列碰巧多了一个未来点，回测收益会变好看而没人发现。报错点名**日期与下标**
以及两条出路（剪掉越界点 / 抬高 as_of），把"要不要剪掉它"变成调用方的显式决定；
返回一份清单则容易被当成提示而继续跑。这与全仓纪律一致：越界拒绝且点名，不静默截断。
`validate_no_lookahead` 是本族里唯一需要自制越界文本的工具（要同时报日期与下标，
`check_series` 的通用文本只报日期）；其余工具一律走上层的 `check_as_of` /
`check_series`，不各写一套。

**为什么窗口切分要保证"测试窗严格晚于训练窗"**：这是 walk-forward 的定义本身。
把它做成返回值的性质（切分规则：测试窗紧接训练窗，中间无缝、无重叠）而非文档里的
忠告，调用方才能在拿到窗口表时直接信任；`invariant` 键回带这条被断言的性质名。

**为什么留出集取最后一段而不是随机抽**：随机切分会让训练集含有晚于某些留出点的
观测——模型训练时已经见过"未来"，留出分数虚高。这是时间序列里最典型的未来函数，
故 `split_holdout` 只提供一种切法（最后一段），并以 `embargo` 切断重叠期泄露。

**为什么"数据够不够"要单独成一个工具**：样本不足不是输入错误，只是结论不成立。
`check_observation_sufficiency` 因此返回布尔值 + 差额（而非抛错），让 pack 在跑昂贵
流程前先问一句——不足时可换口径或放弃，而不是等算完才发现在硬凑。

**as_of 在本族各是什么（不假装有序列）**：

- `split_walk_forward_windows` / `check_observation_sufficiency` / `split_holdout`：
  `as_of` 是**日期轴的口径截止**——轴上任一日期不得晚于它；切分按下标进行，故每个
  窗口的边界日期都落在轴内（`test_end <= as_of` 由构造保证）。外壳回带的起止/长度
  就是这条轴本身的真实数字。
- `validate_no_lookahead` / `check_series_as_of_bounds`：`as_of` 是**被检序列声称的信息
  截止**。合规序列是"截止 as_of 可用的数据"——过期的点不算违规，只有晚于 as_of 才违规。

**日期-only 工具怎么校验日期**：`check_series` 需要一列与日期等长的数值，而本族多数
工具只吃日期轴。这些工具**传一列恒等数列**（`0.0, 1.0, ...`）走同一个 `check_series`
（见 `_axis_series`），因此"ISO 补齐 / 严格递增 / 晚于 as_of 即拒绝并点名首个日期"
这套语义与全仓逐字相同，不存在本族自己写歪的第二套；恒等数列从不进入返回值。

**确定性**：本族不做浮点归约（不算收益、不求和），因此逐位一致是构造性的；无随机源。
返回值恒可 `json.dumps(..., allow_nan=False)`。
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
    check_positive_window,
    check_series,
    result_envelope,
)

GROUP = "validation"

#: 切分默认口径直接取自 spec 的 Scenario（训练 60 / 测试 20 / 步长 20 切分 200 日范围）：
#: 默认值必须是任何一个"够切一刀"的序列都跑得动的数，且能在 spec 用例里逐字对上。
DEFAULT_TRAIN_WINDOW = 60
DEFAULT_TEST_WINDOW = 20

#: 观测充分性的默认门槛（约一个交易年的日频 bar）：252 是"够算年化波动/回撤"的通行最小样本，
#: 取默认值使调用方不传参数也能得到"够不够一个年度分析"的答案。
DEFAULT_REQUIRED_OBSERVATIONS = 252

#: 短样本策略：默认拒绝（不静默缩放、不返回空表）；显式选 "empty" 才返回空窗口表 + reason。
SHORT_SAMPLE_REFUSE = "refuse"
SHORT_SAMPLE_EMPTY = "empty"

#: 切分/划分输出里"日期轴下标"与"实际日期"的别名映射：本族同时回带两套键名——
#: 规范键（`*_i` 下标 / 裸日期）是主口径，别名键（`*_idx` / `*_date`）是 spec 工具契约
#: 里的命名。两套值恒等，调用方按自己习惯取用即可（多一组键，不是多一套算法）。
_WINDOW_ALIASES: tuple[tuple[str, str], ...] = (
    ("window", "index"),
    ("train_start_idx", "train_start_i"),
    ("train_end_idx", "train_end_i"),
    ("test_start_idx", "test_start_i"),
    ("test_end_idx", "test_end_i"),
    ("train_start_date", "train_start"),
    ("train_end_date", "train_end"),
    ("test_start_date", "test_start"),
    ("test_end_date", "test_end"),
)

#: 规范键 → 别名键（反查表；由 `_WINDOW_ALIASES` 派生，不另写一份）。
_CANON_TO_ALIAS: dict[str, str] = {canon: alias for alias, canon in _WINDOW_ALIASES}


def _builder(tool: str, as_of: str, params: dict, label: str, value: Any) -> EvidenceBuilder:
    """构造并落一条 evidence（空 builder 会产出空 evidence 表，等于没有溯源）。"""
    b = EvidenceBuilder(tool=tool, as_of=as_of)
    b.add(label, value, params=params)
    return b


def _axis_series(dates: Sequence[Any], as_of: str, name: str) -> Series:
    """只有日期轴的输入 → 用**恒等数值列**走 `check_series`（本族的日期-only 工具共用）。

    为什么这么做而不是自己写一套日期校验：`check_series` 就是全仓的日期形状与越界
    口径（ISO 补齐、严格递增、晚于 as_of 即拒绝并点名首个日期）。自己再写一份必然
    慢慢漂移；传一个 `[0.0, 1.0, ...]` 的**等长**数值列即可让本工具复用同一条代码路径，
    而恒等列只用于满足"成对等长"的入参契约，**从不进入返回值**（`result_envelope`
    只从 Series 读起止与长度，不读数值）。日期-only 工具因此不假装持有数值序列。
    """
    identity = [float(i) for i in range(len(dates))]
    return check_series(dates, identity, as_of=as_of, name=name)


def _window_aliases(w: dict[str, Any]) -> dict[str, Any]:
    """给窗口条目补别名键（`*_idx` / `*_date` / `window`）：值与规范键恒等。

    规范键是 `*_i`（下标）与裸日期名（如 `train_end`），别名是 spec 工具契约里的
    `train_end_idx` / `train_end_date` / `window`。多一组键不是多一套算法——两套在
    同一个 dict 里由同一处代码同时写入，调用方按习惯取用，不会出现"两套值不一致"。
    """
    for alias, canonical in _WINDOW_ALIASES:
        w[alias] = w[canonical]
    return w


def _check_nonneg_int(value: Any, *, name: str) -> int:
    """非负整数参数校验（embargo 这类"可以是 0"的计数）。"""
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ComputeError(f"{name} must be a non-negative integer, got {value!r}")
    return value


def _require_finite_param(value: Any, *, name: str) -> float:
    """有限数参数校验（比率类；拒 bool / 非数值 / NaN / inf）。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ComputeError(f"{name} must be a number, got {value!r}")
    f = float(value)
    if not math.isfinite(f):
        raise ComputeError(f"{name} must be finite, got {value!r}")
    return f


def _floor_fraction(n: int, fraction: float) -> int:
    """`floor(n × fraction)`，用**精确整数算术**（分数的既约分子/分母）算。

    为什么不用 `int(n * fraction)`：float 乘法的结果依平台/版本可能落在整数边界两侧
    （`n*0.2` 可能得到 `20.000000000000004` 或 `19.999999999999996`），那会让"100 个点
    切 20%"在不同机器上变成 20 或 19——切分是结构，不能有这种末位运气。
    """
    num, den = fraction.as_integer_ratio()
    return (n * num) // den


def _covered_observations(windows: Sequence[dict[str, Any]]) -> int:
    """被任一**测试窗**覆盖的观测数（重叠的测试窗按并集计一次）。

    直接累加 `test_size` 会把重叠的测试窗重复计数（step < test_window 时很常见），
    使 coverage 超过范围长度；这里先合并区间再数——coverage 是"有多少观测参与了
    样本外评估"这一事实，不是"测试窗一共多长"。
    """
    spans = sorted((w["test_start_i"], w["test_end_i"]) for w in windows)
    total = 0
    cur_start: Optional[int] = None
    cur_end: Optional[int] = None
    for s, e in spans:
        if cur_start is None:
            cur_start, cur_end = s, e
        elif s <= cur_end + 1:  # 相接或重叠 → 并成一段（相接时并集不重复计数）
            cur_end = max(cur_end, e)
        else:
            total += cur_end - cur_start + 1
            cur_start, cur_end = s, e
    if cur_start is not None:
        total += cur_end - cur_start + 1
    return total


def _empty_windows_envelope(
    tool: str, as_of: str, axis: Series, reason: str, params: dict
) -> dict:
    """短样本下的"答案而非异常"外壳：`windows: []` + reason（其余键与正常路径同形状）。

    键集与正常路径**完全一致**（缺的只是窗口）——调用方不必按"有没有窗口"分支处理
    两套返回形状，`for w in r["windows"]` 在两种情况下都成立。
    """
    b = _builder(tool, as_of, params, "window_count", 0)
    b.add("short_sample_refused_to_scale", True, params=params)
    return result_envelope(
        tool,
        as_of,
        axis,
        b,
        {
            "windows": [],
            "window_count": 0,
            "n_windows": 0,
            "coverage": 0,
            "coverage_ratio": 0.0,
            "train_window": params["train_window"],
            "test_window": params["test_window"],
            "step": params["step"],
            "point_count": len(axis),
            "unassigned_tail": len(axis),
            "coverage_start": None,
            "coverage_end": None,
            "range_start": axis.dates[0],
            "range_end": axis.dates[-1],
            "invariant": "test_strictly_after_train",
            "reason": reason,
            "params": params,
            "warmup": {},
        },
    )


def _is_iso_date(value: Any) -> bool:
    """本地实现（不用契约层/计算层的私有函数）：各层各自校验同一形状。"""
    if not isinstance(value, str) or len(value) != 10 or value[4] != "-" or value[7] != "-":
        return False
    if not (value[:4].isdigit() and value[5:7].isdigit() and value[8:10].isdigit()):
        return False
    try:
        _date.fromisoformat(value)
    except ValueError:
        return False
    return True


def _lookahead_report(
    dates: Sequence[str], as_of: str, name: str, *, limit: int = 5
) -> Optional[str]:
    """越界点的可操作文本（无越界 → None）。

    点名**首个**越界点的日期与下标（那是调用方要处理的第一个事实）、违规点数、最新日期，
    并给出两条出路。不复用 `check_series` 的通用文本：护栏的错误文本要能直接照着改。
    """
    idxs = [i for i, d in enumerate(dates) if d > as_of]
    if not idxs:
        return None
    i0 = idxs[0]
    shown = ", ".join(f"{dates[i]} (index {i})" for i in idxs[:limit])
    more = "" if len(idxs) <= limit else f" … (+{len(idxs) - limit} more)"
    latest = max(dates[i] for i in idxs)
    return (
        f"{name}: lookahead detected — {len(idxs)} data point(s) are dated after the declared "
        f"as_of={as_of}. First offending point: {dates[i0]} at index {i0}; offending: "
        f"{shown}{more}; latest point: {latest}. This series is not usable at as_of={as_of}: "
        f"either slice the input to points <= {as_of} (keeping the order), or raise as_of to "
        f">= {latest} if that later data genuinely belongs to this computation. No future data "
        f"is dropped silently."
    )


def _prep_declared(
    dates: Sequence[Any], values: Optional[Sequence[Any]], as_of: str, name: str
) -> Series:
    """被检序列校验：ISO / 等长 / 数值形状 / 严格递增，越界点名**日期与下标**。

    数值位允许 None（缺失不是未来数据，只是没有数）；越界先于递增检查被报出——调用方
    要修的是"数据晚于口径"这件事。形状规则与 `check_series` 完全一致，差别只在文本。
    """
    if values is None:
        raise ComputeError(f"{name}: values is required to validate a series")
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

    out = np.empty(len(values), dtype=np.float64)
    for i, v in enumerate(values):
        if v is None:
            out[i] = np.nan
            continue
        try:
            out[i] = float(v)
        except (TypeError, ValueError):
            raise ComputeError(f"{name}: values[{i}] is not numeric: {v!r}") from None

    report = _lookahead_report(parsed, as_of, name)
    if report is not None:
        raise ComputeError(report)
    for i in range(1, len(parsed)):
        if parsed[i] <= parsed[i - 1]:
            raise ComputeError(
                f"{name}: dates must be strictly increasing; "
                f"{parsed[i - 1]} -> {parsed[i]} at index {i}"
            )
    return Series(dates=parsed, values=out)


def _prep_point_as_of(
    dates: Sequence[str], point_as_of: Optional[Sequence[Any]], as_of: str, name: str
) -> Optional[list[str]]:
    """逐点信息口径日校验：等长、ISO；任一点晚于声明的 as_of → 泄漏，报错并点名。

    这是"在 as_of 时刻使用了未来数据"的可检形态：点的**日期**合法（不晚于 as_of），
    但它的值嵌入了只有更晚才可能知道的信息。`point_as_of=None`（默认）跳过此项检查。
    """
    if point_as_of is None:
        return None
    if len(point_as_of) != len(dates):
        raise ComputeError(
            f"{name}: point_as_of length {len(point_as_of)} != series length {len(dates)} "
            f"(give the information cutoff of every point, or omit point_as_of entirely)"
        )
    parsed: list[str] = []
    for i, d in enumerate(point_as_of):
        if not _is_iso_date(d):
            raise ComputeError(f"{name}: point_as_of[{i}] is not an ISO date: {d!r}")
        parsed.append(str(d))
    for i, d in enumerate(parsed):
        if d > as_of:
            cnt = sum(1 for x in parsed if x > as_of)
            latest = max(parsed)
            raise ComputeError(
                f"{name}: lookahead detected — {cnt} point(s) declare an information cutoff "
                f"later than the declared as_of={as_of}. First offending point: {dates[i]} at "
                f"index {i} declares point_as_of={d}; latest declared cutoff: {latest}. The "
                f"value at {dates[i]} embeds data that only became knowable after as_of={as_of}. "
                f"Recompute that point from information available at as_of, or raise as_of to "
                f">= {latest} if the series is genuinely complete then."
            )
    return parsed


def split_walk_forward_windows(
    dates: Sequence[Any],
    values: Optional[Sequence[Any]],
    as_of: str,
    train_window: Any = DEFAULT_TRAIN_WINDOW,
    test_window: Any = DEFAULT_TEST_WINDOW,
    step: Any = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    short_sample: str = SHORT_SAMPLE_REFUSE,
) -> dict:
    """walk-forward 窗口切分：训练窗 → 紧随其后的测试窗，按 step 前推，输出有序窗口表。

    切分规则（按下标、写死、可复现）：第 k 个训练窗 = `[r0 + k*step, r0 + k*step +
    train_window)`，其测试窗紧接其后且无重叠——**测试窗的数据恒严格晚于其训练窗**
    （`test_start_i = train_end_i + 1`，这一性质以 `invariant` 键回带并逐窗再断言一次）。
    相邻训练窗起点相差恰好 step 个 bar（step > test_window 时相邻测试窗之间会跳过 bar，
    这是允许的：调用方自己选的滚动节奏）。窗口数 = `(n - train_window - test_window)
    // step + 1`（n = 参与切分的 bar 数）。
    窗口条目的键有**两套等值别名**：规范键 `index` / `train_start`(日期) / `train_start_i`
    (下标) 等，与 `window` / `train_start_date` / `train_start_idx` 等（同一 dict 内同时写入，
    值恒等）。逐窗键：index|window、train_start|train_start_date 与 train_start_i|
    train_start_idx、train_end*、test_start*、test_end*、train_size、test_size。
    **短样本策略**（`short_sample`）：默认 `"refuse"`——范围放不下一个"训练+测试"窗即报错
    并点名全部数字（不缩放窗口：缩放等于替调用方发明参数；不返回空表：空表会让回测静默
    跑 0 个窗口）。显式传 `"empty"` 则改答一个问题：返回 `windows: []` + `reason` 字符串
    （"这个范围能切几个窗"的答案是"0 个，因为…"），不抛异常。
    Input: dates (ISO, 严格递增, 全轴, 不得晚于 as_of), values (与 dates 等长; 本工具只用其
      长度, 不参与计算; 传 null = 只按日期轴切分——切分是下标结构, 不依赖数值), as_of (ISO,
      必填; 轴的口径截止), train_window (训练 bar 数, 默认 60, 正整数), test_window (测试 bar
      数, 默认 20, 正整数), step (滚动步长, 单位 bar, 默认 null = 取 test_window; 正整数),
      start / end (可选 ISO 日期闭区间; 把轴裁剪到该范围再切, 必须落在轴内且 start <= end,
      否则报错点名; 本工具**不合成日期轴**——没有的日期不会被发明出来, 只有真实观测才进窗口),
      short_sample ("refuse"(默认) 或 "empty", 见上).
    Returns: windows (列表, 每项见上) / window_count / n_windows (同 window_count) /
      coverage (被任一测试窗覆盖的观测数, 整数; 测试窗重叠时按并集计) / coverage_ratio
      (coverage / 参与切分的 bar 数) / train_window / test_window / step (生效值) /
      point_count (轴长度) / unassigned_tail (末尾未被任何窗口覆盖的 bar 数) /
      coverage_start 与 coverage_end (首个训练窗起点 / 末个测试窗终点) / range_start 与
      range_end (参与切分的首末日期) / invariant ("test_strictly_after_train") / reason
      (窗口非空时恒 null; short_sample="empty" 且放不下时为原因字符串) / params /
      warmup (空字典: 本工具不消耗滚动窗, 无 warmup 概念); 另加统一外壳 as_of 起止与
      evidence。**本工具不改数据**：输入列表从不被写入或重排，窗口只是下标区间的描述。
      **拒绝语义**: as_of 非法 / 有未来数据(点名首个越界日期) / 日期非严格递增 / 长度不等 /
      窗口或步长非正 / start 或 end 落在轴外 / short_sample 取值非法 / (默认策略下)范围
      放不下一窗 → ComputeError 点名全部成因数字。
    """
    a = check_as_of(as_of)
    axis = (check_series(dates, values, as_of=a, name="walk_forward") if values is not None
            else _axis_series(dates, a, "walk_forward"))
    tw = check_positive_window(train_window, name="train_window")
    tew = check_positive_window(test_window, name="test_window")
    st = tew if step is None else check_positive_window(step, name="step")
    if short_sample not in (SHORT_SAMPLE_REFUSE, SHORT_SAMPLE_EMPTY):
        raise ComputeError(
            f"short_sample must be {SHORT_SAMPLE_REFUSE!r} or {SHORT_SAMPLE_EMPTY!r}, "
            f"got {short_sample!r} (use {SHORT_SAMPLE_EMPTY!r} to get an empty window list "
            f"+ reason instead of a refusal)"
        )
    for nm, v in (("start", start), ("end", end)):
        if v is not None and not _is_iso_date(v):
            raise ComputeError(f"{nm} must be an ISO date (YYYY-MM-DD), got {v!r}")
    if start is not None and end is not None and str(start) > str(end):
        raise ComputeError(
            f"start={start} is after end={end}; the split range is empty "
            f"(pass them in chronological order or omit one)"
        )

    i0, i1 = 0, len(axis) - 1
    if start is not None:
        hits = [i for i, d in enumerate(axis.dates) if d >= str(start)]
        if not hits:
            raise ComputeError(
                f"start={start} is after the last date on the axis ({axis.dates[-1]}); "
                f"nothing to split (axis runs {axis.dates[0]} .. {axis.dates[-1]})"
            )
        i0 = hits[0]
    if end is not None:
        hits = [i for i, d in enumerate(axis.dates) if d <= str(end)]
        if not hits:
            raise ComputeError(
                f"end={end} is before the first date on the axis ({axis.dates[0]}); "
                f"nothing to split (axis runs {axis.dates[0]} .. {axis.dates[-1]})"
            )
        i1 = hits[-1]

    n = i1 - i0 + 1
    need = tw + tew
    if n < need:
        msg = (
            f"walk_forward: the range holds {n} bar(s) but one train+test window pair needs "
            f"train_window={tw} + test_window={test_window} = {need}; shrink the windows or "
            f"widen the range (axis has {len(axis)} bar(s) from {axis.dates[0]} to "
            f"{axis.dates[-1]}; selected range {axis.dates[i0]} .. {axis.dates[i1]})"
        )
        if short_sample == SHORT_SAMPLE_REFUSE:
            raise ComputeError(msg)
        return _empty_windows_envelope(
            "split_walk_forward_windows", a, axis, msg,
            {"train_window": tw, "test_window": tew, "step": st,
             "start": start, "end": end, "short_sample": short_sample},
        )

    windows: list[dict[str, Any]] = []
    offset = 0
    while offset + need <= n:
        w = {
            "index": len(windows),
            "train_start": axis.dates[i0 + offset],
            "train_end": axis.dates[i0 + offset + tw - 1],
            "test_start": axis.dates[i0 + offset + tw],
            "test_end": axis.dates[i0 + offset + tw + tew - 1],
            "train_start_i": i0 + offset,
            "train_end_i": i0 + offset + tw - 1,
            "test_start_i": i0 + offset + tw,
            "test_end_i": i0 + offset + tw + tew - 1,
            "train_size": tw,
            "test_size": tew,
        }
        if not (w["test_start_i"] == w["train_end_i"] + 1 and w["test_start"] > w["train_end"]):
            raise ComputeError(
                f"walk_forward: internal invariant broken at window {w['index']} "
                f"(test window does not strictly follow its train window; as_of={a})"
            )
        windows.append(_window_aliases(w))
        offset += st

    covered = _covered_observations(windows)
    uncovered_tail = n - (windows[-1]["test_end_i"] + 1 - i0)
    params = {"train_window": tw, "test_window": tew, "step": st,
              "start": start, "end": end, "short_sample": short_sample}
    builder = _builder(
        "split_walk_forward_windows", a,
        {"train": tw, "test": tew, "step": st}, "window_count", len(windows),
    )
    builder.add("coverage", covered, unit="bars", params={"train": tw, "test": tew, "step": st})
    return result_envelope(
        "split_walk_forward_windows",
        a,
        axis,
        builder,
        {
            "windows": windows,
            "window_count": len(windows),
            "n_windows": len(windows),
            "coverage": covered,
            "coverage_ratio": covered / n,
            "train_window": tw,
            "test_window": tew,
            "step": st,
            "point_count": len(axis),
            "unassigned_tail": uncovered_tail,
            "coverage_start": windows[0]["train_start"],
            "coverage_end": windows[-1]["test_end"],
            "range_start": axis.dates[i0],
            "range_end": axis.dates[i1],
            "invariant": "test_strictly_after_train",
            "reason": None,
            "params": params,
            "warmup": {},
        },
    )


def validate_no_lookahead(
    dates: Sequence[Any],
    values: Optional[Sequence[Any]],
    as_of: str,
    name: str = "values",
    point_as_of: Optional[Sequence[str]] = None,
    *,
    declared_as_of: Optional[str] = None,
    windows: Optional[Sequence[Any]] = None,
) -> dict:
    """无未来函数护栏：断言一条自称 `as_of` 口径的序列不含未来数据（只读不改写）。

    检查三件事（全部只读）：(1) **日期越界**——任一点晚于有效截止日即违规；(2) **泄漏点**——
    若给了逐点信息口径日 `point_as_of`，任一点的信息口径日晚于 `as_of` 即违规（点的日期合法，
    但它的值嵌入了只有更晚才可能知道的数据——这正是"在 as_of 时刻使用了未来数据"的可检
    形态）；(3) **窗口表**——若给了 `windows`（`split_walk_forward_windows` 的输出），逐窗
    断言其测试区间**严格晚于**训练区间（`test_start > train_end`），使调用方能用一个工具
    同时验"原始序列"与"窗口计划"两件东西。
    **有效截止日** = `min(as_of, declared_as_of)`（`declared_as_of` 缺省即 `as_of`）：工具的
    截止日与调用方**声称**的口径日都是约束，取更早者才是保守的读法——若声称的截止日更早，
    夹在两者之间的点仍是"声称口径之后的数据"，必须拒绝。任一点晚于有效截止日即抛
    `ComputeError`，文本含序列名、**首个越界点的日期与下标**、违规点数、最新日期，以及两条
    出路（剪掉越界点 / 抬高 as_of）；通过则返回 ok 报告。
    **本工具是断言器，不是变换器（spec 设计 D7：验证器**不改数据**）**：输入列表从不被写入、
    截断、排序或重排，越界时抛错的同一份输入在调用后逐位不变——"顺手修正"会让被检对象
    与调用方手里的数据不再一致，护栏也就失去了意义。合规序列是"截止 as_of 可用的数据"
    ——过期的点不算违规，缺失（null）只是没有数，不报错。
    Input: dates (ISO, 严格递增), values (与 dates 等长, null 表缺失——缺失只计数; 传 null
      = 只校验日期轴, 此时不检查数值形状; 位置保留是为了不打断既有调用形态), as_of (ISO,
      必填; 被检序列**声称的信息截止**), name (被检序列的名字, 默认 "values"; 只进错误文本与
      params, 多序列调用时用于分辨哪条违规), point_as_of (可选, 与 dates 等长的 ISO 日期列表;
      每个点的值最晚用到了哪天的信息; 默认 null = 不做此项检查), declared_as_of (可选 ISO 日期;
      调用方**声称**其计算使用的口径日; 与 as_of 一起构成有效截止日 min, 见上), windows (可选;
      切分器输出的窗口列表, 键名两套别名都认; 默认 null = 不检查窗口计划).
    Returns: ok (通过恒为 true——失败是抛错, 不是 ok=false) / declared_as_of (有效截止日) /
      requested_as_of (工具自己的 as_of 原值) / point_count / points_checked 与 checked_points
      (同 point_count) / values_provided (是否传了数值列) / missing_points (缺失位数) /
      checked_windows (校验过的窗口数; 未传 windows 时为 0) / window_invariant (未传 windows
      时为 null, 否则 "test_strictly_after_train") / future_points 与 lookahead_points (通过
      时均为 0) / first_future_date 与 first_future_index (通过时为 null) / first_date /
      last_date / span_days (首末相差的自然日) / headroom_days (末个观测到有效截止日的自然日
      数; 0 = 数据正好用到截止日) / is_monotonic (恒为 true; 非严格递增在入口已被拒绝, 回带
      便于下游记录) / params / warmup (空字典: 不消耗滚动窗)；另加统一外壳。
      **拒绝语义**: as_of 非法 / 存在未来数据(点名首个越界点的日期与下标) / 日期非严格递增 /
      长度不等 / 序列为空 / point_as_of 长度不等或非 ISO 日期 / declared_as_of 非法 /
      windows 条目缺键或日期不在轴上或测试区间不晚于训练区间 → ComputeError。
    """
    a = check_as_of(as_of)
    if not isinstance(name, str) or not name.strip():
        raise ComputeError(f"name must be a non-empty string, got {name!r}")
    nm = name.strip()

    cutoff = a
    if declared_as_of is not None:
        if not _is_iso_date(declared_as_of):
            raise ComputeError(
                f"declared_as_of must be an ISO date (YYYY-MM-DD), got {declared_as_of!r}"
            )
        cutoff = min(str(declared_as_of), a)

    if values is None:
        # 只给日期轴：用恒等数值列走 check_series（见 _axis_series 的说明——同一套
        # 日期形状与越界口径，不另写一份），恒等列不进返回值。
        series = _axis_series(dates, cutoff, nm)
        provided = False
    else:
        series = _prep_declared(dates, values, cutoff, nm)
        provided = True
    n = len(series)
    _prep_point_as_of(series.dates, point_as_of, a, nm)
    checked_windows = _check_window_plan(windows, series.dates, nm)

    missing = 0
    if provided:
        for i in range(n):
            v = series.values[i]
            if not np.isfinite(v):
                missing += 1
    span = (_date.fromisoformat(series.dates[-1]) - _date.fromisoformat(series.dates[0])).days
    headroom = (_date.fromisoformat(cutoff) - _date.fromisoformat(series.dates[-1])).days

    params = {"name": nm, "point_as_of": "provided" if point_as_of is not None else None,
              "declared_as_of": cutoff, "values_provided": provided,
              "windows_checked": checked_windows}
    builder = _builder(
        "validate_no_lookahead", a,
        {"name": nm, "points": n}, "verified_points", n,
    )
    builder.add("headroom_days", headroom, unit="days", params={"as_of": cutoff})
    if checked_windows:
        builder.add("windows_verified", checked_windows, params={"name": nm})
    return result_envelope(
        "validate_no_lookahead",
        a,
        series,
        builder,
        {
            "ok": True,
            "declared_as_of": cutoff,
            "requested_as_of": a,
            "point_count": n,
            "points_checked": n,
            "checked_points": n,
            "values_provided": provided,
            "missing_points": missing,
            "checked_windows": checked_windows,
            "window_invariant": "test_strictly_after_train" if checked_windows else None,
            "future_points": 0,
            "lookahead_points": 0,
            "first_future_date": None,
            "first_future_index": None,
            "first_date": series.dates[0],
            "last_date": series.dates[-1],
            "span_days": span,
            "headroom_days": headroom,
            "is_monotonic": True,
            "reason": None,
            "params": params,
            "warmup": {},
        },
    )


def _bound_keys(base: str) -> tuple[str, ...]:
    """一个边界的全部可接受键名：下标拼法在前（规范 + 别名），日期拼法在后（规范 + 别名）。

    拼法由 `_WINDOW_ALIASES` 反查得来，故"切分器写了哪些别名"与"校验器认哪些别名"
    必然一致——两处各写一份列表迟早会漂。
    """
    idx_alias = _CANON_TO_ALIAS.get(f"{base}_i")   # train_start_i  -> train_start_idx
    date_alias = _CANON_TO_ALIAS.get(base)         # train_start    -> train_start_date
    return ((f"{base}_i",) + ((idx_alias,) if idx_alias else ())
            + (base,) + ((date_alias,) if date_alias else ()))


def _window_bound(w: dict, base: str, idx: int) -> Any:
    """从窗口条目里取一个边界（下标拼法优先，日期拼法兜底；两套别名都认）。

    两套键名由切分器同时写入，这里两套都认——调用方把切分器的输出直接回传即可，
    不必先做键名转换。取不到任何一套即报错（不静默当成 0：那会把"缺键"变成"第 0 位"）。
    """
    for k in _bound_keys(base):
        if k in w and w[k] is not None:
            return w[k]
    raise ComputeError(
        f"windows[{idx}]: missing {base}_i (accepted spellings: "
        f"{', '.join(_bound_keys(base))}) — pass the window list returned by "
        f"split_walk_forward_windows, or dicts carrying the same keys"
    )


def _bound_to_index(
    bound: Any, canonical: str, idx: int, index_of: dict[str, int],
    axis_dates: Sequence[str], name: str,
) -> int:
    """把一个边界统一成日期轴下标：整数按下标用，字符串按日期查。

    整数按下标用（`*_i` / `*_idx` 两套都是下标）；字符串必须命中被检序列的日期轴——
    窗口若描述的是另一段历史，这份"窗口计划"根本无从复核，报错点名该日期。
    """
    if isinstance(bound, bool):
        raise ComputeError(f"windows[{idx}] ({name}): {canonical!r} must be an int or ISO date")
    if isinstance(bound, int):
        return bound
    key = str(bound)
    if key not in index_of:
        raise ComputeError(
            f"windows[{idx}] ({name}): boundary date {key} is not on the validated series axis "
            f"({axis_dates[0]} .. {axis_dates[-1]}) — the window plan describes a different "
            f"range than the series, so it cannot be checked"
        )
    return index_of[key]


def _check_window_plan(
    windows: Optional[Sequence[Any]], axis_dates: Sequence[str], name: str
) -> int:
    """逐窗断言测试区间严格晚于训练区间（无 windows → 0）。

    这是"窗口计划本身有没有偷看未来"的检查：把 `test_start > train_end` 从切分器的
    构造性质升格为可被调用方**独立复核**的断言。先建日期→下标索引，使两套键名
    （`*_i` 下标 / `*_date` 日期）都能归一到同一下标坐标系。
    """
    if windows is None:
        return 0
    index_of = {d: i for i, d in enumerate(axis_dates)}
    for k, w in enumerate(windows):
        if not isinstance(w, dict):
            raise ComputeError(
                f"windows[{k}] must be a dict (the window entries returned by "
                f"split_walk_forward_windows), got {type(w).__name__}"
            )
        a0 = _bound_to_index(_window_bound(w, "train_start", k), "train_start_i", k,
                             index_of, axis_dates, name)
        a1 = _bound_to_index(_window_bound(w, "train_end", k), "train_end_i", k,
                             index_of, axis_dates, name)
        b0 = _bound_to_index(_window_bound(w, "test_start", k), "test_start_i", k,
                             index_of, axis_dates, name)
        b1 = _bound_to_index(_window_bound(w, "test_end", k), "test_end_i", k,
                             index_of, axis_dates, name)
        if a0 > a1 or b0 > b1:
            raise ComputeError(
                f"windows[{k}] ({name}): window boundaries are inverted "
                f"(train {a0}..{a1}, test {b0}..{b1}) — pass the window list returned by "
                f"split_walk_forward_windows"
            )
        if not b0 > a1:
            raise ComputeError(
                f"windows[{k}] ({name}): lookahead in the window plan — the test range "
                f"{b0}..{b1} does not start strictly after the train range {a0}..{a1} "
                f"(test_start={b0} must be > train_end={a1}). The test window overlaps its own "
                f"training data, so any score computed from it is not out-of-sample. Rebuild "
                f"the plan with split_walk_forward_windows."
            )
    return len(windows)


def check_series_as_of_bounds(
    dates: Sequence[Any],
    values: Sequence[Any],
    as_of: str,
) -> dict:
    """序列口径体检：在 as_of 口径下报出这段序列的缺失与覆盖情况。

    与 `validate_no_lookahead` 的分工：那个是**护栏**（有未来数据就报错中止），
    这个是**体检**（序列合法但稀疏/前段缺失时，调用方需要知道缺多少才能判断
    结果能不能用）。两者都不做数值计算、都不改数据。

    Input: dates (ISO 日期序列, 严格递增; 晚于 as_of 的点被拒绝并点名),
      values (与之等长的数值序列, null 表示缺失),
      as_of (ISO 日期, 必填).
    Returns: point_count, missing_count (null 点数), missing_ratio (缺失占比, 小数),
      warmup (首个非缺失点之前的缺失个数; 全缺失则为 null), first_date / last_date,
      span_days, is_complete (无缺失时为 true), params; 顶层 warmup 键与前一列同义;
      另加统一外壳与 evidence。
      **拒绝语义**: as_of 非法 / 长度不等 / 日期非严格递增 / 有未来数据(点名首个越界
      日期) / 序列为空 → ComputeError。
    """
    as_of_v = check_as_of(as_of)
    series = check_series(dates, values, as_of=as_of_v, name="as_of_bounds")
    n = len(series)

    missing = 0
    warmup: Optional[int] = None
    for i in range(n):
        v = series.values[i]
        if not np.isfinite(v):
            missing += 1
        elif warmup is None:
            warmup = i

    span = (_date.fromisoformat(series.dates[-1]) - _date.fromisoformat(series.dates[0])).days
    builder = _builder("check_series_as_of_bounds", as_of_v, {"points": n},
                       "missing_count", missing)
    return result_envelope(
        "check_series_as_of_bounds",
        as_of_v,
        series,
        builder,
        {
            "point_count": n,
            "missing_count": missing,
            "missing_ratio": missing / n,
            "warmup": warmup,
            "first_date": series.dates[0],
            "last_date": series.dates[-1],
            "span_days": span,
            "is_complete": missing == 0,
            "params": {},
        },
    )


def check_observation_sufficiency(
    dates: Sequence[Any],
    as_of: str,
    values: Optional[Sequence[Any]] = None,
    required_observations: Any = DEFAULT_REQUIRED_OBSERVATIONS,
) -> dict:
    """观测充分性闸门：这批数据够不够支撑一次分析（先问够不够，再花算力）。

    这是 pack 在跑昂贵流程**之前**问的第一个问题：'我手上这段历史够算年化波动/回撤/
    因子 IC 吗？'。答案是一个布尔值加三个数字，而不是异常——数据不够不是输入错误，
    只是结论不成立，调用方需要据此**换口径或放弃**。
    **可用的观测数**：给了 `values` 时只数**非 null**的点（缺失点不参与任何统计，
    252 个点里 100 个 null 不算"够 252 个观测"）；未给 `values` 时按 `dates` 的点数计。
    Input: dates (ISO, 严格递增, 不得晚于 as_of), as_of (ISO, 必填; 口径截止),
      values (可选; 与 dates 等长, null 表缺失; 默认 null = 按日期点数计), 
      required_observations (所需观测数, 默认 252 —— 约一个交易年的日频 bar, 是"够算
      年化波动与回撤"的通行最小样本; 非负整数, 0 表示不设门槛).
    Returns: sufficient (bool; usable_observations >= required_observations) /
      usable_observations (有效观测数, 见上) / observation_count (同 usable_observations;
      别名) / point_count (日期轴长度) / missing_count (null 点数; 未传 values 时为 0) /
      required_observations (生效值) / shortfall (还差多少; 足够时为 0) /
      surplus (超出多少; 不足时为 0) / ratio (usable / required; required=0 时为 null) /
      first_date / last_date / span_days / params / warmup (空字典: 不消耗滚动窗);
      另加统一外壳 as_of 起止与 evidence。**本工具不改数据**（只读计数）。
      **拒绝语义**: as_of 非法 / 有未来数据(点名首个越界日期) / 日期非严格递增 / 长度不等 /
      序列为空 / required_observations 为负或非整数 → ComputeError。
    """
    a = check_as_of(as_of)
    req = _check_nonneg_int(required_observations, name="required_observations")
    if values is None:
        series = _axis_series(dates, a, "observation_sufficiency")
        provided = False
    else:
        series = _prep_declared(dates, values, a, "observation_sufficiency")
        provided = True

    n = len(series)
    missing = 0
    if provided:
        for i in range(n):
            if not np.isfinite(series.values[i]):
                missing += 1
    usable = n - missing
    sufficient = usable >= req
    shortfall = max(0, req - usable)
    span = (_date.fromisoformat(series.dates[-1]) - _date.fromisoformat(series.dates[0])).days

    params = {"required_observations": req, "values_provided": provided}
    builder = _builder("check_observation_sufficiency", a, {"required": req},
                       "usable_observations", usable)
    builder.add("sufficient", sufficient, params={"required": req})
    return result_envelope(
        "check_observation_sufficiency",
        a,
        series,
        builder,
        {
            "sufficient": sufficient,
            "usable_observations": usable,
            "observation_count": usable,
            "point_count": n,
            "missing_count": missing,
            "required_observations": req,
            "shortfall": shortfall,
            "surplus": max(0, usable - req),
            "ratio": (usable / req) if req else None,
            "first_date": series.dates[0],
            "last_date": series.dates[-1],
            "span_days": span,
            "reason": None if sufficient else (
                f"observation_sufficiency: {usable} usable observation(s) < required "
                f"{req}; short by {shortfall}"
                + (f" ({missing} of {n} points are null and do not count)"
                   if missing else "")
            ),
            "params": params,
            "warmup": {},
        },
    )


def split_holdout(
    dates: Sequence[Any],
    as_of: str,
    values: Optional[Sequence[Any]] = None,
    holdout_fraction: Any = 0.2,
    embargo: Any = 0,
) -> dict:
    """训练/留出二切分：留出集恒取**最后一段**（时间序列不能随机切分）。

    **为什么留出集必须是最后一段（不是随机抽样）**：时间序列的观测有次序依赖，随机抽
    一部分当留出集，训练集里就必然含有**晚于某些留出点**的观测——模型在训练时已经见过
    "未来"，留出分数因此虚高，这正是最典型的未来函数（lookahead leak）。所以本工具只有
    一种切法：`[0, n_test_start)` 为训练、`[n_test_start, n)` 为留出，中间的 `embargo`
    个观测被**丢弃**（不属于任何一侧）。
    **embargo 的作用**：防"重叠期泄露"。若标签是未来 h 期的收益，训练集末尾的点与留出集
    开头的点在时间上相隔不到 h，两者的标签窗口会重叠——训练集于是间接包含留出期的信息。
    丢弃训练与留出之间 `embargo` 个观测即切断这条通道（通常取标签前瞻期 h）。
    切法（n = 可用观测数）：`test_size = floor(n × holdout_fraction)`，
    `test_start_idx = n - test_size`，`test_end_idx = n - 1`，
    `train_end_idx = test_start_idx - embargo - 1`，`train_start_idx = 0`；
    `floor` 用精确整数算术（见 `_floor_fraction`），使 100 个点切 20% 恒为 20 个留出点。
    Input: dates (ISO, 严格递增, 不得晚于 as_of), as_of (ISO, 必填; 口径截止),
      values (可选; 与 dates 等长; 默认 null = 只按日期轴切分, 数值不参与),
      holdout_fraction (留出占比, 默认 0.2; 需在 (0, 1) 内; 用 floor 定留出点数),
      embargo (训练与留出之间丢弃的观测数, 默认 0, 非负整数; 见上).
    Returns: train_start_idx / train_end_idx / test_start_idx / test_end_idx (整数下标,
      含端点) 与其日期别名 train_start_date / train_end_date / test_start_date /
      test_end_date, train_size / test_size (观测数), n_train / n_test (同前), embargo,
      point_count, holdout_fraction (生效值), train_fraction (实际训练占比),
      embargo_dropped (被丢弃的观测数 = min(embargo, 可用空隙)) / params / warmup (空字典);
      另加统一外壳 as_of 起止与 evidence。**本工具不改数据**（只描述下标区间）。
      **拒绝语义**: as_of 非法 / 有未来数据(点名首个越界日期) / 日期非严格递增 / 长度不等 /
      序列为空 / holdout_fraction 不在 (0,1) / embargo 为负或非整数 /
      留出集为空(占比太小) / 训练集为空(embargo 或占比过大) → ComputeError 点名数字。
    """
    a = check_as_of(as_of)
    frac = _require_finite_param(holdout_fraction, name="holdout_fraction")
    if not (0.0 < frac < 1.0):
        raise ComputeError(
            f"holdout_fraction must be in (0, 1) exclusive, got {holdout_fraction!r} "
            f"(0.2 = last 20% of the observations is held out)"
        )
    emb = _check_nonneg_int(embargo, name="embargo")
    if values is None:
        series = _axis_series(dates, a, "holdout")
    else:
        series = _prep_declared(dates, values, a, "holdout")
    n = len(series)

    test_size = _floor_fraction(n, frac)
    if test_size < 1:
        raise ComputeError(
            f"holdout: holdout_fraction={frac} on {n} observation(s) yields 0 holdout "
            f"point(s) (floor({n} x {frac}) = 0); use a larger fraction or a longer series "
            f"(the holdout must contain at least one observation)"
        )
    test_start = n - test_size
    train_end = test_start - emb - 1
    if train_end < 0:
        raise ComputeError(
            f"holdout: embargo={emb} leaves no training observations — the holdout starts at "
            f"index {test_start} (last {test_size} of {n} observation(s)) and dropping {emb} "
            f"observation(s) before it leaves {test_start - emb} for training; reduce embargo "
            f"(<= {max(0, test_start - 1)}) or holdout_fraction"
        )

    train_size = train_end + 1
    params = {"holdout_fraction": frac, "embargo": emb, "values_provided": values is not None}
    builder = _builder("split_holdout", a, {"holdout_fraction": frac, "embargo": emb},
                       "holdout_size", test_size)
    builder.add("train_size", train_size, params={"holdout_fraction": frac, "embargo": emb})
    return result_envelope(
        "split_holdout",
        a,
        series,
        builder,
        {
            "train_start_idx": 0,
            "train_end_idx": train_end,
            "test_start_idx": test_start,
            "test_end_idx": n - 1,
            "train_start_date": series.dates[0],
            "train_end_date": series.dates[train_end],
            "test_start_date": series.dates[test_start],
            "test_end_date": series.dates[-1],
            "train_size": train_size,
            "test_size": test_size,
            "n_train": train_size,
            "n_test": test_size,
            "embargo": emb,
            "embargo_dropped": min(emb, test_start),
            "point_count": n,
            "holdout_fraction": frac,
            "train_fraction": train_size / n,
            "reason": None,
            "params": params,
            "warmup": {},
        },
    )


TOOLS: list[ToolSpec] = [
    ToolSpec("split_walk_forward_windows", GROUP, split_walk_forward_windows),
    ToolSpec("split_holdout", GROUP, split_holdout),
    ToolSpec("validate_no_lookahead", GROUP, validate_no_lookahead),
    ToolSpec("check_observation_sufficiency", GROUP, check_observation_sufficiency),
    ToolSpec("check_series_as_of_bounds", GROUP, check_series_as_of_bounds),
]
