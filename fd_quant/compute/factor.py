"""因子统计族：因子统计的四个基础原子——IC / 分组收益 / 单因子暴露回归 / 因子值排名。

**边界（spec 明文）：本族不内置任何因子定义，也不内置因子库。** 因子值与收益两列
一律由调用方传入；本文件里没有"某个因子怎么算"的构造，也没有因子名到公式的登记表
——造因子属于策略层，计算层只回答"给两列按位对齐的数，它们的关系是什么"。

**口径（两列按位对齐，本层不做移位）**：`factor[i]` 与收益列第 `i` 位同属 `dates[i]`。
IC 的定义是"因子与**下一期**收益的相关系数"，但本层不替调用方做 shift：传进来的第二列
究竟是前瞻收益（已自行对齐）还是同期收益，本层无从也不该猜测（猜错等于静默换定义）。
前瞻收益是否用了 `as_of` 之后才知道的数据，是调用方责任。

**as_of 口径**：每个工具第一步都是 `check_as_of`，随后两列都经 `check_series` 与同一份
`dates` 逐位配对校验；任一列含晚于 `as_of` 的数据点即拒绝并点名**首个**越界日期，
绝不静默截断。

**缺失纪律：逐对删除（pairwise deletion），绝不填充。** 任一侧为 null/±inf 的观测对整对
剔除，不参与任何统计量；剔除规模与位置由 `n_dropped` / `used_indices` 回给调用方（横截面
构成本身是答案的一部分，故删除必须可审计）。只用 0 或前值填充等于替调用方发明数据点。

**退化只拒绝、不产出 inf/NaN**：零方差（常数因子 / 常数收益）、自由度 ≤ 0（可用观测 < 3）、
残差方差为 0 导致的 0/0 一律**拒绝**（`ComputeError`，点名出问题的那一列/那个数），绝不返回
inf/NaN；t 值在标准误恰为 0（完美拟合）时以 null 呈现并附 `reason`。返回恒可
`json.dumps(..., allow_nan=False)`。

**确定性**：归约只用 `deterministic_reduction`（Kahan 从左到右）或显式循环；相关系数、
OLS 系数、R² 全程标量运算，不用 `@` / `np.dot` / `np.matmul` / `np.linalg`（BLAS 分块随
线程数变化）。排名用显式排序 + 并列扫描（不用 `argsort` 的 kind 语义），并列口径（平均名次）
在 `_average_ranks` 里只实现一次。

**名次口径的唯一性**：`compute_information_coefficient(method="spearman")` 与
`rank_factor_scores` 调用同一个 `_average_ranks`，故两者的并列取法必然一致——并列规则
只有一处实现，改一处即两处同时生效。
"""
from __future__ import annotations

import math
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
    deterministic_reduction,
    result_envelope,
    to_none_list,
)

GROUP = "factor_statistics"

#: 默认最少可用观测数：一元统计量少于 5 个点，t 值（自由度 n-2）已薄到不可用。
DEFAULT_MIN_OBSERVATIONS = 5

#: 分组收益的默认分组数：五分位是业界惯例。
DEFAULT_N_GROUPS = 5

#: 允许的 IC 口径（小写入参，大小写不敏感）。
_METHODS: tuple[str, ...] = ("pearson", "spearman")


# ── 内部工具 ─────────────────────────────────────────────────────


def _num(x: Any) -> Optional[float]:
    """标量 → JSON 安全浮点（None / NaN / ±inf → None，np 标量 → python 标量）。"""
    if x is None:
        return None
    f = float(x)
    return f if math.isfinite(f) else None


def _mean(arr: np.ndarray) -> float:
    """固定顺序均值（Kahan 从左到右），不用 np.mean 的成对归约。调用方保证非空。"""
    return deterministic_reduction(arr) / len(arr)


def _leading_nulls(arr: Any) -> int:
    """首个非 NaN 位置之前的 null 个数（全 NaN → len）。"""
    for i in range(len(arr)):
        if np.isfinite(arr[i]):
            return i
    return len(arr)


def _warmup(lines: dict) -> dict[str, int]:
    """逐输出线声明 warmup：首个非 null 位置之前的 null 个数（标量线有数记 0、null 记 1）。

    本族不消耗滚动窗口（不取滚动窗、逐对删除而非填充），故有数时一律 0；只有整条输出
    因退化被置 null（如完美拟合下的 t 值）时才非 0——调用方据此知道"这个键为什么没有数"。
    """
    out: dict[str, int] = {}
    for name, v in lines.items():
        if isinstance(v, np.ndarray):
            out[name] = _leading_nulls(v)
        elif isinstance(v, (list, tuple)):
            out[name] = _leading_nulls(np.asarray(v, dtype=np.float64)) if len(v) else 0
        else:
            out[name] = 0 if _num(v) is not None else 1
    return out


def _prep_pair(
    dates: Sequence[Any],
    left: Sequence[Any],
    right: Sequence[Any],
    as_of: str,
    *,
    left_name: str,
    right_name: str,
) -> tuple[Series, Series]:
    """两列按同一份 `dates` 逐位配对校验（两列共享日期轴，日期不各读一遍）。

    顺序：`as_of` 已由调用方校验 → check_series（ISO / 长度配对 / 不晚于 as_of / 数值）。
    越界由 check_series 点名**首个**越界日期；两列长度不一致时点名两个长度。
    """
    left_s = check_series(dates, left, as_of=as_of, name=left_name)
    right_s = check_series(dates, right, as_of=as_of, name=right_name)
    return left_s, right_s


def _usable_positions(left: np.ndarray, right: np.ndarray) -> list[int]:
    """逐对删除：两侧都有限（非 null/±inf）的观测索引，保持输入顺序。

    返回索引而非压实后的数组——索引是"哪些对参与了计算"的唯一事实源，由调用方
    以 `used_indices` 回给用户；`members`（分组）也用它，故整套统计量可逐位审计。
    """
    return [i for i in range(len(left)) if math.isfinite(float(left[i]))
            and math.isfinite(float(right[i]))]


def _require_observations(usable: int, minimum: int, *, dropped: int, tool: str) -> None:
    """可用观测数下限：不足时点名该数与下限（可操作的事实），不静默降级。"""
    if usable < minimum:
        raise ComputeError(
            f"{tool}: only {usable} usable observation(s) after pairwise deletion "
            f"({dropped} pair(s) dropped for a null/non-finite side), but min_observations="
            f"{minimum}; supply at least {minimum} aligned finite pairs"
        )


def _require_spread(arr: np.ndarray, *, name: str, n: int) -> None:
    """零方差拒绝：常数序列的相关系数/斜率是 0/0 —— 拒绝并**点名该列**（不是返回 NaN）。

    判据是平方离差和恰为 0（float 精确判定，不设容差：设容差等于在这里偷偷引入一个
    "多接近算常数"的口径，而那个阈值只能由调用方定）。
    """
    m = _mean(arr)
    if deterministic_reduction((arr - m) ** 2) == 0.0:
        raise ComputeError(
            f"{name} is constant over the {n} usable observation(s) (zero variance), so the "
            f"correlation / slope denominator is 0; the statistic is undefined here — not 0 "
            f"and not NaN. Pass a {name} series that actually varies"
        )


def _require_degrees_of_freedom(usable: int, *, tool: str) -> int:
    """残差自由度为 n - 2：≤ 0 时拒绝并点名该自由度（点名"df = 0"而不是只说"太短"）。"""
    df = usable - 2
    if df <= 0:
        raise ComputeError(
            f"{tool}: degrees of freedom n - 2 = {df} is not positive (n = {usable} usable "
            f"observation(s)); the residual standard deviation divides by it, so at least "
            f"3 usable pairs are required"
        )
    return df


def _check_method(method: Any) -> str:
    """IC 口径校验：只认 pearson / spearman（大小写不敏感），否则点名该值。"""
    if not isinstance(method, str) or method.lower() not in _METHODS:
        raise ComputeError(
            f"method must be one of {list(_METHODS)}, got {method!r} "
            f"(pearson = 原始值相关, spearman = 平均名次相关)"
        )
    return method.lower()


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson 相关系数（固定顺序 Kahan 归约）。调用方已保证两侧非常数、长度相等。

    不做夹取：Cauchy–Schwarz 保证 |r| ≤ 1，越界只可能来自 float64 舍入，而舍入幅度
    远小于本族任何判定阈值；夹取会掩盖"分母算错"这类真 bug。
    """
    mx = _mean(x)
    my = _mean(y)
    dx = x - mx
    dy = y - my
    sxx = deterministic_reduction(dx * dx)
    syy = deterministic_reduction(dy * dy)
    sxy = deterministic_reduction(dx * dy)
    return sxy / math.sqrt(sxx * syy)


def _sorted_order(values: Sequence[float]) -> list[int]:
    """按 (值升序, 输入索引) 排定观测，返回索引列表——分组与排名的唯一顺序事实源。

    键里带索引是为了让并列值的相对次序由输入位置决定（而不是由排序算法的稳定性
    隐含决定），因此"同一输入 → 同一顺序"是写死的性质，与排序实现无关。
    """
    return sorted(range(len(values)), key=lambda i: (float(values[i]), i))


def _average_ranks(values: Sequence[float]) -> list[float]:
    """平均名次（midrank）排名，1 基：名次 1 = 最小值；并列共享其名次区间的平均值。

    并列规则（写死、可复现，全库唯一实现）：
    - 相等判定用 **float 精确相等**，不设容差——设容差等于偷偷引入一个"近邻合并"的口径，
      而"多近算近"只能由调用方定；
    - 排序键为 `(值, 输入索引)`（见 `_sorted_order`），故并列值的先后由输入位置决定；
    - 一段并列（排序后第 i..j 位，0 基）内的观测各得 `((i+1) + (j+1)) / 2`，即 midrank
      （例：同为排序后第 3、4、5 位 → 各得 4.0）。
    这就是 Spearman 的通行并列处理，因此"对名次做 Pearson"与"Spearman 的定义"逐点一致；
    `compute_information_coefficient(method="spearman")` 与本族 `rank_factor_scores`
    都调用本函数，并列口径只有一处实现。
    """
    n = len(values)
    order = _sorted_order(values)
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        v = float(values[order[i]])
        while j + 1 < n and float(values[order[j + 1]]) == v:
            j += 1
        mid = ((i + 1) + (j + 1)) / 2.0  # 1 基名次 i+1 .. j+1 的平均
        for k in range(i, j + 1):
            ranks[order[k]] = mid
        i = j + 1
    return ranks


def _tie_counts(values: Sequence[float]) -> tuple[int, int]:
    """并列规模：返回 (被并列的观测数, 并列组数)。

    "被并列的观测数"= 其值在本序列里出现至少两次的观测个数（Spearman 受影响的规模）；
    "并列组数"= 由同一值构成的并列段的个数。两个数都回给调用方，因为"3 组各 2 个"
    与"1 组 6 个"对名次分布的影响不同。
    """
    counts: dict[float, int] = {}
    for v in values:
        f = float(v)
        counts[f] = counts.get(f, 0) + 1
    groups = [c for c in counts.values() if c > 1]
    return sum(groups), len(groups)


def _emit_pair_counts(
    b: EvidenceBuilder, n: int, usable: int, dropped: int, params: dict
) -> None:
    """观测构成的公共证据：输入长度 / 可用配对数 / 被删除配对数。"""
    b.add("input length", int(n), unit="points", params=params)
    b.add("usable observations", int(usable), unit="points", params=params)
    b.add("dropped pairs", int(dropped), unit="points", params=params)


def _emit_seqs(
    b: EvidenceBuilder, name: str, arr: np.ndarray, params: dict, unit: Optional[str] = None
) -> None:
    """序列型输出的证据：长度 + 首值 + 末值（点数不是信息，首末才是）。

    族作者契约 §2.11 要求序列型至少出这三个事实。缺失位不参与取值（取首个/末个有限值），
    全缺失时首末为 null——此时长度本身就是那条可操作的事实。
    """
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


# ── IC ───────────────────────────────────────────────────────────


def compute_information_coefficient(
    dates: Sequence[str],
    factor: Sequence[Optional[float]],
    forward_returns: Sequence[Optional[float]],
    as_of: str,
    method: str = "pearson",
    min_observations: int = DEFAULT_MIN_OBSERVATIONS,
) -> dict:
    """信息系数 IC：因子值与收益（通常为已对齐的前瞻收益）两列的相关系数，Pearson 或 Spearman。

    定义：Pearson IC = Σ(f-f̄)(r-r̄) / sqrt(Σ(f-f̄)² Σ(r-r̄)²)（用原始值，因子单位由调用方定，
    收益为小数）；Spearman IC = 先对两列各自取平均名次（midrank，并列共享名次），再对名次做
    同一个 Pearson。名次实现全库唯一（`_average_ranks`），故与 `rank_factor_scores` 的并列
    口径必然一致。本工具不做 shift：第二列是前瞻还是同期收益由调用方负责。
    Input: dates (ISO 日期, 严格递增, 与两列等长), factor (因子值序列, 与 dates 等长; null/±inf 表缺失),
      forward_returns (收益序列, 与 dates 等长, 小数口径; null/±inf 表缺失),
      as_of (ISO 日期, 必填; 晚于它的数据点一律拒绝并点名首个越界日期),
      method ("pearson" 默认 | "spearman", 大小写不敏感),
      min_observations (最少可用配对数, 默认 5, 正整数; 不足则拒绝并点名该数)。
    Returns: ic (相关系数, 无量纲, [-1, 1]) / method (生效口径, 小写回显) / n_observations
      (逐对删除后实际参与计算的配对数) / n_dropped (因任一侧缺失而剔除的配对数) /
      used_indices (参与计算的观测在输入中的索引, 与 ranked_* 同序) / params / warmup / evidence;
      method="spearman" 时另含 ranked_factor 与 ranked_returns (长度 = n_observations 的平均名次
      列表, 1 = 最小, 供审计; pearson 时省略这两个键)。
      无滚动窗口, 故有数时 warmup = 0。NaN 语义: 任一侧 null/±inf 的观测对逐对删除（绝不填充 0
      或前值), 删除数与位置由 n_dropped / used_indices 给出。拒绝(不返回 NaN): 可用配对数 <
      min_observations（点名该数）; 两列长度不一致（点名两个长度）; 任一列在可用观测上零方差
      （点名该列）; method 非法（点名该值）。
    """
    a = check_as_of(as_of)
    m = _check_method(method)
    mo = check_positive_window(min_observations, name="min_observations")
    f, r = _prep_pair(
        dates, factor, forward_returns, a,
        left_name="factor", right_name="forward_returns",
    )
    n = len(f)
    used = _usable_positions(f.values, r.values)
    usable = len(used)
    dropped = n - usable
    _require_observations(
        usable, mo, dropped=dropped, tool="compute_information_coefficient"
    )

    x = f.values[used]
    y = r.values[used]
    _require_spread(x, name="factor", n=usable)
    _require_spread(y, name="forward_returns", n=usable)

    params = {"method": m, "min_observations": mo}
    b = EvidenceBuilder("compute_information_coefficient", a)
    _emit_pair_counts(b, n, usable, dropped, params)
    b.add("method", m, unit=None, params=params)

    payload: dict[str, Any] = {
        "method": m,
        "n_observations": usable,
        "n_dropped": dropped,
        "used_indices": [int(i) for i in used],
        "params": params,
    }
    if m == "spearman":
        ranks_factor = _average_ranks(x)
        ranks_returns = _average_ranks(y)
        rank_f = np.asarray(ranks_factor, dtype=np.float64)
        rank_r = np.asarray(ranks_returns, dtype=np.float64)
        ic = _pearson(rank_f, rank_r)
        payload["ranked_factor"] = to_none_list(rank_f)
        payload["ranked_returns"] = to_none_list(rank_r)
        _emit_seqs(b, "ranked factor", rank_f, params, unit="rank")
        b.add("ic", _num(ic), unit="correlation", params=params)
    else:
        ic = _pearson(x, y)
        b.add("ic", _num(ic), unit="correlation", params=params)

    payload["ic"] = _num(ic)
    payload["warmup"] = _warmup({"ic": ic})
    return result_envelope("compute_information_coefficient", a, f, b, payload)


# ── 分组收益 ─────────────────────────────────────────────────────


def compute_grouped_returns(
    dates: Sequence[str],
    factor: Sequence[Optional[float]],
    forward_returns: Sequence[Optional[float]],
    as_of: str,
    n_groups: int = DEFAULT_N_GROUPS,
) -> dict:
    """分组收益：按因子值升序把观测分成 n_groups 个等权组合，逐组算收益均值，并给最高减最低。

    分组规则（写死、可复现）：先按 (因子值升序, 输入索引) 稳定排定；设可用观测 n、组数 g，
    则 `n = g * base + rem`（0 ≤ rem < g），**前 rem 组（因子值最低的那几组）各多拿一个名额**，
    即组大小 = base+1（前 rem 组）或 base（其余），最多相差 1——余数只进低位组，与
    `np.array_split` 的惯例一致。组内等权即取组内收益的算术平均（Kahan 固定顺序）。
    并列因子值不合并：边界按**名次**定（按值定会让组大小不均，或需要一个任意的并列归组规则）。
    组编号 1 = 因子值最低组（与 `rank_factor_scores` 的"1 = 最小"一致）。
    Input: dates (ISO 日期, 严格递增, 与两列等长), factor (因子值序列, 与 dates 等长; null/±inf 表缺失),
      forward_returns (收益序列, 与 dates 等长, 小数口径; null/±inf 表缺失),
      as_of (ISO 日期, 必填; 晚于它的数据点一律拒绝并点名首个越界日期),
      n_groups (分组数, 默认 5; 必须是 ≥ 2 的整数, 且不超过逐对删除后的可用观测数)。
    Returns: groups (长度 n_groups 的列表, 每项 {group: 1 基组号, 1 = 因子最低组; n_members:
      组内成员数; mean_return: 组内等权收益均值, 小数口径; members: 成员在**输入**中的索引列表,
      升序按因子值排定, 供审计}) / top_minus_bottom (最高组均值 - 最低组均值, 小数) /
      monotonic (逐组均值是否非递减, bool; 精确比较, 相等不算违反) / n_observations (可用配对数) /
      n_dropped (逐对删除的对数) / params / warmup / evidence。
      无滚动窗口, 故有数时 warmup = 0。NaN 语义: 任一侧 null/±inf 的观测对整对剔除（绝不填充),
      组均值只在组内成员上计算; 每个观测最多出现在一个组（groups 构成可用观测的一个划分）。
      拒绝: n_groups < 2 或 n_groups > 可用观测数（点名两个数）; 两列长度不一致（点名两个长度）。
    """
    a = check_as_of(as_of)
    g = check_positive_window(n_groups, name="n_groups")
    if g < 2:
        raise ComputeError(
            f"n_groups must be at least 2 so a highest group can be compared with a lowest "
            f"group (got {n_groups!r})"
        )
    f, r = _prep_pair(
        dates, factor, forward_returns, a,
        left_name="factor", right_name="forward_returns",
    )
    n = len(f)
    used = _usable_positions(f.values, r.values)
    usable = len(used)
    dropped = n - usable
    if g > usable:
        raise ComputeError(
            f"n_groups={g} exceeds usable observations={usable} after pairwise deletion "
            f"({dropped} pair(s) dropped for a null/non-finite side); every group needs at "
            f"least one observation (reduce n_groups or supply more aligned finite pairs)"
        )

    params = {"n_groups": g}
    b = EvidenceBuilder("compute_grouped_returns", a)
    _emit_pair_counts(b, n, usable, dropped, params)
    b.add("n groups", g, unit="count", params=params)

    order = [used[k] for k in _sorted_order([float(f.values[i]) for i in used])]
    base, rem = divmod(usable, g)
    groups: list[dict[str, Any]] = []
    means: list[float] = []
    cursor = 0
    for k in range(g):
        size = base + (1 if k < rem else 0)
        members = order[cursor:cursor + size]
        cursor += size
        returns = np.asarray([float(r.values[i]) for i in members], dtype=np.float64)
        mean_return = _mean(returns)
        means.append(mean_return)
        groups.append({
            "group": k + 1,                      # 1 = 因子值最低组
            "n_members": int(size),
            "mean_return": _num(mean_return),
            "members": [int(i) for i in members],
        })
    top_minus_bottom = means[-1] - means[0]
    monotonic = all(means[i] <= means[i + 1] for i in range(len(means) - 1))

    b.add("top minus bottom", _num(top_minus_bottom), unit=None, params=params)
    b.add("monotonic", bool(monotonic), unit=None, params=params)
    _emit_seqs(b, "group mean return", np.asarray(means, dtype=np.float64), params,
               unit="per-period decimal")
    b.add("largest group", int(max(x["n_members"] for x in groups)), unit="members",
          params=params)
    return result_envelope("compute_grouped_returns", a, f, b, {
        "groups": groups,
        "top_minus_bottom": _num(top_minus_bottom),
        "monotonic": bool(monotonic),
        "n_observations": usable,
        "n_dropped": dropped,
        "params": params,
        "warmup": _warmup({
            "group_mean_return": means,
            "top_minus_bottom": top_minus_bottom,
            "monotonic": monotonic,
        }),
    })


# ── 单因子暴露回归 ───────────────────────────────────────────────


def compute_factor_exposure(
    dates: Sequence[str],
    factor: Sequence[Optional[float]],
    returns: Sequence[Optional[float]],
    as_of: str,
    min_observations: int = DEFAULT_MIN_OBSERVATIONS,
) -> dict:
    """单因子暴露回归（OLS，含截距）：returns = alpha + beta × factor + e。

    闭式最小二乘，全程标量 + 固定顺序 Kahan 归约（**无矩阵运算**）：
    beta = Σ(f-f̄)(r-r̄) / Σ(f-f̄)²；alpha = r̄ - beta × f̄；
    residual_i = r_i - alpha - beta × f_i；residual_stdev = sqrt(Σresidual² / (n - 2))（ddof = n - 2，
    即残差自由度；故需 n ≥ 3，否则拒绝并点名该自由度）；
    se_beta = residual_stdev / sqrt(Σ(f-f̄)²)；se_alpha = residual_stdev × sqrt(1/n + f̄²/Σ(f-f̄)²)；
    t_beta = beta / se_beta、t_alpha = alpha / se_alpha（H0: 该系数 = 0；自由度 n - 2）。
    R² = 1 - Σresidual² / Σ(r-r̄)²，含截距的 OLS 数学上保证 R² ∈ [0, 1]：这里**断言而非夹取**
    ——越界说明实现有 bug（或响应接近退化时出现灾难性抵消），让它响亮地失败，不静默改数。
    不输出 p 值（查表需要分布函数，本层不引入 scipy）。
    Input: dates (ISO 日期, 严格递增, 与两列等长), factor (因子值序列, 与 dates 等长; null/±inf 表缺失),
      returns (收益序列——被解释变量, 与 dates 等长, 小数口径; null/±inf 表缺失),
      as_of (ISO 日期, 必填; 晚于它的数据点一律拒绝并点名首个越界日期),
      min_observations (最少可用配对数, 默认 5, 正整数; 不足则拒绝并点名该数)。
    Returns: alpha (截距; 与 returns 同单位, 每期) / beta (因子暴露, 倍率, 无量纲) / t_alpha 与
      t_beta (H0: 系数 = 0 的 t 值, 自由度 n - 2; 残差标准差恰为 0 的完美拟合下标准误为 0,
      两值以 null 呈现并附 reason, 绝不返回 inf) / r_squared (决定系数, [0, 1]) / residual_stdev
      (残差标准差, ddof = n - 2, 与 returns 同单位) / n_observations (可用配对数) / n_dropped
      (被剔除的缺失对数) / used_indices / reason (t 值为 null 时给原因, 否则 null) / params /
      warmup / evidence。无滚动窗口, 故有数时 warmup = 0（t 值为 null 时该线为 1）。
      NaN 语义: 任一侧 null/±inf 的观测对逐对删除（绝不填充）后估计, 删除规模与位置可审计。
      拒绝: 可用配对数 < min_observations（点名该数）; 自由度 n - 2 ≤ 0（点名该自由度）;
      任一列在可用观测上零方差（点名该列）; 两列长度不一致（点名两个长度）。
    """
    a = check_as_of(as_of)
    mo = check_positive_window(min_observations, name="min_observations")
    f, r = _prep_pair(dates, factor, returns, a, left_name="factor", right_name="returns")
    n = len(f)
    used = _usable_positions(f.values, r.values)
    usable = len(used)
    dropped = n - usable
    _require_observations(usable, mo, dropped=dropped, tool="compute_factor_exposure")
    df = _require_degrees_of_freedom(usable, tool="compute_factor_exposure")

    x = f.values[used]
    y = r.values[used]
    _require_spread(x, name="factor", n=usable)
    _require_spread(y, name="returns", n=usable)

    params = {"min_observations": mo}
    b = EvidenceBuilder("compute_factor_exposure", a)
    _emit_pair_counts(b, n, usable, dropped, params)

    mx = _mean(x)
    my = _mean(y)
    dx = x - mx
    dy = y - my
    sxx = deterministic_reduction(dx * dx)
    sxy = deterministic_reduction(dx * dy)
    syy = deterministic_reduction(dy * dy)

    beta = sxy / sxx
    alpha = my - beta * mx
    residuals = y - alpha - beta * x
    rss = deterministic_reduction(residuals * residuals)
    residual_stdev = math.sqrt(rss / df)

    r_squared = 1.0 - rss / syy
    if not (0.0 <= r_squared <= 1.0):
        # 含截距的 OLS 下 R² 数学上必在 [0, 1]；越界只可能是实现 bug 或灾难性抵消。
        # 夹取会把它藏起来，所以这里拒绝：响应的有效信息不足以支撑一次回归。
        raise ComputeError(
            f"compute_factor_exposure: r_squared={r_squared!r} fell outside [0, 1] "
            f"(rss={rss!r}, syy={syy!r}, {usable} usable observation(s)); OLS with an intercept "
            f"cannot do that mathematically, so the response carries no resolvable signal "
            f"here (catastrophic cancellation) and no regression is reported"
        )

    se_beta = residual_stdev / math.sqrt(sxx)
    se_alpha = residual_stdev * math.sqrt(1.0 / usable + (mx * mx) / sxx)
    # 标准误恰为 0（完美拟合）→ t 是 0/0：以 null + reason 呈现, 绝不返回 inf。
    t_beta = beta / se_beta if se_beta != 0.0 else None
    t_alpha = alpha / se_alpha if se_alpha != 0.0 else None
    reason: Optional[str] = None
    if t_beta is None or t_alpha is None:
        reason = (
            "residual standard deviation is 0 (perfect fit): alpha and beta are exact, their "
            "standard errors are 0, so the t-values are undefined here — not infinite"
        )

    b.add("alpha", _num(alpha), unit="per-period decimal", params=params)
    b.add("beta", _num(beta), unit="ratio", params=params)
    b.add("t beta", _num(t_beta), unit=None, params=params)
    b.add("r squared", _num(r_squared), unit="ratio", params=params)
    b.add("residual stdev", _num(residual_stdev), unit="per-period decimal", params=params)
    return result_envelope("compute_factor_exposure", a, f, b, {
        "alpha": _num(alpha),
        "beta": _num(beta),
        "t_alpha": _num(t_alpha),
        "t_beta": _num(t_beta),
        "r_squared": _num(r_squared),
        "residual_stdev": _num(residual_stdev),
        "n_observations": usable,
        "n_dropped": dropped,
        "used_indices": [int(i) for i in used],
        "reason": reason,
        "params": params,
        "warmup": _warmup({
            "alpha": alpha, "beta": beta, "t_alpha": t_alpha, "t_beta": t_beta,
            "r_squared": r_squared, "residual_stdev": residual_stdev,
        }),
    })


# ── 因子值排名 ───────────────────────────────────────────────────


def rank_factor_scores(
    dates: Sequence[str],
    factor: Sequence[Optional[float]],
    as_of: str,
) -> dict:
    """因子值排名：对因子值取 1 基平均名次（midrank），1 = 最小值，并给出排定顺序与并列规模。

    这不是"因子"构造，只是把一列调用方给的值换成名次：`compute_information_coefficient(
    method="spearman")` 用的就是同一份 `_average_ranks`，故并列规则（float 精确相等判定、
    排序键 (值, 输入索引)、并列共享 midrank）在本族**只有一处实现**，两者的名次必然一致。
    缺失位不参与排名（名次为 null，不占名次位也不被填值）：名次是相对的，替缺失位发明一个
    位置等于发明数据。
    Input: dates (ISO 日期, 严格递增, 与 factor 等长), factor (因子值序列, 与 dates 等长;
      null/±inf 视为缺失), as_of (ISO 日期, 必填; 晚于它的数据点一律拒绝并点名首个越界日期)。
    Returns: ranks (与输入等长的列表, 1 = 最小值; 有限值处为平均名次 float, 缺失处为 null) /
      order (参与排名的有限观测按 (值升序, 输入索引) 排定后的**输入索引**列表, 长度 =
      n_observations) / n_ties (其值在序列里出现至少两次的观测数) / n_tie_groups (并列段数) /
      n_observations (参与排名的有限观测数) / n_dropped (缺失观测数) / params / warmup / evidence。
      无滚动窗口, 故 warmup = 首个 null 之前的 null 个数（缺失全在首位时为 n_dropped, 否则 0）。
      拒绝: factor 全是缺失（无值可排, 点名该列）; 日期越界 / 非 ISO / 非严格递增（点名首个
      出问题的日期）。
    """
    a = check_as_of(as_of)
    f = check_series(dates, factor, as_of=a, name="factor")
    n = len(f)
    finite = [i for i in range(n) if math.isfinite(float(f.values[i]))]
    dropped = n - len(finite)
    if not finite:
        raise ComputeError(
            f"factor has no finite value at all ({n} point(s), all null/non-finite); there is "
            f"nothing to rank — supply at least one finite factor value"
        )

    values = [float(f.values[i]) for i in finite]
    ranks_sub = _average_ranks(values)
    ranks = np.full(n, np.nan)
    for k, i in enumerate(finite):
        ranks[i] = ranks_sub[k]
    order = [finite[k] for k in _sorted_order(values)]
    n_ties, n_tie_groups = _tie_counts(values)

    params: dict = {}
    b = EvidenceBuilder("rank_factor_scores", a)
    _emit_pair_counts(b, n, len(finite), dropped, params)
    b.add("n ties", int(n_ties), unit="observations", params=params)
    _emit_seqs(b, "ranks", ranks, params, unit="rank")
    return result_envelope("rank_factor_scores", a, f, b, {
        "ranks": to_none_list(ranks),
        "order": [int(i) for i in order],
        "n_ties": int(n_ties),
        "n_tie_groups": int(n_tie_groups),
        "n_observations": len(finite),
        "n_dropped": dropped,
        "params": params,
        "warmup": _warmup({"ranks": ranks}),
    })


TOOLS: list[ToolSpec] = [
    ToolSpec("compute_information_coefficient", GROUP, compute_information_coefficient),
    ToolSpec("compute_grouped_returns", GROUP, compute_grouped_returns),
    ToolSpec("compute_factor_exposure", GROUP, compute_factor_exposure),
    ToolSpec("rank_factor_scores", GROUP, rank_factor_scores),
]
