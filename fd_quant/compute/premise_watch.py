"""前提可监控性分拣：把 `preconditions[]` 分成「能变告警规则」与「不可监控」。

为什么需要单独一个纯函数
------------------------
`add-decision-consistency-and-premise-watch` 的核心风险不是"映射失败"，而是
**失败被静默吞掉**：一条前提没变成规则，如果不进清单，它就等于"已监控"——
于是「前提失效」永远不会响，而所有人都以为有人在看。

所以分拣必须是显式的、可计数的、可断言的：

    monitorable + unmonitorable == 前提总数

这个等式是清单完整性的全部依据。

纯计算：不取数、不读账本、不写规则。真正的规则创建在 wire 侧（复用
`alert-evaluation` 既有调度与状态机），本模块只回答"这条前提能不能核、不能核为什么"。
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

from . import ToolSpec
from .base import ComputeError, EvidenceBuilder, result_envelope

GROUP = "decision_consistency"

COMPARATORS = ("lt", "lte", "gt", "gte", "eq", "neq")

#: 不可监控的原因码。分码而非自由文本：只有分码才能被计数、被巡检、被断言。
REASON_MISSING_METRIC = "missing_metric_ref"
REASON_METRIC_NOT_RESOLVABLE = "metric_not_resolvable"
REASON_BAD_COMPARATOR = "invalid_comparator"
REASON_BAD_THRESHOLD = "invalid_threshold"
REASON_MISSING_AS_OF = "missing_as_of"
REASON_NO_WINDOW = "window_undetermined"
REASON_NOT_A_MAPPING = "not_a_mapping"


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def classify_preconditions(
    preconditions: Optional[Sequence[Mapping[str, Any]]],
    *,
    as_of: str,
    #: 二元区间 `[start, end]`。**不是**字符串：契约里的 `time_horizon` 是字符串
    #: （`Rating`/`Decision` 都要求 string|null），但那个字符串无法表达窗口边界，
    #: 本工具要的是显式区间。标注曾写成 `Optional[str]`，于是工具面告诉 agent
    #: "传字符串"，而实现只认列表——每条前提都会被判 `window_undetermined`。
    time_horizon: Optional[Sequence[str]] = None,
    decision_as_of: Optional[str] = None,
    resolvable_metrics: Optional[Sequence[str]] = None,
) -> dict:
    """把前提分成可监控与不可监控两组。

    Input:
      preconditions —— 决策记录里的 `preconditions` 数组，元素形如
        `{claim, metric_ref, comparator, threshold, as_of}`。
      as_of —— 口径日（必填）。晚于该日的前提**拒绝计算并点名**（计算面纪律：
        不得用晚于口径日的信息判定当时的可监控性）。
      time_horizon —— 观察窗的**唯一**来源：须为显式区间 `[start, end]`；否则判
        `window_undetermined`（**不**默认取无限窗：那会让规则永不触发却看起来在监控）。
      decision_as_of —— **保留但不参与窗口推导**（历史参数，曾出现在错误文案里，
        见 `_derive_window` 的说明）。传它不会让前提变可监控。
      resolvable_metrics —— 可选；给出即对 `metric_ref` 做白名单校验，
        不在其中判 `metric_not_resolvable`。不给则只做非空校验
        （可解析性由调用方在创建规则时再核）。

    Returns: 字典，含
      monitorable —— [{index, metric_ref, comparator, threshold, as_of, window}]
      unmonitorable —— [{index, reason, detail, raw}]
      counts —— {total, monitorable, unmonitorable}
      complete —— 恒等于 `monitorable + unmonitorable == total`（自证用）
    """
    if not isinstance(as_of, str) or len(as_of) < 10:
        raise ComputeError("as_of 必填（ISO 日期），计算面要求显式口径日")

    items = list(preconditions or ())
    future = sorted({
        str((raw or {}).get("as_of")) for raw in items
        if isinstance(raw, Mapping) and str(raw.get("as_of") or "") > as_of[:10]
    })
    if future:
        raise ComputeError(
            f"前提口径日晚于 as_of={as_of}：{', '.join(future)}；"
            "用未来前提判定当时可监控性会产生当时并不成立的结论，故拒绝而非截断"
        )
    monitorable: list[dict] = []
    unmonitorable: list[dict] = []
    allow = set(resolvable_metrics) if resolvable_metrics is not None else None

    for idx, raw in enumerate(items):
        if not isinstance(raw, Mapping):
            unmonitorable.append({
                "index": idx, "reason": REASON_NOT_A_MAPPING,
                "detail": f"前提不是对象（{type(raw).__name__}）", "raw": raw,
            })
            continue

        metric_ref = raw.get("metric_ref")
        if not isinstance(metric_ref, str) or not metric_ref.strip():
            unmonitorable.append({
                "index": idx, "reason": REASON_MISSING_METRIC,
                "detail": "metric_ref 缺失或为空，无法定位被评测指标", "raw": dict(raw),
            })
            continue
        metric_ref = metric_ref.strip()
        if allow is not None and metric_ref not in allow:
            unmonitorable.append({
                "index": idx, "reason": REASON_METRIC_NOT_RESOLVABLE,
                "detail": f"metric_ref {metric_ref!r} 不在可解析集合内",
                "raw": dict(raw),
            })
            continue

        comparator = raw.get("comparator")
        if comparator not in COMPARATORS:
            unmonitorable.append({
                "index": idx, "reason": REASON_BAD_COMPARATOR,
                "detail": f"比较子非法：{comparator!r}；合法集合 {list(COMPARATORS)}",
                "raw": dict(raw),
            })
            continue

        threshold = raw.get("threshold")
        if not _is_number(threshold):
            unmonitorable.append({
                "index": idx, "reason": REASON_BAD_THRESHOLD,
                "detail": f"阈值不是有限数值：{threshold!r}",
                "raw": dict(raw),
            })
            continue

        as_of = raw.get("as_of")
        if not isinstance(as_of, str) or not as_of.strip():
            unmonitorable.append({
                "index": idx, "reason": REASON_MISSING_AS_OF,
                "detail": "缺少口径日 as_of，无法判定评测基准时点",
                "raw": dict(raw),
            })
            continue

        window = _derive_window(time_horizon)
        if window is None:
            unmonitorable.append({
                "index": idx, "reason": REASON_NO_WINDOW,
                "detail": ("观察窗无法确定：time_horizon 未给出显式区间 [start, end]；"
                           "不默认取无限窗（那会让规则永不触发却看起来在监控）"),
                "raw": dict(raw),
            })
            continue

        monitorable.append({
            "index": idx,
            "metric_ref": metric_ref,
            "comparator": comparator,
            "threshold": float(threshold),
            "as_of": as_of.strip(),
            "window": window,
        })

    total = len(items)
    b = EvidenceBuilder("decision_consistency:check_preconditions_monitorable", as_of)
    b.add("preconditions_total", total, unit="count")
    b.add("monitorable_count", len(monitorable), unit="count")
    return result_envelope(
        "decision_consistency:check_preconditions_monitorable",
        as_of,
        None,
        b,
        {
            "monitorable": monitorable,
            "unmonitorable": unmonitorable,
            "counts": {
                "total": total,
                "monitorable": len(monitorable),
                "unmonitorable": len(unmonitorable),
            },
            # 自证：清单必须覆盖每一条前提，一条都不能凭空消失。
            "complete": len(monitorable) + len(unmonitorable) == total,
        },
    )


def _derive_window(time_horizon: Optional[Sequence[str]]) -> Optional[list[str]]:
    """推导观察窗 ``[start, end]``。

    **唯一**来源是 `time_horizon` 的显式区间形式 ``[start, end]``（且 start <= end）；
    其余一律返回 None。

    刻意不再接受 `premise_as_of` / `decision_as_of`：本函数曾带着这两个参数却从不
    读取，而 docstring 写着"起点取前提的口径日"、错误文案写着"既无 time_horizon
    也无 decision_as_of"——照着文档传 `decision_as_of` 的人会拿到
    `window_undetermined`，进而误判"前提不可监控"。参数删掉比留在签名里撒谎好。
    """
    if isinstance(time_horizon, (list, tuple)) and len(time_horizon) == 2:
        start, end = str(time_horizon[0]), str(time_horizon[1])
        if start <= end:
            return [start, end]
    return None


#: 工具面声明。工具名用 "check" 起头（计算面 verb_noun 约定），实现沿用
#: `classify_preconditions` —— 名实一致：它就是在检查"这些前提能不能被监控"。
TOOLS: list[ToolSpec] = [
    ToolSpec("check_preconditions_monitorable", GROUP, classify_preconditions),
]

SAMPLES: dict[str, dict] = {
    "check_preconditions_monitorable": {
        "as_of": "2026-11-10",
        "preconditions": [
            {"claim": "毛利率不低于 42%", "metric_ref": "financials.gross_margin",
             "comparator": "gte", "threshold": 0.42, "as_of": "2026-11-10"},
        ],
        "time_horizon": ["2026-11-10", "2027-05-10"],
    },
}
