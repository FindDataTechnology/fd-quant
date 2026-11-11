"""决策一致性：跨 pack 结论冲突的纯计算检出。

设计边界
--------
* **纯计算**：只对传入的记录作答，不取数、不联网、不调 LLM、不读账本。与计算面
  其余族同一姿态（`tests/test_offline_boundary.py` 钉住）。
* **只呈现不裁决**：输出冲突清单，不含建议、排序或"哪一方更可信"。把两个有纪律的
  判断压成一个没纪律的判断，是比冲突本身更糟的结果。
* **三类冲突，其余差异不算冲突**：方向互斥、证据反向引用、前提矛盾。
* **时序演进与冲突必须分开**：同一标的 3 月看多、6 月看空是观点变化，不是矛盾；
  把它当冲突会淹没真正的冲突。

注册状态（2026-11-10）
----------------------
本模块**尚未**加入 `compute/__init__.py::_MODULES`：加入即要求 `tool-surface.json`
同步重生成，而该文件当时被并发会话占用。逻辑与单测已就绪，注册是纯接线。
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional, Sequence

from . import ToolSpec
from .base import ComputeError, EvidenceBuilder, result_envelope

GROUP = "decision_consistency"

#: 各主体种类下的动作极性：+1 看多/扩张，-1 看空/收缩，0 中性，None 表示「判定的缺席」。
#: `review` 之所以是 None 而不是 0：它是「还没能形成判断」，不是「判断为中性」；
#: 把它当 0 会让"没形成判断"混进"判断中性"的统计里（与账本 review≠hold 同一条纪律）。
ACTION_POLARITY: dict[str, dict[str, Optional[int]]] = {
    "instrument": {
        "buy": 1, "strong_buy": 1, "hold": 0,
        "sell": -1, "strong_sell": -1, "review": None,
    },
    "store": {"expand": 1, "relocate": 1, "reprice": 0, "close": -1},
    "product_line": {"launch": 1, "invest": 1, "stop": -1, "divest": -1},
    "contract": {"accept": 1, "renegotiate": 0, "reject": -1},
    "pricing": {"raise": 1, "hold": 0, "cut": -1},
}

DIRECTION_CONFLICT = "direction_conflict"
EVIDENCE_CONFLICT = "evidence_conflict"
PREMISE_CONFLICT = "premise_conflict"
TEMPORAL_EVOLUTION = "temporal_evolution"
UNRESOLVABLE = "unresolvable"

#: 单次可以判定的记录上限。超限**显式拒绝**——静默截断后报"无冲突"是最坏的结果：
#: 它看起来像一个结论，实际只是一次没做完的扫描。
MAX_RECORDS = 500


def _polarity(record: Mapping[str, Any]) -> Optional[int]:
    kind = record.get("subject_kind") or "instrument"
    action = str(record.get("action") or "").strip()
    return ACTION_POLARITY.get(kind, {}).get(action)


def _axis_known(record: Mapping[str, Any]) -> bool:
    """这条记录的 action 是否落在该 `subject_kind` 已登记的语义轴上。

    区分三态，别把"中性"和"判不了"混成一件事：
      - 登记且极性 ≠ None → 可判方向；
      - 登记且极性 = None（`hold` / `review`）→ **有意的中性**，算已知；
      - kind 未登记，或该 kind 下没有这个 action（如 `stock` + `buy`）→ 判不了，
        必须点名而不是当成"没冲突"。
    """
    kind = record.get("subject_kind") or "instrument"
    table = ACTION_POLARITY.get(kind)
    if table is None:
        return False
    return str(record.get("action") or "").strip() in table


def _subject_key(record: Mapping[str, Any]) -> tuple[str, str]:
    return (
        str(record.get("subject_kind") or "instrument"),
        str(record.get("subject") or ""),
    )


def _evidence_ids(record: Mapping[str, Any]) -> set[str]:
    out: set[str] = set()
    for ev in record.get("evidence") or ():
        if isinstance(ev, Mapping) and ev.get("evidence_id"):
            out.add(str(ev["evidence_id"]))
    return out


def _unresolvable(record: Mapping[str, Any]) -> list[str]:
    """该记录里指向自身证据集之外、因而无法解析的引用。

    只有 `target_evidence_ids`（结论所依赖的证据）会被核——`evidence` 数组本身
    就是证据集，不构成对自身的引用。
    """
    own = _evidence_ids(record)
    bad = []
    for ref in record.get("target_evidence_ids") or ():
        if str(ref) not in own:
            bad.append(str(ref))
    return bad


def _interval(comparator: str, threshold: float) -> Optional[tuple[float, bool, float, bool]]:
    """把比较子表达成区间 ``(lo, lo_closed, hi, hi_closed)``；``neq`` 返回 None 特殊处理。"""
    inf = float("inf")
    c = str(comparator)
    if c == "lt":
        return (-inf, False, threshold, False)
    if c == "lte":
        return (-inf, False, threshold, True)
    if c == "gt":
        return (threshold, False, inf, False)
    if c == "gte":
        return (threshold, True, inf, False)
    if c == "eq":
        return (threshold, True, threshold, True)
    return None


def comparators_conflict(c1: str, t1: float, c2: str, t2: float) -> bool:
    """两组 (比较子, 阈值) 是否互斥——即不存在同时满足两者的取值。

    `neq` 单独处理：它与 `eq` 同值时互斥，与一切区间比较子相容。
    """
    c1, c2 = str(c1), str(c2)
    if c1 == "neq" or c2 == "neq":
        if c1 == "neq" and c2 == "neq":
            return False
        if c1 == "neq":
            return c2 == "eq" and float(t1) == float(t2)
        return c1 == "eq" and float(t2) == float(t1)

    a, b = _interval(c1, float(t1)), _interval(c2, float(t2))
    if a is None or b is None:
        return False
    lo = max(a[0], b[0])
    hi = min(a[2], b[2])
    if lo > hi:
        return True
    if lo == hi:
        # 端点重合：只有两侧都闭合时才存在交集（恰好等于该点）。
        lo_closed = (a[1] if a[0] == lo else True) and (b[1] if b[0] == lo else True)
        hi_closed = (a[3] if a[2] == hi else True) and (b[3] if b[2] == hi else True)
        return not (lo_closed and hi_closed)
    return False


def _windows_overlap(r1: Mapping[str, Any], r2: Mapping[str, Any]) -> bool:
    """两条结论的观察窗是否重叠。

    同一 `as_of` 视为重叠。不同 `as_of` 时，若双方都给了显式窗口则按区间判断，
    否则**不**算重叠（保守：不把可能的观点演进误判成冲突）。
    """
    if str(r1.get("as_of")) == str(r2.get("as_of")):
        return True
    w1, w2 = r1.get("window"), r2.get("window")
    if not (isinstance(w1, (list, tuple)) and len(w1) == 2
            and isinstance(w2, (list, tuple)) and len(w2) == 2):
        return False
    return max(str(w1[0]), str(w2[0])) <= min(str(w1[1]), str(w2[1]))


def detect_conflicts(
    records: Sequence[Mapping[str, Any]],
    *,
    as_of: str,
    max_records: int = MAX_RECORDS,
) -> dict:
    """检出传入决策记录之间的冲突。

    Input: records —— 决策记录序列，每条至少含
      `as_of`、`subject_kind`、`subject`、`action`；
      as_of —— 口径日（必填）。晚于该日的记录**拒绝计算并点名**，不静默丢弃：
        "用未来记录判当前冲突"会制造一个当时并不存在的矛盾，且看起来完全正常；
      可选 `evidence`（数组，元素含 `evidence_id`）、`target_evidence_ids`、
      `preconditions`（元素含 `metric_ref`/`comparator`/`threshold`）、
      `producer`、`window`（两元素数组 [start, end]）、`decision_id`。
      max_records —— 上限，超限显式拒绝。

    Returns: 字典，含
      conflicts —— 冲突清单，每条 {kind, subject_kind, subject, parties:[...], detail}；
      temporal_evolution —— 时序演进清单（不算冲突）；
      unresolvable —— 引用不可解且被排除在冲突判定之外的记录；
      counts —— 各类计数；
      truncated —— 恒为 False（本实现要么全判要么拒绝，不截断）。

    每条 party 含 `decision_id` / `as_of` / `action` / `producer` / `evidence_ids`，
    便于回溯到具体产出方。输出按 (kind, subject) 稳定排序，同输入逐位一致。
    """
    if not isinstance(as_of, str) or len(as_of) < 10:
        # 计算面统一错误类型：调用方按 ComputeError 捕获，不必区分是哪个工具抛的。
        raise ComputeError("as_of 必填（ISO 日期），计算面要求显式口径日")

    records = list(records or ())
    future = sorted({
        str(r.get("as_of")) for r in records
        if isinstance(r, Mapping) and str(r.get("as_of") or "") > as_of[:10]
    })
    if future:
        raise ComputeError(
            f"记录口径日晚于 as_of={as_of}：{', '.join(future)}；"
            "用未来记录判当前冲突会制造当时并不存在的矛盾，故拒绝而非截断"
        )

    if len(records) > max_records:
        raise ComputeError(
            f"records 超出上限：{len(records)} > {max_records}；"
            "请缩小范围后再判定（不静默截断，那会让'未扫完'伪装成'无冲突'）"
        )

    unresolvable: list[dict] = []
    unjudged: list[dict] = []
    alive: list[Mapping[str, Any]] = []
    for rec in records:
        bad = _unresolvable(rec)
        if bad:
            unresolvable.append({
                "decision_id": rec.get("decision_id"),
                "as_of": rec.get("as_of"),
                "subject": rec.get("subject"),
                "unresolved_ids": sorted(bad),
            })
            continue  # 引用不可解 → 不参与冲突判定
        # 方向判不了的记录必须**点名**：未登记的 subject_kind（如 "stock"）或该 kind 下
        # 不存在的 action，会让 `_polarity` 返回 None。此前这种记录默默进了 alive，
        # 于是 records_judged 把它算成"已判"、conflicts 为空——**一份什么都没扫的
        # 绿报告**。仍留在 alive 里参与前提矛盾判定（那条不需要方向极性）。
        if not _axis_known(rec):
            unjudged.append({
                "decision_id": rec.get("decision_id"),
                "as_of": rec.get("as_of"),
                "subject_kind": rec.get("subject_kind"),
                "subject": rec.get("subject"),
                "action": rec.get("action"),
                "reason": (
                    f"action {rec.get('action')!r} is not on a registered axis for "
                    f"subject_kind {rec.get('subject_kind')!r}; direction and "
                    "evidence conflicts cannot be judged for this record "
                    f"(known kinds: {sorted(ACTION_POLARITY)})"
                ),
            })
        alive.append(rec)

    conflicts: list[dict] = []
    evolution: list[dict] = []

    def party(rec):
        return {
            "decision_id": rec.get("decision_id"),
            "as_of": rec.get("as_of"),
            "action": rec.get("action"),
            "producer": rec.get("producer"),
            "evidence_ids": sorted(_evidence_ids(rec)),
        }

    for i in range(len(alive)):
        for j in range(i + 1, len(alive)):
            r1, r2 = alive[i], alive[j]
            if _subject_key(r1) != _subject_key(r2):
                continue
            p1, p2 = _polarity(r1), _polarity(r2)
            if p1 is None or p2 is None:
                continue  # 判定的缺席不参与方向判定
            if p1 * p2 < 0:
                if not _windows_overlap(r1, r2):
                    evolution.append({
                        "subject_kind": r1.get("subject_kind"),
                        "subject": r1.get("subject"),
                        "parties": [party(r1), party(r2)],
                    })
                    continue
                shared = _evidence_ids(r1) & _evidence_ids(r2)
                # 同证据得出相反结论，比单纯方向相反更值得报：至少一方误读了证据。
                kind = EVIDENCE_CONFLICT if shared else DIRECTION_CONFLICT
                conflicts.append({
                    "kind": kind,
                    "subject_kind": r1.get("subject_kind"),
                    "subject": r1.get("subject"),
                    "parties": [party(r1), party(r2)],
                    "shared_evidence_ids": sorted(shared),
                })

    # 前提矛盾：同一 subject 内、同一 metric_ref 上互斥的比较子/阈值
    for i in range(len(alive)):
        for j in range(i + 1, len(alive)):
            r1, r2 = alive[i], alive[j]
            if _subject_key(r1) != _subject_key(r2):
                continue
            for pc1 in r1.get("preconditions") or ():
                for pc2 in r2.get("preconditions") or ():
                    if not isinstance(pc1, Mapping) or not isinstance(pc2, Mapping):
                        continue
                    if str(pc1.get("metric_ref")) != str(pc2.get("metric_ref")):
                        continue
                    try:
                        clash = comparators_conflict(
                            pc1.get("comparator"), pc1.get("threshold"),
                            pc2.get("comparator"), pc2.get("threshold"),
                        )
                    except (TypeError, ValueError):
                        continue
                    if clash:
                        conflicts.append({
                            "kind": PREMISE_CONFLICT,
                            "subject_kind": r1.get("subject_kind"),
                            "subject": r1.get("subject"),
                            "parties": [party(r1), party(r2)],
                            "metric_ref": pc1.get("metric_ref"),
                            "conditions": [
                                {"comparator": pc1.get("comparator"),
                                 "threshold": pc1.get("threshold")},
                                {"comparator": pc2.get("comparator"),
                                 "threshold": pc2.get("threshold")},
                            ],
                        })

    conflicts.sort(key=lambda c: (c["kind"], str(c.get("subject")),
                                  str(c["parties"][0].get("decision_id")),
                                  str(c["parties"][1].get("decision_id"))))
    evolution.sort(key=lambda e: (str(e.get("subject")),
                                  str(e["parties"][0].get("as_of")),
                                  str(e["parties"][1].get("as_of"))))

    counts: dict[str, int] = {}
    for c in conflicts:
        counts[c["kind"]] = counts.get(c["kind"], 0) + 1
    input_dates = sorted(str(r.get("as_of")) for r in records if r.get("as_of"))

    # 统一外壳：as_of + 起止 + evidence。本工具吃的是"记录集合"而非单条序列，
    # 输入起止由记录口径日的 min/max 表达；无记录时三者全 None（快照型语义）。
    b = EvidenceBuilder("decision_consistency:detect_conflicts", as_of)
    b.add("records_judged", len(alive), unit="count")
    b.add("records_unjudged", len(unjudged), unit="count")
    b.add("conflicts_found", len(conflicts), unit="count")
    return result_envelope(
        "decision_consistency:detect_conflicts",
        as_of,
        None,
        b,
        {
            "input_range": [input_dates[0], input_dates[-1]] if input_dates else None,
            "conflicts": conflicts,
            "temporal_evolution": evolution,
            "unresolvable": unresolvable,
            "unjudged": unjudged,
            "counts": counts,
            "truncated": False,
            "records_judged": len(alive),
            #: 真正参与了方向判定（或方向上有意中性，如 hold/review）的记录数。
            #: `records_judged` 含"进来了但方向判不了"的记录，故两者相等才算全判。
            "direction_judged": len(alive) - len(unjudged),
            "records_excluded": len(unresolvable),
        },
    )


#: 工具面声明。`detect_conflicts` 以 "detect" 起头，符合计算面 verb_noun 约定。
TOOLS: list[ToolSpec] = [
    ToolSpec("detect_conflicts", GROUP, detect_conflicts),
]

#: 跨族 conformance 门要求"快照型工具"（签名含 Mapping 输入）声明一组完整合法 kwargs。
SAMPLES: dict[str, dict] = {
    "detect_conflicts": {
        "as_of": "2026-11-10",
        "records": [
            {"decision_id": "a", "as_of": "2026-11-10", "subject_kind": "instrument",
             "subject": "600519.SS", "action": "buy",
             "evidence": [{"evidence_id": "e1"}], "target_evidence_ids": ["e1"],
             "producer": {"pack": "wire-equity-research",
                          "skill": "equity-research-thesis"}},
            {"decision_id": "b", "as_of": "2026-11-10", "subject_kind": "instrument",
             "subject": "600519.SS", "action": "sell",
             "evidence": [{"evidence_id": "e2"}], "target_evidence_ids": ["e2"],
             "producer": {"pack": "wire-portfolio-risk",
                          "skill": "portfolio-budget"}},
        ],
    },
}
