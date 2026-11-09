"""`Decision` 契约（schema_version="2"）：把判定从"只装得下证券"放宽到"任何可结算的判断"。

为什么需要 v2
-------------
v1 的 `Rating` 把三件事绑死在证券语汇上：`subject` 是标的、`side` 是买卖、
结算只认「相对基准的 alpha」。于是一家店要不要开第二家、这个报价要不要接、这条产线
要不要停——这些判断没有任何可以落账的形状。

v2 不取代 v1：`Rating` 是它在 `subject_kind="instrument"` 上的特化。v1 实例继续可
校验、可落账、可检索、可聚合；本模块只新增一条入口。

与 v1 的实质差别是 `preconditions`
---------------------------------
v1 的 `basis` 是一段自由文本，机器核不了。于是"这个判断的前提失效了"这件事，系统
永远发现不了——它只能等人自己想起来的。v2 把前提结构化：
`{claim, metric_ref, comparator, threshold, as_of}`，五项齐全才算可核。

`basis` 与 `preconditions` **并存，互不替代**：前者给人读，后者给机器核。把条件只
写进 `basis` 等于放弃监控。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

SCHEMA_VERSION = "2"

#: 主体种类。决定 `action` 的合法取值集合，也决定结算口径是否说得通。
SUBJECT_KINDS = ("instrument", "store", "product_line", "contract", "pricing")

#: 按 `subject_kind` 分派的动作集合。
#: `review` 在 instrument 里的语义是「判定的缺席」（还没能形成判断），不是「判断为中性」——
#: 与账本 review≠hold 同一条纪律；把它当 hold 会让"没判断"混进"判断中性"的统计。
ACTIONS_BY_KIND: dict[str, tuple[str, ...]] = {
    "instrument": ("buy", "strong_buy", "hold", "sell", "strong_sell", "review"),
    "store": ("expand", "close", "relocate", "reprice"),
    "product_line": ("launch", "stop", "invest", "divest"),
    "contract": ("accept", "reject", "renegotiate"),
    "pricing": ("raise", "cut", "hold"),
}

COMPARATORS = ("lt", "lte", "gt", "gte", "eq", "neq")

SETTLEMENT_RULES = ("alpha_vs_benchmark", "metric_threshold")

#: 结算数据一律**由调用方传入**：账本不取数、不联网、不读客户库。
#: 经营结算的序列由客户 cell 内 agent 取好后传入，账本只留 metric_ref + 序列指纹 + 结论。
SETTLEMENT_SOURCES = ("caller_supplied_series",)


def _is_iso_date(value: Any) -> bool:
    if not isinstance(value, str) or len(value) < 10:
        return False
    try:
        from datetime import date
        date.fromisoformat(value[:10])
        return True
    except ValueError:
        return False


def _is_number(value: Any) -> bool:
    # bool 是 int 的子类：不排除它，threshold=True 会被当成 1 —— 一个永远不报错的错误。
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def check_precondition(d: Any, path: str = "preconditions[0]") -> list[dict]:
    """校验单条前提。五字段必填，逐项给路径与原因。"""
    errs: list[dict] = []
    if not isinstance(d, Mapping):
        return [{"path": path, "reason": f"前提必须是对象，收到 {type(d).__name__}"}]

    claim = d.get("claim")
    if not isinstance(claim, str) or not claim.strip():
        errs.append({"path": f"{path}.claim",
                     "reason": "claim 必填且非空：前提要写成一句可判定的话"})

    metric_ref = d.get("metric_ref")
    if not isinstance(metric_ref, str) or not metric_ref.strip():
        errs.append({"path": f"{path}.metric_ref",
                     "reason": "metric_ref 必填：没有可定位的指标，这条前提无法核"})

    comparator = d.get("comparator")
    if comparator not in COMPARATORS:
        errs.append({"path": f"{path}.comparator",
                     "reason": f"comparator 非法：{comparator!r}；合法集合 {list(COMPARATORS)}"})

    if not _is_number(d.get("threshold")):
        errs.append({"path": f"{path}.threshold",
                     "reason": f"threshold 必须是有限数值，收到 {d.get('threshold')!r}"})

    if not _is_iso_date(d.get("as_of")):
        errs.append({"path": f"{path}.as_of",
                     "reason": "as_of 必填且为 ISO 日期（'YYYY-MM-DD'）"})
    return errs


def check_preconditions(items: Any, *, required: bool) -> list[dict]:
    """校验前提数组。`required=True` 时至少一条（证券类）。"""
    if items is None:
        items = []
    if not isinstance(items, Sequence) or isinstance(items, (str, bytes)):
        return [{"path": "preconditions", "reason": "preconditions 必须是数组"}]
    if required and len(items) == 0:
        return [{"path": "preconditions",
                 "reason": "subject_kind='instrument' 必须给出至少一条可核前提"}]
    errs: list[dict] = []
    for i, item in enumerate(items):
        errs.extend(check_precondition(item, f"preconditions[{i}]"))
    return errs


def _check_expected_outcome(d: Any, path: str = "expected_outcome") -> list[dict]:
    if not isinstance(d, Mapping):
        return [{"path": path, "reason": "expected_outcome 必填且为对象"}]
    errs: list[dict] = []
    if not isinstance(d.get("metric_ref"), str) or not d.get("metric_ref", "").strip():
        errs.append({"path": f"{path}.metric_ref",
                     "reason": "结算所依据的指标必填"})
    if d.get("comparator") not in COMPARATORS:
        errs.append({"path": f"{path}.comparator",
                     "reason": f"comparator 非法：{d.get('comparator')!r}"})
    if not _is_number(d.get("threshold")):
        errs.append({"path": f"{path}.threshold",
                     "reason": "threshold 必须是有限数值"})
    source = d.get("settlement_source")
    if source not in SETTLEMENT_SOURCES:
        errs.append({"path": f"{path}.settlement_source",
                     "reason": (f"settlement_source 非法：{source!r}；"
                                f"合法集合 {list(SETTLEMENT_SOURCES)}"
                                "（账本不取数，序列一律由调用方传入）")})
    horizon = d.get("horizon")
    if not isinstance(horizon, Mapping) or not horizon.get("unit") or not _is_number(horizon.get("count")):
        errs.append({"path": f"{path}.horizon",
                     "reason": "horizon 必填，形如 {unit: 'month'|'day'|'quarter'|'year', count: n}"})
    return errs


def check_decision(d: Any) -> list[dict]:
    """校验 `Decision` v2 实例。返回错误列表（空 = 通过）。纯本地、确定性、不取数。"""
    from fd_quant.contracts.models import _check_evidence, _check_producer

    if isinstance(d, (str, bytes)):
        try:
            d = json.loads(d)
        except (TypeError, ValueError) as e:
            return [{"path": "$", "reason": f"invalid JSON: {e}"}]
    if isinstance(d, Decision):
        d = d.to_dict()
    if not isinstance(d, Mapping):
        return [{"path": "$", "reason": f"Decision 必须是对象，收到 {type(d).__name__}"}]

    errs: list[dict] = []

    if str(d.get("schema_version")) != SCHEMA_VERSION:
        errs.append({"path": "schema_version",
                     "reason": f"Decision 需要 schema_version=\"{SCHEMA_VERSION}\""})

    if not _is_iso_date(d.get("as_of")):
        errs.append({"path": "as_of", "reason": "as_of 必填且为 ISO 日期"})

    subject = d.get("subject")
    if not isinstance(subject, str) or not subject.strip():
        errs.append({"path": "subject", "reason": "subject 必填"})

    kind = d.get("subject_kind")
    if kind not in SUBJECT_KINDS:
        errs.append({"path": "subject_kind",
                     "reason": f"subject_kind 非法：{kind!r}；合法集合 {list(SUBJECT_KINDS)}"})
        kind = None

    action = d.get("action")
    if kind is not None:
        legal = ACTIONS_BY_KIND[kind]
        if action not in legal:
            errs.append({"path": "action",
                         "reason": (f"subject_kind={kind!r} 的 action 非法：{action!r}；"
                                    f"合法集合 {list(legal)}")})

    errs.extend(check_preconditions(d.get("preconditions"),
                                   required=(kind == "instrument")))

    rule = d.get("settlement_rule_id")
    if rule not in SETTLEMENT_RULES:
        errs.append({"path": "settlement_rule_id",
                     "reason": f"settlement_rule_id 非法：{rule!r}；合法集合 {list(SETTLEMENT_RULES)}"})

    # `metric_threshold` 必须给 expected_outcome（结算要按它判 hit/miss）；
    # `alpha_vs_benchmark` 给了也校验——但它是可选的：证券口径的"预期"通常已经
    # 由 side + price_target 表达，硬性再要一份指标阈值属于重复建模。
    if rule == "metric_threshold" or d.get("expected_outcome") is not None:
        errs.extend(_check_expected_outcome(d.get("expected_outcome")))

    confidence = d.get("confidence")
    if confidence is not None and not (0 <= (confidence if _is_number(confidence) else -1) <= 1):
        errs.append({"path": "confidence",
                     "reason": "confidence 为 0..1 的数值或 null"})

    basis = d.get("basis")
    if not isinstance(basis, str) or not basis.strip():
        errs.append({"path": "basis",
                     "reason": "basis 必填：为什么这么判（与 preconditions 并存，互不替代）"})

    horizon = d.get("time_horizon")
    if horizon is not None and not isinstance(horizon, str):
        errs.append({"path": "time_horizon", "reason": "time_horizon 为字符串或 null"})

    # evidence 是**数组**，逐元素校验；`_check_evidence` 吃的是单个对象。
    # 直接把它喂给数组会得到 "must be an object"，那是把形状错当成内容错。
    ev = d.get("evidence")
    if not isinstance(ev, list):
        errs.append({"path": "evidence", "reason": "evidence 必须是数组"})
        ev = []
    ids: list[str] = []
    for i, item in enumerate(ev):
        errs.extend(_check_evidence(item, f"evidence[{i}]"))
        if isinstance(item, Mapping) and isinstance(item.get("evidence_id"), str):
            ids.append(item["evidence_id"])
    for dup in sorted({i for i in ids if ids.count(i) > 1}):
        errs.append({"path": "evidence", "reason": f"duplicate evidence_id: {dup}"})

    if "producer" in d:
        errs.extend(_check_producer(d["producer"], "producer"))
    else:
        errs.append({"path": "producer", "reason": "required field missing"})
    return errs


@dataclass
class Decision:
    """v2 决策的轻量载体（与 `Rating` 同级，供调用方构造后再落账）。"""

    as_of: str
    subject: str
    subject_kind: str
    action: str
    basis: str
    settlement_rule_id: str
    preconditions: list[dict] = field(default_factory=list)
    expected_outcome: Optional[dict] = None
    confidence: Optional[float] = None
    time_horizon: Optional[str] = None
    evidence: list[dict] = field(default_factory=list)
    producer: Optional[dict] = None
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict:
        out: dict[str, Any] = {
            "schema_version": self.schema_version,
            "as_of": self.as_of,
            "subject": self.subject,
            "subject_kind": self.subject_kind,
            "action": self.action,
            "preconditions": list(self.preconditions),
            "settlement_rule_id": self.settlement_rule_id,
            "basis": self.basis,
            "evidence": list(self.evidence),
        }
        if self.expected_outcome is not None:
            out["expected_outcome"] = self.expected_outcome
        if self.confidence is not None:
            out["confidence"] = self.confidence
        if self.time_horizon is not None:
            out["time_horizon"] = self.time_horizon
        if self.producer is not None:
            out["producer"] = self.producer
        return out
