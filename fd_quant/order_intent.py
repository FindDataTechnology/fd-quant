"""纸面订单意图（openspec change: deepen-trading-benchmark-and-risk, task 4）。

## 为什么需要这个对象

订单原本只存在于模拟交易服务内部。事后再看一笔成交，答不出三件事：它依据哪一期权重？
当时过没过风控闸？属于哪一轮团队运行？`OrderIntent` 把这三样变成字段。

## 为什么只有模拟通道

适配器接口只提供模拟实现，真实通道**不提供**（不是"未实现"，是明确的拒绝）。这既是合规
红线，也省掉长期维护负担。`paper_only` 恒为真且不可关闭——没有把它设为假的合法路径。

## 为什么对账要覆盖三态

`已成交 / 被拒 / 未触发`。第三态最容易被漏（限价单挂到收盘没成交），而它恰恰是"策略看起来
在跑但什么都没发生"的来源。孤儿意图（既无成交也无拒绝）单独报出，且不计入任何成交量统计
——否则统计口径会被静默稀释。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

SCHEMA_VERSION = "1"

ORDER_SIDES = ("buy", "sell")
ORDER_TYPES = ("market", "limit")

#: 对账三态 + 孤儿。孤儿不是第四态，是"三态都不成立"的异常。
FILLED = "filled"
REJECTED = "rejected"
NOT_TRIGGERED = "not_triggered"
ORPHAN = "orphan"

#: 允许出口的通道。真实通道**不在其中**——它没有被实现，也不许被加进来。
ALLOWED_CHANNELS = ("simulated",)


class OrderIntentError(ValueError):
    """意图本身不合法。"""


class ChannelNotAvailable(PermissionError):
    """请求了一个不提供的执行通道（真实通道不提供）。"""


@dataclass(frozen=True)
class OrderIntent:
    """一条纸面订单意图。见 openspec spec `order-intent-export`。"""

    intent_id: str
    as_of: str
    account_id: str
    symbol: str
    side: str
    order_type: str
    derived_from: Sequence[str]
    risk_verdict: Mapping[str, Any]
    producer: Mapping[str, Any]
    quantity: Optional[float] = None
    target_weight: Optional[float] = None
    limit_price: Optional[float] = None
    paper_only: bool = True
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict:
        d: dict[str, Any] = {
            "schema_version": self.schema_version,
            "intent_id": self.intent_id,
            "as_of": self.as_of,
            "account_id": self.account_id,
            "symbol": self.symbol,
            "side": self.side,
            "order_type": self.order_type,
            "derived_from": list(self.derived_from),
            "risk_verdict": dict(self.risk_verdict),
            "producer": dict(self.producer),
            # paper_only 恒真：没有合法路径把它写成 false
            "paper_only": True,
        }
        if self.quantity is not None:
            d["quantity"] = self.quantity
        if self.target_weight is not None:
            d["target_weight"] = self.target_weight
        if self.limit_price is not None:
            d["limit_price"] = self.limit_price
        return d


def check_order_intent(d: Any) -> list[dict]:
    """校验 OrderIntent。返回 `[{path, reason}]`（空 = 通过）。"""
    errs: list[dict] = []
    if not isinstance(d, Mapping):
        return [{"path": "$", "reason": "must be an object"}]

    required = ("schema_version", "intent_id", "as_of", "account_id", "symbol", "side",
                "order_type", "derived_from", "risk_verdict", "producer", "paper_only")
    for k in required:
        if k not in d:
            errs.append({"path": k, "reason": "required field missing"})

    if d.get("schema_version") != SCHEMA_VERSION:
        errs.append({"path": "schema_version", "reason": f'must be "{SCHEMA_VERSION}"'})

    for k in ("intent_id", "as_of", "account_id", "symbol"):
        if k in d and (not isinstance(d[k], str) or not d[k].strip()):
            errs.append({"path": k, "reason": "must be a non-empty string"})

    if d.get("side") not in ORDER_SIDES:
        errs.append({"path": "side",
                     "reason": f"must be one of {ORDER_SIDES} (got {d.get('side')!r})"})
    if d.get("order_type") not in ORDER_TYPES:
        errs.append({"path": "order_type",
                     "reason": f"must be one of {ORDER_TYPES} (got {d.get('order_type')!r})"})

    # 出处必填：没有出处的订单意图无法回答"这笔单是谁的决定"
    df = d.get("derived_from")
    if not isinstance(df, (list, tuple)) or not df:
        errs.append({"path": "derived_from",
                     "reason": "must be a non-empty array: an order without a traceable "
                               "origin cannot be reviewed"})
    elif any(not isinstance(x, str) or not x.strip() for x in df):
        errs.append({"path": "derived_from", "reason": "every entry must be a non-empty string"})

    # 数量与目标权重至少其一
    qty, tw_ = d.get("quantity"), d.get("target_weight")
    if qty is None and tw_ is None:
        errs.append({"path": "quantity",
                     "reason": "either quantity or target_weight is required"})
    for k, v in (("quantity", qty), ("target_weight", tw_)):
        if v is not None and (not isinstance(v, (int, float)) or isinstance(v, bool) or v <= 0):
            errs.append({"path": k, "reason": "must be a positive number"})

    if d.get("order_type") == "limit":
        lp = d.get("limit_price")
        if lp is None:
            errs.append({"path": "limit_price",
                         "reason": "a limit order must carry limit_price"})
        elif not isinstance(lp, (int, float)) or isinstance(lp, bool) or lp <= 0:
            errs.append({"path": "limit_price", "reason": "must be a positive number"})

    rv = d.get("risk_verdict")
    if not isinstance(rv, Mapping):
        errs.append({"path": "risk_verdict",
                     "reason": "must be an object (the gate verdict this intent was cleared by)"})
    else:
        status = rv.get("status")
        if status not in ("ok", "breached", "unavailable"):
            errs.append({"path": "risk_verdict.status",
                         "reason": 'must be "ok" | "breached" | "unavailable"'})

    # paper_only 不可关闭
    if d.get("paper_only") is not True:
        errs.append({"path": "paper_only",
                     "reason": "must be true: only simulated execution exists, so a "
                               "non-paper intent has no legal path"})

    return errs


def check_order_intent_provenance(intent: Any, known_source_ids: Sequence[str]) -> dict:
    """校验 `derived_from` 的每条都能解析到一个**真实存在**的来源（权重实例或判定标识）。

    spec「OrderIntent 的形状与出处」后半句：`derived_from` 非空只是形状；**可解析**才是
    "这笔单是谁的决定"能被回答的前提。契约层不知道你那边有哪些权重实例与判定，所以
    决议需要调用方给集合。

    三态，缺省不冒充通过：
      - `checked=False`  —— 没校验（形状都不合法，或没给集合）
      - `checked=True, valid=True`  —— 每条都命中
      - `checked=True, valid=False` —— 逐条点名悬空项

    与 `portfolio.check_intent_provenance` 同一手法、同一返回值形状——两处出处要求
    本来就是同一条规矩，别各写一套。
    """
    payload = intent.to_dict() if isinstance(intent, OrderIntent) else dict(intent or {})
    errs = check_order_intent(payload)
    if errs:
        return {"checked": False, "valid": False, "resolved": [], "unresolved": [],
                "errors": errs}

    known = {str(i) for i in (known_source_ids or []) if str(i)}
    derived = [str(i) for i in (payload.get("derived_from") or [])]
    resolved = [i for i in derived if i in known]
    unresolved = [i for i in derived if i not in known]
    return {
        "checked": True,
        "valid": not unresolved,
        "resolved": resolved,
        "unresolved": unresolved,
        "errors": [
            {"path": f"derived_from[{derived.index(i)}]",
             "reason": f"{i!r} does not resolve to a known weight instance or decision; "
                       "an order whose origin cannot be read back must not reach execution"}
            for i in unresolved
        ],
    }


def export_order_intent(intent: OrderIntent | Mapping[str, Any],
                        *, channel: str = "simulated",
                        known_source_ids: Optional[Sequence[str]] = None) -> dict:
    """把意图交给执行通道。只提供模拟通道；其余一律拒绝。

    拒绝是**明确的拒绝**（`ChannelNotAvailable`），不是"未实现"——两者在合规与合同上
    含义不同：前者说"我们不提供这条路"，后者暗示"以后会有"。

    `known_source_ids`：你这边真实存在的来源标识（权重实例 id、判定 id）。给了就验
    `derived_from` 可解析，悬空即拒绝——"不得进入执行"必须发生在**这道门**上，
    而不是事后审计时才发现。不给则返回体里 `provenance.checked=false`，
    **不假装查过**。
    """
    if channel not in ALLOWED_CHANNELS:
        raise ChannelNotAvailable(
            f"channel {channel!r} is not provided (available: {ALLOWED_CHANNELS}); "
            "only simulated execution exists"
        )
    payload = intent.to_dict() if isinstance(intent, OrderIntent) else dict(intent)
    errs = check_order_intent(payload)
    if errs:
        raise OrderIntentError(f"{errs[0]['path']}: {errs[0]['reason']}")

    # 风控判据不可用 → 不许导出（宁可不下单，也不在不知道风险偏好的情况下下单）
    status = payload["risk_verdict"].get("status")
    if status == "unavailable":
        missing = payload["risk_verdict"].get("inputs", {}).get("missing")
        raise OrderIntentError(
            f"risk_verdict indicates an unavailable gate (missing: {missing}); "
            "refusing to export an order intent while the risk thresholds are unknown"
        )

    provenance: dict[str, Any] = {"checked": False}
    if known_source_ids is not None:
        prov = check_order_intent_provenance(payload, known_source_ids)
        provenance = {k: prov[k] for k in ("checked", "valid", "unresolved")}
        if not prov["valid"]:
            raise OrderIntentError(
                f"{prov['errors'][0]['path']}: {prov['errors'][0]['reason']} "
                f"(unresolved: {prov['unresolved']})"
            )

    return {
        "channel": channel,
        "exported": True,
        "paper_only": True,
        "provenance": provenance,
        "intent": payload,
    }


def reconcile(intents: Sequence[Mapping[str, Any]],
              outcomes: Mapping[str, Mapping[str, Any]]) -> dict:
    """意图 ↔ 结果对账。覆盖 `已成交 / 被拒 / 未触发` 三态，孤儿单独报出。

    `outcomes` = `{intent_id: {"state": filled|rejected|not_triggered, "reason"?: ...}}`。
    三态都不成立的意图进 `orphans`，**且不计入任何成交量统计**——否则统计口径会被静默稀释。
    """
    buckets: dict[str, list[str]] = {FILLED: [], REJECTED: [], NOT_TRIGGERED: [], ORPHAN: []}
    details: dict[str, dict] = {}
    for it in intents:
        iid = it.get("intent_id")
        if not isinstance(iid, str) or not iid:
            raise OrderIntentError("every intent must carry a non-empty intent_id")
        state = (outcomes.get(iid) or {}).get("state")
        if state not in (FILLED, REJECTED, NOT_TRIGGERED):
            buckets[ORPHAN].append(iid)
            details[iid] = {"state": ORPHAN,
                            "reason": "no outcome recorded for this intent"}
            continue
        buckets[state].append(iid)
        details[iid] = dict(outcomes[iid])

    return {
        "counts": {k: len(v) for k, v in buckets.items()},
        # 孤儿不计入成交量：只报三态之和
        "accounted": len(buckets[FILLED]) + len(buckets[REJECTED]) + len(buckets[NOT_TRIGGERED]),
        "orphans": buckets[ORPHAN],
        "details": details,
    }
