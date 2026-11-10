"""评级 → 目标权重的确定性转换器（add-portfolio-intent-contract, tasks 2.1-2.6）。

## 三条纪律

1. **纯函数**：不取数、不读库、不联网、不调模型。同输入逐位一致（含 rationale 文本）。
2. **不做最优化**：只实现三种规则。任何"帮我算最优权重"都拒绝，并说明为什么。
3. **不编数**：入选 0 条就返回空权重 + 精确原因，不返回均匀占位权重；`price_target`
   为 null 的标的按"扣值不扣结论"处理——它在组合层表现为**不配仓**（进 excluded 的
   `data_gap`），而不是给它猜一个点位。
"""
from __future__ import annotations

import datetime as dt
from typing import Any, Iterable, Mapping, Optional, Sequence

from fd_quant.contracts.models import (
    EXCLUSION_REASONS,
    RULE_KINDS,
    SELECTED_SIDES,
    ContractError,
    PortfolioIntent,
    Producer,
    check_portfolio_intent,
)

DEFAULT_PRODUCER = {"pack": "fd-quant", "skill": "portfolio-construction"}

#: 浮点比较容差（与契约层同一把尺）。
_EPS = 1e-9

#: 权重求和后按此位数舍入，保证"同输入逐位一致"不受浮点累加顺序影响。
_WEIGHT_DECIMALS = 12


class IntentBuildError(ValueError):
    """转换不可成立。单独成类型，便于调用方把"输入不合格"与"契约不合格"分开。"""

    def __init__(self, path: str, reason: str):
        self.path = path
        self.reason = reason
        super().__init__(f"{path}: {reason}")


def _rating_get(r: Any, key: str, default: Any = None) -> Any:
    if isinstance(r, Mapping):
        return r.get(key, default)
    return getattr(r, key, default)


def _rating_id(r: Any, index: int) -> str:
    """评级标识：优先取显式 id，其次由 (as_of, subject) 合成，最后退回序号。

    合成而非要求调用方预先生成 id，是为了让转换器能直接吃 `Rating` 实例；
    显式 id 永远优先，因为它才是账本里的那个键。
    """
    rid = _rating_get(r, "id") or _rating_get(r, "decision_id") or _rating_get(r, "evidence_id")
    if isinstance(rid, str) and rid:
        return rid
    as_of = _rating_get(r, "as_of")
    subject = _rating_get(r, "subject")
    if as_of and subject:
        return f"{subject}@{as_of}"
    return f"rating[{index}]"


def _parse_iso(value: Any, path: str) -> dt.date:
    try:
        return dt.date.fromisoformat(str(value))
    except (TypeError, ValueError) as exc:
        raise IntentBuildError(path, f"{value!r} is not an ISO date") from exc


def _validate_rules(rules: Mapping[str, Any]) -> None:
    if not isinstance(rules, Mapping):
        raise IntentBuildError("rules", "must be an object")
    kind = rules.get("kind")
    if kind not in RULE_KINDS:
        raise IntentBuildError(
            "rules.kind",
            f"must be one of {RULE_KINDS} (got {kind!r}); optimising rules are not accepted — "
            "without out-of-sample validation an 'optimal' weight is in-sample fitting",
        )
    if rules.get("cash_floor") is None:
        raise IntentBuildError("rules.cash_floor", f"required for kind={kind!r}")
    cf = rules["cash_floor"]
    if not isinstance(cf, (int, float)) or isinstance(cf, bool) or not (0 <= float(cf) <= 1):
        raise IntentBuildError("rules.cash_floor", "must be a number in [0, 1]")
    if kind != "equal_weight" and rules.get("conviction_floor") is None:
        raise IntentBuildError("rules.conviction_floor", f"required for kind={kind!r}")
    if kind == "target_upside_weighted":
        pr = rules.get("price_reference")
        if not isinstance(pr, Mapping) or not pr:
            raise IntentBuildError(
                "rules.price_reference",
                "required for kind=target_upside_weighted and must be symbol → price",
            )


def _reference_price(value: Any) -> Optional[float]:
    """从参考价条目里取出价格。

    两种形态都收（change2 task 5.4）：
    - 裸数字：``{"600519": 1500.0}`` —— 老写法，继续支持，零破坏；
    - 带出处的条目：``{"600519": {"price": 1500.0, "series_ref": {...}}}`` ——
      参考价来自哪一路、哪天的水位，随 intent 一起落账（``rules`` 是逐字快照的）。

    带出处时**只认合法的 ref**：形状不对就拒绝，不静默退回"只有价格"——
    否则"我带了出处"与"我以为我带了出处"在报告上看不出区别。
    """
    if isinstance(value, Mapping):
        ref = value.get("series_ref")
        if ref is not None:
            from .contracts.models import check_price_series_ref

            errs = check_price_series_ref(ref)
            if errs:
                raise IntentBuildError(
                    f"rules.price_reference.series_ref.{errs[0].get('path', '?')}",
                    f"invalid series_ref: {errs[0].get('reason', '')}",
                )
        return value.get("price")
    return value


def _upside(price_target: Any, reference: Any) -> Optional[float]:
    if price_target is None or reference is None:
        return None
    try:
        pt, ref = float(price_target), float(_reference_price(reference))
    except (TypeError, ValueError):
        return None
    if ref <= 0:
        return None
    return pt / ref - 1.0


def _rationale_for(kind: str, rules: Mapping[str, Any], n_selected: int, cash_floor: float) -> str:
    cf = f"{cash_floor:g}"
    if kind == "equal_weight":
        body = (
            f"规则 equal_weight：入选 {n_selected} 只，每只 (1 - cash_floor) / N = "
            f"(1 - {cf}) / {n_selected}；其余 {cf} 为现金。"
        )
    elif kind == "conviction_weighted":
        body = (
            f"规则 conviction_weighted：按 conviction 占比分配 (1 - cash_floor) = "
            f"{1 - cash_floor:g}，conviction 下限 {rules.get('conviction_floor')!r}；"
            f"其余 {cf} 为现金。**conviction 是判定强度（多确定），不是预期收益（能赚多少）**。"
        )
    else:
        body = (
            f"规则 target_upside_weighted：按目标涨幅 u = price_target / price_reference - 1 "
            f"占比分配 (1 - cash_floor) = {1 - cash_floor:g}，仅纳入 u > 0 的标的，"
            f"conviction 下限 {rules.get('conviction_floor')!r}；其余 {cf} 为现金。"
            "参考价口径日由 rules.price_reference_as_of 给出。"
        )
    return body + " 权重是规则的产物，不是研究结论；不含任何权重最优化。"


def build_intent(
    ratings: Sequence[Any],
    rules: Mapping[str, Any],
    *,
    as_of: str,
    universe: Optional[Sequence[str]] = None,
    cash_floor: Optional[float] = None,
    producer: Optional[Mapping[str, Any] | Producer] = None,
    rebalance: Optional[Mapping[str, Any]] = None,
) -> PortfolioIntent:
    """把一批 Rating 压成一期 PortfolioIntent。

    Args:
        ratings: `Rating` 实例或同形 dict，每个至少含 `as_of` / `subject` / `side`。
        rules: `{"kind": ..., "cash_floor": ..., ...}`，见模块文档。
        as_of: 本期决策日。晚于此日的评级被拒绝并点名第一个越界标的。
        universe: 参与标的；缺省取全部评级标的并按首次出现顺序去重。
        cash_floor: 覆盖 rules 里的 cash_floor（便于"同一规则不同现金档"对比）。
        producer: 产出方标识。
        rebalance: 调仓口径，如 `{"every_n_bars": 20}`。

    Raises:
        IntentBuildError: 输入不合格（越界日期、规则非法、参考价缺失等）。
        ContractError: 结果未通过契约校验（理论上不该发生；发生即为实现缺陷）。
    """
    _validate_rules(rules)
    kind = rules["kind"]
    cf = float(cash_floor if cash_floor is not None else rules["cash_floor"])
    if not (0 <= cf <= 1):
        raise IntentBuildError("cash_floor", "must be in [0, 1]")
    conv_floor = rules.get("conviction_floor")
    conv_floor = None if conv_floor is None else float(conv_floor)
    decision_day = _parse_iso(as_of, "as_of")

    # 1) as_of 守卫：先判越界，再谈权重。
    for i, r in enumerate(ratings):
        r_date = _parse_iso(_rating_get(r, "as_of"), f"ratings[{i}].as_of")
        if r_date > decision_day:
            raise IntentBuildError(
                f"ratings[{i}].as_of",
                f"rating for {_rating_get(r, 'subject')!r} is dated {r_date.isoformat()}, "
                f"later than as_of={decision_day.isoformat()}; refusing to use future "
                "information (not truncated silently)",
            )

    # 2) universe
    if universe is None:
        seen: list[str] = []
        for r in ratings:
            s = _rating_get(r, "subject")
            if isinstance(s, str) and s and s not in seen:
                seen.append(s)
        universe_list = seen
    else:
        universe_list = [str(s) for s in universe]
    uni_set = set(universe_list)

    # 3) 逐条判定入选 / 排除
    derived_from: list[str] = []
    excluded: list[dict] = []
    selected: list[dict[str, Any]] = []
    price_reference = dict(rules.get("price_reference") or {})

    for i, r in enumerate(ratings):
        rid = _rating_id(r, i)
        derived_from.append(rid)
        subject = _rating_get(r, "subject")
        side = _rating_get(r, "side")
        conv = _rating_get(r, "conviction")
        ptarget = _rating_get(r, "price_target")

        if subject not in uni_set:
            excluded.append({"id": rid, "symbol": subject, "reason": "outside_universe"})
            continue
        if side not in SELECTED_SIDES:
            excluded.append({"id": rid, "symbol": subject, "reason": "side_not_selected"})
            continue
        if conv_floor is not None and (conv is None or float(conv) < conv_floor):
            excluded.append({"id": rid, "symbol": subject, "reason": "below_conviction_floor"})
            continue

        upside = None
        if kind == "target_upside_weighted":
            ref = _reference_price(price_reference.get(subject))
            if ref is None:
                # 参考价缺失是**环境**问题，不是这条评级的问题：整条规则拒绝，
                # 不退回等权（静默换规则＝换了策略而报告上看不出来）。
                raise IntentBuildError(
                    "rules.price_reference",
                    f"missing reference price for {subject!r}; refusing the whole rule "
                    "rather than falling back to equal_weight",
                )
            if ptarget is None:
                excluded.append({"id": rid, "symbol": subject, "reason": "data_gap"})
                continue
            upside = _upside(ptarget, ref)
            if upside is None:
                excluded.append({"id": rid, "symbol": subject, "reason": "data_gap"})
                continue
            if upside <= 0:
                excluded.append({"id": rid, "symbol": subject, "reason": "negative_upside"})
                continue

        if kind == "conviction_weighted" and conv is None:
            excluded.append({"id": rid, "symbol": subject, "reason": "data_gap"})
            continue

        selected.append(
            {"id": rid, "symbol": subject, "conviction": conv, "upside": upside}
        )

    # 4) 权重
    alloc = 1.0 - cf
    weights: dict[str, float] = {}
    empty_reason = ""
    if not selected:
        empty_reason = (
            f"入选 0 条（输入 {len(derived_from)} 条，阈值 conviction_floor="
            f"{conv_floor!r}），不返回占位权重"
        )
    elif kind == "equal_weight":
        per = alloc / len(selected)
        for item in selected:
            weights[item["symbol"]] = round(per, _WEIGHT_DECIMALS)
    elif kind == "conviction_weighted":
        total = sum(float(item["conviction"]) for item in selected)
        if total <= _EPS:
            empty_reason = "入选标的的 conviction 合计为 0，无法按占比分配"
        else:
            for item in selected:
                weights[item["symbol"]] = round(
                    alloc * float(item["conviction"]) / total, _WEIGHT_DECIMALS
                )
    else:  # target_upside_weighted
        total = sum(float(item["upside"]) for item in selected)
        if total <= _EPS:
            empty_reason = "入选标的的目标涨幅合计为 0，无法按占比分配"
        else:
            for item in selected:
                weights[item["symbol"]] = round(
                    alloc * float(item["upside"]) / total, _WEIGHT_DECIMALS
                )

    if empty_reason:
        weights = {}

    rationale = _rationale_for(kind, rules, len(selected), cf)
    if empty_reason:
        rationale += f" 本期未产出权重：{empty_reason}。"

    prod = producer if producer is not None else DEFAULT_PRODUCER
    intent = PortfolioIntent(
        as_of=as_of,
        universe=universe_list,
        rules={k: v for k, v in rules.items()},
        weights=weights,
        producer=prod if isinstance(prod, Producer) else Producer.from_dict(dict(prod)),
        cash_floor=cf,
        derived_from=derived_from,
        excluded=excluded,
        rebalance=dict(rebalance) if rebalance is not None else None,
        rationale=rationale,
    )

    errs = check_portfolio_intent(intent.to_dict())
    if errs:
        raise ContractError(errs[0]["path"], errs[0]["reason"])
    return intent


__all__ = ["build_intent", "IntentBuildError", "RULE_KINDS", "SELECTED_SIDES", "EXCLUSION_REASONS"]


# ── MCP 工具外壳（add-portfolio-intent-contract task 5.1）─────────────
#
# 为什么要有外壳而不是让技能直接调 Python：技能只能调 MCP 工具。技能稿里写了
# `mcp__fd-quant__build_portfolio_intent`，这个名字就必须真的在工具面上——
# `check_tool_surface.py` 会逐名核对，写了不存在的工具即红。


def build_portfolio_intent(
    ratings: Sequence[Any],
    rules: Mapping[str, Any],
    as_of: str,
    universe: Optional[Sequence[str]] = None,
    cash_floor: Optional[float] = None,
    pack: str = "wire-equity-research",
    skill: str = "portfolio-construction",
    every_n_bars: Optional[int] = None,
) -> dict:
    """评级 → PortfolioIntent（纯确定性，不取数、不联网、不调模型）。

    Input: ratings（`Rating` 实例或同形 dict 数组，每条至少含 as_of / subject / side）,
      rules（{"kind": "equal_weight"|"conviction_weighted"|"target_upside_weighted",
      "cash_floor": 0..1, 可选 "conviction_floor" / "price_reference" /
      "price_reference_as_of"}）, as_of（本期决策日，ISO）,
      universe（可选，缺省取全部评级标的）, cash_floor（可选，覆盖 rules 里的值）,
      pack / skill（产出方标识）, every_n_bars（可选，调仓口径）。
    Returns: PortfolioIntent 契约实例（dict）+ rationale（规则名、参数与
      "conviction 是判定强度不是预期收益"的原话）+ counts（入选/排除/输入）。
      权重为小数（0.4 = 40%），Σ权重 + cash_floor ≤ 1，**拒绝而不归一化**。
      不做权重最优化；请求最优化时抛错并列出允许的三种规则。
    """
    rules_d = {k: v for k, v in dict(rules).items()}
    reb = {"every_n_bars": int(every_n_bars)} if every_n_bars is not None else None
    intent = build_intent(
        ratings, rules_d, as_of=as_of, universe=universe, cash_floor=cash_floor,
        producer={"pack": pack, "skill": skill}, rebalance=reb,
    )
    return {
        "tool": "build_portfolio_intent",
        "as_of": as_of,
        "intent": intent.to_dict(),
        "rationale": intent.rationale,
        "counts": {
            "input": len(intent.derived_from),
            "selected": len(intent.weights),
            "excluded": len(intent.excluded),
        },
    }


def validate_portfolio_intent_tool(intent: Any) -> dict:
    """校验一个 PortfolioIntent 实例（严格：未知字段、不变式、出处、排除原因）。

    Input: intent（实例 dict 或 JSON 字符串）。
    Returns: {"valid": bool, "errors": [{"path", "reason"}]}。
      errors 每条都带字段路径与可读原因，**从不是**一句 "invalid"。
    """
    from fd_quant.contracts.models import validate_portfolio_intent as _v

    return {"tool": "validate_portfolio_intent", **_v(intent)}


def to_target_weights(intent: Any) -> dict:
    """把 PortfolioIntent 无损导出为可执行的 TargetWeights。

    Input: intent（实例 dict 或 JSON 字符串）。
    Returns: {"tool", "target_weights": {...}, "valid": bool}。
      导出逐位无损（权重 / universe / cash_floor 相同）；执行侧只认 TargetWeights，
      故这一步是"研究结论 → 可执行权重"的唯一出口。
    """
    from fd_quant.contracts.models import PortfolioIntent as _PI
    from fd_quant.contracts.models import validate_target_weights

    inst = _PI.from_json(intent) if isinstance(intent, (str, bytes)) else _PI.from_dict(
        intent.to_dict() if hasattr(intent, "to_dict") else dict(intent)
    )
    tw = inst.to_target_weights()
    return {
        "tool": "to_target_weights",
        "target_weights": tw.to_dict(),
        "valid": validate_target_weights(tw)["valid"],
    }


#: 工具面声明：组合构建三件（纯确定性、只读、幂等）。名字必须与技能稿逐字一致。
from .compute import ToolSpec as _ToolSpec  # noqa: E402

_PORTFOLIO_READ = {
    "readOnlyHint": True,
    "destructiveHint": False,
    "idempotentHint": True,
    "openWorldHint": False,
}

PORTFOLIO_TOOLS: list[_ToolSpec] = [
    _ToolSpec("build_portfolio_intent", "portfolio_construction", build_portfolio_intent,
              tool_annotations=_PORTFOLIO_READ),
    _ToolSpec("validate_portfolio_intent", "portfolio_construction",
              validate_portfolio_intent_tool, tool_annotations=_PORTFOLIO_READ),
    _ToolSpec("to_target_weights", "portfolio_construction", to_target_weights,
              tool_annotations=_PORTFOLIO_READ),
]
