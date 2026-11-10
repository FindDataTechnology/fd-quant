#!/usr/bin/env python3
"""执行岗"下单前强制过闸"的链路验证（change3 task 3.2）。

## 这个脚本证明什么、不证明什么

**证明**（全本地、无需编成、无需模型）：
1. 缺任一阈值时，门返回 `unavailable` 并点名缺的那一项 —— 不是"当作通过"；
2. 该判定下订单意图**导不出去**（`export_order_intent` 抛错），
   所以"闸没配置还照下单"在契约层是做不到的；
3. 执岗的稿子（种子 persona + 节奏 prompt）把"阈值未配置"列为不行动条件之一。

**不证明**：某一轮真实回合真的因此产生了零订单。那需要舰队被实化 + 模型跑一轮，
属于编成侧（`openspec/changes/add-investment-trading-fleet` 的 §6 卡点 table 里挂起）。
本脚本把"机制链"钉住，实跑那一步只需在实化后看产出。

用法：
    PYTHONPATH=. python3 scripts/verify_executor_gate_chain.py [--seed PATH]
退出码：0 = 三段全绿；1 = 有段落不成立（逐段打印原因）；2 = 种子读不到（跑不完整）
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

DEFAULT_SEED = pathlib.Path(__file__).resolve().parents[2] / (
    "paas/docs/fleet-templates/invest-trading-fleet.seed.json"
)


def _gate_refuses_when_a_threshold_is_missing() -> list[str]:
    """① 缺阈值 → unavailable 且点名缺项（不是"通过"）。"""
    from fd_quant.risk import evaluate_portfolio_gate

    problems: list[str] = []
    verdict = evaluate_portfolio_gate(
        weights={"600519": 0.4, "000001": 0.4},
        thresholds={"per_symbol_max": 0.5},          # 故意只给一项
        as_of="2026-06-30",
    )
    if verdict.get("status") != "unavailable":
        problems.append(f"缺阈值时门应报 unavailable，实际 {verdict.get('status')!r}")

    # 缺失项要**点名**：从 inputs.missing 里读，别只信 status
    missing = set((verdict.get("inputs") or {}).get("missing") or [])
    for want in ("industry_max", "drawdown_max", "daily_loss_max"):
        if want not in missing:
            problems.append(f"缺项未被点名：{want}（实际 {sorted(missing)}）")

    # 反向：四项齐全且都不触达时必须是 ok，否则"恒 unavailable"也能骗过上面两条
    full = evaluate_portfolio_gate(
        weights={"600519": 0.4, "000001": 0.4},
        thresholds={"per_symbol_max": 0.5, "industry_max": 0.9,
                    "drawdown_max": 0.9, "daily_loss_max": 0.9},
        as_of="2026-06-30",
    )
    if full.get("status") != "ok":
        problems.append(f"四项齐全且未触达时应为 ok，实际 {full.get('status')!r}")
    return problems


def _order_cannot_be_exported_without_a_usable_verdict() -> list[str]:
    """② unavailable 判定下订单意图导不出去 —— "没配阈值还下单"在契约层不成立。"""
    from fd_quant.order_intent import OrderIntentError, export_order_intent

    problems: list[str] = []
    intent = {
        "schema_version": "1",
        "intent_id": "verify-1",
        "as_of": "2026-06-30",
        "account_id": "acc-1",
        "symbol": "600519",
        "side": "buy",
        "order_type": "market",
        "quantity": 100.0,
        "derived_from": ["2026-06-30-eq"],
        "risk_verdict": {"status": "unavailable",
                         "inputs": {"missing": ["industry_max"]}},
        "producer": {"pack": "wire-invest-trading", "skill": "portfolio-construction"},
        "paper_only": True,
    }
    try:
        export_order_intent(intent)
    except OrderIntentError as exc:
        if "unavailable" not in str(exc):
            problems.append(f"拒绝理由应点名 unavailable，实际：{exc}")
    else:
        problems.append("口径不可用时订单意图竟然导出成功")
    return problems


def _executor_prompt_lists_thresholds_unset_as_a_no_go(seed_path: pathlib.Path) -> list[str]:
    """③ 执岗稿子把"阈值未配置"列为不行动条件（否则前两步只是理论上的）。"""
    problems: list[str] = []
    try:
        seed = json.loads(seed_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise SystemExit(2)

    by_id = {a.get("id"): a for a in seed.get("agents", [])}
    executor = by_id.get("trade-execute")
    if executor is None:
        return [f"种子里没有 trade-execute（实际 {sorted(by_id)}）"]

    persona = executor.get("persona") or ""
    if "阈值未配置" not in persona:
        problems.append("执岗 persona 未把「阈值未配置」列为不行动条件")

    member = next((m for m in seed.get("fleet", {}).get("members", [])
                   if m.get("ref") == "execute"), None)
    if member is None:
        return problems + ["fleet.members 里没有 execute 成员"]
    prompt = ((member.get("cadence") or {}).get("promptTemplate") or "")
    if "不行动条件" not in prompt:
        problems.append("执岗节奏 prompt 没有提到不行动条件（每轮读到的就是这段）")
    if "evaluate_portfolio_gate" not in prompt:
        problems.append("执岗节奏 prompt 没有点名 evaluate_portfolio_gate（过闸动作不可见）")
    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=pathlib.Path, default=DEFAULT_SEED)
    args = ap.parse_args(argv)

    steps = [
        ("① 缺阈值 → unavailable 且点名缺项", _gate_refuses_when_a_threshold_is_missing),
        ("② unavailable 下订单意图导不出去", _order_cannot_be_exported_without_a_usable_verdict),
        ("③ 执岗稿把「阈值未配置」列为不行动条件",
         lambda: _executor_prompt_lists_thresholds_unset_as_a_no_go(args.seed)),
    ]

    failed = 0
    for title, fn in steps:
        try:
            problems = fn()
        except SystemExit:
            print(f'✗ 种子读不到：{args.seed}（跑不完整，不等于「没有违规」）', file=sys.stderr)
            return 2
        except Exception as exc:  # noqa: BLE001 - 闸自己崩了也是"这一段不成立"
            # 崩栈不是"证明通过"，也不是"跑不完整"：它是这一段明确不成立，
            # 所以计一次失败并继续跑后面的段（别让第一段崩掉就看不到全貌）。
            failed += 1
            print(f"✗ {title}（抛异常）")
            print(f"    · {type(exc).__name__}: {exc}")
            continue
        if problems:
            failed += 1
            print(f"✗ {title}")
            for p in problems:
                print(f"    · {p}")
        else:
            print(f"✓ {title}")

    if failed:
        print(f"\n✗ {failed} 段不成立：机制链有断口，实跑前必须先修", file=sys.stderr)
        return 1
    print("\n✓ 三段全绿：闸会拒、意图导不出、执岗稿点名了它")
    print("  仍未证明：真实回合是否零订单（需舰队实化工，见 change3 §6 卡点表）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
