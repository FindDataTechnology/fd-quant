#!/usr/bin/env python3
"""最小闭合演练（openspec change: close-first-decision-loop, tasks 3.x）。

**为什么要有这个脚本**：2026-11-13 生产实测 `decisions = 0`、`settlements = 0`——
闭环一次都没通过电，而所有闸门都是绿的（它们验的是"实现符合规格"，不是"这条链跑过"）。
本脚本用**不依赖基准序列**的 `metric_threshold` 口径结出第一笔，证明"链路在技术上可以
闭合"；它不证明"业务上有人在用"，后者只有真实客户落账才能证明（探针的 production 档
在之前会持续红，那是刻意的）。

三条纪律：

1. **演练记录可识别**：`pack` 前缀 `drill:`、caller 默认 `drill:close-first-decision-loop`。
   探针默认 `--mode production` 会排除它们——演练数据进客户命中率就是污染。
2. **不依赖基准与最新价**：阈值口径的观测值由本脚本传入（确定性序列），账本不取数。
   指数族零观测、A 股价格停更都不妨碍这一步闭合。
3. **幂等**：同一键重复执行返回既有记录，不产生第二行（账本自己的幂等键保证）。

用法：

    python scripts/close_first_decision_drill.py                 # 临时库，跑完打印路径
    python scripts/close_first_decision_drill.py --db /tmp/x.db
    python scripts/close_first_decision_drill.py --verdict miss  # 造一条 miss 对照
    python scripts/close_first_decision_drill.py --json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Optional, Sequence

#: 演练前缀（与探针的 DRILL_PREFIX 同一常量语义）。
DRILL_PREFIX = "drill:"
DRILL_CALLER = "drill:close-first-decision-loop"
DRILL_PACK = f"{DRILL_PREFIX}close-first-decision-loop"
DRILL_SKILL = "decision-loop-closure-drill"

#: 固定口径：可逐位复现，不依赖当天日期。
AS_OF = "2026-11-13"
WINDOW_DATES = ["2026-08-31", "2026-09-30", "2026-10-31"]
THRESHOLD = 0.42
COMPARATOR = "gte"


def _payload(subject: str) -> dict:
    """v2 决策实例：结构由 `generalize-decision-contract` 定义。

    `expected_outcome.metric_ref` 是阈值结算必须知道的东西（缺它结算会拒绝并点名），
    所以演练也必须给全——不是为了迁就校验，而是因为真实调用方同样拿不到结算口径。
    """
    return {
        "schema_version": "2",
        "as_of": AS_OF,
        "subject": subject,
        "subject_kind": "store",
        "action": "expand",
        "preconditions": [],
        "settlement_rule_id": "metric_threshold",
        "expected_outcome": {
            "metric_ref": subject,
            "comparator": COMPARATOR,
            "threshold": THRESHOLD,
            "horizon": {"unit": "month", "count": 3},
            "settlement_source": "caller_supplied_series",
        },
        "basis": "演练判定：门店月均流水连续三个月高于保本线（示例数据，非真实业务）。",
        "evidence": [],
        "producer": {"pack": DRILL_PACK, "skill": DRILL_SKILL},
    }


def run(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="最小闭合演练（不依赖基准序列）")
    ap.add_argument("--db", help="账本路径（默认临时库，跑完保留并打印）")
    ap.add_argument("--subject", default="drill-store-0001", help="演练主体")
    ap.add_argument("--verdict", choices=("hit", "miss"), default="hit",
                    help="期望结论（默认 hit：末值超过阈值）")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    # caller 由环境变量解析（账本不接受入参声明身份）。显式设成演练身份，
    # 保证演练不会混进任何客户口径的聚合里。
    os.environ.setdefault("FD_QUANT_CALLER", DRILL_CALLER)
    caller = os.environ["FD_QUANT_CALLER"]
    if not caller.startswith(DRILL_PREFIX):
        print(
            f"[fatal] FD_QUANT_CALLER={caller!r} 不以 {DRILL_PREFIX!r} 开头；"
            "演练必须在演练身份下运行，否则会污染真实调用者的统计",
            file=sys.stderr,
        )
        return 2

    db_path = Path(args.db).expanduser() if args.db else Path(
        tempfile.mkdtemp(prefix="fd-quant-drill-")
    ) / "drill_ledger.db"

    from fd_quant.caller import resolve_caller  # noqa: PLC0415
    from fd_quant.ledger.store import LedgerStore  # noqa: PLC0415

    resolved = resolve_caller()
    if not resolved.startswith(DRILL_PREFIX):
        print(f"[fatal] 解析出的 caller={resolved!r} 不是演练身份", file=sys.stderr)
        return 2

    store = LedgerStore(db_path)
    subject = args.subject
    rec = store.record(
        caller=resolved, as_of=AS_OF, symbol=subject,
        pack=DRILL_PACK, skill=DRILL_SKILL, payload=_payload(subject), evidence=[],
    )
    decision_id = rec["decision_id"]

    # 末值决定结论：0.45 过 0.42 → hit；0.38 不过 → miss。
    values = [0.36, 0.40, 0.45] if args.verdict == "hit" else [0.36, 0.40, 0.31]
    settled = store.settle_metric_threshold(
        decision_id=decision_id,
        dates=WINDOW_DATES,
        values=values,
        as_of=AS_OF,
        comparator=COMPARATOR,
        threshold=THRESHOLD,
        reflection=(
            "演练结论：阈值口径可在无基准序列时给出明确结论；"
            "这一步只证明链路可闭合，不构成任何真实业务判断。"
        ),
    )

    out = {
        "ok": settled.get("verdict") in ("hit", "miss") and bool(settled.get("reflection")),
        "db_path": str(db_path),
        "caller": resolved,
        "decision_id": decision_id,
        "created": rec.get("created"),
        "verdict": settled.get("verdict"),
        "rule_id": settled.get("rule_id"),
        "observed_value": settled.get("observed_value"),
        "threshold": settled.get("threshold"),
        "gap_kinds": settled.get("gap_kinds"),
        "reflection_present": bool(settled.get("reflection")),
        "counts": settled.get("counts"),
    }
    if args.json:
        print(json.dumps(out, ensure_ascii=False, indent=1))
    else:
        print(f"演练账本：{db_path}")
        print(f"caller={resolved}  decision_id={decision_id}  created={out['created']}")
        print(
            f"结算：verdict={out['verdict']} rule={out['rule_id']} "
            f"observed={out['observed_value']} threshold={out['threshold']} "
            f"gap_kinds={out['gap_kinds']}"
        )
        print(f"反思非空：{out['reflection_present']}")
        print(f"行数快照：{out['counts']}")
        print(
            "\n下一步：\n"
            f"  python scripts/decision_loop_probe.py --mode drill --db {db_path}"
        )
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    sys.exit(run())
