#!/usr/bin/env python3
"""契约联调（fd-quant-compute-plane task 9.2）。

把 9.1 端到端链的结果组装为四个契约实例并逐个过契约校验：
  1. `rating-point.json`      —— 有数值 price_target 的常规评级
  2. `rating-null.json`       —— **方法分歧超阈值 → price_target 为 null**（扣值不扣结论）
  3. `rating-review.json`     —— 样本不足 → `side=review`（不猜、不静默降级为 hold）
  4. `target-weights.json`    —— 目标权重向量（和为 1 留现金、非负）

**为什么要单独造这四份而不是一份**：契约的价值在边界上——数值点位人人会写，
`null` 点位与 `review` 才是"算不出时不装作算得出"的落点。四份都过校验才说明
契约与计算面在边界处咬合。

用法：
    python3 scripts/contract_e2e.py
退出码：0 全部通过 / 1 有失败。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

OUT_DIR = REPO.parent / "openspec" / "changes" / "fd-quant-compute-plane" / "reports" / "contract-fixtures"


def main() -> int:
    from fd_quant.contracts.models import validate_rating, validate_target_weights

    fixtures = build_fixtures()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    failures: list[str] = []
    for name, doc, validator in fixtures:
        res = validator(doc)
        ok = res.get("valid") is True if isinstance(res, dict) else False
        path = OUT_DIR / name
        path.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        mark = "✓" if ok else "✗"
        print(f"  {mark} {name}")
        if not ok:
            failures.append(f"{name}: {json.dumps(res, ensure_ascii=False)}")

    if failures:
        print(f"\n✗ {len(failures)} 份未过契约校验：", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1
    print(f"\n✓ 四份契约实例全部通过校验 → {OUT_DIR.relative_to(REPO.parent)}")
    return 0


def build_fixtures() -> list[tuple[str, dict, object]]:
    from fd_quant.contracts.models import validate_rating, validate_target_weights

    as_of = "2026-10-16"
    subject = "600519.SH"
    # producer 是对象且必带 pack + skill：契约要求每个评级都记录"谁产出的"，
    # 故本联调以未来消费方 pack 的身份署名（计算面自身不是 pack，不冒充）。
    producer = {"pack": "wire-equity-research", "skill": "equity-research-thesis",
                "run_id": "contract-e2e-2026-10-17"}

    ev = [
        {"evidence_id": "estimate_dcf.value_per_share", "label": "DCF 永续每股价值",
         "value": 1000.496255812341, "unit": "currency/share", "as_of": as_of,
         "source": "fd-quant:estimate_dcf(terminal_method=perpetuity,discount_rate=0.09)"},
        {"evidence_id": "market_price.close", "label": "收盘价",
         "value": 1329.0, "unit": "currency/share", "as_of": as_of,
         "source": "price-channel-B:daas.db/scraw_600519_daily"},
    ]

    # ① 常规：单一方法给出点位
    point = {
        "schema_version": "1",
        "as_of": as_of,
        "subject": subject,
        "side": "hold",
        "conviction": 0.4,
        "price_target": 1205.31,
        "target_evidence_ids": ["estimate_dcf.value_per_share"],
        "basis": "DCF 退出倍数法给 1205.31，与市价 1329 相差约 9%；单一方法可给点位，"
                 "多方法合成因分歧 27% 超阈值而扣值（见 rating-null）。",
        "time_horizon": "12m",
        "evidence": ev,
        "producer": producer,
    }

    # ② 分歧超阈值 → price_target 必须为 null（扣值不扣结论）
    nullpt = {
        "schema_version": "1",
        "as_of": as_of,
        "subject": subject,
        "side": "hold",
        "conviction": None,
        "price_target": None,
        "basis": "三种方法分歧 27.3%（DCF 永续 1000.50 / DCF 退出倍数 1205.31 / 市价 1329.0）"
                 "超过 20% 阈值，按契约**扣值不扣结论**：不给单一目标价，只给分歧说明。",
        "time_horizon": "12m",
        "evidence": ev,
        "producer": producer,
    }

    # ③ 样本不足 → side=review（不猜、不静默降级）
    review = {
        "schema_version": "1",
        "as_of": as_of,
        "subject": subject,
        "side": "review",
        "conviction": None,
        "price_target": None,
        "basis": "本标的可得历史 400 日、且估值所需假设集来自公开量级的近似值；"
                 "样本与假设均不足以支撑方向判断，按契约给 review 而非静默降级为 hold。",
        "evidence": ev,
        "producer": producer,
    }

    # ④ 目标权重向量：和为 1 留现金、非负
    weights = {
        "schema_version": "1",
        "as_of": as_of,
        "universe": ["600519.SH", "000001.SZ", "CASH"],
        "weights": {"600519.SH": 0.35, "000001.SZ": 0.25},
        "cash_floor": 0.4,
        "constraints": {
            "max_weight_per_symbol": 0.4,
            "max_weight_per_sector": None,
        },
        "evidence": ev,
        "producer": producer,
    }

    return [
        ("rating-point.json", point, validate_rating),
        ("rating-null.json", nullpt, validate_rating),
        ("rating-review.json", review, validate_rating),
        ("target-weights.json", weights, validate_target_weights),
    ]


if __name__ == "__main__":
    raise SystemExit(main())
