#!/usr/bin/env python3
"""端到端演练（fd-quant-decision-ledger task 5.2）：落账 → 结算 → 点内检索 → 聚合。

链路（全离线，无网络、无外部数据源）：
    构造 5 条决策（含 1 次修订、1 次重复提交被拒、1 次提前结算被拒）
      → 全部结算（含一次幂等重结算）
      → 点内检索（含一条"as_of 之后才结算"的对照，证明它被过滤）
      → 按 pack 聚合（含一个空组）
    输出 reports/ledger-e2e.md，每一步的数字与 decision_id 都落在报告里。

**为什么"离线"是这条演练的核心断言**：账本的纪律是"不取数"——价格与基准序列由
调用方传入。这条演练把"调用方"演成一个确定性构造的序列源（`_SeriesSource`），
于是整条链路可以在封网环境下重跑，且每次输出的数字逐位相同（可用 `--check` 验）。
若有人给账本加了取数逻辑，封网运行会立刻失败。

用法：
    python3 scripts/ledger_e2e.py            # 跑并写报告
    python3 scripts/ledger_e2e.py --check    # 只验复跑一致，不写文件
退出码：0 成功 / 1 失败。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))

import ledgerdata as ld  # noqa: E402  （tests/ 下的确定性夹具）
from fd_quant.contracts.models import validate  # noqa: E402
from fd_quant.ledger import (  # noqa: E402
    LedgerStore,
    RevisionConflict,
    SettlementRefused,
    resolve_benchmark,
)

OUT = (REPO.parent / "openspec" / "changes" / "fd-quant-decision-ledger"
       / "reports" / "ledger-e2e.md")

#: 演练用的 pack / skill（两条 pack 线：一条有结算、一条只有 pending，用于空组演示）。
PACK_A = "pack-equity-research"
PACK_B = "pack-portfolio-risk"
SKILL_A = "equity-research-thesis"
SKILL_B = "portfolio-risk-budget"


class _SeriesSource:
    """"调用方"的价格源：确定性构造，不联网、不读库。

    每个标的由 (起点, 每交易日涨幅, 交易日数) 定义，日期轴 = 决策 as_of 起的连续自然日
    （账本的交易日历来自传入序列的日期集合，故这里"自然日"就是这套夹具的交易日）。
    """

    def __init__(self) -> None:
        self.spec: dict[str, tuple[float, float, int]] = {}

    def define(self, symbol: str, start_price: float, daily_return: float,
               days: int = 40) -> None:
        self.spec[symbol] = (start_price, daily_return, days)

    def series(self, symbol: str, start: str, days: int) -> dict:
        base, rate, _cap = self.spec[symbol]
        prices = [round(base * (1.0 + rate) ** i, 6) for i in range(days)]
        return {"dates": ld.dates(days, start=start), "prices": prices}


def _dump(obj: object) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)


def run(db_path: Path) -> dict:
    """跑完整条链路，返回每一步的结果（供报告与 `--check` 使用）。

    `id_factory=ld.sequential_id_factory()` 让 `decision_id` 确定：报告要能复跑比对，
    而 uuid4 派生的 id 每次都不同（默认值对并发是对的，对可复现的报告不是）。
    """
    store = LedgerStore(db_path, id_factory=ld.sequential_id_factory("dec-e2e"))
    src = _SeriesSource()
    src.define("600519.SS", 1600.0, 0.004)    # 标的 A：上行
    src.define("000001.SZ", 12.0, -0.002)     # 标的 B：下行
    src.define("300750.SZ", 220.0, 0.006)     # 标的 C：上行
    src.define("601318.SS", 48.0, 0.001)      # 标的 D：微升
    src.define("000300.SS", 3800.0, 0.002)    # 显式基准（沪深 300）
    # 映射基准：.SS→000001.SS、.SZ→399001.SZ（账本默认映射）
    src.define("000001.SS", 3100.0, 0.0015)
    src.define("399001.SZ", 9800.0, 0.001)

    steps: dict[str, object] = {}

    # ── 步骤 1：落 5 条决策（含一次修订）─────────────────────────
    decisions = [
        ("600519.SS", "2025-01-06", PACK_A, SKILL_A, "buy", 0.7, "消费复苏主线，估值中位。"),
        ("000001.SZ", "2025-01-06", PACK_A, SKILL_A, "sell", 0.55, "息差收窄压制盈利。"),
        ("300750.SZ", "2025-01-06", PACK_B, SKILL_B, "buy", 0.6, "储能出货超预期。"),
        ("601318.SS", "2025-01-06", PACK_B, SKILL_B, "hold", 0.4, "负债端改善但资产端未验证。"),
    ]
    recorded = []
    for symbol, as_of, pack, skill, side, conviction, basis in decisions:
        r = store.record(
            as_of=as_of, symbol=symbol, pack=pack, skill=skill,
            payload=ld.rating(as_of, symbol, side=side, conviction=conviction, basis=basis,
                              pack=pack, skill=skill),
            now=f"{as_of}T09:00:00+00:00",
        )
        assert r["created"] is True, r
        recorded.append(r)
    steps["1_recorded_initial"] = {
        "count": len(recorded),
        "decisions": [{"decision_id": r["decision_id"], "symbol": r["symbol"],
                       "pack": r["pack"], "revision_seq": r["revision_seq"],
                       "status": r["status"]} for r in recorded],
        "row_counts": store.row_counts(),
    }

    # ── 步骤 2：第 5 条决策 + 一次修订（新行、revises 指回、原行不变）──
    target = recorded[3]
    before_revision = store.get(target["decision_id"])
    revision = store.record(
        as_of=target["as_of"], symbol=target["symbol"], pack=target["pack"],
        skill=target["skill"],
        payload=ld.rating(target["as_of"], target["symbol"], side="buy", conviction=0.5,
                          basis="负债端与资产端双双改善，上调至看多。",
                          pack=target["pack"], skill=target["skill"]),
        revises=target["decision_id"], now="2025-01-07T09:00:00+00:00",
    )
    assert revision["created"] is True and revision["revises"] == target["decision_id"]
    after_revision = store.get(target["decision_id"])
    assert after_revision["payload"] == before_revision["payload"]
    assert after_revision["created_at"] == before_revision["created_at"]
    steps["2_revision"] = {
        "original_decision_id": target["decision_id"],
        "revision_decision_id": revision["decision_id"],
        "revision_seq": revision["revision_seq"],
        "revises": revision["revises"],
        "original_payload_unchanged": after_revision["payload"] == before_revision["payload"],
        "original_created_at": after_revision["created_at"],
        "revision_row_counts": store.row_counts(),
    }

    # ── 步骤 3：重复提交被拒（返回既有 id、行数不变）─────────────
    counts_before_duplicate = store.row_counts()
    duplicate = store.record(
        as_of=decisions[0][1], symbol=decisions[0][0], pack=PACK_A, skill=SKILL_A,
        payload=ld.rating(decisions[0][1], decisions[0][0], side="buy", conviction=0.7,
                          basis="消费复苏主线，估值中位。", pack=PACK_A, skill=SKILL_A),
    )
    assert duplicate["created"] is False and duplicate["reason"] == "duplicate_key"
    assert store.row_counts() == counts_before_duplicate
    steps["3_duplicate_refused"] = {
        "created": duplicate["created"],
        "reason": duplicate["reason"],
        "returned_existing_id": duplicate["decision_id"],
        "same_as_original": duplicate["decision_id"] == recorded[0]["decision_id"],
        "row_counts_before": counts_before_duplicate,
        "row_counts_after": store.row_counts(),
        "rows_unchanged": store.row_counts() == counts_before_duplicate,
    }

    # ── 步骤 4：提前结算被拒（序列不足持有期，报缺口天数）────────
    premature_id = recorded[0]["decision_id"]
    short = src.series("600519.SS", start=decisions[0][1], days=3)
    premature: dict
    try:
        store.settle(decision_id=premature_id, symbol_dates=short["dates"],
                     symbol_prices=short["prices"], benchmark_dates=short["dates"],
                     benchmark_prices=short["prices"], as_of=short["dates"][-1],
                     holding_days=5)
        raise AssertionError("premature settlement should have been refused")
    except SettlementRefused as exc:
        premature = {"refused": True, "message": str(exc),
                     "missing_trading_days": exc.missing_trading_days}
    assert store.row_counts()["settlements"] == 0
    assert store.get(premature_id)["status"] == "pending"
    steps["4_premature_refused"] = {
        **premature,
        "settlements_after": store.row_counts()["settlements"],
        "status_still": store.get(premature_id)["status"],
    }

    # ── 步骤 5：全部结算（含一次幂等重结算、一条显式基准、一条 10 日持有期）──
    settlements = []
    plan = [
        # (decision_id, holding_days, benchmark_symbol, reflection)
        (target["decision_id"], 5, None, "保险负债端改善兑现，资产端仍未验证；下调持有期内的仓位。"),
        (revision["decision_id"], 5, None, None),
        (recorded[0]["decision_id"], 5, None, "消费复苏兑现幅度高于预期，渠道库存是先行指标。"),
        (recorded[1]["decision_id"], 3, "000300.SS",
         "息差压力如期，但下行幅度被拨备回补抵消——基准口径换成宽基后 alpha 更正。"),
        (recorded[2]["decision_id"], 10, None, "储能出货确认，长持有期给了估值修复更多时间。"),
    ]
    for did, hd, bench, reflection in plan:
        item = store.get(did)
        days = hd + 6  # 多给几点，覆盖持有期之外
        sym = src.series(item["symbol"], start=item["as_of"], days=days)
        # 基准用**账本自己的映射**解析（演练不该复制一份映射表：复制的那份会与账本漂移）
        bench_symbol = resolve_benchmark(item["symbol"], explicit=bench)["benchmark_symbol"]
        if bench_symbol not in src.spec:  # 未预定义：显式失败而非静默换个基准
            raise AssertionError(f"no benchmark series defined for {bench_symbol}")
        bs = src.series(bench_symbol, start=item["as_of"], days=days)
        out = store.settle(decision_id=did, symbol_dates=sym["dates"],
                           symbol_prices=sym["prices"], benchmark_dates=bs["dates"],
                           benchmark_prices=bs["prices"], as_of=sym["dates"][-1],
                           holding_days=hd, benchmark_symbol=bench,
                           reflection=reflection, now="2025-02-01T00:00:00+00:00")
        assert out["created"] is True, out
        settlements.append({"decision_id": did, "symbol": out["symbol"],
                            "holding_days": out["holding_days"],
                            "entry_price": out["entry_price"], "exit_price": out["exit_price"],
                            "raw_return": out["raw_return"],
                            "benchmark_symbol": out["benchmark_symbol"],
                            "benchmark_mapping": out["benchmark_mapping"],
                            "benchmark_return": out["benchmark_return"], "alpha": out["alpha"],
                            "settlement_date": out["settlement_date"],
                            "reflection": out["reflection"]})
    steps["5_settlements"] = {"count": len(settlements), "items": settlements,
                              "row_counts": store.row_counts()}

    # 幂等重结算：行数不变、数字逐位一致
    first = store.get(recorded[0]["decision_id"])["settlement"]
    again = store.settle(decision_id=recorded[0]["decision_id"],
                         symbol_dates=src.series("600519.SS", start=decisions[0][1],
                                                 days=11)["dates"],
                         symbol_prices=src.series("600519.SS", start=decisions[0][1],
                                                  days=11)["prices"],
                         benchmark_dates=src.series("000001.SS", start=decisions[0][1],
                                                    days=11)["dates"],
                         benchmark_prices=src.series("000001.SS", start=decisions[0][1],
                                                     days=11)["prices"],
                         as_of=ld.dates(11, start=decisions[0][1])[-1])
    assert again["created"] is False and again["reason"] == "duplicate_settlement"
    second = store.get(recorded[0]["decision_id"])["settlement"]
    assert second == first
    assert store.row_counts()["settlements"] == len(settlements)
    steps["6_settlement_idempotent"] = {
        "created": again["created"], "reason": again["reason"],
        "alpha_unchanged": again["alpha"] == settlements[2]["alpha"],
        "settlements_row_count": store.row_counts()["settlements"],
    }

    # ── 步骤 6：点内检索（含"未来结算被过滤"的对照）─────────────
    # 三个口径：① 早于任何结算日 → 全空；② 落在中间 → 只看到已结算的那几条；
    # ③ 全历史 → 全部可见。②是本演练的关键对照：10 日持有期的那条结算日在
    # 2025-01-16，故以 2025-01-12 检索时它必须缺席（点内纪律）。
    cutoffs = ["2025-01-08", "2025-01-12", "2025-12-31"]
    pit = {c: store.retrieve(symbol="600519.SS", as_of=c, n_same=5, m_cross=3)
           for c in cutoffs}
    assert pit["2025-01-08"]["counts"]["total"] == 0, pit["2025-01-08"]

    def _items(result: dict) -> list[dict]:
        return result["same_symbol"] + result["cross_symbol"]

    full, mid = pit["2025-12-31"], pit["2025-01-12"]
    # ① 严格点内：任何口径下都不出现结算日 > as_of 的记录
    for cutoff in cutoffs:
        assert all(x["settlement"]["settlement_date"] <= cutoff for x in _items(pit[cutoff])), (
            cutoff, [x["settlement"]["settlement_date"] for x in _items(pit[cutoff])]
        )
    # ② 未来结算**结构性缺席**：全历史里 as_of 之后才结算的那条，在中间口径里一条都不出现。
    #    （注意不能说"只少那一条"：N/M 限额会让被挤掉的更早记录补位——限额与过滤是
    #     两件不同的事，混着断言会把限额行为误报成过滤失败。）
    late_after_mid = {x["decision_id"] for x in _items(full)
                      if x["settlement"]["settlement_date"] > "2025-01-12"}
    assert late_after_mid, "the 10-trading-day settlement should be later than 2025-01-12"
    assert len(late_after_mid) == 1
    assert not (late_after_mid & {x["decision_id"] for x in _items(mid)})
    # ③ 该记录在更早的口径里也不出现，在更晚的口径里出现（时间轴两端的对照）
    assert not (late_after_mid & {x["decision_id"] for x in _items(pit["2025-01-08"])})
    assert late_after_mid <= {x["decision_id"] for x in _items(full)}
    steps["7_point_in_time"] = {
        "cuts": [
            {
                "as_of": c,
                "counts": pit[c]["counts"],
                "settlement_dates": sorted(
                    x["settlement"]["settlement_date"]
                    for x in pit[c]["same_symbol"] + pit[c]["cross_symbol"]
                ),
            }
            for c in cutoffs
        ],
        "future_settlement_hidden": {
            "decision_ids": sorted(late_after_mid),
            "settled_after": "2025-01-12",
            "absent_from_cutoffs": ["2025-01-08", "2025-01-12"],
            "present_at_cutoff": cutoffs[-1],
        },
        "note": ("结算日晚于 as_of 的记录一律不返回：10 日持有期那条在 2025-01-16 结算，"
                 "在 2025-01-08 与 2025-01-12 两个口径里都不出现，只有在 2025-12-31 口径才可见。"
                 "注意 N/M 限额会让被挤掉的更早记录补位，故各口径的条数可能相同——"
                 "被过滤是结构性的（那条记录本身不在），与限额是两件事。"),
        "full_history": {
            "as_of": cutoffs[-1],
            "n_same": pit[cutoffs[-1]]["n_same"],
            "m_cross": pit[cutoffs[-1]]["m_cross"],
            "counts": pit[cutoffs[-1]]["counts"],
            "same_symbol": [
                {"decision_id": x["decision_id"], "symbol": x["symbol"],
                 "settlement_date": x["settlement"]["settlement_date"],
                 "alpha": x["settlement"]["alpha"],
                 "holding_days": x["settlement"]["holding_days"],
                 "reflection": x["settlement"]["reflection"],
                 "payload_side": x["payload"]["side"],
                 "revision_seq": x["revision_seq"]}
                for x in pit[cutoffs[-1]]["same_symbol"]
            ],
            "cross_symbol": [
                {"decision_id": x["decision_id"], "symbol": x["symbol"],
                 "settlement_date": x["settlement"]["settlement_date"],
                 "alpha": x["settlement"]["alpha"],
                 "holding_days": x["settlement"]["holding_days"]}
                for x in pit[cutoffs[-1]]["cross_symbol"]
            ],
            "ordered_desc": all(
                a["settlement"]["settlement_date"] >= b["settlement"]["settlement_date"]
                for group in (pit[cutoffs[-1]]["same_symbol"],
                              pit[cutoffs[-1]]["cross_symbol"])
                for a, b in zip(group, group[1:])
            ),
            "latest_first_same_symbol": (
                [x["settlement"]["settlement_date"]
                 for x in pit[cutoffs[-1]]["same_symbol"]]
            ),
        },
    }

    # ── 步骤 7：聚合（含一个空组）────────────────────────────────
    stats_all = store.stats_by_pack(as_of="2025-12-31")
    stats_by_skill = store.stats_by_pack(as_of="2025-12-31", group_by_skill=True)
    stats_by_holding = store.stats_by_pack(as_of="2025-12-31", group_by_holding=True,
                                            holding_days=[3, 5, 10])
    stats_empty = store.stats_by_pack(as_of="2025-12-31", packs=["pack-never-used"])
    steps["8_aggregate"] = {
        "by_pack": stats_all["rows"],
        "by_pack_and_skill": stats_by_skill["rows"],
        "by_pack_and_holding": stats_by_holding["rows"],
        "empty_group": stats_empty["rows"],
    }

    # ── 步骤 8：契约联调（payload 落账 → 查回 → 过契约校验）──────
    contract_checks = []
    for r in [*recorded, revision]:
        item = store.get(r["decision_id"])
        result = validate(item["payload"], kind="rating")
        contract_checks.append({"decision_id": r["decision_id"],
                                "valid": result["valid"],
                                "errors": result["errors"]})
    assert all(c["valid"] for c in contract_checks), contract_checks
    steps["9_contract_roundtrip"] = contract_checks

    return steps


def _fingerprint(steps: dict) -> str:
    """结果的指纹：用于 `--check` 验证复跑逐位一致。

    值得说清"为什么它可以是逐位确定的"：演练**注入了**两处不确定性——确定性 `id_factory`
    （decision_id）与显式 `now`（created_at / settled_at）。账本的默认行为是 uuid4 +
    当前 UTC 时刻（对并发与审计是对的），但报告要能比对，故在接缝上替换掉。
    """
    return hashlib.sha256(_dump(steps).encode("utf-8")).hexdigest()


def render(steps: dict, fingerprint: str) -> str:
    """把每一步写成可读报告（数字原样，不重新计算）。"""
    s = steps
    lines: list[str] = []
    add = lines.append
    add("# 决策账本端到端演练（fd-quant-decision-ledger task 5.2）")
    add("")
    add("**离线**运行：无网络、无外部数据源。价格与基准序列由演练自带的确定性价格源")
    add("（`_SeriesSource`）构造后**传入**账本——账本不取数，这是它的核心纪律。")
    add("")
    add("- 复跑脚本：`python3 scripts/ledger_e2e.py`（`--check` 只验复跑一致，不写文件）")
    add("- 库路径：临时单文件 SQLite（生产用 `FD_QUANT_LEDGER_DB` 或 `db_path` 参数）")
    add("- 确定性来源：演练注入顺序 `decision_id` 生成器与显式时间戳，故复跑逐位一致")
    add(f"- 结果指纹（sha256）：`{fingerprint}`")
    add("")
    add("## 步骤 1 — 落账 5 条决策（4 条初始 + 1 条修订，见步骤 2）")
    add("")
    add(f"落账 {s['1_recorded_initial']['count']} 条初始记录；行数 "
        f"`{_dump(s['1_recorded_initial']['row_counts'])}`。第 5 条由步骤 2 的修订产生"
        "（修订是新记录，不是改写）。")
    add("")
    add("| decision_id | symbol | pack | revision_seq | status |")
    add("|---|---|---|---|---|")
    for d in s["1_recorded_initial"]["decisions"]:
        add(f"| `{d['decision_id']}` | {d['symbol']} | {d['pack']} | {d['revision_seq']} | {d['status']} |")
    add("")
    add("## 步骤 2 — 一次修订（新行 + revises 指回 + 原行逐字不变）")
    add("")
    r = s["2_revision"]
    add(f"- 原记录 `{r['original_decision_id']}` → 修订 `{r['revision_decision_id']}`")
    add(f"- `revises` = `{r['revises']}`，`revision_seq` = {r['revision_seq']}")
    add(f"- 原记录 `payload` 不变：**{r['original_payload_unchanged']}**；"
        f"`created_at` = `{r['original_created_at']}`")
    add(f"- 修订后行数：`{_dump(r['revision_row_counts'])}`")
    add("")
    add("## 步骤 3 — 重复提交被拒（返回既有 id、行数不变）")
    add("")
    d = s["3_duplicate_refused"]
    add(f"- `created` = **{d['created']}**，`reason` = `{d['reason']}`")
    add(f"- 返回既有 `decision_id` = `{d['returned_existing_id']}`"
        f"（与原记录相同：**{d['same_as_original']}**）")
    add(f"- 行数：`{_dump(d['row_counts_before'])}` → `{_dump(d['row_counts_after'])}`"
        f"（不变：**{d['rows_unchanged']}**）")
    add("")
    add("## 步骤 4 — 提前结算被拒（序列 3 日 < 持有期 5 日）")
    add("")
    p = s["4_premature_refused"]
    add(f"- 被拒：**{p['refused']}**，缺口交易日数 = **{p['missing_trading_days']}**")
    add(f"- 错误文本：`{p['message']}`")
    add(f"- 结算行数仍为 {p['settlements_after']}，该决策状态仍为 `{p['status_still']}`"
        "（半窗 alpha 不入账）")
    add("")
    add("## 步骤 5 — 全部结算")
    add("")
    add(f"共 {s['5_settlements']['count']} 条结算；行数 `{_dump(s['5_settlements']['row_counts'])}`。")
    add("")
    add("| decision_id | symbol | 持有交易日 | 入场价 | 出场价 | 原始收益 | 基准 | 基准来源 | 基准收益 | alpha | 结算日 |")
    add("|---|---|---|---|---|---|---|---|---|---|---|")
    for it in s["5_settlements"]["items"]:
        add(f"| `{it['decision_id']}` | {it['symbol']} | {it['holding_days']} |"
            f" {it['entry_price']} | {it['exit_price']} | {it['raw_return']:.6f} |"
            f" {it['benchmark_symbol']} | {it['benchmark_mapping']} |"
            f" {it['benchmark_return']:.6f} | {it['alpha']:.6f} | {it['settlement_date']} |")
    add("")
    add("反思（反思是账本的一等公民，但只在点内检索里带出）：")
    add("")
    for it in s["5_settlements"]["items"]:
        refl = it["reflection"] if it["reflection"] else "（未提供——反思可选，不阻塞结算）"
        add(f"- `{it['decision_id']}`：{refl}")
    add("")
    add("## 步骤 6 — 重复结算幂等")
    add("")
    i = s["6_settlement_idempotent"]
    add(f"- `created` = **{i['created']}**，`reason` = `{i['reason']}`")
    add(f"- alpha 与首次逐位一致：**{i['alpha_unchanged']}**"
        f"；`settlements` 行数仍为 {i['settlements_row_count']}")
    add("")
    add("## 步骤 7 — 点内检索")
    add("")
    pit = s["7_point_in_time"]
    add("三个口径的可见性（只回结算日 ≤ as_of 的记录）：")
    add("")
    add("| 检索 as_of | 同标的 | 跨标的 | 合计 | 可见记录的结算日 |")
    add("|---|---|---|---|---|")
    for cut in pit["cuts"]:
        add(f"| {cut['as_of']} | {cut['counts']['same_symbol']} |"
            f" {cut['counts']['cross_symbol']} | {cut['counts']['total']} |"
            f" {', '.join(cut['settlement_dates']) or '（无）'} |")
    add("")
    fsh = pit["future_settlement_hidden"]
    add(f"**未来结算被过滤**：`{'`, `'.join(fsh['decision_ids'])}` 在 {fsh['settled_after']} 之后才结算，")
    add(f"在 `{'` / `'.join(fsh['absent_from_cutoffs'])}` 两个口径里都不出现（仅在"
        f" `{fsh['present_at_cutoff']}` 口径可见）。")
    add("")
    add(f"{pit['note']}")
    add("")
    f = pit["full_history"]
    add(f"**全历史（as_of={f['as_of']}）**：同标的 {f['counts']['same_symbol']} 条"
        f"（N={f['n_same']}）、跨标的教训 {f['counts']['cross_symbol']} 条（M={f['m_cross']}）。")
    add(f"按结算日倒序：**{f['ordered_desc']}**（同标的结算日：{f['latest_first_same_symbol']}）。")
    add("")
    add("| 组 | decision_id | symbol | 结算日 | 持有交易日 | alpha | 判定 side | 反思 |")
    add("|---|---|---|---|---|---|---|---|")
    for group, items in (("同标的", f["same_symbol"]), ("跨标的", f["cross_symbol"])):
        for x in items:
            refl = (x.get("reflection") or "—").replace("\n", " ")
            add(f"| {group} | `{x['decision_id']}` | {x['symbol']} |"
                f" {x['settlement_date']} | {x['holding_days']} | {x['alpha']:.6f} |"
                f" {x.get('payload_side', '—')} | {refl} |")
    add("")
    add("## 步骤 8 — 聚合统计")
    add("")
    for label, key in (("按 pack", "by_pack"), ("按 pack × skill", "by_pack_and_skill"),
                       ("按 pack × 持有期", "by_pack_and_holding")):
        add(f"**{label}**")
        add("")
        rows = s["8_aggregate"][key]
        keys = sorted({k for row in rows for k in row})
        add("| " + " | ".join(keys) + " |")
        add("|" + "---|" * len(keys))
        for row in rows:
            cells = []
            for k in keys:
                v = row.get(k, "—")
                cells.append(f"{v:.6f}" if isinstance(v, float) else str(v))
            add("| " + " | ".join(cells) + " |")
        add("")
    empty = next(g for g in s["8_aggregate"]["empty_group"]
                 if g["pack"] == "pack-never-used")
    add("**空组明确返回零**（`pack-never-used` 从未落账）：")
    add("")
    add(f"`samples` = **{empty['samples']}**，`mean_alpha` = `{empty['mean_alpha']}`，"
        f"`win_rate` = `{empty['win_rate']}`——行在场，而不是被省略。")
    add("")
    add("（注意：请求 `packs=['pack-never-used']` 时返回的行集是"
        "**已注册 pack ∪ 请求的 pack**——只请求一个空 pack 不会把已有的 pack 藏起来，"
        "这正是「空组出行、非空组不消失」的同一条规则。）")
    add("")
    add("## 步骤 9 — 契约联调（payload 落账 → 查回 → 过契约校验）")
    add("")
    add("| decision_id | 契约校验 |")
    add("|---|---|")
    for chk in s["9_contract_roundtrip"]:
        add(f"| `{chk['decision_id']}` | {'valid' if chk['valid'] else chk['errors']} |")
    add("")
    add("## 复跑")
    add("")
    add("```bash")
    add("cd /Users/chengsishi/finddata/fd-quant")
    add("python3 scripts/ledger_e2e.py           # 跑并写本报告")
    add("python3 scripts/ledger_e2e.py --check   # 只验复跑一致（指纹比对）")
    add("PYTHONPATH=scripts python3 -m pytest tests/ -q -p netblock   # 全套测试在封网下通过")
    add("```")
    add("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="只验复跑一致，不写报告")
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "ledger-e2e.db"
        steps = run(db_path)
        fingerprint = _fingerprint(steps)
        report = render(steps, fingerprint)

    if args.check:
        existing = OUT.read_text(encoding="utf-8") if OUT.exists() else ""
        marker = "结果指纹（sha256）"
        ok = marker in existing and fingerprint in existing
        print(("OK: 复跑指纹与报告一致 " if ok else "MISMATCH: 报告缺失或指纹不一致 ")
              + fingerprint)
        return 0 if ok else 1

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(report, encoding="utf-8")
    print(f"✓ 已写入 {OUT}")
    print(f"  步数 {len(steps)}｜结算 {steps['5_settlements']['count']} 条｜"
          f"指纹 {fingerprint[:16]}…")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
