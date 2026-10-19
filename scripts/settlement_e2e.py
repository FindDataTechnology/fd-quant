#!/usr/bin/env python3
"""结算取数端到端演练（finance-legal-value-realization 组 3，tasks 3.3–3.5）。

与 `ledger_e2e.py` 的分工：那条演练**全离线**（序列由夹具确定性构造），证明的是
"账本不取数"这条纪律；这条演练反过来——**序列来自真实通道 B**（本地 DAAS 库），
证明的是"结算取数这条链路真的能走通"，以及"结算结果可被独立复核"。

链路（三步，每步都落在报告里）：
    造一条到期决策 → 从 DAAS 读标的与基准的结算窗序列 → 调 settle_decision
    → 用结算记录里的价格与日期重算 alpha，逐位比对

**为什么必须真取数一次**：技能 `decision-review-settlement` 声明"结算窗数据只能来自
通道 B"，但这句话此前没有被执行过一次。声明未被执行的取数路径，与不存在的取数
路径在调用方看来是一样的——组 3 要销的就是这个"纸面可达"。

用法：
    python3 scripts/settlement_e2e.py [--daas-db PATH] [--as-of DATE] [--holding-days N]
退出码：0 全部通过 / 1 有断言失败。
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from fd_quant.contracts.models import Producer, Rating  # noqa: E402
from fd_quant.ledger import LedgerStore  # noqa: E402

DEFAULT_DAAS = "/Users/chengsishi/code/DAAS/daas.db"

#: 标的与基准的 DAAS 源表（结算窗的两条序列来源）。
SYMBOL_TABLE = "scraw_600519_daily"
BENCHMARK_TABLE = "scraw_000001_daily"

CALLER = "e2e-group3"


def read_series(db_path: str, table: str, start: str, n: int) -> tuple[list[str], list[float]]:
    """从 DAAS 读一条 (日期, 收盘价) 序列。

    **只读 `date` 与 `Close` 两列**：账本要求 (ISO 日期, 价格) 对，多读一列都是
    多余的耦合。缺失价格不补——DAAS 里若真有 null，就该由账本点名拒绝，而不是
    在这里插值（插值出来的 alpha 无法复核，也就没有价值）。
    """
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    try:
        rows = list(con.execute(
            f"SELECT date, Close FROM {table} WHERE date >= ? ORDER BY date LIMIT ?",
            (start, n),
        ))
    finally:
        con.close()
    return ([r["date"] for r in rows], [float(r["Close"]) for r in rows])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="结算取数端到端演练（真实通道 B）")
    ap.add_argument("--daas-db", default=DEFAULT_DAAS, help="DAAS 库路径")
    ap.add_argument("--as-of", default="2026-10-09", help="决策 as_of（入场日）")
    ap.add_argument("--holding-days", type=int, default=5, help="持有期（交易日）")
    ap.add_argument("--json", action="store_true", help="输出 JSON 而非人读文本")
    args = ap.parse_args(argv)

    if not Path(args.daas_db).exists():
        print(f"DAAS 库不存在: {args.daas_db}", file=sys.stderr)
        return 1

    # 结算窗：入场日起至少 holding_days + 1 个对齐点（多取几个，让对齐规则真正生效）。
    n = args.holding_days + 6
    sym_dates, sym_prices = read_series(args.daas_db, SYMBOL_TABLE, args.as_of, n)
    bench_dates, bench_prices = read_series(args.daas_db, BENCHMARK_TABLE, args.as_of, n)
    if len(sym_dates) < args.holding_days + 1:
        print(f"标的序列不足：{len(sym_dates)} 点，需要 {args.holding_days + 1}", file=sys.stderr)
        return 1
    if len(bench_dates) < args.holding_days + 1:
        print(f"基准序列不足：{len(bench_dates)} 点，需要 {args.holding_days + 1}", file=sys.stderr)
        return 1

    store = LedgerStore(Path(tempfile.mkdtemp()) / "settlement-e2e.db")
    rating = Rating(
        as_of=args.as_of, subject="600519.SS", side="buy", conviction=0.6,
        price_target=None, target_evidence_ids=[],
        basis="组 3 端到端演练：真实通道 B 序列结算。", evidence=[], time_horizon="6-12 个月",
        producer=Producer(pack="wire-decision-review", skill="decision-review-settlement"),
    )
    rec = store.record(
        caller=CALLER, as_of=args.as_of, symbol="600519.SS",
        pack="wire-decision-review", skill="decision-review-settlement", payload=rating,
    )
    assert rec["created"] is True and rec["status"] == "pending", rec

    res = store.settle(
        decision_id=rec["decision_id"], symbol_dates=sym_dates, symbol_prices=sym_prices,
        benchmark_dates=bench_dates, benchmark_prices=bench_prices, as_of=sym_dates[-1],
        holding_days=args.holding_days,
    )
    assert res["status"] == "settled", res

    # ── 复核：只用结算记录里的价格与日期，重算一遍 ──────────────────
    # 这是 task 3.5 的全部内容：结算结果必须能被一个没看过账本代码的人算回来。
    entry_date, exit_date = args.as_of, res["settlement_date"]
    b_entry = bench_prices[bench_dates.index(entry_date)]
    b_exit = bench_prices[bench_dates.index(exit_date)]
    raw = res["exit_price"] / res["entry_price"] - 1.0
    bench_return = b_exit / b_entry - 1.0
    recomputed_alpha = raw - bench_return
    alpha_matches = abs(recomputed_alpha - res["alpha"]) < 1e-15

    # ── 3.4 无取数时状态诚实：到期但无序列 → 仍 pending ──────────────
    no_data = store.record(
        caller=CALLER, as_of=args.as_of, symbol="000333.SZ",
        pack="wire-decision-review", skill="decision-review-settlement",
        payload=Rating(
            as_of=args.as_of, subject="000333.SZ", side="buy", conviction=0.5,
            price_target=None, target_evidence_ids=[], basis="无序列对照。", evidence=[],
            time_horizon="6-12 个月",
            producer=Producer(pack="wire-decision-review", skill="decision-review-settlement"),
        ),
    )
    pending_holds = no_data["status"] == "pending"

    out = {
        "daas_db": args.daas_db,
        "symbol_table": SYMBOL_TABLE,
        "benchmark_table": BENCHMARK_TABLE,
        "as_of": args.as_of,
        "holding_days": args.holding_days,
        "settlement": {
            "decision_id": rec["decision_id"],
            "entry_date": entry_date,
            "settlement_date": exit_date,
            "entry_price": res["entry_price"],
            "exit_price": res["exit_price"],
            "raw_return": res["raw_return"],
            "benchmark_symbol": res["benchmark_symbol"],
            "benchmark_mapping": res["benchmark_mapping"],
            "benchmark_entry_price": b_entry,
            "benchmark_exit_price": b_exit,
            "benchmark_return": res["benchmark_return"],
            "alpha": res["alpha"],
        },
        "recheck": {
            "raw_return": raw,
            "benchmark_return": bench_return,
            "recomputed_alpha": recomputed_alpha,
            "alpha_matches": alpha_matches,
        },
        "no_data_stays_pending": pending_holds,
        "counts": store.row_counts(),
    }

    if args.json:
        print(json.dumps(out, ensure_ascii=False, indent=2, default=str))
    else:
        s = out["settlement"]
        print("结算取数端到端（真实通道 B）")
        print(f"  标的序列  : {SYMBOL_TABLE}  {len(sym_dates)} 点  {sym_dates[0]}..{sym_dates[-1]}")
        print(f"  基准序列  : {BENCHMARK_TABLE}  {len(bench_dates)} 点  "
              f"{bench_dates[0]}..{bench_dates[-1]}")
        print(f"  决策      : {s['decision_id']}  as_of={entry_date}")
        print(f"  结算      : {s['settlement_date']}  holding_days={args.holding_days}")
        print(f"  入场/出场 : {s['entry_price']} → {s['exit_price']}")
        print(f"  raw_return: {s['raw_return']:.12f}")
        print(f"  基准      : {s['benchmark_symbol']} ({s['benchmark_mapping']})  "
              f"{b_entry} → {b_exit}  bench_return={s['benchmark_return']:.12f}")
        print(f"  alpha     : {s['alpha']:.12f}")
        print(f"  复核 alpha: {recomputed_alpha:.12f}  逐位一致={alpha_matches}")
        print(f"  无序列对照: status={no_data['status']}（应为 pending）={pending_holds}")
        print(f"  counts    : {out['counts']}")

    ok = alpha_matches and pending_holds and res["status"] == "settled"
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
