#!/usr/bin/env python3
"""决策闭环探针（openspec change: close-first-decision-loop, tasks 1.x）。

**它判的不是"实现是否符合规格"，而是"这条链是否真的通过电"。** 既有各道闸门
（contract e2e、settlement e2e、price_caliber_gate、allowTools 实探）检验的都是
前者；2026-11-13 在生产实测到的状态是：全部为绿，而账本 `decisions = 0`、
`settlements = 0`——一条记录都没有。没有任何一条判据会因此变红，这就是本探针存在的理由。

四条判据（缺一不可）：

1. **存在性**——账本里有 `verdict ∈ {hit, miss}` 的结算。`insufficient_data` 不计入：
   它表达"没能定论"，与"闭环跑通"是相反的结论。
2. **新鲜度**——最近 N 天有新落账。超期报「链路停摆」，不是「本期无新决策」——
   这两句对运维是相反的指令（一个要修，一个可以不管）。
3. **可归因**——每条已结算记录能解析出 pack / skill / record_kind。空则无法回答
   "这个结论是哪份方法的产出"，命中率就失去主语。
4. **水位**——账本涉及标的的价格是否足够新。这条**独立于**"多路是否指向同一来源"：
   三路一致地指向同一个 8 周前的源时，口径判据为绿而数据不可用（2026-11-13 实测
   的正是这个形态）。三态输出：`ok` / `violation` / `unavailable`——拿不到语义库时
   报 `unavailable`，**不计为通过**。

只读：只发 SELECT，不建表、不写行、不落缓存、不写报告文件。全部输出走标准输出。
退出码：0 = 零违规；1 = 有违规；2 = 跑不起来（账本不可达 / 参数非法）。

用法：

    python scripts/decision_loop_probe.py
    python scripts/decision_loop_probe.py --json
    python scripts/decision_loop_probe.py --mode drill --db /tmp/drill.db
    python scripts/decision_loop_probe.py --days 30 --max-stale-days 10
    FD_OPEN_DATA_DSN=postgresql://... python scripts/decision_loop_probe.py   # 带水位判据

演练与生产分档（design D2）：`pack` 以 `drill:` 开头的是演练记录。默认 `--mode
production` **排除**它们（演练数据进客户命中率就是污染），`--mode drill` 只看它们。
P0 验收走 drill 档；production 档在第一个真实客户落账前会持续红——那是刻意的，
它表达的是"这条线还没有真实使用"。
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sqlite3
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

#: 默认库路径与账本同源：先 `FD_QUANT_LEDGER_DB`，再工作目录下的单文件。
DEFAULT_DB_FILENAME = "fd_quant_ledger.db"

#: 演练记录的 pack 前缀（design D2）。写进契约常量，不靠调用方自觉。
DRILL_PREFIX = "drill:"

#: 语义库 DSN 的候选环境变量名（与 fd-open-data-mcp / paper-trading 部署一致）。
SEMANTIC_DSN_ENVS = ("FD_OPEN_DATA_DSN", "FD_OPEN_DATA_MCP_DATABASE_URL")

#: 闭环判据认为"有结论"的 verdict 集合。
DECIDED_VERDICTS = ("hit", "miss")


@dataclass
class Violation:
    code: str
    judgment: str
    detail: str
    fix: str
    evidence: dict[str, Any] = field(default_factory=dict)

    def line(self) -> str:
        return (
            f"[{self.judgment}] {self.code}: {self.detail}\n"
            f"    修法: {self.fix}"
        )


@dataclass
class Report:
    mode: str
    db_path: str
    checked: dict[str, Any] = field(default_factory=dict)
    violations: list[Violation] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.violations

    def add(self, v: Violation) -> None:
        self.violations.append(v)

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "mode": self.mode,
            "db_path": self.db_path,
            "violation_count": len(self.violations),
            "violations": [
                {
                    "code": v.code,
                    "judgment": v.judgment,
                    "detail": v.detail,
                    "fix": v.fix,
                    "evidence": v.evidence,
                }
                for v in self.violations
            ],
            "checked": self.checked,
        }


# ── 账本读取（只读、短连接）──────────────────────────────────


def resolve_db_path(explicit: Optional[str]) -> Path:
    if explicit:
        return Path(explicit).expanduser()
    env = (os.environ.get("FD_QUANT_LEDGER_DB") or "").strip()
    if env:
        return Path(env).expanduser()
    return Path.cwd() / DEFAULT_DB_FILENAME


def _connect_ro(db_path: Path) -> sqlite3.Connection:
    """只读连接：`mode=ro` 让"探针不该写"成为驱动层的事实，而不是纪律。"""
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5.0)
    con.row_factory = sqlite3.Row
    return con


def _mode_filter(mode: str) -> tuple[str, tuple[Any, ...]]:
    """把 mode 翻成 SQL 片段。演练/生产互斥，`all` 不过滤。"""
    if mode == "drill":
        return " AND d.pack LIKE ?", (f"{DRILL_PREFIX}%",)
    if mode == "production":
        return " AND d.pack NOT LIKE ?", (f"{DRILL_PREFIX}%",)
    return "", ()


def load_ledger(db_path: Path, mode: str) -> dict:
    """读一次账本，把四条判据需要的原料都取出来（一次快照，避免判据之间自相矛盾）。"""
    con = _connect_ro(db_path)
    try:
        where, params = _mode_filter(mode)
        decided = con.execute(
            "SELECT COUNT(*) AS n FROM decisions d JOIN settlements s"
            " ON s.decision_id = d.decision_id"
            f" WHERE s.verdict IN ('hit','miss'){where}",
            params,
        ).fetchone()["n"]
        insufficient = con.execute(
            "SELECT COUNT(*) AS n FROM decisions d JOIN settlements s"
            " ON s.decision_id = d.decision_id"
            f" WHERE s.verdict = 'insufficient_data'{where}",
            params,
        ).fetchone()["n"]
        totals = con.execute(
            "SELECT COUNT(*) AS decisions FROM decisions d WHERE 1=1" + where, params
        ).fetchone()
        settlements = con.execute(
            "SELECT COUNT(*) AS n FROM decisions d JOIN settlements s"
            f" ON s.decision_id = d.decision_id WHERE 1=1{where}",
            params,
        ).fetchone()["n"]
        last_created = con.execute(
            "SELECT MAX(d.created_at) AS last FROM decisions d WHERE 1=1" + where, params
        ).fetchone()["last"]
        # 可归因：已结算记录里 pack / skill / record_kind 任一为空的行。
        unattributable = con.execute(
            "SELECT d.decision_id, d.pack, d.skill, d.record_kind"
            " FROM decisions d JOIN settlements s ON s.decision_id = d.decision_id"
            f" WHERE (d.pack IS NULL OR TRIM(d.pack) = ''"
            f" OR d.skill IS NULL OR TRIM(d.skill) = ''"
            f" OR d.record_kind IS NULL OR TRIM(d.record_kind) = ''){where}",
            params,
        ).fetchall()
        # `gap_kinds` 是 close-first-decision-loop 新加的列。**探针必须能读旧账本**：
        # 它最该能跑的时刻正是"迁移还没上/刚回滚"的时候，那时这张表没有该列。
        # 缺列就跳过缺口统计并显式标注——不能因此整体退出（那会把"读不了"伪装成"跑不起来"）。
        scols = {r[1] for r in con.execute("PRAGMA table_info(settlements)")}
        has_gap_column = "gap_kinds" in scols
        gap_rows = (
            con.execute(
                "SELECT s.gap_kinds AS gap_kinds, COUNT(*) AS n FROM decisions d"
                " JOIN settlements s ON s.decision_id = d.decision_id"
                " WHERE s.verdict = 'insufficient_data'" + where
                + " GROUP BY 1 ORDER BY 2 DESC",
                params,
            ).fetchall()
            if has_gap_column
            else []
        )
        subjects = con.execute(
            "SELECT DISTINCT d.symbol AS symbol FROM decisions d"
            " WHERE d.symbol IS NOT NULL AND TRIM(d.symbol) <> ''" + where,
            params,
        ).fetchall()
        return {
            "decided": int(decided),
            "insufficient": int(insufficient),
            "decisions": int(totals["decisions"]),
            "settlements": int(settlements),
            "last_created_at": last_created,
            "unattributable": [dict(r) for r in unattributable],
            "gap_counts": {r["gap_kinds"]: int(r["n"]) for r in gap_rows},
            "gap_column_present": has_gap_column,
            "subjects": [r["symbol"] for r in subjects],
        }
    finally:
        con.close()


# ── 判据 1-3：账本自身 ───────────────────────────────────────


def judge_existence(rep: Report, snap: dict) -> None:
    if snap["decided"] > 0:
        rep.checked["existence"] = {"ok": True, "decided": snap["decided"]}
        return
    rep.checked["existence"] = {
        "ok": False,
        "decided": 0,
        "insufficient": snap["insufficient"],
        "decisions": snap["decisions"],
        "settlements": snap["settlements"],
    }
    detail = (
        f"账本 {snap['decisions']} 条决策 / {snap['settlements']} 条结算行，"
        f"其中 verdict∈{{hit,miss}} 为 0"
    )
    if snap["insufficient"]:
        detail += f"（{snap['insufficient']} 条为 insufficient_data，不算闭合）"
    rep.add(
        Violation(
            code="LOOP_NEVER_CLOSED",
            judgment="存在性",
            detail=detail,
            fix=(
                "先用不依赖基准序列的口径结一笔：rule_id='metric_threshold'"
                "（见 scripts/close_first_decision_drill.py 的演练形态），"
                "再查该方法为何在生产没被调用"
            ),
            evidence={"insufficient": snap["insufficient"]},
        )
    )


def judge_freshness(rep: Report, snap: dict, *, now: dt.datetime, days: int) -> None:
    last = snap["last_created_at"]
    if not last:
        rep.checked["freshness"] = {"ok": False, "last_created_at": None, "days": days}
        rep.add(
            Violation(
                code="NO_DECISIONS_AT_ALL",
                judgment="新鲜度",
                detail=f"该档（mode={rep.mode}）下没有任何落账记录，无法判断链路是否在跑",
                fix="确认落账入口（record_decision）是否被任何岗位/技能调用",
            )
        )
        return
    try:
        last_dt = dt.datetime.fromisoformat(str(last).replace("Z", "+00:00"))
    except ValueError:
        rep.checked["freshness"] = {"ok": False, "last_created_at": last, "days": days}
        rep.add(
            Violation(
                code="UNPARSABLE_TIMESTAMP",
                judgment="新鲜度",
                detail=f"最近落账时间 {last!r} 不是可解析的 ISO 时间戳",
                fix="检查 created_at 的写入路径（应为 ISO-8601）",
            )
        )
        return
    if last_dt.tzinfo is None:
        last_dt = last_dt.replace(tzinfo=dt.timezone.utc)
    age_days = (now - last_dt).total_seconds() / 86400.0
    rep.checked["freshness"] = {
        "ok": age_days <= days,
        "last_created_at": last,
        "age_days": round(age_days, 1),
        "window_days": days,
    }
    if age_days > days:
        rep.add(
            Violation(
                code="LOOP_STALLED",
                judgment="新鲜度",
                detail=(
                    f"链路停摆：最近一次落账 {last} 距今 {age_days:.1f} 天"
                    f"（窗口 {days} 天）"
                ),
                fix="查落账入口的调度与上游取数；停摆与「本期无新决策」是两件事",
                evidence={"last_created_at": last, "age_days": round(age_days, 1)},
            )
        )


def judge_attribution(rep: Report, snap: dict) -> None:
    rows = snap["unattributable"]
    rep.checked["attribution"] = {"ok": not rows, "unattributable": len(rows)}
    if not rows:
        return
    sample = ", ".join(r["decision_id"] for r in rows[:5])
    rep.add(
        Violation(
            code="SETTLEMENT_UNATTRIBUTABLE",
            judgment="可归因",
            detail=f"{len(rows)} 条已结算记录缺 pack / skill / record_kind：{sample}",
            fix="落账时补齐 pack 与 skill（它们是命中率的分母主语），并回填既有行",
            evidence={"sample": rows[:5]},
        )
    )


# ── 判据 4：价格水位（三态）───────────────────────────────────


def normalize_symbol(symbol: str) -> str:
    """`600519.SS` / `000333.SZ` / `600519` → 语义库里的 6 位实体 code。

    语义库按裸码建实体，账本按带后缀的标的建；不归一化会让判据永远查不到人，
    而"查不到"会被读成"没有这个问题"。
    """
    s = (symbol or "").strip().upper()
    for suffix in (".SH", ".SS", ".SZ", ".BJ", ".HK"):
        if s.endswith(suffix):
            return s[: -len(suffix)]
    return s.split(".", 1)[0]


def _semantic_dsn() -> Optional[str]:
    for name in SEMANTIC_DSN_ENVS:
        v = (os.environ.get(name) or "").strip()
        if v:
            return v.replace("+psycopg2", "").replace("+asyncpg", "")
    return None


def read_water_level(symbols: Sequence[str]) -> dict:
    """按标的读价格最新日期与最近抓取时间。

    返回 `{"state": "ok"|"unavailable", "symbols": {...}, "reason": str}`。
    拿不到语义库（无 DSN / 无驱动 / 连不上）一律 `unavailable`——**不当作通过**。
    """
    dsn = _semantic_dsn()
    if not dsn:
        return {
            "state": "unavailable",
            "reason": f"未配置语义库 DSN（{ ' / '.join(SEMANTIC_DSN_ENVS) }）",
            "symbols": {},
        }
    try:
        import psycopg2  # noqa: PLC0415 — 可选依赖，缺了就降级为 unavailable
    except Exception:  # noqa: BLE001
        return {"state": "unavailable", "reason": "本环境未安装 psycopg2", "symbols": {}}

    codes = sorted({normalize_symbol(s) for s in symbols if normalize_symbol(s)})
    if not codes:
        return {"state": "ok", "reason": "账本无标的，无需判定水位", "symbols": {}}
    try:
        con = psycopg2.connect(dsn, connect_timeout=8)
    except Exception as exc:  # noqa: BLE001
        return {"state": "unavailable", "reason": f"语义库不可达：{exc}", "symbols": {}}
    try:
        cur = con.cursor()
        cur.execute(
            "SELECT e.code, MAX(o.date) FROM entities e"
            " LEFT JOIN semantic_observations o"
            "   ON o.entity_id = e.id AND o.entity_type = e.entity_type"
            " JOIN concepts c ON c.id = o.concept_id AND c.code = 'price.close'"
            " WHERE e.entity_type = 'stock' AND e.code = ANY(%s)"
            " GROUP BY e.code",
            (codes,),
        )
        as_of = {r[0]: r[1] for r in cur.fetchall()}
        cur.execute(
            "SELECT e.code, MAX(f.timestamp) FROM entities e"
            " JOIN concepts c ON c.code = 'price.close' AND c.entity_type = 'stock'"
            " JOIN fetch_log f ON f.concept_id = c.id AND f.entity_id = e.id"
            " WHERE e.entity_type = 'stock' AND e.code = ANY(%s)"
            " GROUP BY e.code",
            (codes,),
        )
        fetched = {r[0]: r[1] for r in cur.fetchall()}
    finally:
        con.close()
    out = {}
    for code in codes:
        out[code] = {
            "price_as_of_max": as_of.get(code),
            "last_fetch_at": (
                fetched[code].isoformat() if fetched.get(code) is not None else None
            ),
        }
    return {"state": "ok", "reason": "", "symbols": out}


def judge_water(rep: Report, snap: dict, *, now: dt.date, max_stale_days: int) -> None:
    water = read_water_level(snap["subjects"])
    rep.checked["water_level"] = {
        "state": water["state"],
        "max_stale_days": max_stale_days,
        "subjects": len(snap["subjects"]),
    }
    if water["state"] == "unavailable":
        # 不计入违规（探针要能在只有账本的机器上跑），但必须出现在输出里：
        # 静默当成"通过"就是把"没查"写成"没问题"。
        rep.checked["water_level"]["reason"] = water["reason"]
        rep.checked["water_level"]["judged"] = False
        return
    rep.checked["water_level"]["judged"] = True
    if not water["symbols"]:
        rep.checked["water_level"]["ok"] = True
        return

    stale, never, no_fetch = [], [], []
    for code, info in water["symbols"].items():
        raw_date = info.get("price_as_of_max")
        if not raw_date:
            never.append(code)
        else:
            try:
                d = dt.date.fromisoformat(str(raw_date)[:10])
            except ValueError:
                never.append(code)
                continue
            age = (now - d).days
            if age > max_stale_days:
                stale.append((code, str(raw_date), age))
        if not info.get("last_fetch_at"):
            no_fetch.append(code)
    rep.checked["water_level"].update({
        "ok": not (stale or never or no_fetch),
        "stale": stale,
        "no_observation": never,
        "never_fetched": no_fetch,
    })
    if stale:
        code, d, age = stale[0]
        rep.add(
            Violation(
                code="PRICE_WATERMARK_STALE",
                judgment="水位",
                detail=(
                    f"{len(stale)} 个标的价格水位落后超过 {max_stale_days} 天；"
                    f"例 {code} 最新 {d}（落后 {age} 天）"
                ),
                fix=(
                    "口径一致不等于数据可用：检查该概念族的日更派发"
                    "（fetch_log 有无尝试），不要靠同一份旧序列继续结算"
                ),
                evidence={"stale": stale[:5]},
            )
        )
    if never:
        rep.add(
            Violation(
                code="PRICE_NEVER_OBSERVED",
                judgment="水位",
                detail=f"{len(never)} 个标的在语义库里没有任何 price.close 观测：{', '.join(never[:5])}",
                fix="确认这些标的的实体与价格概念绑定是否已登记（实体在册 ≠ 有观测）",
                evidence={"symbols": never[:5]},
            )
        )
    if no_fetch:
        rep.add(
            Violation(
                code="SUPPLY_NOT_PRODUCING",
                judgment="水位",
                detail=(
                    f"{len(no_fetch)} 个标的的价格概念从未被抓取：{', '.join(no_fetch[:5])}"
                ),
                fix=(
                    "供给未产出，不是水位过旧：查该概念族是否在日更派发范围内"
                    "（2026-11-13 实测：A 股价格 60 天仅 7 次抓取，基金 2 天 12991 次）"
                ),
                evidence={"symbols": no_fetch[:5]},
            )
        )


# ── 供给归因（design D5）────────────────────────────────────


#: 归因判据的三行表：尝试次数 × 产出观测数 → 结论。顺序即优先级。
ATTRIBUTION_RULES = (
    ("not_dispatched", "窗口内没有任何抓取尝试 → 该概念族不在派发范围内（不是故障，是范围）"),
    ("fetch_failing", "有抓取尝试但零产出 → 抓取失败（再看 classification 分 transient/ban）"),
    ("write_path_broken", "有尝试也有产出但水位停旧 → 写路径或映射断（新数据落到别的口径）"),
    ("producing", "有尝试且有新产出 → 供给正常"),
)


def attribute_supply(*, days: int = 60, dsn: Optional[str] = None) -> dict:
    """按「抓取次数 vs 产出观测数」给价格供给定性（只读）。

    **为什么不用 `schedules` 表判"有没有被调度"**：那张表在生产里**全表 0 行**
    （2026-11-13 实测），任何"某概念没有 schedule 行"的推论对每个概念都成立，
    等于什么都没说。空表不能证明不存在——这是本项目反复踩过的同一个坑。
    `fetch_log` 才是"有没有真的去抓"的事实源。
    """
    dsn = dsn or _semantic_dsn()
    if not dsn:
        return {
            "state": "unavailable",
            "reason": f"未配置语义库 DSN（{ ' / '.join(SEMANTIC_DSN_ENVS) }）",
            "families": [],
        }
    try:
        import psycopg2  # noqa: PLC0415
    except Exception:  # noqa: BLE001
        return {"state": "unavailable", "reason": "本环境未安装 psycopg2", "families": []}
    try:
        con = psycopg2.connect(dsn, connect_timeout=8)
    except Exception as exc:  # noqa: BLE001
        return {"state": "unavailable", "reason": f"语义库不可达：{exc}", "families": []}

    out: list[dict] = []
    try:
        cur = con.cursor()
        for entity_type in ("stock", "fund", "index"):
            cur.execute(
                "SELECT COUNT(*) FROM fetch_log f"
                " JOIN concepts c ON c.id = f.concept_id"
                " WHERE c.code = 'price.close' AND c.entity_type = %s"
                "   AND f.timestamp > now() - (%s || ' days')::interval",
                (entity_type, str(days)),
            )
            attempts = int(cur.fetchone()[0])
            cur.execute(
                "SELECT COUNT(*), MAX(o.date) FROM semantic_observations o"
                " JOIN concepts c ON c.id = o.concept_id"
                " JOIN entities e ON e.id = o.entity_id AND e.entity_type = o.entity_type"
                " WHERE c.code = 'price.close' AND c.entity_type = %s"
                "   AND e.entity_type = %s",
                (entity_type, entity_type),
            )
            produced, latest = cur.fetchone()
            produced = int(produced)
            if attempts == 0:
                kind = "not_dispatched"
            elif produced == 0:
                kind = "fetch_failing"
            else:
                kind = "producing"
            out.append({
                "entity_type": entity_type,
                "attempts": attempts,
                "observations": produced,
                "latest_observation": str(latest) if latest else None,
                "attribution": kind,
                "reading": dict(ATTRIBUTION_RULES)[kind],
            })
    finally:
        con.close()
    return {"state": "ok", "reason": "", "days": days, "families": out}


def print_attribution(result: dict) -> None:
    if result["state"] == "unavailable":
        print(f"  供给归因未判定（{result['reason']}）")
        return
    print(f"  供给归因（近 {result['days']} 天，price.close）：")
    for f in result["families"]:
        print(
            f"    {f['entity_type']:6} 尝试 {f['attempts']:>6} 次 / 观测 {f['observations']:>7} 行"
            f" / 最新 {f['latest_observation'] or '—'} → {f['attribution']}"
        )
        print(f"           {f['reading']}")


# ── 入口 ─────────────────────────────────────────────────────


def run(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="决策闭环探针（只读）")
    ap.add_argument("--db", help="账本 SQLite 路径（默认 FD_QUANT_LEDGER_DB 或工作目录单文件）")
    ap.add_argument("--mode", choices=("production", "drill", "all"), default="production",
                    help="production 排除 drill: 前缀的演练记录（默认）；drill 只看演练")
    ap.add_argument("--days", type=int, default=14, help="新鲜度窗口（自然日，默认 14）")
    ap.add_argument("--max-stale-days", type=int, default=7, help="价格水位容忍天数（默认 7）")
    ap.add_argument("--as-of", help="判定用的当天日期（默认今天，便于可复现测试）")
    ap.add_argument("--json", action="store_true", help="机读输出")
    ap.add_argument("--attribution", action="store_true",
                    help="额外输出供给归因（按抓取次数 vs 产出观测数定性；需语义库）")
    ap.add_argument("--attribution-days", type=int, default=60,
                    help="供给归因窗口（天，默认 60）")
    args = ap.parse_args(argv)

    db_path = resolve_db_path(args.db)
    if not db_path.exists():
        msg = f"账本不存在：{db_path}（先跑一次落账，或用 --db 指定库）"
        if args.json:
            print(json.dumps({"ok": False, "fatal": msg}, ensure_ascii=False))
        else:
            print(f"[fatal] {msg}")
        return 2
    if args.days <= 0 or args.max_stale_days < 0:
        print("[fatal] --days 必须 > 0，--max-stale-days 必须 ≥ 0")
        return 2

    now_date = dt.date.fromisoformat(args.as_of) if args.as_of else dt.date.today()
    now_dt = dt.datetime.combine(now_date, dt.time(0, 0), tzinfo=dt.timezone.utc)

    rep = Report(mode=args.mode, db_path=str(db_path))
    try:
        snap = load_ledger(db_path, args.mode)
    except sqlite3.Error as exc:
        msg = f"账本不可读：{exc}"
        if args.json:
            print(json.dumps({"ok": False, "fatal": msg}, ensure_ascii=False))
        else:
            print(f"[fatal] {msg}")
        return 2
    rep.checked["ledger_schema"] = {
        "gap_column_present": bool(snap.get("gap_column_present", True))
    }

    judge_existence(rep, snap)
    judge_freshness(rep, snap, now=now_dt, days=args.days)
    judge_attribution(rep, snap)
    judge_water(rep, snap, now=now_date, max_stale_days=args.max_stale_days)

    attribution = (
        attribute_supply(days=args.attribution_days) if args.attribution else None
    )

    if args.json:
        payload = rep.to_dict()
        if attribution is not None:
            payload["supply_attribution"] = attribution
        print(json.dumps(payload, ensure_ascii=False, indent=1))
    else:
        print(f"决策闭环探针 · mode={args.mode} · {db_path}")
        wl = rep.checked.get("water_level", {})
        if wl.get("judged") is False:
            print(f"  [水位] 未判定（{wl.get('reason')}）")
        if snap["gap_counts"]:
            gaps = ", ".join(f"{k}={v}" for k, v in snap["gap_counts"].items())
            print(f"  未定论缺口：{gaps}")
        if not snap.get("gap_column_present", True):
            print("  [说明] 该账本还没有 gap_kinds 列（早于本 change），缺口统计已跳过")
        if attribution is not None:
            print_attribution(attribution)
        if rep.ok:
            # 判据全绿也不许夸大：水位未判定时明说，否则"没查"会被读成"查过没问题"。
            tail = (
                "（水位未判定，见上）"
                if wl.get("judged") is False
                else "（四项均通过，含水位可判定）"
            )
            print(f"  ✓ 闭环成立：账本已有可结算的结论 {tail}")
        else:
            for v in rep.violations:
                print("  ✗ " + v.line())
    return 0 if rep.ok else 1


if __name__ == "__main__":
    sys.exit(run())
