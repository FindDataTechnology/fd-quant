"""决策账本存储层：SQLite 单文件、追加式两表、幂等键唯一约束。

**为什么是 stdlib `sqlite3` + 单文件**：见 design.md D1（蓝本 TradingAgents 的
ChromaDB→BM25→全删演进史证明账本的查询是关系型过滤，不是相似度检索）。零新增依赖：
本模块只用标准库 `sqlite3`，不引入 SQLAlchemy/aiosqlite。

表与键
------
`decisions`：`decision_id` 主键 + `caller / as_of / symbol / pack / skill / payload /
target_weights / evidence / status / revises / revision_seq / created_at`。
**幂等键**是 `(caller, as_of, symbol, pack, skill)`——**含调用者**。

`caller` 是调用者标识（账本按它隔离，见 `fd_quant.caller`）。它由调用上下文解析，
**不接受入参声明**：能由入参声明的"隔离"不是隔离，只是请调用方自觉遵守。隔离前
的旧行在迁移中一律标为 `legacy`（归属未知），且不被任何具名调用者看到——归属未知
的记录若对所有人可见，就是把无主样本灌进每个调用者的胜率与平均 alpha。

`settlements`：`decision_id` 主键（一决策最多一条结算）+ 结算窗的价格、原始收益、
基准（符号 + 映射来源 + 基准收益）、alpha、持有天数、反思。

**"唯一键"与"修订产生新行"的张力怎么解**（本层唯一的非平凡设计决定）：spec 同时要求
"幂等键唯一 → 重复提交被拒"与"修订产生**新记录**且 `revises` 指向被修订记录"。
若把唯一性直接做成 `(as_of, symbol, pack, skill)` 的全表唯一约束，任何修订都必然撞键。
解法是**两条部分唯一索引**：

- `ux_decisions_initial`：`(caller, as_of, symbol, pack, skill) WHERE revises IS NULL`
  ——初始记录在同一调用者内唯一。重复提交由**数据库**拒绝，而不是靠应用层"先查后插"。
- `ux_decisions_revision`：`(caller, as_of, symbol, pack, skill, revision_seq)
  WHERE revises IS NOT NULL`
  ——修订按同键的递增序号唯一，链是线性的（v1→v2→v3），且并发安全。
- `CHECK` 约束把两者绑死：`revises IS NULL ⇔ revision_seq = 0`（不可能出现
  "无指针的 v2"或"指针存在却是 v0"的半态），且禁止自指。

于是"同键两条初始记录"在库层不可能出现，而修订是新行、原行逐字节不动（追加语义的
可证伪形式，由 tests 断言 `payload` 与 `created_at` 不变）。

原子性与中断模型
----------------
写路径一律显式事务并用 `BEGIN IMMEDIATE`：它立刻取写锁，使"先查既有键 → 再插入"成为
**原子**判断。若用默认的 deferred 事务，两个并发写者都会读到"键不存在"，第二个在
COMMIT 时才以 `SQLITE_BUSY`/`IntegrityError` 收场——冲突被拖到提交期才暴露。写锁前置
+ 部分唯一索引 = 应用层与数据库层两道屏障。

**"进程中途中断不产生半行"的模型是"未提交的事务"**：SQLite 的原子性单位是事务，写在
COMMIT 之前对任何读者都不可见；进程在 COMMIT 前死掉（`os._exit`），WAL 里未提交的帧在
下次打开时被丢弃，表里不存在该行。这就是"半行不可能出现"的机械保证，也是我们选事务
而不是"临时文件 + 替换"的原因：SQLite 已经提供了它，替换法在 SQLite 上只是把这份保证
自己重写一遍，且并发下更容易写错。`tests/test_ledger_store.py` 用子进程在 `record()`
的**真实路径**上于 INSERT 之后、COMMIT 之前 `os._exit(1)` 把这条钉住（不是读代码相信它）。

无推送
------
本模块（与整个 `fd_quant.ledger`）没有任何 push / notify / broadcast / subscribe 类接口：
反思只存、只能被检索带出；注入决策是消费侧（P5/P4）的行为（design.md D6）。
`tests/test_ledger_retrieval.py` 对公开 API 面做结构性断言钉住这一点。
"""
from __future__ import annotations

import json
import math
import os
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import date as _date
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Optional, Sequence

from ..contracts.models import ContractError, validate
from ..caller import LEGACY
from .errors import LedgerError, RevisionConflict, SettlementRefused, UnknownDecision
from .settlement import (
    DEFAULT_BENCHMARK_MAP,
    DEFAULT_HOLDING_DAYS,
    build_series,
    check_holding_days,
    check_iso_date,
    portfolio_settlement_window,
    resolve_benchmark,
    settlement_window,
)

#: 默认库文件名（零配置时落当前工作目录；`FD_QUANT_LEDGER_DB` 可覆盖）。
DEFAULT_DB_FILENAME = "fd_quant_ledger.db"

#: 写锁等待上限（毫秒）：并发写者排队，而不是立刻 SQLITE_BUSY 失败。
BUSY_TIMEOUT_MS = 5000

#: DDL 逐条列出（不拼成一个字符串再按 ";" 切——切分是不可靠的 SQL 解析）。
SCHEMA_STATEMENTS: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS decisions (
        decision_id     TEXT PRIMARY KEY,
        -- caller：调用者标识（账本按此隔离）。由调用上下文解析，不接受入参声明。
        caller          TEXT NOT NULL DEFAULT 'anonymous',
        -- record_kind：单体判定 or 组合一期（add-portfolio-intent-contract）。
        -- 两类记录共存于一张表，但主体字段互斥，且聚合时不得混算。
        record_kind     TEXT NOT NULL DEFAULT 'single'
                        CHECK (record_kind IN ('single', 'portfolio')),
        as_of           TEXT NOT NULL,
        -- 单体记录的标；组合记录为 NULL（主体改由 portfolio_key 承担）。
        symbol          TEXT,
        -- 组合记录的稳定键；单体记录为 NULL。
        portfolio_key   TEXT,
        pack            TEXT NOT NULL,
        skill           TEXT NOT NULL,
        payload         TEXT NOT NULL,
        target_weights  TEXT,
        -- 取价路径的序列引用（add-price-supply-and-caliber 4.5）：落账时由调用方传入，
        -- 结算时与本次传入的 ref 比对——口径不一致在 warn 下标记不可比、refuse 下拒绝。
        series_ref      TEXT,
        evidence        TEXT NOT NULL,
        status          TEXT NOT NULL CHECK (status IN ('pending', 'settled')),
        revises         TEXT REFERENCES decisions(decision_id),
        revision_seq    INTEGER NOT NULL CHECK (revision_seq >= 0),
        created_at      TEXT NOT NULL,
        -- v2 决策（generalize-decision-contract）：NULL 表示 v1 记录（Rating/TargetWeights）。
        -- 刻意不加 NOT NULL：v1 记录没有主体种类与动作，portfolio 记录也没有单一主体；
        -- 强行回填会把一个错误的语义固化进历史数据。
        subject_kind    TEXT,
        subject_key     TEXT,
        action          TEXT,
        preconditions   TEXT,   -- JSON array
        settlement_spec TEXT,   -- JSON object
        CHECK (
            (revises IS NULL AND revision_seq = 0)
            OR (revises IS NOT NULL AND revision_seq > 0)
        ),
        CHECK (revises IS NULL OR revises <> decision_id),
        -- 主体字段与记录种类必须一致：同时给出或同时缺失都是错误，不是"取其一"。
        CHECK (
            (record_kind = 'single' AND symbol IS NOT NULL AND portfolio_key IS NULL)
            OR
            (record_kind = 'portfolio' AND portfolio_key IS NOT NULL AND symbol IS NULL)
        ),
        -- v2 记录必须给齐 subject_kind；v1 记录三列全空。不许"半套"。
        CHECK (
            (subject_kind IS NULL AND subject_key IS NULL AND action IS NULL)
            OR (subject_kind IS NOT NULL AND subject_key IS NOT NULL AND action IS NOT NULL)
        )
    )
    """,
    # 初始记录：幂等键全表唯一（重复提交的最终屏障）。
    # subject_key 由 COALESCE(symbol, portfolio_key) 表达——不为它另立一列，
    # 因为那是把同一个事实写两遍，而两份迟早分叉。
    """
    CREATE UNIQUE INDEX IF NOT EXISTS ux_decisions_initial
        ON decisions (caller, record_kind, as_of, COALESCE(symbol, portfolio_key), pack, skill)
        WHERE revises IS NULL
    """,
    # 修订：同键按递增序号唯一，线性链不可能出现兄弟分叉
    """
    CREATE UNIQUE INDEX IF NOT EXISTS ux_decisions_revision
        ON decisions (caller, record_kind, as_of, COALESCE(symbol, portfolio_key), pack, skill,
                      revision_seq)
        WHERE revises IS NOT NULL
    """,
    "CREATE INDEX IF NOT EXISTS ix_decisions_symbol_as_of ON decisions (caller, symbol, as_of)",
    "CREATE INDEX IF NOT EXISTS ix_decisions_pack_skill ON decisions (caller, pack, skill)",
    "CREATE INDEX IF NOT EXISTS ix_decisions_kind ON decisions (caller, record_kind, as_of)",
    """
    CREATE TABLE IF NOT EXISTS settlements (
        decision_id           TEXT PRIMARY KEY REFERENCES decisions(decision_id),
        settled_at            TEXT NOT NULL,
        settlement_date       TEXT NOT NULL,
        -- rule_id / verdict（generalize-decision-contract）：一个口径位 + 一个三态结论。
        -- 历史行按 rule_id='alpha_vs_benchmark' 回填，verdict 由 alpha>0 推导——那次推导是
        -- **新增判断**，不是"恢复原值"；迁移与复盘输出都必须注明这一点。
        rule_id               TEXT NOT NULL DEFAULT 'alpha_vs_benchmark'
                              CHECK (rule_id IN ('alpha_vs_benchmark', 'metric_threshold')),
        verdict               TEXT NOT NULL DEFAULT 'miss'
                              CHECK (verdict IN ('hit', 'miss', 'insufficient_data')),
        -- 价格口径（rule_id='alpha_vs_benchmark' 时非空）
        entry_price           REAL,
        exit_price            REAL,
        raw_return            REAL,
        benchmark_symbol      TEXT,
        benchmark_mapping     TEXT,
        benchmark_entry_price REAL,
        benchmark_exit_price  REAL,
        benchmark_return      REAL,
        alpha                 REAL,
        holding_days          INTEGER,
        -- 阈值口径（rule_id='metric_threshold' 时非空）
        metric_ref            TEXT,
        observed_value        REAL,
        comparator            TEXT,
        threshold             REAL,
        series_fingerprint    TEXT,
        reflection            TEXT,
        -- 两种口径各自要求的字段必须齐；不许"都没有"。
        CHECK (
            (rule_id = 'alpha_vs_benchmark' AND alpha IS NOT NULL AND benchmark_symbol IS NOT NULL)
            OR
            (rule_id = 'metric_threshold' AND metric_ref IS NOT NULL AND verdict IS NOT NULL)
        )
    )
    """,
    "CREATE INDEX IF NOT EXISTS ix_settlements_date ON settlements (settlement_date)",
    # 组合结算是**另一张表**：单票结算的 entry/exit 是"两笔成交价"，组合的是
    # "加权收益 + 现金部分"，硬塞进同一张表会让两类语义共用一个形状。
    # 分开也顺带把"不得以单票结果拼装组合收益"变成结构事实——没有那样的拼装路径。
    """
    CREATE TABLE IF NOT EXISTS portfolio_settlements (
        decision_id       TEXT PRIMARY KEY REFERENCES decisions(decision_id),
        settled_at        TEXT NOT NULL,
        settlement_date   TEXT NOT NULL,
        raw_return        REAL NOT NULL,
        invested_weight   REAL NOT NULL,
        cash_weight       REAL NOT NULL,
        cash_return       REAL NOT NULL,
        benchmark_symbol  TEXT NOT NULL,
        benchmark_mapping TEXT NOT NULL,
        benchmark_return  REAL NOT NULL,
        alpha             REAL NOT NULL,
        holding_days      INTEGER NOT NULL,
        legs              TEXT NOT NULL,
        reflection        TEXT
    )
    """,
    "CREATE INDEX IF NOT EXISTS ix_portfolio_settlements_date"
    " ON portfolio_settlements (settlement_date)",
    # 风控阈值变更留痕（change3 task 3.5）：**追加式**，最新一行即当前生效值。
    # 为什么不就地改写：改了阈值就改了策略，而"当时用的是哪套阈值"必须能查回来——
    # 否则历史轮次的产出会看起来像用新阈值跑出来的。
    """
    CREATE TABLE IF NOT EXISTS risk_thresholds (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        caller       TEXT NOT NULL,
        scope        TEXT NOT NULL DEFAULT 'default',
        thresholds   TEXT NOT NULL,
        reason       TEXT NOT NULL,
        recorded_at  TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS ix_risk_thresholds_latest"
    " ON risk_thresholds (caller, scope, id)",
)


def default_db_path() -> Path:
    """零配置默认库路径：`FD_QUANT_LEDGER_DB` → 否则当前工作目录下的单文件。"""
    env = os.environ.get("FD_QUANT_LEDGER_DB")
    if env:
        return Path(env).expanduser()
    return Path.cwd() / DEFAULT_DB_FILENAME


def _json_dumps(value: Any) -> str:
    """存库前统一序列化：`allow_nan=False` 让 NaN/Inf 在**写入期**就炸，
    而不是静默存成 JSON 非法字面量、出库时才污染调用方。"""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def _json_loads(text: Optional[str]) -> Any:
    return None if text is None else json.loads(text)


def _clean_float(value: Any, *, name: str) -> float:
    """出库浮点的 JSON 安全检查（库被外部工具写入 NaN 也不外泄）。"""
    f = float(value)
    if not math.isfinite(f):
        raise LedgerError(f"{name} in the ledger is not finite ({value!r}); refusing to return it")
    return f


def _clean_float_or_none(value: Any, *, name: str):
    """结算读数：阈值口径下价格类列**本来就空**，空是合法值而非脏数据。

    与 `_clean_float` 的分工：那个表达"这一列必须有限，否则说明库被写坏了"；
    这个表达"这一列可能为空（另一种口径不用它），有值时仍必须是有限数"。
    不把两者合并——合并会让真正的脏数据被静默当成"空"。
    """
    return None if value is None else _clean_float(value, name=name)


def _mean(values: Sequence[float]) -> float:
    """算术均值：显式从左到右 + Kahan 补偿（不用 `np.mean` 的成对归约——那是
    唯一随数组长度/线程数变化的路径）。"""
    total = 0.0
    c = 0.0
    for x in values:
        y = float(x) - c
        t = total + y
        c = (t - total) - y
        total = t
    return total / len(values)


def _new_decision_id() -> str:
    return f"dec-{uuid.uuid4().hex[:16]}"


def _utc_now() -> str:
    """ISO-8601 UTC 时间戳。仅作审计记录：测试不得断言其跨运行相等
    （需要确定性时调用方传 `now=`）。"""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class _Merged:
    """`sqlite3.Row` 的最小替身（只需 `__getitem__` / `keys()`），
    用于把 separately-fetched 的结算行合并进记录行的读取路径。"""

    def __init__(self, data: Mapping[str, Any]) -> None:
        self._d = dict(data)

    def __getitem__(self, key: str) -> Any:
        return self._d[key]

    def keys(self) -> list[str]:
        return list(self._d)


def _merge_settlement(decision_row: Any, settlement_row: Any) -> _Merged:
    merged = {k: decision_row[k] for k in decision_row.keys()}
    for key in ("settled_at", "settlement_date", "rule_id", "verdict", "entry_price",
                "exit_price", "raw_return", "benchmark_symbol", "benchmark_mapping",
                "benchmark_return", "alpha", "holding_days", "metric_ref",
                "observed_value", "comparator", "threshold", "series_fingerprint",
                "reflection"):
        merged[f"s_{key}"] = settlement_row[key]
    return _Merged(merged)


def build_premise_invalidated_event(
    *, decision_id: str, precondition_index: int, metric_ref: str,
    observed_value: Optional[float], comparator: str, threshold: float,
    evaluated_at: str, series_fingerprint: str,
) -> dict:
    """构造 `premise_invalidated` 事件（add-decision-consistency-and-premise-watch 3.3）。

    纯函数：不评测、不落库、不通知。六个字段是"三十秒内可回答"的最小事实——
    缺任何一个，看事件的人就得去翻账本才能知道该重新看哪个判断。

    注意 `observed_value` 允许为 None（指标点缺失也是一种失效形态），但
    `series_fingerprint` 不允许——没有指纹，"当时凭什么判失效"就无法复核。
    """
    if not series_fingerprint:
        raise ValueError("series_fingerprint 必填：没有指纹的失效判定无法复核")
    return {
        "event": "premise_invalidated",
        "decision_id": decision_id,
        "precondition_index": precondition_index,
        "metric_ref": metric_ref,
        "observed_value": observed_value,
        "comparator": comparator,
        "threshold": threshold,
        "evaluated_at": evaluated_at,
        "series_fingerprint": series_fingerprint,
    }


def _compare(observed: float, comparator: str, threshold: float) -> bool:
    """阈值比较。`eq`/`neq` 用容差判等——浮点严格相等几乎总是错的。"""
    tol = 1e-9
    if comparator == "lt":
        return observed < threshold
    if comparator == "lte":
        return observed <= threshold
    if comparator == "gt":
        return observed > threshold
    if comparator == "gte":
        return observed >= threshold
    if comparator == "eq":
        return abs(observed - threshold) <= tol
    if comparator == "neq":
        return abs(observed - threshold) > tol
    raise LedgerError(f"comparator 非法：{comparator!r}")


def _series_fingerprint(series: Sequence[tuple[str, Optional[float]]]) -> str:
    """序列指纹：让"当时凭什么判 hit/miss"可复核，同时不留客户原始数据。

    只对 (日期, 值) 做摘要——值本身是客户数据，指纹不是。复核方拿同一序列重算
    指纹即可确认判定依据没被改动。
    """
    import hashlib

    h = hashlib.sha256()
    for d, v in series:
        h.update(f"{d}|{'' if v is None else v!r};".encode("utf-8"))
    return h.hexdigest()[:32]


def _migrate_to_caller(con: sqlite3.Connection) -> None:
    """把 caller 维度加进既有库（幂等：可重复执行）。

    **为什么要重建表而不是 ALTER TABLE ADD COLUMN**：`caller` 要进唯一索引，而
    SQLite 无法把列加进一个已存在的索引——只能按新 schema 建表再搬行。故这里显式
    rename → 建新表 → INSERT ... SELECT（旧行的 caller 一律为 `LEGACY`）→ drop 旧表。

    幂等的落点：`PRAGMA table_info` 里已经有 `caller` 就直接返回。第二次运行因此是
    空操作，不会产生第二份数据，也不会因为"新表已存在"而报错。
    """
    cols = {r[1] for r in con.execute("PRAGMA table_info(decisions)")}
    if not cols or "caller" in cols:
        return  # 新库（建表已带 caller）或已迁移

    con.execute("ALTER TABLE decisions RENAME TO decisions_old")
    for stmt in SCHEMA_STATEMENTS:
        if "CREATE TABLE IF NOT EXISTS decisions" in stmt or " ON decisions" in stmt:
            con.execute(stmt)
    # LEGACY：这些行产生于"账本还没有调用者维度"的时期。显式标注而非猜——运维据此
    # 知道这批记录不属于任何已知调用者，也不会被任何调用者的检索看到。
    con.execute(
        "INSERT INTO decisions (decision_id, caller, as_of, symbol, pack, skill,"
        " payload, target_weights, evidence, status, revises, revision_seq, created_at)"
        " SELECT decision_id, ?, as_of, symbol, pack, skill, payload, target_weights,"
        " evidence, status, revises, revision_seq, created_at FROM decisions_old",
        (LEGACY,),
    )
    con.execute("DROP TABLE decisions_old")


def _migrate_to_portfolio(con: sqlite3.Connection) -> None:
    """把 record_kind / portfolio_key 加进既有库（幂等：可重复执行）。

    **为什么要重建表而不是 ALTER TABLE**：`symbol` 原先 `NOT NULL`，而组合记录的 symbol
    必须为 NULL。SQLite 无法放宽列约束，只能按新 schema 建表再搬行——与
    `_migrate_to_caller` 同一手法。既有行一律 `record_kind='single'`、`portfolio_key=NULL`，
    故旧幂等键在新索引下逐条等价（`COALESCE(symbol, NULL)` 就是 symbol）。
    """
    cols = {r[1] for r in con.execute("PRAGMA table_info(decisions)")}
    if not cols or "record_kind" in cols:
        return  # 新库（建表已带）或已迁移

    con.execute("ALTER TABLE decisions RENAME TO decisions_old")
    for stmt in SCHEMA_STATEMENTS:
        if "CREATE TABLE IF NOT EXISTS decisions" in stmt or " ON decisions" in stmt:
            con.execute(stmt)
    con.execute(
        "INSERT INTO decisions (decision_id, caller, record_kind, as_of, symbol, portfolio_key,"
        " pack, skill, payload, target_weights, evidence, status, revises, revision_seq,"
        " created_at)"
        " SELECT decision_id, caller, 'single', as_of, symbol, NULL, pack, skill, payload,"
        " target_weights, evidence, status, revises, revision_seq, created_at"
        " FROM decisions_old"
    )
    con.execute("DROP TABLE decisions_old")


def _migrate_to_decision_v2(con: sqlite3.Connection) -> None:
    """把 v2 决策所需的列与阈值口径结算加进既有库（幂等：可重复执行）。

    两处改动都要**重建表**：
    * `decisions` 加 5 列——单靠 ALTER 加列无法同时加"三列全空或全齐"的 CHECK，
      而那条 CHECK 正是防止出现"半个 v2 记录"的唯一屏障。
    * `settlements` 要放宽 `entry_price` / `benchmark_*` / `alpha` 的 NOT NULL——SQLite
      不支持 `ALTER COLUMN ... DROP NOT NULL`，只能按新 schema 建表再搬行。

    历史结算行按 `rule_id='alpha_vs_benchmark'` 回填、`verdict` 由 `alpha > 0` 推导。
    **那次推导是新增判断，不是恢复原值**：老行本来没有 verdict 概念。迁移日志与任何
    基于历史数据的复盘输出都必须带这句，否则三个月后会有人把回填值当成当时就有的结论。
    """
    dcols = {r[1] for r in con.execute("PRAGMA table_info(decisions)")}
    if dcols and "subject_kind" not in dcols:
        con.execute("ALTER TABLE decisions RENAME TO decisions_old_v2")
        for stmt in SCHEMA_STATEMENTS:
            if "CREATE TABLE IF NOT EXISTS decisions" in stmt or " ON decisions" in stmt:
                con.execute(stmt)
        con.execute(
            # series_ref 由 add-price-supply-and-caliber 4.5 引入；本迁移重建表时若不搬它，
            # 已落过 ref 的行会静默变 NULL（实测复现过）。两条迁移链叠在同一条 _init_schema 上，
            # 谁都没错，但缺口会丢数据——所以这里必须显式列出。
            "INSERT INTO decisions (decision_id, caller, record_kind, as_of, symbol,"
            " portfolio_key, pack, skill, payload, target_weights, series_ref, evidence,"
            " status, revises, revision_seq, created_at, subject_kind, subject_key, action,"
            " preconditions, settlement_spec)"
            " SELECT decision_id, caller, record_kind, as_of, symbol, portfolio_key,"
            " pack, skill, payload, target_weights, series_ref, evidence, status, revises,"
            " revision_seq, created_at, NULL, NULL, NULL, NULL, NULL"
            " FROM decisions_old_v2"
        )
        con.execute("DROP TABLE decisions_old_v2")

    scols = {r[1] for r in con.execute("PRAGMA table_info(settlements)")}
    if scols and "rule_id" not in scols:
        con.execute("ALTER TABLE settlements RENAME TO settlements_old_v2")
        for stmt in SCHEMA_STATEMENTS:
            if "CREATE TABLE IF NOT EXISTS settlements" in stmt or " ON settlements" in stmt:
                con.execute(stmt)
        con.execute(
            "INSERT INTO settlements (decision_id, settled_at, settlement_date, rule_id,"
            " verdict, entry_price, exit_price, raw_return, benchmark_symbol,"
            " benchmark_mapping, benchmark_entry_price, benchmark_exit_price,"
            " benchmark_return, alpha, holding_days, metric_ref, observed_value,"
            " comparator, threshold, series_fingerprint, reflection)"
            " SELECT decision_id, settled_at, settlement_date, 'alpha_vs_benchmark',"
            " CASE WHEN alpha > 0 THEN 'hit' ELSE 'miss' END,"
            " entry_price, exit_price, raw_return, benchmark_symbol, benchmark_mapping,"
            " benchmark_entry_price, benchmark_exit_price, benchmark_return, alpha,"
            " holding_days, NULL, NULL, NULL, NULL, NULL, reflection"
            " FROM settlements_old_v2"
        )
        con.execute("DROP TABLE settlements_old_v2")


def _migrate_to_series_ref(con: sqlite3.Connection) -> None:
    """给既有库补 `series_ref` 列（add-price-supply-and-caliber task 4.5）。

    列可空、无索引 → `ALTER TABLE ADD COLUMN` 即可，不必重建表。幂等：列已在就返回。
    新库由建表语句直接带该列，故这一步对新库是空操作。
    """
    cols = {r[1] for r in con.execute("PRAGMA table_info(decisions)")}
    if not cols or "series_ref" in cols:
        return
    con.execute("ALTER TABLE decisions ADD COLUMN series_ref TEXT")


class LedgerStore:
    """决策账本：追加式记录（幂等 + 修订）+ 到期结算 + 点内检索 + 聚合。

    每次操作开一条短连接（WAL + `foreign_keys=ON` + `busy_timeout`），用完即关：
    SQLite 连接不可跨线程共享，短连接让线程/进程并发落到同一套锁语义上，
    也让"进程中断"测试里的连接天然干净（没有跨调用的残留状态）。
    """

    def __init__(
        self,
        db_path: Optional[os.PathLike | str] = None,
        *,
        benchmark_map: Optional[Mapping[str, str]] = None,
        holding_days: int = DEFAULT_HOLDING_DAYS,
        timeout: float = BUSY_TIMEOUT_MS / 1000.0,
        id_factory: Optional[Callable[[], str]] = None,
    ) -> None:
        self.db_path = Path(db_path) if db_path is not None else default_db_path()
        self.benchmark_map: dict[str, str] = dict(
            DEFAULT_BENCHMARK_MAP if benchmark_map is None else benchmark_map
        )
        self.holding_days = check_holding_days(holding_days)
        self.timeout = float(timeout)
        #: `decision_id` 生成器。缺省是 uuid4 派生（无需协调、并发下天然不撞）。
        #: 需要**逐位可复现**的调用方（端到端演练 / 报告比对）传一个确定性生成器——
        #: 与"注入时钟"同一个手法：把不确定性收到一个可替换的接缝上，而不是让
        #: 随机 id 渗进被比较的结果里。
        self.id_factory: Callable[[], str] = id_factory or _new_decision_id
        self._init_schema()

    # ── 连接与事务 ────────────────────────────────────────────

    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        con = sqlite3.connect(
            str(self.db_path), timeout=self.timeout,
            isolation_level=None,  # 显式事务：BEGIN 由本层发，不靠驱动隐式开
        )
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA foreign_keys=ON")
        con.execute(f"PRAGMA busy_timeout={BUSY_TIMEOUT_MS}")
        return con

    def _init_schema(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        con = self._connect()
        try:
            con.execute("BEGIN IMMEDIATE")
            # 两个迁移都在建表之前跑：它们各自幂等，旧库走"rename → 按新 schema 建表
            # → 搬行"，新库直接返回。索引必须等迁移完成后再建，否则旧表上会缺列。
            _migrate_to_caller(con)
            _migrate_to_portfolio(con)
            _migrate_to_series_ref(con)
            _migrate_to_decision_v2(con)
            for stmt in SCHEMA_STATEMENTS:
                con.execute(stmt)
            con.execute("COMMIT")
        except BaseException:
            try:
                con.execute("ROLLBACK")
            except sqlite3.Error:
                pass
            raise
        finally:
            con.close()

    @contextmanager
    def _write_tx(self) -> Iterator[sqlite3.Connection]:
        """写事务：`BEGIN IMMEDIATE`（立刻取写锁，见模块文档）→ COMMIT / ROLLBACK。

        任何异常（含 `SettlementRefused`）都回滚：被拒的写入不留下任何痕迹
        （"半窗 alpha 不入账"与"半行不可能出现"是同一个机制）。
        """
        con = self._connect()
        try:
            con.execute("BEGIN IMMEDIATE")
            yield con
            con.execute("COMMIT")
        except BaseException:
            try:
                con.execute("ROLLBACK")
            except sqlite3.Error:
                pass
            raise
        finally:
            con.close()

    @contextmanager
    def _read(self) -> Iterator[sqlite3.Connection]:
        con = self._connect()
        try:
            yield con
        finally:
            con.close()

    # ── 追加（含幂等与修订）────────────────────────────────────

    def record(
        self,
        *,
        caller: str,
        as_of: str,
        symbol: Optional[str] = None,
        pack: str,
        skill: str,
        payload: Any,
        record_kind: str = "single",
        portfolio_key: Optional[str] = None,
        target_weights: Any = None,
        evidence: Optional[Sequence[str]] = None,
        series_ref: Optional[Any] = None,
        revises: Optional[str] = None,
        now: Optional[str] = None,
    ) -> dict:
        """追加一条决策：`payload` 是契约实例（`Rating`，可选关联 `TargetWeights`）。

        - 幂等键 = `(caller, as_of, symbol, pack, skill)`。同键重复提交**不抛异常**：
          返回既有 `decision_id` 且 `created=False`（spec 要求"被拒"与"给出既有 id"
          同时可见，抛异常会让后者无法表达），库行数逐字不变（结果带 `counts`）。
        - `revises=<decision_id>`：提交修订 → 新行、`revises` 指向被修订记录、
          `revision_seq` 递增。被修订的必须是**当前 tip**（链是线性的，账本不裁决分叉）；
          内容与 tip 逐字相同 → 幂等返回 tip（`reason="revision_duplicate"`）。
        - `payload` 必须是合法 `Rating`；其 `as_of/subject/producer.pack/skill`
          必须与记录的键逐项一致（不一致即拒绝并点名两边，不静默取信其中一边）。
          同一判定要挂组合权重时传 `target_weights`（`TargetWeights`）。
        """
        a = check_iso_date(as_of, name="as_of")
        kind = (record_kind or "single").strip()
        if kind not in ("single", "portfolio"):
            raise LedgerError(
                f"record_kind={record_kind!r} must be 'single' or 'portfolio'"
            )
        pk = self._require_text(pack, name="pack")
        sk = self._require_text(skill, name="skill")
        created_at = now if now is not None else _utc_now()

        # 主体字段与种类必须一致：同时给出或同时缺失都是错误。这里显式判而不是靠
        # CHECK 约束兜底，因为要给出"哪两个字段冲突"的可读错误（约束只报 constraint failed）。
        has_sym = symbol is not None and str(symbol).strip() != ""
        has_pkey = portfolio_key is not None and str(portfolio_key).strip() != ""
        if kind == "single":
            if not has_sym:
                raise LedgerError("record_kind='single' requires symbol")
            if has_pkey:
                raise LedgerError(
                    "record_kind='single' must not carry portfolio_key "
                    f"(got {portfolio_key!r})"
                )
        else:
            if not has_pkey:
                raise LedgerError("record_kind='portfolio' requires portfolio_key")
            if has_sym:
                raise LedgerError(
                    "record_kind='portfolio' must not carry symbol "
                    f"(got {symbol!r}); the subject of a portfolio record is portfolio_key"
                )

        subject = self._require_text(symbol if kind == "single" else portfolio_key, name="subject")
        sym = symbol if kind == "single" else None

        checked = self._check_payload(
            payload, as_of=a, symbol=subject, pack=pk, skill=sk, record_kind=kind
        )
        ref_json = self._check_series_ref(series_ref)
        tw_json = self._check_target_weights(target_weights)
        refs = self._check_evidence(evidence, payload=checked)
        payload_json = _json_dumps(checked)
        evidence_json = _json_dumps(refs)

        key = {
            "caller": caller, "record_kind": kind, "as_of": a, "subject": subject,
            "symbol": sym, "portfolio_key": (subject if kind == "portfolio" else None),
            "pack": pk, "skill": sk,
            "series_ref_json": ref_json,
        }
        if revises is not None:
            return self._append_revision(
                key, payload_json=payload_json, target_weights_json=tw_json,
                evidence_json=evidence_json, revises=self._require_text(revises, name="revises"),
                created_at=created_at,
            )
        return self._append_initial(
            key, payload_json=payload_json, target_weights_json=tw_json,
            evidence_json=evidence_json, created_at=created_at,
        )

    def _append_initial(self, key: dict, *, payload_json: str, target_weights_json: Optional[str],
                        evidence_json: str, created_at: str) -> dict:
        with self._write_tx() as con:
            existing = self._initial_row(con, **key)
            if existing is not None:
                return self._result(con, existing["decision_id"], created=False,
                                    reason="duplicate_key")
            decision_id = self.id_factory()
            con.execute(
                "INSERT INTO decisions (decision_id, caller, record_kind, as_of, symbol,"
                " portfolio_key, pack, skill, payload, target_weights, series_ref, evidence,"
                " status, revises, revision_seq, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', NULL, 0, ?)",
                (decision_id, key["caller"], key["record_kind"], key["as_of"], key["symbol"],
                 key["portfolio_key"], key["pack"], key["skill"], payload_json,
                 target_weights_json, key.get("series_ref_json"), evidence_json, created_at),
            )
            return self._result(con, decision_id, created=True, reason="created")

    def _append_revision(self, key: dict, *, payload_json: str, target_weights_json: Optional[str],
                         evidence_json: str, created_at: str, revises: str) -> dict:
        with self._write_tx() as con:
            target = con.execute(
                "SELECT * FROM decisions WHERE decision_id = ?", (revises,)
            ).fetchone()
            if target is None:
                raise UnknownDecision(
                    f"revises={revises!r}: no such decision in the ledger; a revision must "
                    f"point at an existing decision_id"
                )
            for field in ("record_kind", "as_of", "symbol", "portfolio_key", "pack", "skill"):
                if target[field] != key[field]:
                    raise RevisionConflict(
                        f"revises={revises!r} has {field}={target[field]!r} but the revision was "
                        f"submitted with {field}={key[field]!r}; a revision keeps the idempotency "
                        f"key of the record it revises"
                    )
            tip = self._tip_row(con, **key)
            same_content = (
                tip["payload"] == payload_json
                and tip["target_weights"] == target_weights_json
                and tip["evidence"] == evidence_json
            )
            if tip["decision_id"] != revises and not same_content:
                raise RevisionConflict(
                    f"revises={revises!r} is not the current revision tip "
                    f"({tip['decision_id']!r}, revision_seq={tip['revision_seq']}); a revision "
                    f"must target the tip so the chain stays linear"
                )
            if same_content:
                # 内容逐字相同：这是重复提交（不是一次真修订），幂等返回 tip。
                return self._result(con, tip["decision_id"], created=False,
                                    reason="revision_duplicate")
            decision_id = self.id_factory()
            con.execute(
                "INSERT INTO decisions (decision_id, caller, record_kind, as_of, symbol,"
                " portfolio_key, pack, skill, payload, target_weights, series_ref, evidence,"
                " status, revises, revision_seq, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?, ?)",
                (decision_id, key["caller"], key["record_kind"], key["as_of"], key["symbol"],
                 key["portfolio_key"], key["pack"], key["skill"], payload_json,
                 target_weights_json, key.get("series_ref_json"), evidence_json,
                 tip["decision_id"], int(tip["revision_seq"]) + 1, created_at),
            )
            return self._result(con, decision_id, created=True, reason="revision_created",
                                revises_override=tip["decision_id"])

    # ── 结算 ─────────────────────────────────────────────────

    def settle(
        self,
        *,
        decision_id: str,
        symbol_dates: Sequence[Any],
        symbol_prices: Sequence[Any],
        benchmark_dates: Sequence[Any],
        benchmark_prices: Sequence[Any],
        as_of: str,
        reflection: Optional[str] = None,
        benchmark_symbol: Optional[str] = None,
        holding_days: Optional[int] = None,
        series_ref: Optional[Any] = None,
        now: Optional[str] = None,
    ) -> dict:
        """到期结算一条决策：价格序列由**调用方传入**，账本不取数、不联网。

        - 幂等：已有结算行 → 返回既有结果、不插入第二行（`created=False`）。
          若本次传了与既有**不同**的反思，既有行不被改写，结果里
          `reflection_ignored=True` 提示未生效（追加语义：已落记录不就地改写）。
        - 持有期未满 → `SettlementRefused`（带 `missing_trading_days`），**不落任何行**、
          状态保持 `pending`——"半窗 alpha"比"没有 alpha"更坏（design.md D4）。
        - 基准：显式 `benchmark_symbol` 永远优先；否则按市场后缀映射（可配置），
          结果与库行都标注基准是怎么定的（`benchmark_mapping`）。
        - 持有期按**交易日**计，交易日历由传入序列的日期集合隐式给出。
        """
        a = check_iso_date(as_of, name="as_of")
        did = self._require_text(decision_id, name="decision_id")
        hd = check_holding_days(self.holding_days if holding_days is None else holding_days)
        settled_at = now if now is not None else _utc_now()

        with self._write_tx() as con:
            row = con.execute(
                "SELECT * FROM decisions WHERE decision_id = ?", (did,)
            ).fetchone()
            if row is None:
                raise UnknownDecision(
                    f"no decision {did!r} in the ledger; settle() works on a recorded decision"
                )
            existing = self._settlement_row(con, did)
            if existing is not None:
                ignored = reflection is not None and reflection != existing["reflection"]
                return self._settlement_result(con, existing, created=False,
                                               reason="duplicate_settlement",
                                               reflection_ignored=ignored)

            # 结算侧口径断言（add-price-supply-and-caliber 4.5）：
            # 落账时记了 ref、本次结算也带了 ref → 比对；不一致按开关处置。
            caliber = self._assert_caliber(row, series_ref)

            bench = resolve_benchmark(
                row["symbol"], explicit=benchmark_symbol, mapping=self.benchmark_map
            )
            window = settlement_window(
                entry_date=row["as_of"],
                symbol_dates=symbol_dates,
                symbol_values=symbol_prices,
                benchmark_dates=benchmark_dates,
                benchmark_values=benchmark_prices,
                as_of=a,
                holding_days=hd,
                benchmark_symbol=bench["benchmark_symbol"],
            )
            # verdict 由 alpha 推导。这是**判定**，不是状态字段：alpha>0 记 hit，
            # 反之为 miss；alpha 恰为 0 记 miss（不算赢，与 summarize 的口径一致）。
            verdict = "hit" if window["alpha"] > 0 else "miss"
            con.execute(
                "INSERT INTO settlements (decision_id, settled_at, settlement_date, rule_id, "
                "verdict, entry_price, exit_price, raw_return, benchmark_symbol, "
                "benchmark_mapping, benchmark_entry_price, benchmark_exit_price, "
                "benchmark_return, alpha, holding_days, reflection)"
                " VALUES (?, ?, ?, 'alpha_vs_benchmark', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (did, settled_at, window["exit_date"], verdict, window["entry_price"],
                 window["exit_price"], window["raw_return"], window["benchmark_symbol"],
                 bench["mapping"], window["benchmark_entry_price"],
                 window["benchmark_exit_price"], window["benchmark_return"], window["alpha"],
                 window["holding_days"], reflection),
            )
            con.execute("UPDATE decisions SET status = 'settled' WHERE decision_id = ?", (did,))
            stored = self._settlement_row(con, did)
            out = self._settlement_result(con, stored, created=True, reason="settled",
                                          reflection_ignored=False)
            out["caliber"] = caliber
            return out

    def settle_metric_threshold(
        self,
        *,
        decision_id: str,
        dates: Sequence[Any],
        values: Sequence[Any],
        as_of: str,
        comparator: str,
        threshold: float,
        reflection: Optional[str] = None,
        window_end: Optional[str] = None,
        now: Optional[str] = None,
    ) -> dict:
        """按指标阈值结算一条决策（`metric_threshold` 口径）。

        与价格口径的分工：那个问"跑赢基准多少"，这个问"指标有没有达到阈值"。
        经营决策没有基准、没有成交价，只有这一种问法。

        **三条不可退让**：

        1. 序列由**调用方传入**，账本不取数、不联网。经营指标住在客户 cell 里，
           账本只留 `metric_ref`、观测值、阈值与序列指纹——客户数据一行不出 cell。
        2. 数据不足记 `insufficient_data`，**MUST NOT** 记 `miss`。把"不知道"当成
           "判断错了"，会让复盘系统性地低估一个模式的准确率，而且再也查不出来。
        3. 数据不足时**照样落一行结算**（`verdict='insufficient_data'`），不让记录停在
           `pending` 而不给原因——那会让复盘误以为"还没到期"，一直等一个永不到来的结算。

        与价格口径不同，这里**没有"持有期未满就拒结算"**：阈值结算的观察窗由
        `window_end`（或 `as_of`）界定，窗未走完就如实记 `insufficient_data`。
        拒绝落行的语义只适用于"价格口径不能用半窗算 alpha"那种情形。
        """
        a = check_iso_date(as_of, name="as_of")
        did = self._require_text(decision_id, name="decision_id")
        settled_at = now if now is not None else _utc_now()
        if comparator not in ("lt", "lte", "gt", "gte", "eq", "neq"):
            raise LedgerError(
                f"comparator 非法：{comparator!r}；合法集合 "
                "['lt', 'lte', 'gt', 'gte', 'eq', 'neq']"
            )
        thr = _clean_float(threshold, name="threshold")
        end = check_iso_date(window_end or a, name="window_end")
        if end > a:
            raise LedgerError(f"window_end ({end}) 晚于 as_of ({a})：不得用未来窗口结算")

        series = build_series(dates, values, name="metric")
        # 未来点拒绝：与价格口径同一姿态——结算只能用 as_of 当时已知的数据。
        # 静默截断会让"用了未来数据"看起来像"窗口就这么长"，两种情形在结果里同形。
        future = [d for d, _v in series if d > a]
        if future:
            raise LedgerError(
                f"序列含晚于 as_of={a} 的数据点：{', '.join(future[:5])}"
                + (" 等" if len(future) > 5 else "")
                + "；结算不得使用未来数据"
            )
        fingerprint = _series_fingerprint(series)

        with self._write_tx() as con:
            row = con.execute(
                "SELECT * FROM decisions WHERE decision_id = ?", (did,)
            ).fetchone()
            if row is None:
                raise UnknownDecision(
                    f"no decision {did!r} in the ledger; settle() works on a recorded decision"
                )
            existing = self._settlement_row(con, did)
            if existing is not None:
                ignored = reflection is not None and reflection != existing["reflection"]
                return self._settlement_result(con, existing, created=False,
                                               reason="duplicate_settlement",
                                               reflection_ignored=ignored)

            in_window = [p for p in series if p[0] <= end]
            if not in_window:
                verdict = "insufficient_data"
                observed = None
                reason = "insufficient_data"
            else:
                observed = in_window[-1][1] if in_window[-1][1] is not None else None
                if observed is None:
                    verdict = "insufficient_data"
                    reason = "insufficient_data"
                else:
                    verdict = "hit" if _compare(observed, comparator, thr) else "miss"
                    reason = "settled"

            # metric_ref 的真源是决策 payload 的 expected_outcome.metric_ref。
            # 此前用 subject 兜底：subject 是"判断的对象"（门店），metric_ref 是
            # "结算所依据的指标"——两者语义不同，兜底会把门店 id 写进指标位，
            # 指纹对不上、复核也找不到数据源。
            payload = _json_loads(row["payload"]) or {}
            metric_ref = ((payload.get("expected_outcome") or {}).get("metric_ref")
                          if isinstance(payload.get("expected_outcome"), dict)
                          else None)
            if not metric_ref:
                raise LedgerError(
                    f"决策 {did} 的 payload 缺少 expected_outcome.metric_ref；"
                    "阈值结算必须知道依据哪个指标"
                )
            con.execute(
                "INSERT INTO settlements (decision_id, settled_at, settlement_date, rule_id, "
                "verdict, metric_ref, observed_value, comparator, threshold, "
                "series_fingerprint, reflection)"
                " VALUES (?, ?, ?, 'metric_threshold', ?, ?, ?, ?, ?, ?, ?)",
                (did, settled_at, end, verdict,
                 metric_ref, observed, comparator, thr,
                 fingerprint, reflection),
            )
            con.execute("UPDATE decisions SET status = 'settled' WHERE decision_id = ?", (did,))
            stored = self._settlement_row(con, did)
            return self._settlement_result(con, stored, created=True, reason=reason,
                                           reflection_ignored=False)

    def settle_portfolio(
        self,
        *,
        decision_id: str,
        legs: Mapping[str, tuple[Sequence[Any], Sequence[Any]]],
        benchmark_dates: Sequence[Any],
        benchmark_prices: Sequence[Any],
        as_of: str,
        benchmark_symbol: str,
        reflection: Optional[str] = None,
        holding_days: Optional[int] = None,
        now: Optional[str] = None,
    ) -> dict:
        """到期结算一条**组合**记录：整期权重 + 全部标的价格序列 + 基准序列。

        - 权重取自记录的 `payload`（`PortfolioIntent`），**不接受调用方另传一份**：
          否则"这笔单依据哪份权重"会有两个答案，而两份迟早不一致。
        - 整期原子：任一腿不覆盖持有期 → 整次拒绝，记录保持 `pending`（不落半行）。
        - 幂等：已有组合结算行 → 返回既有结果，不插第二行。
        - 现金部分收益记零并单独报出，不摊进标的收益。
        - 基准必须显式给出：组合没有单一标的可做后缀映射。
        """
        a = check_iso_date(as_of, name="as_of")
        did = self._require_text(decision_id, name="decision_id")
        hd = check_holding_days(self.holding_days if holding_days is None else holding_days)
        settled_at = now if now is not None else _utc_now()

        with self._write_tx() as con:
            row = con.execute(
                "SELECT * FROM decisions WHERE decision_id = ?", (did,)
            ).fetchone()
            if row is None:
                raise UnknownDecision(
                    f"no decision {did!r} in the ledger; settle_portfolio() works on a "
                    "recorded decision"
                )
            if row["record_kind"] != "portfolio":
                raise LedgerError(
                    f"decision {did!r} has record_kind={row['record_kind']!r}; "
                    "settle_portfolio() only settles portfolio records "
                    "(use settle() for a single-symbol record)"
                )

            existing = con.execute(
                "SELECT * FROM portfolio_settlements WHERE decision_id = ?", (did,)
            ).fetchone()
            if existing is not None:
                ignored = reflection is not None and reflection != existing["reflection"]
                return self._portfolio_settlement_result(
                    existing, created=False, reason="duplicate_settlement",
                    reflection_ignored=ignored,
                )

            payload = _json_loads(row["payload"]) or {}
            weights = payload.get("weights") or {}
            if not weights:
                raise SettlementRefused(
                    f"decision {did!r} carries no weights; an empty portfolio cannot be settled"
                )

            window = portfolio_settlement_window(
                entry_date=row["as_of"],
                legs=legs,
                weights=weights,
                benchmark_dates=benchmark_dates,
                benchmark_values=benchmark_prices,
                as_of=a,
                holding_days=hd,
                benchmark_symbol=benchmark_symbol,
            )
            con.execute(
                "INSERT INTO portfolio_settlements (decision_id, settled_at, settlement_date,"
                " raw_return, invested_weight, cash_weight, cash_return, benchmark_symbol,"
                " benchmark_mapping, benchmark_return, alpha, holding_days, legs, reflection)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (did, settled_at, window["exit_date"], window["raw_return"],
                 window["invested_weight"], window["cash_weight"], window["cash_return"],
                 window["benchmark_symbol"], "explicit", window["benchmark_return"],
                 window["alpha"], window["holding_days"],
                 _json_dumps(window["leg_returns"]), reflection),
            )
            con.execute("UPDATE decisions SET status = 'settled' WHERE decision_id = ?", (did,))
            stored = con.execute(
                "SELECT * FROM portfolio_settlements WHERE decision_id = ?", (did,)
            ).fetchone()
            return self._portfolio_settlement_result(
                stored, created=True, reason="settled", reflection_ignored=False
            )

    @staticmethod
    def _portfolio_settlement_result(row: sqlite3.Row, *, created: bool, reason: str,
                                     reflection_ignored: bool) -> dict:
        return {
            "decision_id": row["decision_id"],
            "created": created,
            "reason": reason,
            "reflection_ignored": reflection_ignored,
            "settlement": {
                "settled_at": row["settled_at"],
                "settlement_date": row["settlement_date"],
                "raw_return": _clean_float_or_none(row["raw_return"], name="raw_return"),
                "invested_weight": _clean_float(row["invested_weight"], name="invested_weight"),
                "cash_weight": _clean_float(row["cash_weight"], name="cash_weight"),
                "cash_return": _clean_float(row["cash_return"], name="cash_return"),
                "benchmark_symbol": row["benchmark_symbol"],
                "benchmark_mapping": row["benchmark_mapping"],
                "benchmark_return": _clean_float_or_none(row["benchmark_return"], name="benchmark_return"),
                "alpha": _clean_float_or_none(row["alpha"], name="alpha"),
                "holding_days": int(row["holding_days"]),
                "leg_returns": _json_loads(row["legs"]) or {},
                "reflection": row["reflection"],
            },
        }

    # ── 查询 ─────────────────────────────────────────────────

    def get(self, decision_id: str) -> Optional[dict]:
        """单条快照：判定快照 + 结算结果（`pending` 记录的 `settlement` 为 null）。"""
        with self._read() as con:
            row = con.execute(
                "SELECT * FROM decisions WHERE decision_id = ?", (str(decision_id),)
            ).fetchone()
            if row is None:
                return None
            return self._decision_item(con, row)

    def retrieve(
        self,
        *,
        caller: str,
        symbol: str,
        as_of: str,
        n_same: int = 5,
        m_cross: int = 3,
        pack: Optional[str] = None,
    ) -> dict:
        """点内检索：只回**结算日 ≤ `as_of`** 的记录，按结算日倒序。

        只回**本人（`caller`）**的记录——账本按调用者隔离，一个调用者的判定对另一个
        调用者不可见（隔离落在 SQL 的 WHERE 里，不是应用层过滤）。
        同标的最多 `n_same` 条、跨标的教训最多 `m_cross` 条（默认 5 / 3，可配）。
        `as_of` 之后才结算的记录不出现在结果里——点内过滤落在 SQL 层，不依赖调用方
        自觉（design.md D5：这正是蓝本到 v0.6.0 才修的 #1251 号 bug）。
        空账本 / 无匹配 → 两个空数组，不报错。
        """
        a = check_iso_date(as_of, name="as_of")
        sym = self._require_text(symbol, name="symbol")
        n = self._require_limit(n_same, name="n_same")
        m = self._require_limit(m_cross, name="m_cross")
        pack_f = None if pack is None else self._require_text(pack, name="pack")

        where = ["s.settlement_date <= ?", "d.caller = ?"]
        args: list[Any] = [a, caller]
        if pack_f is not None:
            where.append("d.pack = ?")
            args.append(pack_f)
        sql = (
            "SELECT d.*, s.settled_at AS s_settled_at, s.settlement_date AS s_settlement_date, "
            "s.entry_price AS s_entry_price, s.exit_price AS s_exit_price, "
            "s.raw_return AS s_raw_return, s.benchmark_symbol AS s_benchmark_symbol, "
            "s.benchmark_mapping AS s_benchmark_mapping, s.benchmark_return AS s_benchmark_return, "
            "s.alpha AS s_alpha, s.holding_days AS s_holding_days, s.reflection AS s_reflection, "
            "s.rule_id AS s_rule_id, s.verdict AS s_verdict, s.metric_ref AS s_metric_ref, "
            "s.observed_value AS s_observed_value, s.comparator AS s_comparator, "
            "s.threshold AS s_threshold, s.series_fingerprint AS s_series_fingerprint "
            "FROM decisions d JOIN settlements s ON s.decision_id = d.decision_id "
            f"WHERE {' AND '.join(where)} "
        )
        order = "ORDER BY s.settlement_date DESC, d.created_at DESC, d.decision_id DESC LIMIT ?"
        with self._read() as con:
            same_rows = con.execute(
                sql + f"AND d.symbol = ? {order}", (*args, sym, n)
            ).fetchall()
            cross_rows = con.execute(
                sql + f"AND d.symbol <> ? {order}", (*args, sym, m)
            ).fetchall()
            same = [self._decision_item(con, r) for r in same_rows]
            cross = [self._decision_item(con, r) for r in cross_rows]
        return {
            "as_of": a,
            "symbol": sym,
            "n_same": n,
            "m_cross": m,
            "pack": pack_f,
            "same_symbol": same,
            "cross_symbol": cross,
            "counts": {"same_symbol": len(same), "cross_symbol": len(cross),
                       "total": len(same) + len(cross)},
        }

    def stats_by_pack(
        self,
        *,
        caller: str,
        as_of: str,
        packs: Optional[Sequence[str]] = None,
        group_by_skill: bool = False,
        group_by_holding: bool = False,
        holding_days: Optional[Sequence[int]] = None,
    ) -> dict:
        """按 pack 的结算统计：样本数 / 平均 alpha / 胜率 / 原始收益均值。

        - **只统计 `settled` 且结算日 ≤ `as_of`** 的记录；未结算的决策只贡献"组存在"。
        - 组集合 = 已注册 pack（decisions 表里出现过的）∪ 显式请求的 `packs`；
          **零样本的组显式返回 0 行**（`samples=0`，比值为 null），不省略——
          "这个 pack 还没结算"与"这个 pack 不存在"含义完全不同（design.md D7）。
        - `group_by_skill=True`：组细化为 (pack, skill)，skill 取自已落记录的 producer。
        - `group_by_holding=True`：再按持有期分档，档位取 `holding_days`
          （默认只有本库配置的持有期），故 3/5/10 日可并列比较。
        """
        a = check_iso_date(as_of, name="as_of")
        requested = None if packs is None else [
            self._require_text(p, name="packs[]") for p in packs
        ]
        hd_list: Optional[list[int]] = None
        if group_by_holding:
            hd_list = [check_holding_days(h) for h in (
                [self.holding_days] if holding_days is None else list(holding_days)
            )]

        with self._read() as con:
            keys = self._group_keys(con, requested=requested, group_by_skill=group_by_skill,
                                     caller=caller)
            rows = self._settled_rows(con, as_of=a, caller=caller)

        groups: list[dict] = []
        for pack_name, skill_name, kind in keys:
            base = {"pack": pack_name, "skill": skill_name, "record_kind": kind}
            if hd_list is None:
                groups.append(self._stats_row(base, rows, holding=None))
            else:
                for h in hd_list:
                    groups.append(self._stats_row(base, rows, holding=h))

        return {
            "as_of": a,
            "only_settled": True,
            # record_kind 恒在分组维度里：组合 alpha 与单票 alpha 混入同一个胜率，
            # 那个胜率就没有主语了（spec: 聚合不得混两类）。
            "group_by": (["pack", "skill"] if group_by_skill else ["pack"])
            + ["record_kind"]
            + (["holding_days"] if group_by_holding else []),
            "holding_days": hd_list,
            "registered_packs": sorted({k[0] for k in keys}),
            "rows": groups,
            "counts": {"rows": len(groups), "settled_rows_in_scope": len(rows)},
        }

    def list_pending(
        self,
        *,
        caller: str,
        pack: Optional[str] = None,
        record_kind: Optional[str] = None,
        as_of_from: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
        holding_days: Optional[int] = None,
        today: Optional[str] = None,
    ) -> dict:
        """列出**待结算**记录（分页 + 总数），供定时复盘使用。

        ## 为什么必须要这个入口

        `retrieve()` 只回**已结算且结算日 ≤ as_of** 的记录（点内纪律），所以它天然
        看不到待结算清单。此前做到期复盘只能靠"落账时自己记住 decision_id"——对定时
        任务等于没有。没有列举入口，复盘岗结构性做不了。

        ## 未到期 ≠ 不存在

        未到期的记录**照样在清单里**，带 `due` 与 `missing_trading_days` 标注。把它们
        隐去会让"这个账本没东西要结"与"有 12 条还没到期"变成同一个输出——而这两件事
        的处置完全不同（前者收工，后者等）。

        到期日按交易日推定：`due` = `as_of` + `holding_days` 个**自然日**的粗估，
        精确交易日以结算时传入序列的日期集合为准（账本不内置日历）；故这里给的是
        `due_estimate` 而不是承诺值。
        """
        lim = self._require_limit(limit, name="limit")
        off = self._require_limit(offset, name="offset")
        pk = self._require_text(pack, name="pack") if pack is not None else None
        kind = None
        if record_kind is not None:
            kind = record_kind.strip()
            if kind not in ("single", "portfolio"):
                raise LedgerError(
                    f"record_kind={record_kind!r} must be 'single' or 'portfolio'"
                )
        since = check_iso_date(as_of_from, name="as_of_from") if as_of_from else None
        hd = check_holding_days(self.holding_days if holding_days is None else holding_days)
        ref_day = check_iso_date(today, name="today") if today else None

        where = ["caller = ?", "status = 'pending'"]
        params: list[Any] = [caller]
        if pk is not None:
            where.append("pack = ?")
            params.append(pk)
        if kind is not None:
            where.append("record_kind = ?")
            params.append(kind)
        if since is not None:
            where.append("as_of >= ?")
            params.append(since)
        clause = " AND ".join(where)

        with self._read() as con:
            total = con.execute(
                f"SELECT COUNT(*) AS n FROM decisions WHERE {clause}", params
            ).fetchone()["n"]
            rows = con.execute(
                f"SELECT * FROM decisions WHERE {clause} "
                "ORDER BY as_of ASC, created_at ASC, decision_id ASC LIMIT ? OFFSET ?",
                [*params, lim, off],
            ).fetchall()

            items = []
            for row in rows:
                entry = self._decision_item(con, row)
                due = self._due_estimate(row["as_of"], hd)
                entry["due_estimate"] = due
                entry["holding_days"] = hd
                entry["matured"] = bool(ref_day and due and due <= ref_day)
                if ref_day and due and not entry["matured"]:
                    gap = (_date.fromisoformat(due) - _date.fromisoformat(ref_day)).days
                    entry["missing_trading_days_estimate"] = max(0, gap)
                else:
                    entry["missing_trading_days_estimate"] = 0
                items.append(entry)

        return {
            "caller": caller,
            "filters": {"pack": pk, "record_kind": kind, "as_of_from": since},
            "pagination": {"limit": lim, "offset": off, "total": int(total),
                           "returned": len(items)},
            "holding_days": hd,
            "today": ref_day,
            "items": items,
        }

    @staticmethod
    def _due_estimate(as_of: str, holding_days: int) -> str:
        """到期日的粗估（自然日）。精确值取决于调用方结算窗的交易日集合。"""
        return (_date.fromisoformat(as_of) + timedelta(days=holding_days)).isoformat()

    def record_risk_thresholds(
        self,
        *,
        caller: str,
        thresholds: Mapping[str, Any],
        reason: str,
        scope: str = "default",
        now: Optional[str] = None,
    ) -> dict:
        """追加一条阈值变更记录。返回本条 + 上一条（便于核对旧值）。

        阈值必须四项齐全才允许落账：只写一半的阈值表，读出来就是"不可用"，
        而那与"从没配过"在运维上是两件事。
        """
        from ..risk import THRESHOLD_KEYS

        missing = [k for k in THRESHOLD_KEYS if (thresholds or {}).get(k) is None]
        if missing:
            raise LedgerError(
                f"thresholds missing {missing}; a partial threshold set reads as "
                "unavailable, which is indistinguishable from never configured"
            )
        why = self._require_text(reason, name="reason")
        sc = self._require_text(scope, name="scope")
        stamp = now if now is not None else _utc_now()
        payload = _json_dumps({k: thresholds[k] for k in THRESHOLD_KEYS
                               if k in thresholds}
                              | {k: thresholds[k] for k in thresholds
                                 if k not in THRESHOLD_KEYS})

        with self._write_tx() as con:
            prev = con.execute(
                "SELECT * FROM risk_thresholds WHERE caller = ? AND scope = ? "
                "ORDER BY id DESC LIMIT 1", (caller, sc)
            ).fetchone()
            con.execute(
                "INSERT INTO risk_thresholds (caller, scope, thresholds, reason, recorded_at)"
                " VALUES (?, ?, ?, ?, ?)", (caller, sc, payload, why, stamp),
            )
            row = con.execute(
                "SELECT * FROM risk_thresholds WHERE caller = ? AND scope = ? "
                "ORDER BY id DESC LIMIT 1", (caller, sc)
            ).fetchone()
            return {
                "caller": caller,
                "scope": sc,
                "current": {"thresholds": _json_loads(row["thresholds"]),
                            "reason": row["reason"], "recorded_at": row["recorded_at"],
                            "id": int(row["id"])},
                "previous": (
                    None if prev is None else
                    {"thresholds": _json_loads(prev["thresholds"]),
                     "reason": prev["reason"], "recorded_at": prev["recorded_at"],
                     "id": int(prev["id"])}
                ),
                "counts": self._counts(con),
            }

    def latest_risk_thresholds(self, *, caller: str,
                               scope: str = "default") -> Optional[dict]:
        """当前生效阈值（最新一条）。没有记录时返回 None——**不返回空默认值**。"""
        sc = self._require_text(scope, name="scope")
        with self._read() as con:
            row = con.execute(
                "SELECT * FROM risk_thresholds WHERE caller = ? AND scope = ? "
                "ORDER BY id DESC LIMIT 1", (caller, sc)
            ).fetchone()
        if row is None:
            return None
        return {"thresholds": _json_loads(row["thresholds"]), "reason": row["reason"],
                "recorded_at": row["recorded_at"], "id": int(row["id"]), "scope": sc}

    def row_counts(self) -> dict:
        """行数快照——幂等断言"行数不变"时可核对的显式事实。"""
        with self._read() as con:
            return self._counts(con)

    # ── 内部：行读取 ──────────────────────────────────────────

    def _initial_row(self, con: sqlite3.Connection, *, caller: str, record_kind: str,
                     as_of: str, subject: str, symbol: Optional[str] = None,
                     portfolio_key: Optional[str] = None, pack: str, skill: str,
                     series_ref_json: Any = None,
                     ) -> Optional[sqlite3.Row]:
        """按幂等键取初始行。主体用 `subject` 统一寻址——单体走 symbol，组合走 portfolio_key。"""
        return con.execute(
            "SELECT * FROM decisions WHERE caller = ? AND record_kind = ? AND as_of = ? "
            "AND COALESCE(symbol, portfolio_key) = ? AND pack = ? AND skill = ? "
            "AND revises IS NULL",
            (caller, record_kind, as_of, subject, pack, skill),
        ).fetchone()

    def _tip_row(self, con: sqlite3.Connection, *, caller: str, record_kind: str,
                 as_of: str, subject: str, symbol: Optional[str] = None,
                 portfolio_key: Optional[str] = None, pack: str, skill: str,
                 series_ref_json: Any = None,
                 ) -> sqlite3.Row:
        row = con.execute(
            "SELECT * FROM decisions WHERE caller = ? AND record_kind = ? AND as_of = ? "
            "AND COALESCE(symbol, portfolio_key) = ? AND pack = ? AND skill = ? "
            "ORDER BY revision_seq DESC LIMIT 1",
            (caller, record_kind, as_of, subject, pack, skill),
        ).fetchone()
        if row is None:
            raise UnknownDecision(
                f"no decision with key (caller={caller}, record_kind={record_kind}, "
                f"as_of={as_of}, subject={subject}, pack={pack}, skill={skill}) in the ledger"
            )
        return row

    def _settlement_row(self, con: sqlite3.Connection,
                        decision_id: str) -> Optional[sqlite3.Row]:
        return con.execute(
            "SELECT * FROM settlements WHERE decision_id = ?", (decision_id,)
        ).fetchone()

    def _counts(self, con: sqlite3.Connection) -> dict:
        decisions = con.execute("SELECT COUNT(*) AS n FROM decisions").fetchone()["n"]
        settlements = con.execute("SELECT COUNT(*) AS n FROM settlements").fetchone()["n"]
        pending = con.execute(
            "SELECT COUNT(*) AS n FROM decisions WHERE status = 'pending'"
        ).fetchone()["n"]
        settled = con.execute(
            "SELECT COUNT(*) AS n FROM decisions WHERE status = 'settled'"
        ).fetchone()["n"]
        return {"decisions": int(decisions), "settlements": int(settlements),
                "pending": int(pending), "settled": int(settled)}

    def _decision_item(self, con: sqlite3.Connection, row: Any) -> dict:
        """记录 → 返回项：decision 快照（payload / evidence / 修订指针）+ 结算结果。"""
        has_settlement = "s_settlement_date" in row.keys()
        if not has_settlement and row["status"] == "settled":
            srow = self._settlement_row(con, row["decision_id"])
            if srow is not None:
                row = _merge_settlement(row, srow)
                has_settlement = True
        settlement = None
        if has_settlement:
            settlement = {
                "settled_at": row["s_settled_at"],
                "settlement_date": row["s_settlement_date"],
                "rule_id": row["s_rule_id"],
                "verdict": row["s_verdict"],
                "entry_price": _clean_float_or_none(row["s_entry_price"], name="entry_price"),
                "exit_price": _clean_float_or_none(row["s_exit_price"], name="exit_price"),
                "raw_return": _clean_float_or_none(row["s_raw_return"], name="raw_return"),
                "benchmark_symbol": row["s_benchmark_symbol"],
                "benchmark_mapping": row["s_benchmark_mapping"],
                "benchmark_return": _clean_float_or_none(row["s_benchmark_return"],
                                                         name="benchmark_return"),
                "alpha": _clean_float_or_none(row["s_alpha"], name="alpha"),
                "holding_days": (int(row["s_holding_days"])
                                 if row["s_holding_days"] is not None else None),
                "metric_ref": row["s_metric_ref"],
                "observed_value": _clean_float_or_none(row["s_observed_value"],
                                                       name="observed_value"),
                "comparator": row["s_comparator"],
                "threshold": _clean_float_or_none(row["s_threshold"], name="threshold"),
                "series_fingerprint": row["s_series_fingerprint"],
                "reflection": row["s_reflection"],
            }
        return {
            "decision_id": row["decision_id"],
            "as_of": row["as_of"],
            # 主体按种类二选一：单体看 symbol，组合看 portfolio_key。复盘岗必须知道
            # 该用哪个结算入口，所以种类与主体都要出现在条目里。
            "record_kind": row["record_kind"],
            "subject": row["symbol"] if row["record_kind"] == "single" else row["portfolio_key"],
            "symbol": row["symbol"],
            "portfolio_key": row["portfolio_key"],
            "pack": row["pack"],
            "skill": row["skill"],
            "status": row["status"],
            "revises": row["revises"],
            "revision_seq": int(row["revision_seq"]),
            "created_at": row["created_at"],
            "payload": _json_loads(row["payload"]),
            "target_weights": _json_loads(row["target_weights"]),
            # 取价口径出处：结算与复盘要能回答"这条判定用的是哪批价格"。
            "series_ref": _json_loads(row["series_ref"]) if "series_ref" in row.keys() else None,
            "evidence": _json_loads(row["evidence"]),
            "settlement": settlement,
        }

    def _result(self, con: sqlite3.Connection, decision_id: str, *, created: bool,
                reason: str, revises_override: Optional[str] = None) -> dict:
        row = con.execute(
            "SELECT * FROM decisions WHERE decision_id = ?", (decision_id,)
        ).fetchone()
        if row is None:  # 理论不可达：刚插入或刚查到的行
            raise LedgerError(f"decision {decision_id!r} vanished while reading it back")
        return {
            "created": created,
            "reason": reason,
            "decision_id": decision_id,
            "as_of": row["as_of"],
            "symbol": row["symbol"],
            "pack": row["pack"],
            "skill": row["skill"],
            "status": row["status"],
            "revises": row["revises"] if revises_override is None else revises_override,
            "revision_seq": int(row["revision_seq"]),
            "created_at": row["created_at"],
            "counts": self._counts(con),
        }

    def _settlement_result(self, con: sqlite3.Connection, row: sqlite3.Row, *,
                           created: bool, reason: str, reflection_ignored: bool) -> dict:
        decision = con.execute(
            "SELECT as_of, symbol, pack, skill, status FROM decisions WHERE decision_id = ?",
            (row["decision_id"],),
        ).fetchone()
        return {
            "created": created,
            "reason": reason,
            "decision_id": row["decision_id"],
            "decision_as_of": decision["as_of"],
            "symbol": decision["symbol"],
            "pack": decision["pack"],
            "skill": decision["skill"],
            "status": decision["status"],
            "settled_at": row["settled_at"],
            "settlement_date": row["settlement_date"],
            "entry_price": _clean_float_or_none(row["entry_price"], name="entry_price"),
            "exit_price": _clean_float_or_none(row["exit_price"], name="exit_price"),
            "raw_return": _clean_float_or_none(row["raw_return"], name="raw_return"),
            "benchmark_symbol": row["benchmark_symbol"],
            "benchmark_mapping": row["benchmark_mapping"],
            "benchmark_entry_price": _clean_float_or_none(row["benchmark_entry_price"],
                                                  name="benchmark_entry_price"),
            "benchmark_exit_price": _clean_float_or_none(row["benchmark_exit_price"],
                                                 name="benchmark_exit_price"),
            "benchmark_return": _clean_float_or_none(row["benchmark_return"], name="benchmark_return"),
            "rule_id": row["rule_id"],
            "verdict": row["verdict"],
            "alpha": _clean_float_or_none(row["alpha"], name="alpha"),
            "holding_days": int(row["holding_days"]) if row["holding_days"] is not None else None,
            "metric_ref": row["metric_ref"],
            "observed_value": _clean_float_or_none(row["observed_value"],
                                                  name="observed_value"),
            "comparator": row["comparator"],
            "threshold": _clean_float_or_none(row["threshold"], name="threshold"),
            "series_fingerprint": row["series_fingerprint"],
            "reflection": row["reflection"],
            "reflection_ignored": bool(reflection_ignored),
            "counts": self._counts(con),
        }

    # ── 内部：聚合 ────────────────────────────────────────────

    def _group_keys(self, con: sqlite3.Connection, *, requested: Optional[list[str]],
                    group_by_skill: bool,
                    caller: str) -> list[tuple[str, Optional[str], str]]:
        """组集合 = 已注册（decisions 表里出现过的）∪ 显式请求。

        显式请求一个**从未落过记录**的 pack 时给一行 (pack, None) 且样本数 0——这正是
        spec "空组明确返回零"的场景。skill 分组下 skill 只能来自已落记录（账本不臆造
        skill 名）；未注册 pack 以 skill=None 出现。
        """
        if group_by_skill:
            rows = con.execute(
                "SELECT DISTINCT pack, skill, record_kind FROM decisions WHERE caller = ? "
                "ORDER BY pack, skill, record_kind",
                (caller,),
            ).fetchall()
            keys = [(r["pack"], r["skill"], r["record_kind"]) for r in rows]
        else:
            rows = con.execute(
                "SELECT DISTINCT pack, record_kind FROM decisions WHERE caller = ? "
                "ORDER BY pack, record_kind",
                (caller,),
            ).fetchall()
            keys = [(r["pack"], None, r["record_kind"]) for r in rows]
        if requested:
            # 显式请求但从未落过记录的 pack：按两种记录种类各补一行零样本，
            # 而不是只补一行——"这个 pack 的组合记录还没样本"与"这个 pack 不存在"
            # 是两件事，合并成一行会把前者抹掉。
            known = {(k[0], k[2]) for k in keys}
            # 只对本调用者的账本里**实际出现过**的记录种类补零：纯单体账本的输出
            # 因此与引入 record_kind 之前逐行相同（不为一个从未有组合记录的账本
            # 凭空造出一行组合零样本）。账本为空时退回 'single'。
            observed = sorted({k[2] for k in keys}) or ["single"]
            for p in requested:
                for kind in observed:
                    if (p, kind) not in known:
                        keys.append((p, None, kind))
        seen: set[tuple[str, Optional[str], str]] = set()
        out: list[tuple[str, Optional[str], str]] = []
        for k in keys:
            if k not in seen:
                seen.add(k)
                out.append(k)
        return sorted(out, key=lambda k: (k[0], k[1] or "", k[2]))

    def _settled_rows(self, con: sqlite3.Connection, *, as_of: str,
                      caller: str) -> list[dict]:
        """统计口径内的全部已结算行（结算日 ≤ as_of）。

        **刻意不在 SQL 层按 `packs` 过滤**：`packs` 参数决定的是"要出现哪些组"，
        不是"丢掉了别的组的样本"——若在这里过滤，显式请求 `pack A` 会让已注册的
        pack B 显示 samples=0（把"没查"伪装成"没样本"）。组过滤在 `_stats_row` 里做。
        """
        # 两类结算各在自己的表里，统计口径必须都取——否则组合记录的组会出现
        # samples=0（"有这个组但没样本"），把"类别存在"误报成"类别没样本"。
        # rule_id / verdict 必须一起取出来：阈值口径的结算**没有 alpha**，把它和
        # 价格口径混在同一个胜率里，那个胜率就没有主语了。portfolio_settlements 是
        # 独立的表且恒为价格口径，故补常量 'alpha_vs_benchmark'（它没有 verdict 概念）。
        rows = con.execute(
            "SELECT d.pack, d.skill, d.record_kind, s.holding_days, s.raw_return, s.alpha, "
            "s.rule_id AS rule_id, s.verdict AS verdict "
            "FROM decisions d JOIN settlements s ON s.decision_id = d.decision_id "
            "WHERE d.status = 'settled' AND s.settlement_date <= ? AND d.caller = ? "
            "UNION ALL "
            "SELECT d.pack, d.skill, d.record_kind, p.holding_days, p.raw_return, p.alpha, "
            "'alpha_vs_benchmark' AS rule_id, NULL AS verdict "
            "FROM decisions d JOIN portfolio_settlements p ON p.decision_id = d.decision_id "
            "WHERE d.status = 'settled' AND p.settlement_date <= ? AND d.caller = ?",
            (as_of, caller, as_of, caller),
        ).fetchall()
        return [
            {
                "pack": r["pack"],
                "skill": r["skill"],
                "record_kind": r["record_kind"],
                # 阈值口径的结算没有持有期概念（它的窗口是观察窗，不是持有期）。
                "holding_days": (int(r["holding_days"])
                                 if r["holding_days"] is not None else None),
                "raw_return": _clean_float_or_none(r["raw_return"], name="raw_return"),
                "alpha": _clean_float_or_none(r["alpha"], name="alpha"),
                "rule_id": r["rule_id"],
                "verdict": r["verdict"],
            }
            for r in rows
        ]

    @staticmethod
    def _rule_breakdown(rows: list[dict]) -> dict:
        """按结算口径分开计数：两种口径的问法不同，不能合成一个数。

        价格口径回答"跑赢基准多少"；阈值口径回答"指标有没有达到阈值"。后者的
        `insufficient_data` **既不是命中也不是未命中**，必须单列——把它算进 miss
        就等于把"不知道"记成"判断错了"。
        """
        out: dict[str, dict] = {}
        for rule in sorted({r.get("rule_id") or "alpha_vs_benchmark" for r in rows}):
            sel = [r for r in rows if (r.get("rule_id") or "alpha_vs_benchmark") == rule]
            if rule == "metric_threshold":
                verdicts = [r.get("verdict") for r in sel]
                hits = sum(1 for v in verdicts if v == "hit")
                misses = sum(1 for v in verdicts if v == "miss")
                insufficient = sum(1 for v in verdicts if v == "insufficient_data")
                decided = hits + misses
                out[rule] = {
                    "samples": len(sel),
                    "hits": hits,
                    "misses": misses,
                    "insufficient_data": insufficient,
                    # 命中率的分母是"有结论的那些"，不含 insufficient_data。
                    "hit_rate": (hits / decided) if decided else None,
                }
            else:
                alphas = [r["alpha"] for r in sel if r["alpha"] is not None]
                wins = sum(1 for a in alphas if a > 0)
                out[rule] = {
                    "samples": len(sel),
                    "alpha_samples": len(alphas),
                    "wins": wins,
                    "mean_alpha": _mean(alphas) if alphas else None,
                    "win_rate": (wins / len(alphas)) if alphas else None,
                }
        return out

    @staticmethod
    def _stats_row(base: dict, rows: list[dict], *, holding: Optional[int]) -> dict:
        sel = [
            r for r in rows
            if r["pack"] == base["pack"]
            and r["record_kind"] == base["record_kind"]
            and (base["skill"] is None or r["skill"] == base["skill"])
            and (holding is None or r["holding_days"] == holding)
        ]
        out = dict(base)
        if holding is not None:
            out["holding_days"] = holding
        out["samples"] = len(sel)
        if not sel:
            # 空组：显式 0 + null 比值（"没有样本"既不是"不存在"，也不是"胜率 0"）
            out.update({"wins": 0, "alpha_samples": 0, "mean_alpha": None,
                        "win_rate": None, "mean_raw_return": None, "by_rule": {}})
            return out

        # 只有 alpha 口径的行才能进"胜率 / 平均 alpha"。阈值口径的结算没有 alpha，
        # 把它的 NULL 当成 0 参与均值，等于把"另一种问法"记成"跑输基准"——胜率会被
        # 系统性拉低，而且再也无法从输出里看出是哪种口径造成的。
        alpha_rows = [r for r in sel if r["alpha"] is not None]
        alphas = [r["alpha"] for r in alpha_rows]
        raws = [r["raw_return"] for r in alpha_rows]
        wins = sum(1 for a in alphas if a > 0)
        out.update({
            "wins": wins,
            # 显式给出 alpha 口径的样本数：否则 win_rate 的分母是谁无从判断，
            # 而"3 个样本里赢 2 个"与"6 个里赢 2 个"读起来一样。
            "alpha_samples": len(alpha_rows),
            "mean_alpha": _mean(alphas) if alphas else None,
            "win_rate": (wins / len(alphas)) if alphas else None,
            "mean_raw_return": _mean(raws) if raws else None,
            "by_rule": LedgerStore._rule_breakdown(sel),
        })
        return out

    # ── 内部：入参校验 ────────────────────────────────────────

    @staticmethod
    def _require_text(value: Any, *, name: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise LedgerError(f"{name} must be a non-empty string, got {value!r}")
        return value.strip()

    @staticmethod
    def _require_limit(value: Any, *, name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise LedgerError(f"{name} must be a non-negative integer, got {value!r}")
        return int(value)

    @staticmethod
    def _check_payload(payload: Any, *, as_of: str, symbol: str, pack: str,
                       skill: str, record_kind: str = "single") -> dict:
        """`payload` 必须是合法契约实例，且与记录键逐项一致（不一致即拒绝并点名两边）。

        `record_kind='single'` → `Rating`（键维度 `as_of` / `subject`）；
        `record_kind='portfolio'` → `PortfolioIntent`（无 `subject` 字段，键维度只有 `as_of`）。
        两类都要 `producer.pack/skill` 与记录键一致——产出方可追溯是两类记录的共同要求。
        """
        if hasattr(payload, "to_dict"):
            payload = payload.to_dict()
        # v2 决策（有 subject_kind）走 Decision 校验；其余仍按 Rating/PortfolioIntent。
        # 判据用字段存在性而不是 schema_version：v2 要求的正是"主体种类 + 动作"这一对，
        # 缺了它就不是 v2，哪怕版本号写着 2。
        if isinstance(payload, dict) and "subject_kind" in payload:
            kind = "decision"
        else:
            kind = "portfolio_intent" if record_kind == "portfolio" else "rating"
        result = validate(payload, kind=kind)
        if not result["valid"]:
            first = result["errors"][0]
            raise ContractError(first["path"], first["reason"])
        id_fields = (("as_of", as_of),) if record_kind == "portfolio" else (
            ("as_of", as_of), ("subject", symbol)
        )
        # v2 的 subject 同样必须等于记录键里的主体，否则"记录说的是门店 A、payload 说的是
        # 门店 B"这种错配会静默通过，日后按 subject 聚合时才发现对不上。
        for field, given in id_fields:
            if payload[field] != given:
                raise ContractError(
                    field,
                    f"payload.{field}={payload[field]!r} disagrees with the ledger's "
                    f"{field}={given!r}; a record's key dimensions must equal the contract "
                    f"instance's own values (refused, not silently resolved)",
                )
        producer = payload["producer"]
        for field, given in (("pack", pack), ("skill", skill)):
            if producer[field] != given:
                raise ContractError(
                    f"producer.{field}",
                    f"payload.producer.{field}={producer[field]!r} disagrees with the "
                    f"ledger's {field}={given!r}",
                )
        return payload

    @staticmethod
    def _check_series_ref(series_ref: Any) -> Optional[str]:
        """校验并序列化取价路径的 `PriceSeriesRef`（可选）。

        为什么在账本里校验而不是信调用方：口径是**账本的记录维度**，一旦落进库就要能被
        结算时比对。不合法就当场拒绝，不留一个"看起来有 ref 其实缺字段"的行。
        """
        if series_ref is None:
            return None
        from ..contracts.models import check_price_series_ref

        payload = series_ref.to_dict() if hasattr(series_ref, "to_dict") else series_ref
        errs = check_price_series_ref(payload)
        if errs:
            raise ContractError(errs[0]["path"], errs[0]["reason"])
        return _json_dumps(payload)

    @staticmethod
    def _assert_caliber(row: Any, series_ref: Any) -> dict:
        """比对落账 ref 与结算 ref；按 `PRICE_CALIBER_ENFORCEMENT` 决定 warn 还是 refuse。

        三态结论（这是本方法的全部意义）：
          - `comparable: true`  → 两侧同源同口径，数字可直接比
          - `comparable: false` → 两侧不同源/不同口径，或仅一侧有 ref → **不可比**，
            `warn` 下产出告警、`refuse` 下抛错
          - `checked: false`    → 两侧都没有 ref（历史记录），既不告警也不判定为可比

        **为什么"仅一侧有 ref"也算不可比**：只有一侧能说出自己的源与锚点，就没有办法
        断言两边是同一批数据；把它当"可比"等于默认缺失的那侧与另一侧同源——那是本仓
        最忌讳的那类推断（读不到当成不存在）。

        **为什么不把 `warn` 当成足够**：设计里 `warn` 只是迁移期的观察窗；收敛完成后必须
        切 `refuse`，否则开关会变成永久的静默降级口。探针（task 6.3）会断言该值。
        """
        import os

        mode = (os.environ.get("PRICE_CALIBER_ENFORCEMENT") or "warn").strip().lower()
        if mode not in ("warn", "refuse"):
            raise LedgerError(
                f"PRICE_CALIBER_ENFORCEMENT={mode!r} must be 'warn' or 'refuse'"
            )

        recorded = _json_loads(row["series_ref"]) if "series_ref" in row.keys() else None
        incoming = (series_ref.to_dict() if hasattr(series_ref, "to_dict") else series_ref)

        if recorded is None and incoming is None:
            return {"checked": False, "comparable": None, "mode": mode,
                    "reason": "neither side carries a series_ref (legacy record)"}

        if recorded is None or incoming is None:
            missing = "record" if recorded is None else "settlement"
            msg = (f"series_ref present only on the {('settlement' if missing == 'record' else 'record')} "
                   f"side; cannot assert the two price series share one caliber")
            if mode == "refuse":
                raise SettlementRefused(msg)
            return {"checked": True, "comparable": False, "mode": mode,
                    "reason": msg, "warning": True}

        diffs = [k for k in ("source_id", "adjustment", "adjustment_anchor", "as_of_max")
                 if (recorded or {}).get(k) != (incoming or {}).get(k)]
        if not diffs:
            return {"checked": True, "comparable": True, "mode": mode,
                    "source_id": (recorded or {}).get("source_id")}

        detail = {k: {"recorded": (recorded or {}).get(k), "settlement": (incoming or {}).get(k)}
                  for k in diffs}
        msg = f"caliber mismatch on {', '.join(diffs)}; results are not comparable"
        if mode == "refuse":
            raise SettlementRefused(msg + f" ({detail})")
        return {"checked": True, "comparable": False, "mode": mode, "reason": msg,
                "difference": detail, "warning": True}

    @staticmethod
    def _check_target_weights(target_weights: Any) -> Optional[str]:
        if target_weights is None:
            return None
        if hasattr(target_weights, "to_dict"):
            target_weights = target_weights.to_dict()
        result = validate(target_weights, kind="target_weights")
        if not result["valid"]:
            first = result["errors"][0]
            raise ContractError(first["path"], first["reason"])
        return _json_dumps(target_weights)

    @staticmethod
    def _check_evidence(evidence: Optional[Sequence[str]], *, payload: Any) -> list[str]:
        """引用数组：缺省取 `payload.target_evidence_ids`（判定自带的引用来源）。"""
        if evidence is None:
            refs = payload.get("target_evidence_ids") if isinstance(payload, Mapping) else []
            return [str(r) for r in (refs or [])]
        if isinstance(evidence, (str, bytes)) or not isinstance(evidence, Sequence):
            raise LedgerError(
                f"evidence must be an array of reference strings, got {evidence!r}"
            )
        out: list[str] = []
        for i, r in enumerate(evidence):
            if not isinstance(r, str) or not r.strip():
                raise LedgerError(f"evidence[{i}] must be a non-empty string, got {r!r}")
            out.append(r.strip())
        return out
