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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Optional, Sequence

from ..contracts.models import ContractError, validate
from ..caller import LEGACY
from .errors import LedgerError, RevisionConflict, UnknownDecision
from .settlement import (
    DEFAULT_BENCHMARK_MAP,
    DEFAULT_HOLDING_DAYS,
    check_holding_days,
    check_iso_date,
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
        as_of           TEXT NOT NULL,
        symbol          TEXT NOT NULL,
        pack            TEXT NOT NULL,
        skill           TEXT NOT NULL,
        payload         TEXT NOT NULL,
        target_weights  TEXT,
        evidence        TEXT NOT NULL,
        status          TEXT NOT NULL CHECK (status IN ('pending', 'settled')),
        revises         TEXT REFERENCES decisions(decision_id),
        revision_seq    INTEGER NOT NULL CHECK (revision_seq >= 0),
        created_at      TEXT NOT NULL,
        CHECK (
            (revises IS NULL AND revision_seq = 0)
            OR (revises IS NOT NULL AND revision_seq > 0)
        ),
        CHECK (revises IS NULL OR revises <> decision_id)
    )
    """,
    # 初始记录：幂等键全表唯一（重复提交的最终屏障）
    """
    CREATE UNIQUE INDEX IF NOT EXISTS ux_decisions_initial
        ON decisions (caller, as_of, symbol, pack, skill) WHERE revises IS NULL
    """,
    # 修订：同键按递增序号唯一，线性链不可能出现兄弟分叉
    """
    CREATE UNIQUE INDEX IF NOT EXISTS ux_decisions_revision
        ON decisions (caller, as_of, symbol, pack, skill, revision_seq) WHERE revises IS NOT NULL
    """,
    "CREATE INDEX IF NOT EXISTS ix_decisions_symbol_as_of ON decisions (caller, symbol, as_of)",
    "CREATE INDEX IF NOT EXISTS ix_decisions_pack_skill ON decisions (caller, pack, skill)",
    """
    CREATE TABLE IF NOT EXISTS settlements (
        decision_id           TEXT PRIMARY KEY REFERENCES decisions(decision_id),
        settled_at            TEXT NOT NULL,
        settlement_date       TEXT NOT NULL,
        entry_price           REAL NOT NULL,
        exit_price            REAL NOT NULL,
        raw_return            REAL NOT NULL,
        benchmark_symbol      TEXT NOT NULL,
        benchmark_mapping     TEXT NOT NULL,
        benchmark_entry_price REAL NOT NULL,
        benchmark_exit_price  REAL NOT NULL,
        benchmark_return      REAL NOT NULL,
        alpha                 REAL NOT NULL,
        holding_days          INTEGER NOT NULL,
        reflection            TEXT
    )
    """,
    "CREATE INDEX IF NOT EXISTS ix_settlements_date ON settlements (settlement_date)",
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
    for key in ("settled_at", "settlement_date", "entry_price", "exit_price", "raw_return",
                "benchmark_symbol", "benchmark_mapping", "benchmark_return", "alpha",
                "holding_days", "reflection"):
        merged[f"s_{key}"] = settlement_row[key]
    return _Merged(merged)


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
            _migrate_to_caller(con)
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
        symbol: str,
        pack: str,
        skill: str,
        payload: Any,
        target_weights: Any = None,
        evidence: Optional[Sequence[str]] = None,
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
        sym = self._require_text(symbol, name="symbol")
        pk = self._require_text(pack, name="pack")
        sk = self._require_text(skill, name="skill")
        created_at = now if now is not None else _utc_now()

        rating = self._check_payload(payload, as_of=a, symbol=sym, pack=pk, skill=sk)
        tw_json = self._check_target_weights(target_weights)
        refs = self._check_evidence(evidence, payload=rating)
        payload_json = _json_dumps(rating)
        evidence_json = _json_dumps(refs)

        key = {"caller": caller, "as_of": a, "symbol": sym, "pack": pk, "skill": sk}
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
                "INSERT INTO decisions (decision_id, caller, as_of, symbol, pack, skill, payload, "
                "target_weights, evidence, status, revises, revision_seq, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', NULL, 0, ?)",
                (decision_id, key["caller"], key["as_of"], key["symbol"], key["pack"],
                 key["skill"], payload_json, target_weights_json, evidence_json, created_at),
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
            for field in ("as_of", "symbol", "pack", "skill"):
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
                "INSERT INTO decisions (decision_id, caller, as_of, symbol, pack, skill, payload, "
                "target_weights, evidence, status, revises, revision_seq, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?, ?)",
                (decision_id, key["caller"], key["as_of"], key["symbol"], key["pack"],
                 key["skill"], payload_json, target_weights_json, evidence_json,
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
            con.execute(
                "INSERT INTO settlements (decision_id, settled_at, settlement_date, entry_price, "
                "exit_price, raw_return, benchmark_symbol, benchmark_mapping, "
                "benchmark_entry_price, benchmark_exit_price, benchmark_return, alpha, "
                "holding_days, reflection) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (did, settled_at, window["exit_date"], window["entry_price"],
                 window["exit_price"], window["raw_return"], window["benchmark_symbol"],
                 bench["mapping"], window["benchmark_entry_price"],
                 window["benchmark_exit_price"], window["benchmark_return"], window["alpha"],
                 window["holding_days"], reflection),
            )
            con.execute("UPDATE decisions SET status = 'settled' WHERE decision_id = ?", (did,))
            stored = self._settlement_row(con, did)
            return self._settlement_result(con, stored, created=True, reason="settled",
                                           reflection_ignored=False)

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
            "s.alpha AS s_alpha, s.holding_days AS s_holding_days, s.reflection AS s_reflection "
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
        for pack_name, skill_name in keys:
            base = {"pack": pack_name, "skill": skill_name}
            if hd_list is None:
                groups.append(self._stats_row(base, rows, holding=None))
            else:
                for h in hd_list:
                    groups.append(self._stats_row(base, rows, holding=h))

        return {
            "as_of": a,
            "only_settled": True,
            "group_by": (["pack", "skill"] if group_by_skill else ["pack"])
            + (["holding_days"] if group_by_holding else []),
            "holding_days": hd_list,
            "registered_packs": sorted({k[0] for k in keys}),
            "rows": groups,
            "counts": {"rows": len(groups), "settled_rows_in_scope": len(rows)},
        }

    def row_counts(self) -> dict:
        """行数快照——幂等断言"行数不变"时可核对的显式事实。"""
        with self._read() as con:
            return self._counts(con)

    # ── 内部：行读取 ──────────────────────────────────────────

    def _initial_row(self, con: sqlite3.Connection, *, caller: str, as_of: str, symbol: str,
                     pack: str, skill: str) -> Optional[sqlite3.Row]:
        return con.execute(
            "SELECT * FROM decisions WHERE caller = ? AND as_of = ? AND symbol = ? "
            "AND pack = ? AND skill = ? AND revises IS NULL",
            (caller, as_of, symbol, pack, skill),
        ).fetchone()

    def _tip_row(self, con: sqlite3.Connection, *, caller: str, as_of: str, symbol: str,
                 pack: str, skill: str) -> sqlite3.Row:
        row = con.execute(
            "SELECT * FROM decisions WHERE caller = ? AND as_of = ? AND symbol = ? "
            "AND pack = ? AND skill = ? ORDER BY revision_seq DESC LIMIT 1",
            (caller, as_of, symbol, pack, skill),
        ).fetchone()
        if row is None:
            raise UnknownDecision(
                f"no decision with key (caller={caller}, as_of={as_of}, symbol={symbol}, "
                f"pack={pack}, skill={skill}) in the ledger"
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
                "entry_price": _clean_float(row["s_entry_price"], name="entry_price"),
                "exit_price": _clean_float(row["s_exit_price"], name="exit_price"),
                "raw_return": _clean_float(row["s_raw_return"], name="raw_return"),
                "benchmark_symbol": row["s_benchmark_symbol"],
                "benchmark_mapping": row["s_benchmark_mapping"],
                "benchmark_return": _clean_float(row["s_benchmark_return"],
                                                 name="benchmark_return"),
                "alpha": _clean_float(row["s_alpha"], name="alpha"),
                "holding_days": int(row["s_holding_days"]),
                "reflection": row["s_reflection"],
            }
        return {
            "decision_id": row["decision_id"],
            "as_of": row["as_of"],
            "symbol": row["symbol"],
            "pack": row["pack"],
            "skill": row["skill"],
            "status": row["status"],
            "revises": row["revises"],
            "revision_seq": int(row["revision_seq"]),
            "created_at": row["created_at"],
            "payload": _json_loads(row["payload"]),
            "target_weights": _json_loads(row["target_weights"]),
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
            "entry_price": _clean_float(row["entry_price"], name="entry_price"),
            "exit_price": _clean_float(row["exit_price"], name="exit_price"),
            "raw_return": _clean_float(row["raw_return"], name="raw_return"),
            "benchmark_symbol": row["benchmark_symbol"],
            "benchmark_mapping": row["benchmark_mapping"],
            "benchmark_entry_price": _clean_float(row["benchmark_entry_price"],
                                                  name="benchmark_entry_price"),
            "benchmark_exit_price": _clean_float(row["benchmark_exit_price"],
                                                 name="benchmark_exit_price"),
            "benchmark_return": _clean_float(row["benchmark_return"], name="benchmark_return"),
            "alpha": _clean_float(row["alpha"], name="alpha"),
            "holding_days": int(row["holding_days"]),
            "reflection": row["reflection"],
            "reflection_ignored": bool(reflection_ignored),
            "counts": self._counts(con),
        }

    # ── 内部：聚合 ────────────────────────────────────────────

    def _group_keys(self, con: sqlite3.Connection, *, requested: Optional[list[str]],
                    group_by_skill: bool,
                    caller: str) -> list[tuple[str, Optional[str]]]:
        """组集合 = 已注册（decisions 表里出现过的）∪ 显式请求。

        显式请求一个**从未落过记录**的 pack 时给一行 (pack, None) 且样本数 0——这正是
        spec "空组明确返回零"的场景。skill 分组下 skill 只能来自已落记录（账本不臆造
        skill 名）；未注册 pack 以 skill=None 出现。
        """
        if group_by_skill:
            rows = con.execute(
                "SELECT DISTINCT pack, skill FROM decisions WHERE caller = ? ORDER BY pack, skill",
                (caller,),
            ).fetchall()
            keys: list[tuple[str, Optional[str]]] = [(r["pack"], r["skill"]) for r in rows]
        else:
            rows = con.execute(
                "SELECT DISTINCT pack FROM decisions WHERE caller = ? ORDER BY pack", (caller,)
            ).fetchall()
            keys = [(r["pack"], None) for r in rows]
        if requested:
            known = {k[0] for k in keys}
            for p in requested:
                if p not in known:
                    keys.append((p, None))
        seen: set[tuple[str, Optional[str]]] = set()
        out: list[tuple[str, Optional[str]]] = []
        for k in keys:
            if k not in seen:
                seen.add(k)
                out.append(k)
        return sorted(out, key=lambda k: (k[0], k[1] or ""))

    def _settled_rows(self, con: sqlite3.Connection, *, as_of: str,
                      caller: str) -> list[dict]:
        """统计口径内的全部已结算行（结算日 ≤ as_of）。

        **刻意不在 SQL 层按 `packs` 过滤**：`packs` 参数决定的是"要出现哪些组"，
        不是"丢掉了别的组的样本"——若在这里过滤，显式请求 `pack A` 会让已注册的
        pack B 显示 samples=0（把"没查"伪装成"没样本"）。组过滤在 `_stats_row` 里做。
        """
        rows = con.execute(
            "SELECT d.pack, d.skill, s.holding_days, s.raw_return, s.alpha "
            "FROM decisions d JOIN settlements s ON s.decision_id = d.decision_id "
            "WHERE d.status = 'settled' AND s.settlement_date <= ? AND d.caller = ?",
            (as_of, caller),
        ).fetchall()
        return [
            {
                "pack": r["pack"],
                "skill": r["skill"],
                "holding_days": int(r["holding_days"]),
                "raw_return": _clean_float(r["raw_return"], name="raw_return"),
                "alpha": _clean_float(r["alpha"], name="alpha"),
            }
            for r in rows
        ]

    @staticmethod
    def _stats_row(base: dict, rows: list[dict], *, holding: Optional[int]) -> dict:
        sel = [
            r for r in rows
            if r["pack"] == base["pack"]
            and (base["skill"] is None or r["skill"] == base["skill"])
            and (holding is None or r["holding_days"] == holding)
        ]
        out = dict(base)
        if holding is not None:
            out["holding_days"] = holding
        out["samples"] = len(sel)
        if not sel:
            # 空组：显式 0 + null 比值（"没有样本"既不是"不存在"，也不是"胜率 0"）
            out.update({"wins": 0, "mean_alpha": None, "win_rate": None,
                        "mean_raw_return": None})
            return out
        alphas = [r["alpha"] for r in sel]
        raws = [r["raw_return"] for r in sel]
        wins = sum(1 for a in alphas if a > 0)
        out.update({
            "wins": wins,
            "mean_alpha": _mean(alphas),
            "win_rate": wins / len(alphas),
            "mean_raw_return": _mean(raws),
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
                       skill: str) -> dict:
        """`payload` 必须是合法 `Rating`，且与记录键逐项一致（不一致即拒绝并点名两边）。"""
        if hasattr(payload, "to_dict"):
            payload = payload.to_dict()
        result = validate(payload, kind="rating")
        if not result["valid"]:
            first = result["errors"][0]
            raise ContractError(first["path"], first["reason"])
        for field, given in (("as_of", as_of), ("subject", symbol)):
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
