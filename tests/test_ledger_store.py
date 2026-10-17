"""账本存储层：建表与唯一约束、写入原子性、幂等、修订、并发（tasks 1.1–1.5）。

每个用例的 docstring 引用它钉住的 spec Scenario / task 措辞——这份文件就是
`quant-decision-ledger` spec 的"追加式决策记录与幂等"与"存储原子性与并发安全"
两条 Requirement 的可执行形式。
"""
from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import textwrap
import threading
from pathlib import Path

import pytest

import ledgerdata as ld
from fd_quant.contracts.models import ContractError
from fd_quant.ledger import LedgerStore, RevisionConflict, UnknownDecision

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture()
def db_path(tmp_path) -> Path:
    return tmp_path / "ledger.db"


@pytest.fixture()
def store(db_path) -> LedgerStore:
    return LedgerStore(db_path)


def _record(store: LedgerStore, *, as_of: str = "2025-01-06", symbol: str = "600519.SS",
            side: str = "buy", revises=None, pack: str = ld.PACK, skill: str = ld.SKILL,
            now: str | None = None):
    return store.record(
        as_of=as_of, symbol=symbol, pack=pack, skill=skill,
        payload=ld.rating(as_of, symbol, side=side, pack=pack, skill=skill),
        revises=revises, now=now,
    )


# ── 1.1 建表与唯一约束 ──────────────────────────────────────────


def test_schema_creates_both_tables_and_rejects_a_duplicate_key_by_hand(store, db_path):
    """task 1.1：建表成功；手工插入两条同键记录时数据库拒绝第二条。

    "手工插入"是刻意的：不用 `record()` 走应用层——若唯一性只由应用层的"先查后插"
    提供，这条会过而并发下会烂。这里直接对库插，证明**约束本身**在场。
    """
    con = sqlite3.connect(db_path)
    try:
        tables = {r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )}
        assert {"decisions", "settlements"} <= tables
        cols = {r[1] for r in con.execute("PRAGMA table_info(decisions)")}
        assert {"decision_id", "as_of", "symbol", "pack", "skill", "payload", "evidence",
                "status", "revises", "created_at"} <= cols
        scol = {r[1] for r in con.execute("PRAGMA table_info(settlements)")}
        assert {"decision_id", "settled_at", "settlement_date", "entry_price", "exit_price",
                "raw_return", "benchmark_symbol", "benchmark_return", "alpha",
                "holding_days", "reflection"} <= scol

        payload = json.dumps(ld.rating("2025-01-06", "600519.SS").to_dict())
        row = ("d1", "2025-01-06", "600519.SS", ld.PACK, ld.SKILL, payload, None, "[]",
               "pending", None, 0, "2025-01-06T00:00:00+00:00")
        con.execute(
            "INSERT INTO decisions (decision_id, as_of, symbol, pack, skill, payload, "
            "target_weights, evidence, status, revises, revision_seq, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", row,
        )
        con.commit()
        with pytest.raises(sqlite3.IntegrityError):
            con.execute(
                "INSERT INTO decisions (decision_id, as_of, symbol, pack, skill, payload, "
                "target_weights, evidence, status, revises, revision_seq, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                ("d2", *row[1:]),
            )
    finally:
        con.close()


def test_second_settlement_row_is_refused_by_the_primary_key(store, db_path):
    """`settlements.decision_id` 是主键：库里不可能出现同一决策的两条结算。"""
    r = _record(store)
    _settle(store, r["decision_id"])
    con = sqlite3.connect(db_path)
    try:
        with pytest.raises(sqlite3.IntegrityError):
            con.execute(
                "INSERT INTO settlements (decision_id, settled_at, settlement_date, entry_price, "
                "exit_price, raw_return, benchmark_symbol, benchmark_mapping, "
                "benchmark_entry_price, benchmark_exit_price, benchmark_return, alpha, "
                "holding_days, reflection) VALUES (?, ?, ?, 1, 1, 0, 'X', 'explicit', 1, 1, 0, "
                "0, 5, NULL)", (r["decision_id"], "now", "2025-01-13"),
            )
    finally:
        con.close()


def test_wal_mode_and_foreign_keys_are_on(store, db_path):
    """spec: 存储原子性与并发安全——WAL + foreign_keys 是本层声明的运行配置。

    两点要分清：WAL 是**库级**属性（持久在文件里），`foreign_keys` 是**连接级**
    pragma——故对后者断言的是"账本自己开的每条连接都设了它"（`_connect` 是唯一入口），
    而不是"随便一条裸连接也有"。
    """
    con = sqlite3.connect(db_path)
    try:
        assert con.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
    finally:
        con.close()

    led = store._connect()
    try:
        assert led.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert led.execute("PRAGMA busy_timeout").fetchone()[0] == 5000
    finally:
        led.close()


def test_foreign_keys_are_enforced_so_a_dangling_decision_id_cannot_be_inserted(store, db_path):
    """`settlements.decision_id` 外键在场且被强制：给不存在的 decision 插结算行失败。"""
    con = store._connect()
    try:
        with pytest.raises(sqlite3.IntegrityError):
            con.execute(
                "INSERT INTO settlements (decision_id, settled_at, settlement_date, entry_price, "
                "exit_price, raw_return, benchmark_symbol, benchmark_mapping, "
                "benchmark_entry_price, benchmark_exit_price, benchmark_return, alpha, "
                "holding_days, reflection) VALUES ('dec-ghost', 'now', '2025-01-13', 1, 1, 0, "
                "'X', 'explicit', 1, 1, 0, 0, 5, NULL)"
            )
    finally:
        con.close()


def test_schema_is_idempotent_across_reopen(db_path):
    """重开同一库不报错、也不清空现有数据（IF NOT EXISTS 的可见后果）。"""
    s1 = LedgerStore(db_path)
    r = _record(s1)
    s2 = LedgerStore(db_path)
    assert s2.get(r["decision_id"]) is not None
    assert s2.row_counts()["decisions"] == 1


# ── 1.2 写入中断不产生半行 ──────────────────────────────────────


def test_interrupted_write_leaves_no_half_row(db_path, tmp_path):
    """task 1.2 / spec Scenario「写入中断不产生半行」。

    模型：**未提交的事务**。子进程走 `record()` 的**真实路径**（`_write_tx` 开
    BEGIN IMMEDIATE、INSERT、COMMIT），靠 SQLite 的 trace 回调在 COMMIT 语句上
    `os._exit(1)`——这是进程死亡的诚实模拟：不跑 finally、不回滚、不关连接、不做任何
    清理。重开库后：该行不存在、库未损坏、且仍可正常写入、且同键可以正常重录。

    为什么这样模拟而不是"构造一个半行"：半行在 SQLite 里**写不出来**——写只有 COMMIT
    之后才可见，未提交的帧在下次打开时被丢弃。测试要证的就是这一点，故必须让进程在
    真正的提交点上死掉，而不是自己伪造一个不存在的状态。
    """
    script = textwrap.dedent(
        """
        import os, sys
        sys.path.insert(0, {repo!r})
        sys.path.insert(0, {tests!r})
        import ledgerdata as ld
        from fd_quant.ledger import LedgerStore

        class Interrupting(LedgerStore):
            \"\"\"在生产写路径上装一个"见 COMMIT 就猝死"的钩子（覆盖 _connect）。\"\"\"

            def _connect(self):
                con = super()._connect()
                def hook(sql):
                    if sql.strip().upper().startswith("COMMIT"):
                        os._exit(1)
                con.set_trace_callback(hook)
                return con

        store = Interrupting({db!r})
        result = store.record(
            as_of="2025-01-06", symbol="600519.SS", pack={pack!r}, skill={skill!r},
            payload=ld.rating("2025-01-06", "600519.SS").to_dict(),
        )
        print("UNREACHABLE", result)
        """
    ).format(repo=str(REPO), tests=str(REPO / "tests"), db=str(db_path),
             pack=ld.PACK, skill=ld.SKILL)

    proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert proc.returncode == 1, (proc.returncode, proc.stdout, proc.stderr)
    assert "UNREACHABLE" not in proc.stdout

    store = LedgerStore(db_path)
    con = sqlite3.connect(db_path)
    try:
        assert con.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert con.execute("SELECT COUNT(*) FROM decisions").fetchone()[0] == 0
    finally:
        con.close()
    # 中断没有损坏库：之后照常追加（连同一个幂等键都能重录，证明那次写没留下任何痕迹）
    r = _record(store)
    assert r["created"] is True
    assert store.row_counts()["decisions"] == 1
    assert store.get(r["decision_id"])["payload"]["basis"]


def test_interruption_mid_settlement_leaves_no_settlement_row(store, db_path):
    """结算被拒（持有期未满）时事务整体回滚：不落结算行、状态仍 pending。

    （这是同一个原子性保证的另一面：被拒绝的写入与被打断的写入都不留痕迹。）
    """
    from fd_quant.ledger import SettlementRefused

    r = _record(store)
    d = ld.flat_then_move(100.0, 110.0, 6)
    short = ld.series([100.0, 101.0, 102.0])
    with pytest.raises(SettlementRefused):
        store.settle(decision_id=r["decision_id"], symbol_dates=short["dates"],
                     symbol_prices=short["prices"], benchmark_dates=short["dates"],
                     benchmark_prices=short["prices"], as_of=short["dates"][-1])
    assert store.row_counts() == {"decisions": 1, "settlements": 0, "pending": 1, "settled": 0}
    assert store.get(r["decision_id"])["status"] == "pending"
    assert d  # 夹具自足性：正常窗序列与短序列来自同一构造函数


# ── 1.3 幂等拒绝 ────────────────────────────────────────────────


def test_duplicate_submission_is_refused_and_returns_the_existing_id(store):
    """task 1.3 / spec Scenario「重复提交被拒且返回既有记录」。

    两件事必须同时可观察：**被拒**（created=False）与**给出既有 decision_id**，
    且 `SELECT COUNT(*)` 逐字不变（用 row_counts 核对，而不是相信返回值）。
    """
    first = _record(store, now="2025-01-06T09:00:00+00:00")
    before = store.row_counts()
    assert first["created"] is True
    assert first["reason"] == "created"
    assert first["revision_seq"] == 0

    second = _record(store, now="2025-01-07T09:00:00+00:00")
    assert second["created"] is False
    assert second["reason"] == "duplicate_key"
    assert second["decision_id"] == first["decision_id"]
    assert store.row_counts() == before == {"decisions": 1, "settlements": 0,
                                            "pending": 1, "settled": 0}
    assert second["created_at"] == "2025-01-06T09:00:00+00:00"  # 原行的时刻未被改写


def test_duplicate_is_not_an_exception_and_the_row_is_untouched(store):
    """幂等路径不抛异常（spec 要"返回既有 id"），且原行逐字不变。"""
    first = _record(store)
    snapshot = store.get(first["decision_id"])
    again = _record(store, side="sell")  # 内容不同、键相同 —— 仍是重复提交
    assert again["decision_id"] == first["decision_id"]
    assert store.get(first["decision_id"]) == snapshot
    assert store.get(first["decision_id"])["payload"]["side"] == "buy"


def test_same_key_with_different_skill_is_a_distinct_record(store):
    """幂等键含 skill：同 pack 不同技能对同一标的同日可各落一条（design.md 的有意设计）。"""
    a = _record(store, skill="equity-research-thesis")
    b = _record(store, skill="equity-research-modeling")
    assert a["created"] and b["created"]
    assert a["decision_id"] != b["decision_id"]
    assert store.row_counts()["decisions"] == 2


def test_payload_must_agree_with_the_key_dimensions(store):
    """payload 与键不一致即拒绝并点名两边（不静默取信其中一边）。"""
    with pytest.raises(ContractError) as ei:
        store.record(as_of="2025-01-06", symbol="000001.SZ", pack=ld.PACK, skill=ld.SKILL,
                     payload=ld.rating("2025-01-06", "600519.SS"))
    assert ei.value.path == "subject"
    assert "600519.SS" in ei.value.reason and "000001.SZ" in ei.value.reason

    with pytest.raises(ContractError) as ei2:
        store.record(as_of="2025-01-06", symbol="600519.SS", pack="other-pack", skill=ld.SKILL,
                     payload=ld.rating("2025-01-06", "600519.SS"))
    assert ei2.value.path == "producer.pack"
    assert store.row_counts()["decisions"] == 0


def test_illegal_payload_is_refused_by_the_contract(store):
    """payload 必须过契约校验：非法实例（未知 side）被拒绝且点名字段。"""
    bad = ld.rating("2025-01-06", "600519.SS").to_dict()
    bad["side"] = "moon"
    with pytest.raises(ContractError) as ei:
        store.record(as_of="2025-01-06", symbol="600519.SS", pack=ld.PACK, skill=ld.SKILL,
                     payload=bad)
    assert ei.value.path == "side"
    assert store.row_counts()["decisions"] == 0


# ── 1.4 修订语义 ────────────────────────────────────────────────


def test_revision_creates_a_new_row_and_points_back_at_the_original(store):
    """task 1.4 / spec Scenario「修订产生新记录」：新行 revises 指向原记录。"""
    original = _record(store, side="buy", now="2025-01-06T09:00:00+00:00")
    before = store.get(original["decision_id"])

    revised = _record(store, side="hold", revises=original["decision_id"],
                      now="2025-01-07T09:00:00+00:00")
    assert revised["created"] is True
    assert revised["reason"] == "revision_created"
    assert revised["revises"] == original["decision_id"]
    assert revised["revision_seq"] == 1
    assert revised["decision_id"] != original["decision_id"]
    assert store.row_counts() == {"decisions": 2, "settlements": 0, "pending": 2, "settled": 0}

    # 原行逐字未变（payload 与 created_at 是最容易被"就地改写"顺手改掉的两项）
    after = store.get(original["decision_id"])
    assert after == before
    assert after["payload"]["side"] == "buy"
    assert after["created_at"] == "2025-01-06T09:00:00+00:00"


def test_revision_chain_is_linear_and_stale_targets_are_refused(store):
    """修订必须指向当前 tip：指向陈旧记录即拒绝（账本不裁决分叉），并点名两条 id。"""
    v1 = _record(store, side="buy")
    v2 = _record(store, side="hold", revises=v1["decision_id"])
    with pytest.raises(RevisionConflict) as ei:
        _record(store, side="sell", revises=v1["decision_id"])
    assert v1["decision_id"] in str(ei.value) and v2["decision_id"] in str(ei.value)
    # 链继续前进是允许的：v3 指向 v2
    v3 = _record(store, side="sell", revises=v2["decision_id"])
    assert v3["revises"] == v2["decision_id"]
    assert v3["revision_seq"] == 2


def test_identical_revision_content_is_idempotent(store):
    """内容与 tip 逐字相同的"修订"是重复提交：幂等返回 tip，不产生第三行。"""
    v1 = _record(store, side="buy")
    v2 = _record(store, side="hold", revises=v1["decision_id"])
    again = _record(store, side="hold", revises=v1["decision_id"])
    assert again["created"] is False
    assert again["reason"] == "revision_duplicate"
    assert again["decision_id"] == v2["decision_id"]
    assert store.row_counts()["decisions"] == 2


def test_revision_cannot_change_the_idempotency_key(store):
    """修订必须保持幂等键：换一个 symbol 提交修订被拒并点名不一致的字段。"""
    v1 = _record(store, symbol="600519.SS")
    with pytest.raises(RevisionConflict) as ei:
        store.record(as_of="2025-01-06", symbol="000001.SZ", pack=ld.PACK, skill=ld.SKILL,
                     payload=ld.rating("2025-01-06", "000001.SZ"), revises=v1["decision_id"])
    assert "symbol" in str(ei.value)


def test_revision_pointing_at_an_unknown_id_is_refused(store):
    """悬空 revises 指向不存在的记录 → UnknownDecision（点名该 id）。"""
    with pytest.raises(UnknownDecision) as ei:
        _record(store, revises="dec-does-not-exist")
    assert "dec-does-not-exist" in str(ei.value)
    assert store.row_counts()["decisions"] == 0


def test_each_generation_is_settleable(store):
    """修订链的每一代都是独立记录：可分别结算，聚合时各算一个样本。"""
    v1 = _record(store, symbol="600519.SZ")
    v2 = _record(store, symbol="600519.SZ", side="hold", revises=v1["decision_id"])
    for r in (v1, v2):
        s = _settle(store, r["decision_id"], symbol="600519.SZ")
        assert s["created"] is True
    assert store.row_counts() == {"decisions": 2, "settlements": 2, "pending": 0, "settled": 2}


# ── 1.5 并发安全 ────────────────────────────────────────────────


def test_two_concurrent_submissions_of_the_same_key_yield_exactly_one_created(db_path):
    """task 1.5 / spec Scenario「并发重复提交只有一条成功」。

    两个真实线程（各开自己的连接）同时提交同一幂等键：恰一条 created=True，
    另一条 created=False 且 decision_id 相同。唯一约束 + BEGIN IMMEDIATE 是屏障，
    不是应用层"先查后插"的时序运气。
    """
    store = LedgerStore(db_path)
    barrier = threading.Barrier(2)
    out: list[dict] = []
    errors: list[BaseException] = []

    def submit():
        try:
            barrier.wait(timeout=10)
            out.append(_record(store))
        except BaseException as exc:  # noqa: BLE001 — 记下来由主线程断言
            errors.append(exc)

    threads = [threading.Thread(target=submit) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert not errors, errors
    assert len(out) == 2
    created = [o for o in out if o["created"]]
    refused = [o for o in out if not o["created"]]
    assert len(created) == 1, out
    assert len(refused) == 1, out
    assert refused[0]["decision_id"] == created[0]["decision_id"]
    assert refused[0]["reason"] == "duplicate_key"
    assert store.row_counts()["decisions"] == 1


def test_concurrent_distinct_keys_all_succeed(db_path):
    """不同键并发全成功（并发安全不等于"大家都失败"——反向对照）。"""
    store = LedgerStore(db_path)
    barrier = threading.Barrier(6)
    out: list[dict] = []
    errors: list[BaseException] = []

    def submit(i: int):
        try:
            barrier.wait(timeout=10)
            out.append(_record(store, symbol=f"60051{i}.SS"))
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=submit, args=(i,)) for i in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert not errors, errors
    assert len(out) == 6 and all(o["created"] for o in out)
    assert store.row_counts()["decisions"] == 6


# ── 库路径可配置 ────────────────────────────────────────────────


def test_db_path_is_configurable_by_argument_and_env(tmp_path, monkeypatch):
    """spec: 本地数据库文件路径 SHALL 可配置（构造参数 + FD_QUANT_LEDGER_DB）。"""
    explicit = tmp_path / "explicit.db"
    s = LedgerStore(explicit)
    assert s.db_path == explicit
    assert explicit.exists()

    env_path = tmp_path / "from-env.db"
    monkeypatch.setenv("FD_QUANT_LEDGER_DB", str(env_path))
    s_env = LedgerStore()
    assert s_env.db_path == env_path
    assert env_path.exists()
    # 参数优先于 env
    s_arg = LedgerStore(explicit)
    assert s_arg.db_path == explicit


def test_nested_db_directory_is_created(tmp_path):
    """库路径的父目录不存在时自动建（零配置部署不因缺目录而失败）。"""
    nested = tmp_path / "a" / "b" / "ledger.db"
    s = LedgerStore(nested)
    assert nested.exists()
    assert _record(s)["created"] is True


def test_env_var_default_is_honoured_by_the_tool_layer(tmp_path, monkeypatch):
    """工具层的默认路径同样支持 env（工具不传 db_path 时走 env）。"""
    from fd_quant.ledger import tools

    env_path = tmp_path / "tool.db"
    monkeypatch.setenv("FD_QUANT_LEDGER_DB", str(env_path))
    r = tools.record_decision(
        as_of="2025-01-06", symbol="600519.SS", pack=ld.PACK, skill=ld.SKILL,
        payload=ld.rating("2025-01-06", "600519.SS").to_dict(),
    )
    assert r["db_path"] == str(env_path)
    assert r["result"]["created"] is True


# ── 辅助 ────────────────────────────────────────────────────────


def _settle(store: LedgerStore, decision_id: str, *, symbol: str = "600519.SS",
            holding_days: int | None = None) -> dict:
    """用与标的匹配的映射基准结算（600519.SS → 000001.SS）。"""
    sym = ld.flat_then_move(100.0, 110.0, 6)
    bench = ld.flat_then_move(3000.0, 3060.0, 6)
    return store.settle(
        decision_id=decision_id, symbol_dates=sym["dates"], symbol_prices=sym["prices"],
        benchmark_dates=bench["dates"], benchmark_prices=bench["prices"],
        as_of=sym["dates"][-1], holding_days=holding_days,
    )
