"""账本调用者隔离（finance-legal-value-realization 组 2，tasks 2.2–2.5）。

这组测试钉住一件事：**账本此前是单用户的**，多个调用者共用一个库时，彼此的判定
互相可见、且会撞同一个幂等键——这不是隐私问题，是**统计污染**：A 的判定会被算进
B 的胜率与平均 alpha，而两边都不会知道。

四条断言，各对应一条任务：

- 2.2 迁移可重复执行（幂等），且 `caller` 进了唯一索引——否则隔离只是应用层约定。
- 2.3 `caller` 由调用上下文解析，**不接受入参声明**（能由入参声明的"隔离"不是隔离）。
- 2.4 旧记录归属 `legacy`，且**不被任何具名调用者看到**。
- 2.5 两个调用者同键各自成功、互不可见。
"""
from __future__ import annotations

import sqlite3

import pytest

import ledgerdata as ld
from fd_quant.caller import ANONYMOUS, LEGACY, resolve_caller
from fd_quant.ledger import LedgerStore

CALLER_A = "caller-a"
CALLER_B = "caller-b"

SYMBOL = "600519.SS"


@pytest.fixture()
def db_path(tmp_path):
    return tmp_path / "ledger.db"


def _record(store: LedgerStore, *, caller: str, as_of: str = "2025-01-06",
            symbol: str = SYMBOL, pack: str = ld.PACK, skill: str = ld.SKILL,
            side: str = "buy", revises: str | None = None) -> dict:
    return store.record(
        caller=caller, as_of=as_of, symbol=symbol, pack=pack, skill=skill,
        payload=ld.rating(as_of, symbol, side=side, pack=pack, skill=skill), revises=revises,
    )


def _settle(store: LedgerStore, decision_id: str, *, start: str = "2025-01-06", n: int = 6,
            sym_p0: float = 100.0, sym_p1: float = 110.0, bench_p0: float = 3000.0,
            bench_p1: float = 3060.0) -> dict:
    """在 `start` 起的 n 日窗内结算（`as_of` 取窗末，故整窗在视野内）。"""
    sym = ld.flat_then_move(sym_p0, sym_p1, n, move_at=n - 1)
    bench = ld.flat_then_move(bench_p0, bench_p1, n, move_at=n - 1)
    return store.settle(
        decision_id=decision_id,
        symbol_dates=sym["dates"], symbol_prices=sym["prices"],
        benchmark_dates=bench["dates"], benchmark_prices=bench["prices"],
        as_of=sym["dates"][-1],
    )


# ── 2.2 迁移：caller 进 schema 与唯一索引，且可重复执行 ────────────


def test_caller_is_a_column_and_part_of_the_idempotency_unique_index(db_path):
    """task 2.2：`caller` 是列，且**进唯一索引**（唯一性必须由库保证，不是应用层先查后插）。

    只加列不够：若唯一索引仍是 (as_of, symbol, pack, skill)，两个调用者的同键记录
    会互相拒绝——隔离就退化成"先到先得"。
    """
    LedgerStore(db_path)
    con = sqlite3.connect(db_path)
    try:
        cols = {r[1] for r in con.execute("PRAGMA table_info(decisions)")}
        assert "caller" in cols
        idx = {r[1] for r in con.execute("PRAGMA index_list(decisions)")}
        sql = {r[0]: r[1] for r in con.execute(
            "SELECT name, sql FROM sqlite_master WHERE type = 'index'")}
        assert "ux_decisions_initial" in idx
        assert "caller" in sql["ux_decisions_initial"], sql["ux_decisions_initial"]
        assert "caller" in sql["ux_decisions_revision"], sql["ux_decisions_revision"]
    finally:
        con.close()


def test_migration_is_idempotent_and_marks_preexisting_rows_as_legacy(db_path):
    """task 2.2 / 2.4：迁移可重复执行，且把既有行标为 `legacy`（不猜归属）。

    先手工造一个**旧 schema** 的库（无 caller 列）并塞两行，再开 LedgerStore——
    迁移应把它们搬成 `legacy`。再开一次：第二次必须是空操作（行数不变、caller 不变）。
    """
    con = sqlite3.connect(db_path)
    con.execute(
        "CREATE TABLE decisions (decision_id TEXT PRIMARY KEY, as_of TEXT NOT NULL, "
        "symbol TEXT NOT NULL, pack TEXT NOT NULL, skill TEXT NOT NULL, payload TEXT NOT NULL, "
        "target_weights TEXT, evidence TEXT NOT NULL, status TEXT NOT NULL, revises TEXT, "
        "revision_seq INTEGER NOT NULL, created_at TEXT NOT NULL)"
    )
    con.execute(
        "INSERT INTO decisions VALUES ('dec-old-1', '2025-01-06', ?, ?, ?, '{}', NULL, "
        "'[]', 'pending', NULL, 0, '2025-01-06T00:00:00+00:00')", (SYMBOL, ld.PACK, ld.SKILL),
    )
    con.execute(
        "INSERT INTO decisions VALUES ('dec-old-2', '2025-01-07', ?, ?, ?, '{}', NULL, "
        "'[]', 'pending', NULL, 0, '2025-01-07T00:00:00+00:00')", (SYMBOL, ld.PACK, ld.SKILL),
    )
    con.commit()
    con.close()

    store = LedgerStore(db_path)          # 第一次：迁移
    assert store.row_counts()["decisions"] == 2
    con = sqlite3.connect(db_path)
    assert {r[0] for r in con.execute("SELECT caller FROM decisions")} == {LEGACY}
    con.close()

    LedgerStore(db_path)                  # 第二次：必须是空操作
    store = LedgerStore(db_path)
    counts = store.row_counts()
    assert counts["decisions"] == 2, counts
    con = sqlite3.connect(db_path)
    assert {r[0] for r in con.execute("SELECT caller FROM decisions")} == {LEGACY}
    con.close()


def test_migrated_legacy_rows_are_invisible_to_a_named_caller(db_path):
    """task 2.4：`legacy` 行**不被任何具名调用者检索到**。

    处置旧数据有两种做法：让所有人都能看到（等于把历史样本灌进每个调用者的统计），
    或让谁都看不到（历史封存）。取后者——前者的"每人多几条样本"正是要防的污染。
    """
    con = sqlite3.connect(db_path)
    con.execute(
        "CREATE TABLE decisions (decision_id TEXT PRIMARY KEY, as_of TEXT NOT NULL, "
        "symbol TEXT NOT NULL, pack TEXT NOT NULL, skill TEXT NOT NULL, payload TEXT NOT NULL, "
        "target_weights TEXT, evidence TEXT NOT NULL, status TEXT NOT NULL, revises TEXT, "
        "revision_seq INTEGER NOT NULL, created_at TEXT NOT NULL)"
    )
    con.execute(
        "INSERT INTO decisions VALUES ('dec-old-1', '2025-01-06', ?, ?, ?, '{}', NULL, "
        "'[]', 'pending', NULL, 0, '2025-01-06T00:00:00+00:00')", (SYMBOL, ld.PACK, ld.SKILL),
    )
    con.commit()
    con.close()

    store = LedgerStore(db_path)
    out = store.retrieve(caller=CALLER_A, symbol=SYMBOL, as_of="2025-12-31")
    assert out["counts"]["total"] == 0, out
    stats = store.stats_by_pack(caller=CALLER_A, as_of="2025-12-31")
    assert stats["registered_packs"] == [], stats


# ── 2.3 caller 从调用上下文解析，不接受入参声明 ──────────────────


def test_caller_is_not_a_tool_parameter(monkeypatch):
    """task 2.3：四个 MCP 工具的签名里**没有** `caller` 参数。

    若 caller 是入参，任何调用方都能声称自己是别人——那不叫隔离，叫"请自觉遵守"。
    断言落在工具签名（对外契约）上，而不是 store 层：store 层拿 caller 是设计如此，
    工具层不该给调用方这个把手。
    """
    monkeypatch.delenv("FD_QUANT_CALLER", raising=False)
    from fd_quant.ledger import tools as ledger_tools

    for spec in ledger_tools.TOOLS:
        names = set(spec.fn.__code__.co_varnames[:spec.fn.__code__.co_argcount])
        assert "caller" not in names, f"{spec.name} 暴露了 caller 入参"


def test_caller_falls_back_to_env_then_anonymous(monkeypatch):
    """task 2.3：解析顺序的可见部分——环境变量 → `anonymous`（不报错、不猜）。"""
    monkeypatch.setenv("FD_QUANT_CALLER", "deployed-identity")
    assert resolve_caller() == "deployed-identity"
    monkeypatch.delenv("FD_QUANT_CALLER", raising=False)
    assert resolve_caller() == ANONYMOUS


def test_bearer_token_is_fingerprinted_not_stored(monkeypatch):
    """task 2.3：凭据本身不落库——落的是它的稳定指纹，且同一 token 恒得同一指纹。

    这是"取身份而非取凭据"：token 原文进库等于把凭据写进一份会被备份、被复制的
    文件里。指纹只需能区分调用者。
    """
    monkeypatch.setenv("FD_QUANT_CALLER", "")
    token = "sk-super-secret-token-value"

    class _Request:
        headers = {"Authorization": f"Bearer {token}"}

    # `resolve_caller` 在函数体内 import fastmcp（账本离线时可能没装），故这里
    # 用 monkeypatch 往 sys.modules 塞一个假依赖，把 HTTP 分支逼出来。
    import sys
    import types

    fake_dep = types.ModuleType("fastmcp.server.dependencies")
    fake_dep.get_http_request = lambda: _Request()
    monkeypatch.setitem(sys.modules, "fastmcp.server.dependencies", fake_dep)

    got = resolve_caller()
    assert got != token
    assert token not in got
    assert len(got) == 12          # SHA-256 前 12 位十六进制
    assert resolve_caller() == got  # 同一 token 恒得同一指纹（可复核）


def test_anonymous_and_legacy_are_distinct_values():
    """两个占位符必须是**不同的可查询取值**，不能都是空串或 None。

    空串看起来像"解析失败"，而 `anonymous` / `legacy` 是可以被 SQL 直接查出来、
    被运维统计的真实取值——"有多少记录没有身份基础设施"这个问题因此可答。
    """
    assert ANONYMOUS and LEGACY
    assert ANONYMOUS != LEGACY


# ── 2.5 隔离：两个调用者同键各自成功、互不可见 ──────────────────


def test_two_callers_submit_the_same_idempotency_key_and_both_succeed(db_path):
    """task 2.5 前半：同键（as_of/symbol/pack/skill）两个调用者**各自成功**。

    隔离前这是不可能的：第二个调用者会被判 duplicate_key 并拿到第一个的
    decision_id——于是 A 的判定被算进 B 的账，且 B 拿到的 id 指向 A 的记录。
    """
    store = LedgerStore(db_path)
    a = _record(store, caller=CALLER_A)
    b = _record(store, caller=CALLER_B)

    assert a["created"] is True and b["created"] is True
    assert a["decision_id"] != b["decision_id"]
    assert store.row_counts()["decisions"] == 2


def test_same_caller_still_refused_on_duplicate_key(db_path):
    """隔离**不能**把幂等键放掉：同一调用者重复提交仍被拒（否则"只算一次账"失效）。"""
    store = LedgerStore(db_path)
    first = _record(store, caller=CALLER_A)
    again = _record(store, caller=CALLER_A)
    assert again["created"] is False and again["reason"] == "duplicate_key"
    assert again["decision_id"] == first["decision_id"]
    assert store.row_counts()["decisions"] == 1


def test_retrieval_returns_only_the_callers_own_records(db_path):
    """task 2.5 后半：检索只回**本人**记录——同标的、同 as_of 也不串。"""
    store = LedgerStore(db_path)
    ra = _record(store, caller=CALLER_A, side="buy")
    rb = _record(store, caller=CALLER_B, side="sell")
    _settle(store, ra["decision_id"])
    _settle(store, rb["decision_id"])

    out_a = store.retrieve(caller=CALLER_A, symbol=SYMBOL, as_of="2025-12-31")
    out_b = store.retrieve(caller=CALLER_B, symbol=SYMBOL, as_of="2025-12-31")

    ids_a = {i["decision_id"] for i in out_a["same_symbol"]}
    ids_b = {i["decision_id"] for i in out_b["same_symbol"]}
    assert ids_a == {ra["decision_id"]}, out_a
    assert ids_b == {rb["decision_id"]}, out_b
    # 方向不同是刻意的：若串了，A 会看到一条"卖出"并以为是自己的判定。
    assert out_a["same_symbol"][0]["payload"]["side"] == "buy"
    assert out_b["same_symbol"][0]["payload"]["side"] == "sell"


def test_aggregate_stats_do_not_mix_callers(db_path):
    """聚合是最容易被污染的地方：跨调用者的样本混进胜率与平均 alpha，且无人察觉。

    A 有一条**盈利**的结算、B 有一条**亏损**的结算；各自聚合只看得到自己的一条。
    若隔离漏了，两边都会看到 samples=2、胜率 50%——一个看起来合理的错数字。
    """
    store = LedgerStore(db_path)
    ra = _record(store, caller=CALLER_A)
    rb = _record(store, caller=CALLER_B)
    # A：标的涨 100→110（+10%），基准涨 3000→3060（+2%）→ alpha 为正
    _settle(store, ra["decision_id"], sym_p0=100.0, sym_p1=110.0,
            bench_p0=3000.0, bench_p1=3060.0)
    # B：标的跌 100→90（-10%），基准不变 → alpha 为负
    _settle(store, rb["decision_id"], sym_p0=100.0, sym_p1=90.0,
            bench_p0=3000.0, bench_p1=3000.0)

    sa = store.stats_by_pack(caller=CALLER_A, as_of="2025-12-31")
    sb = store.stats_by_pack(caller=CALLER_B, as_of="2025-12-31")

    row_a = [r for r in sa["rows"] if r["pack"] == ld.PACK][0]
    row_b = [r for r in sb["rows"] if r["pack"] == ld.PACK][0]
    assert (row_a["samples"], row_a["wins"]) == (1, 1), row_a
    assert (row_b["samples"], row_b["wins"]) == (1, 0), row_b
    assert row_a["mean_alpha"] > 0 > row_b["mean_alpha"]


def test_revision_chain_stays_within_one_caller(db_path):
    """修订也按 caller 隔离：B 不能修订 A 的记录（这会改写别人的历史判定）。

    账本的红线是"历史判定永不改写"。若修订不隔离，B 就能在 A 的判定上挂一条
    自己的更正——A 的账本里从此多了一条不是 A 写的记录。
    """
    from fd_quant.ledger import UnknownDecision

    store = LedgerStore(db_path)
    ra = _record(store, caller=CALLER_A)
    with pytest.raises(UnknownDecision):
        _record(store, caller=CALLER_B, revises=ra["decision_id"])
    assert store.row_counts()["decisions"] == 1
