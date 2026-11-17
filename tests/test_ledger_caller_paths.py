"""账本路径派生（seal-ledger-path-and-caller-isolation 任务 1.1 / 1.2 / 1.3）。

三条判据分别对应三条任务，且都按「能红」写：

1. 六个刁钻 caller 各得**不同且安全**的路径（安全 = 结果里不可能出现 `..`／`/`／控制字符）；
2. `A/B` 与 `A-B` 必须落不同目录 —— 去掉 sha256 后缀该用例转红；
3. 孤立旧库存在时启动提示两条路径，且**不自动搬**。
"""

from __future__ import annotations

import pytest

from fd_quant.ledger import store as store_mod
from fd_quant.ledger.store import (
    DEFAULT_DB_FILENAME,
    LEDGER_DB_FILENAME,
    SLUG_MAX_LEN,
    caller_db_path,
    caller_digest,
    ledger_root,
    migration_hint,
    safe_segment,
)

#: 六个 caller：路径穿越、绝对路径片段、空、超长、非 ASCII、正常邮箱。
TRICKY_CALLERS = [
    "A/B",
    "../x",
    "",
    "x" * 200,
    "中文用户",
    "zhang@example.com",
]


def _segment(path) -> str:
    """取分库目录名（不含根），后面所有安全性断言都落在这一个片段上。"""
    return path.parent.name


# ── 1.1 路径派生 ────────────────────────────────────────────────────────────


def test_six_tricky_callers_get_distinct_paths(tmp_path):
    paths = {c: caller_db_path(c, {"FD_QUANT_LEDGER_ROOT": str(tmp_path)}) for c in TRICKY_CALLERS}
    seen = {str(p) for p in paths.values()}
    assert len(seen) == len(TRICKY_CALLERS), f"六个 caller 必须六个库，实得 {len(seen)}：{paths}"


@pytest.mark.parametrize("caller", TRICKY_CALLERS)
def test_paths_are_safe(tmp_path, caller):
    """安全判据：片段里不出现 `..`、`/`、控制字符；文件名恒为 ledger.db。"""
    path = caller_db_path(caller, {"FD_QUANT_LEDGER_ROOT": str(tmp_path)})
    seg = _segment(path)

    assert path.name == LEDGER_DB_FILENAME
    assert path.parent.parent == tmp_path, "分库必须在根目录下一层，不能更深"
    assert "/" not in seg and "\\" not in seg
    assert ".." not in seg
    assert all(ord(ch) >= 0x20 for ch in seg), f"控制字符泄漏：{seg!r}"
    assert len(seg) <= SLUG_MAX_LEN + 1 + 8, f"片段超长：{len(seg)}"


def test_long_caller_is_truncated_but_still_unique(tmp_path):
    """超长 caller 只截断**可读部分**；唯一性由摘要保证（截断后仍互不相同）。"""
    a = "y" * 60 + "-one"
    b = "y" * 60 + "-two"
    pa = caller_db_path(a, {"FD_QUANT_LEDGER_ROOT": str(tmp_path)})
    pb = caller_db_path(b, {"FD_QUANT_LEDGER_ROOT": str(tmp_path)})
    assert _segment(pa)[:SLUG_MAX_LEN] == _segment(pb)[:SLUG_MAX_LEN], "前提：可读部分确实同形"
    assert pa != pb, "同形前缀必须由摘要分开"


def test_empty_caller_falls_back_to_anon(tmp_path):
    """空 caller 回落 `anon`（任务原文点名的六种之一）；全非法字符**不**回落——
    它替换后仍非空（`!!!` → `---`），白名单里没有的字符变成 `-` 本身就是安全结果，
    于是不必也不该把它并进回退档（并进去会让两条不同的 caller 落一个库）。"""
    env = {"FD_QUANT_LEDGER_ROOT": str(tmp_path)}
    assert safe_segment("") == "anon"
    assert safe_segment("!!!") == "---"
    assert caller_db_path("", env) != caller_db_path("!!!", env), "回退档与替换档仍要按原文区分"


# ── 1.1 根目录回退链 ────────────────────────────────────────────────────────


def test_root_fallback_chain(tmp_path):
    """`FD_QUANT_LEDGER_ROOT` → `FD_QUANT_LEDGER_DB` 的父目录 → CWD。"""
    assert ledger_root({"FD_QUANT_LEDGER_ROOT": str(tmp_path)}) == tmp_path
    assert ledger_root(
        {"FD_QUANT_LEDGER_ROOT": "", "FD_QUANT_LEDGER_DB": str(tmp_path / "old.db")}
    ) == tmp_path
    assert ledger_root({}) == store_mod.Path.cwd()


def test_root_wins_over_legacy_variable(tmp_path):
    """两个变量都在时以新变量为准——否则「加了 ROOT 却没生效」无法从读数上分辨。"""
    other = tmp_path / "other"
    got = ledger_root(
        {"FD_QUANT_LEDGER_ROOT": str(tmp_path), "FD_QUANT_LEDGER_DB": str(other / "old.db")}
    )
    assert got == tmp_path


# ── 1.2 slug 撞名 ───────────────────────────────────────────────────────────


def test_slug_collision_pair_lands_in_different_dirs(tmp_path):
    env = {"FD_QUANT_LEDGER_ROOT": str(tmp_path)}
    left, right = caller_db_path("A/B", env), caller_db_path("A-B", env)
    assert safe_segment("A/B") == safe_segment("A-B"), "前提：两者 slug 同形（这才是撞名）"
    assert left != right, "slug 同形必须由 sha256 后缀分开"


def test_red_proof_dropping_the_digest_collides(monkeypatch, tmp_path):
    """红证：把摘要钉成常量（＝回到「只用 slug」的旧做法），撞名用例必须转红。

    这条不是在测 monkeypatch，是在证明**上一条真能红**——否则「两者不同」可能
    只是 slug 恰好不同，而不是摘要承重。
    """
    monkeypatch.setattr(store_mod, "caller_digest", lambda caller: "00000000")
    env = {"FD_QUANT_LEDGER_ROOT": str(tmp_path)}
    assert caller_db_path("A/B", env) == caller_db_path("A-B", env), (
        "摘要不承重时两者必须同路径；不同则说明还有别的差异在撑着那条断言"
    )


# ── 1.3 迁移提示 ────────────────────────────────────────────────────────────


def test_migration_hint_names_both_paths(tmp_path):
    stray = tmp_path / DEFAULT_DB_FILENAME
    stray.write_bytes(b"SQLite format 3\x00")
    hint = migration_hint(tmp_path)
    assert hint is not None
    assert "legacy" in hint, "提示必须点名 legacy 子目录"
    assert str(stray) in hint, "提示必须给出源路径"
    assert str(tmp_path / "legacy" / LEDGER_DB_FILENAME) in hint, "提示必须给出目标路径"


def test_no_hint_without_a_stray_ledger(tmp_path):
    assert migration_hint(tmp_path) is None


def test_no_hint_once_legacy_dir_exists(tmp_path):
    """已经搬过就不要再提示——否则提示会变成永久噪音。"""
    (tmp_path / DEFAULT_DB_FILENAME).write_bytes(b"SQLite format 3\x00")
    (tmp_path / "legacy").mkdir()
    assert migration_hint(tmp_path) is None


def test_hint_never_moves_anything(tmp_path):
    """判据的另一半：提示不自动搬（文件仍在原处、legacy/ 仍不存在）。"""
    stray = tmp_path / DEFAULT_DB_FILENAME
    stray.write_bytes(b"SQLite format 3\x00")
    assert migration_hint(tmp_path) is not None
    assert stray.exists(), "提示不得搬走源文件"
    assert not (tmp_path / "legacy").exists(), "提示不得自建目标目录"


# ── 1.3 启动时的提示（真打印，不只是纯函数）──────────────────────────────────


def test_startup_announces_hint_on_stderr_never_stdout(tmp_path, monkeypatch, capsys):
    """启动提示必须**落到 stderr**。

    stdio 传输下 stdout 就是 MCP 报文通道——往上写一个字，客户端会表现为
    「连上了但一个工具都没有」。所以这条用例有**两个**断言：stderr 有提示、
    **stdout 一个字都没有**（只断言前者会把「打在 stdout 上」放过去）。
    """
    import io as _io

    from fd_quant.server import announce_ledger_migration

    stray = tmp_path / DEFAULT_DB_FILENAME
    stray.write_bytes(b"SQLite format 3\x00")
    monkeypatch.setenv("FD_QUANT_LEDGER_ROOT", str(tmp_path))

    sink = _io.StringIO()
    hint = announce_ledger_migration(stream=sink)
    assert hint and "legacy" in hint
    assert str(stray) in sink.getvalue()

    captured = capsys.readouterr()
    assert captured.out == "", f"启动提示泄漏到了 stdout：{captured.out!r}"


def test_startup_is_silent_when_nothing_to_migrate(tmp_path, monkeypatch):
    import io as _io

    from fd_quant.server import announce_ledger_migration

    monkeypatch.setenv("FD_QUANT_LEDGER_ROOT", str(tmp_path))
    sink = _io.StringIO()
    assert announce_ledger_migration(stream=sink) is None
    assert sink.getvalue() == ""


# ── 4.1 迁移脚本：幂等、不覆盖、边车一起搬 ───────────────────────────────────


def _seed_legacy_row(path, *, as_of="2025-01-06", symbol="600519.SH"):
    """在 `path` 建一个旧式单文件库，并写一条归属未知（legacy）的记录。"""
    import ledgerdata as ld
    from fd_quant.caller import LEGACY
    from fd_quant.ledger import LedgerStore

    store = LedgerStore(path)
    rec = store.record(
        caller=LEGACY, as_of=as_of, symbol=symbol,
        pack=ld.PACK, skill=ld.SKILL,
        payload=ld.rating(as_of, symbol, side="buy", pack=ld.PACK, skill=ld.SKILL),
    )
    return rec["decision_id"]


def _rows(path):
    import sqlite3

    con = sqlite3.connect(path)
    try:
        return con.execute(
            "SELECT decision_id, caller, as_of, symbol, payload FROM decisions ORDER BY decision_id"
        ).fetchall()
    finally:
        con.close()


def test_migration_moves_the_stray_ledger_and_keeps_rows(tmp_path):
    from fd_quant.ledger.store import migrate_stray_ledger

    stray = tmp_path / DEFAULT_DB_FILENAME
    _seed_legacy_row(stray)
    before = _rows(stray)

    out = migrate_stray_ledger(tmp_path)
    target = tmp_path / "legacy" / LEDGER_DB_FILENAME
    assert "已把" in out
    assert target.is_file() and not stray.exists()
    assert _rows(target) == before, "迁移不得改动任何一行"


def test_migration_is_idempotent(tmp_path):
    """跑两次：第二次跳过，数据与第一次逐条相同（任务判据原话）。"""
    from fd_quant.ledger.store import migrate_stray_ledger

    stray = tmp_path / DEFAULT_DB_FILENAME
    _seed_legacy_row(stray)
    migrate_stray_ledger(tmp_path)
    target = tmp_path / "legacy" / LEDGER_DB_FILENAME
    first = _rows(target)

    out2 = migrate_stray_ledger(tmp_path)
    assert "没有可搬" in out2 or "跳过" in out2
    assert _rows(target) == first


def test_existing_target_is_never_overwritten(tmp_path):
    """目标已存在 ⇒ 跳过、不报错、不改内容。覆盖等于用一份旧数据盖掉另一份。"""
    from fd_quant.ledger.store import migrate_stray_ledger

    stray = tmp_path / DEFAULT_DB_FILENAME
    _seed_legacy_row(stray)
    target_dir = tmp_path / "legacy"
    target_dir.mkdir()
    target = target_dir / LEDGER_DB_FILENAME
    target.write_bytes(b"NOT-A-DB-SENTINEL")

    out = migrate_stray_ledger(tmp_path)
    assert "跳过" in out
    assert target.read_bytes() == b"NOT-A-DB-SENTINEL"
    assert stray.exists(), "跳过时源文件也不该被动"


def test_wal_sidecars_move_with_the_database(tmp_path):
    """`-wal`/`-shm` 必须一起搬：只搬主文件会丢掉已提交但未 checkpoint 的帧。"""
    from fd_quant.ledger.store import migrate_stray_ledger

    stray = tmp_path / DEFAULT_DB_FILENAME
    _seed_legacy_row(stray)
    (tmp_path / f"{DEFAULT_DB_FILENAME}-wal").write_bytes(b"wal-frames")
    (tmp_path / f"{DEFAULT_DB_FILENAME}-shm").write_bytes(b"shm")

    migrate_stray_ledger(tmp_path)
    legacy = tmp_path / "legacy"
    assert (legacy / f"{LEDGER_DB_FILENAME}-wal").read_bytes() == b"wal-frames"
    assert (legacy / f"{LEDGER_DB_FILENAME}-shm").exists()
    assert not (tmp_path / f"{DEFAULT_DB_FILENAME}-wal").exists()


def test_dry_run_moves_nothing(tmp_path):
    from fd_quant.ledger.store import migrate_stray_ledger

    stray = tmp_path / DEFAULT_DB_FILENAME
    stray.write_bytes(b"x")
    out = migrate_stray_ledger(tmp_path, dry_run=True)
    assert "dry-run" in out
    assert stray.exists() and not (tmp_path / "legacy").exists()


def test_cli_prints_the_outcome(tmp_path, capsys):
    """脚本入口能跑通（`--root` + `--dry-run`），且把结局打在 stdout。"""
    import importlib.util

    stray = tmp_path / DEFAULT_DB_FILENAME
    stray.write_bytes(b"x")
    spec = importlib.util.spec_from_file_location(
        "migrate_ledger_to_legacy",
        str(store_mod.Path(__file__).resolve().parents[1] / "scripts" / "migrate_ledger_to_legacy.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.main(["--root", str(tmp_path), "--dry-run"]) == 0
    assert "dry-run" in capsys.readouterr().out


# ── 4.2 legacy 行内语义不受影响 ─────────────────────────────────────────────


def test_legacy_rows_readable_by_legacy_and_invisible_to_named_callers(tmp_path):
    """迁移后两层同时成立：`legacy` 显式查询读得回；具名 caller 一条都看不到。

    这条是 4.2 的判据原话。它在**两层**上都成立才叫双保险：存储层（分库=不同文件）
    与行层（`caller` 过滤）。只验一层的话，把 `caller` 列过滤删掉仍然绿。
    """
    from fd_quant.caller import LEGACY
    from fd_quant.ledger import LedgerStore
    from fd_quant.ledger.store import caller_db_path, migrate_stray_ledger

    stray = tmp_path / DEFAULT_DB_FILENAME
    decision_id = _seed_legacy_row(stray)
    migrate_stray_ledger(tmp_path)

    # 为什么用 `get()` 而不是 `retrieve()`：`retrieve` 只回**已结算**的记录
    # （"点内检索：只回结算日 ≤ as_of 的记录"），而判据问的是"读得回"。
    # 拿 retrieve 当判据会得到"两条列表都空"，看起来像迁移丢了数据。
    legacy_store = LedgerStore(tmp_path / "legacy" / LEDGER_DB_FILENAME)
    got = legacy_store.get(decision_id)
    assert got and got["decision_id"] == decision_id, "以 legacy 显式查询必须完整读回"
    assert got["payload"], "payload 不得在迁移中丢失"

    # `get()` 的返回体**不含 caller 列**（它是快照视图），所以「行内 caller 仍是
    # legacy」只能用 SQL 直接读——这也正是 4.2 说的"行内语义不受影响"：
    # 迁移只动文件位置，一行都不改写。
    import sqlite3

    con = sqlite3.connect(tmp_path / "legacy" / LEDGER_DB_FILENAME)
    try:
        callers = {r[0] for r in con.execute("SELECT caller FROM decisions")}
    finally:
        con.close()
    assert callers == {LEGACY}, f"行内归属被改动了：{callers}"

    # 第二层（存储层）：具名 caller 用的是**另一个文件**，压根没有这条记录。
    named = LedgerStore(caller_db_path("alice@example.com", {"FD_QUANT_LEDGER_ROOT": str(tmp_path)}))
    assert named.get(decision_id) is None, "具名 caller 的库里不得出现旧记录"
