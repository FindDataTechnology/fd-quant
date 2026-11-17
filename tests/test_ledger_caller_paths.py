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
