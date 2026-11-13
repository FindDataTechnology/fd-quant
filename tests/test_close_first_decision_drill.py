"""最小闭合演练（close-first-decision-loop tasks 3.1-3.4）。

**它证明的是"链路在技术上可以闭合"，不是"业务上有人在用"。** 两者的区别就是本文件
反复断言的两件事：演练能结出 `hit`/`miss`，且演练**进不了**任何真实调用者的口径。

三条不可退让：
1. 演练能产出 `verdict ∈ {hit, miss}` 且反思非空（不依赖基准序列、不依赖最新价）；
2. 同一键重复执行不产生第二行（账本自己的幂等键保证）；
3. 演练记录对生产档与真实调用者不可见——否则演练数据会污染客户命中率。
"""
from __future__ import annotations

import importlib.util
import json
import os
import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DRILL = ROOT / "scripts" / "close_first_decision_drill.py"
PROBE = ROOT / "scripts" / "decision_loop_probe.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = mod  # dataclass 需要回查 sys.modules
    spec.loader.exec_module(mod)
    return mod


drill = _load(DRILL, "close_first_decision_drill")
probe = _load(PROBE, "decision_loop_probe_for_drill")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("FD_QUANT_CALLER", raising=False)
    yield


def run_drill(*args: str):
    import io
    from contextlib import redirect_stdout

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = drill.run(list(args) + ["--json"])
    return rc, json.loads(buf.getvalue())


# ── 3.1 结出第一条 ──────────────────────────────────────────


def test_drill_produces_decided_settlement(tmp_path):
    db = tmp_path / "d.db"
    rc, out = run_drill("--db", str(db))
    assert rc == 0, out
    assert out["verdict"] in ("hit", "miss")
    assert out["rule_id"] == "metric_threshold"
    assert out["reflection_present"] is True
    assert out["gap_kinds"] is None, "有结论的记录不许带缺口"
    assert out["counts"]["decisions"] == 1
    assert out["counts"]["settlements"] == 1


def test_drill_miss_variant_is_also_decided(tmp_path):
    """miss 也是"有结论"：闭环不该只能结出 hit。"""
    db = tmp_path / "m.db"
    rc, out = run_drill("--db", str(db), "--verdict", "miss")
    assert rc == 0
    assert out["verdict"] == "miss"
    assert out["gap_kinds"] is None


# ── 3.2 幂等 ────────────────────────────────────────────────


def test_drill_is_idempotent(tmp_path):
    db = tmp_path / "i.db"
    _, first = run_drill("--db", str(db))
    _, second = run_drill("--db", str(db))
    assert first["decision_id"] == second["decision_id"]
    assert second["created"] is False, "第二次必须命中幂等键并返回既有 id"
    con = sqlite3.connect(db)
    assert con.execute("SELECT COUNT(*) FROM decisions").fetchone()[0] == 1
    assert con.execute("SELECT COUNT(*) FROM settlements").fetchone()[0] == 1
    con.close()


# ── 3.3 演练不得进生产口径 ──────────────────────────────────


def test_drill_records_are_not_visible_to_client_caller(tmp_path):
    """真实调用者的聚合里看不到演练记录（caller 隔离，不是靠前缀过滤）。"""
    from fd_quant.ledger.store import LedgerStore

    db = tmp_path / "iso.db"
    run_drill("--db", str(db))
    store = LedgerStore(db)
    out = store.stats_by_pack(caller="alice", as_of="2026-11-13")
    assert out["rows"] == [], "演练记录不得出现在客户口径的聚合里"
    own = store.stats_by_pack(caller=drill.DRILL_CALLER, as_of="2026-11-13")
    assert sum(r["samples"] for r in own["rows"]) == 1


def test_probe_production_mode_ignores_drill(tmp_path):
    """探针 production 档不把演练算作闭合——这是 P0 双层判据的另一半。

    只有演练记录时，production 档看到的是"一条真实落账都没有"，即 `not_started`
    （退出码 3）。判据要的是"绝不判为通过"，别把编码钉死在某一个非零值上——
    否则下次改状态机就要改这个测试，而它想守的其实是"演练不进生产"这一件事。
    """
    db = tmp_path / "pp.db"
    run_drill("--db", str(db))

    import io
    from contextlib import redirect_stdout

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = probe.run(["--db", str(db), "--as-of", "2026-11-13", "--json"])
    out = json.loads(buf.getvalue())
    assert rc != 0 and out["state"] != "ok", out
    assert out["mode"] == "production"
    assert out["checked"]["existence"].get("decided", 0) == 0
    assert out["state"] == "not_started"


# ── 3.4 演练档三条判据全绿 ──────────────────────────────────


def test_probe_drill_mode_passes(tmp_path, monkeypatch):
    monkeypatch.delenv("FD_OPEN_DATA_DSN", raising=False)
    monkeypatch.delenv("FD_OPEN_DATA_MCP_DATABASE_URL", raising=False)
    db = tmp_path / "pd.db"
    run_drill("--db", str(db))

    import io
    from contextlib import redirect_stdout

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = probe.run(["--db", str(db), "--mode", "drill",
                        "--as-of", "2026-11-13", "--json"])
    out = json.loads(buf.getvalue())
    assert rc == 0, out
    assert out["checked"]["existence"]["decided"] == 1
    assert out["checked"]["freshness"]["ok"] is True
    assert out["checked"]["attribution"]["ok"] is True


# ── 身份纪律 ────────────────────────────────────────────────


def test_drill_refuses_non_drill_caller(tmp_path):
    """演练必须在演练身份下跑；否则会污染真实调用者的统计。"""
    os.environ["FD_QUANT_CALLER"] = "alice"
    try:
        rc = drill.run(["--db", str(tmp_path / "x.db")])
    finally:
        os.environ.pop("FD_QUANT_CALLER", None)
    assert rc == 2


def test_default_caller_is_drill_namespaced():
    assert drill.DRILL_CALLER.startswith("drill:")
    assert drill.DRILL_PACK.startswith("drill:")
