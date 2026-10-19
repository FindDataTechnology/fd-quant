"""账本层的离线边界（task 5.2 的"断网运行通过" + 依赖清单核查）。

计算面已有自己的一份边界回归（`test_offline_boundary.py`）；账本是**有状态**的一层，
它的边界问法不同：

1. 账本**不得出网**——它读的是本地 SQLite 文件，价格由调用方传入。所以"封网后整条
   落账→结算→检索→聚合链路仍跑通"才是它"不取数"的运行时证据。
2. 账本**不得引入新依赖**——只用 stdlib `sqlite3` + 既有的 numpy/jsonschema。
   这条以源码 AST 扫描断言（比人工看 requirements 可靠）。
3. 账本**不得出现在计算层**——`fd_quant/compute/` 的封闭性不被本改动打破。
"""
from __future__ import annotations

import ast
import json
import socket
import sys
from pathlib import Path

import pytest

import ledgerdata as ld
from fd_quant.ledger import LedgerStore

#: 单用户本地账本场景的测试调用者（隔离新增维度后必须显式给出）
CALLER = "test-caller"

REPO = Path(__file__).resolve().parents[1]
LEDGER_DIR = REPO / "fd_quant" / "ledger"


@pytest.fixture()
def network_blocked(monkeypatch):
    """封掉一切出网路径（与计算面同一套闸门）。"""

    def _deny(*a, **k):
        raise AssertionError(
            "network access attempted from the ledger — it must be local-only "
            "(prices are supplied by the caller; storage is a local SQLite file)"
        )

    monkeypatch.setattr(socket.socket, "connect", _deny)
    monkeypatch.setattr(socket.socket, "connect_ex", _deny)
    monkeypatch.setattr(socket, "create_connection", _deny)
    monkeypatch.setattr(socket, "getaddrinfo", _deny)
    monkeypatch.setattr(socket, "gethostbyname", _deny)
    yield


def test_full_chain_runs_with_the_network_blocked(tmp_path, network_blocked):
    """封网下走完整链路：落账 → 提前结算被拒 → 结算 → 幂等重结算 → 检索 → 聚合。"""
    from fd_quant.ledger import SettlementRefused

    store = LedgerStore(tmp_path / "offline.db")
    r = store.record(caller=CALLER, as_of="2025-01-06", symbol="600519.SS", pack=ld.PACK, skill=ld.SKILL,
                     payload=ld.rating("2025-01-06", "600519.SS"))
    # 提前结算被拒（序列 3 日 < 持有期 5 日）
    three = ld.dates(3)
    with pytest.raises(SettlementRefused) as ei:
        store.settle(decision_id=r["decision_id"], symbol_dates=three,
                     symbol_prices=[1.0, 1.0, 1.0], benchmark_dates=three,
                     benchmark_prices=[1.0, 1.0, 1.0], as_of=three[-1])
    assert ei.value.missing_trading_days == 3

    six = ld.dates(6)
    s = store.settle(decision_id=r["decision_id"], symbol_dates=six,
                     symbol_prices=[100.0] * 5 + [110.0], benchmark_dates=six,
                     benchmark_prices=[1000.0] * 6, as_of=six[-1],
                     reflection="封网下的结算。")
    assert s["created"] is True and s["alpha"] == pytest.approx(0.10, abs=1e-12)

    again = store.settle(decision_id=r["decision_id"], symbol_dates=six,
                         symbol_prices=[100.0] * 5 + [110.0], benchmark_dates=six,
                         benchmark_prices=[1000.0] * 6, as_of=six[-1])
    assert again["created"] is False
    assert store.row_counts()["settlements"] == 1

    found = store.retrieve(caller=CALLER, symbol="600519.SS", as_of="2025-12-31")
    assert found["counts"]["same_symbol"] == 1
    assert found["same_symbol"][0]["settlement"]["reflection"] == "封网下的结算。"
    stats = store.stats_by_pack(caller=CALLER, as_of="2025-12-31")
    assert stats["rows"][0]["samples"] == 1


def test_ledger_tools_run_with_the_network_blocked(tmp_path, network_blocked):
    """工具面（MCP 调用的形态）同样封网可用。"""
    from fd_quant.ledger import tools

    db = str(tmp_path / "tools.db")
    rec = tools.record_decision(as_of="2025-01-06", symbol="600519.SS", pack=ld.PACK,
                                skill=ld.SKILL,
                                payload=ld.rating("2025-01-06", "600519.SS").to_dict(),
                                db_path=db)
    six = ld.dates(6)
    tools.settle_decision(decision_id=rec["result"]["decision_id"], symbol_dates=six,
                          symbol_prices=[100.0] * 5 + [110.0], benchmark_dates=six,
                          benchmark_prices=[1000.0] * 6, as_of=six[-1], db_path=db)
    out = tools.retrieve_decisions(symbol="600519.SS", as_of="2025-12-31", db_path=db)
    json.dumps(out, allow_nan=False)
    assert out["result"]["counts"]["same_symbol"] == 1


def test_ledger_source_imports_only_stdlib_and_the_existing_layers():
    """依赖清单核查：账本源码的第三方 import 只许是 `fd_quant.*` 与标准库。

    白名单刻意写成**封闭**集合：新增任何第三方依赖都必须显式改这里，从而在评审时
    可见（spec: 不引入新依赖；sqlite3 是 stdlib）。
    """
    stdlib_allowed = {
        "__future__", "json", "math", "os", "sqlite3", "uuid", "contextlib", "datetime",
        "pathlib", "typing", "collections", "abc", "itertools", "functools", "re", "sys",
        "argparse", "logging", "tempfile", "hashlib", "enum", "dataclasses", "inspect",
        "traceback", "subprocess", "time", "copy", "operator", "textwrap", "warnings",
    }
    allowed = stdlib_allowed | {"fd_quant"}
    seen: set[str] = set()
    for py in sorted(LEDGER_DIR.glob("*.py")):
        tree = ast.parse(py.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    seen.add(a.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                if node.level == 0 and node.module:
                    seen.add(node.module.split(".")[0])
                elif node.level > 0:
                    seen.add("fd_quant")  # 相对 import = 本包内部
    unexpected = sorted(seen - allowed)
    assert not unexpected, (
        f"ledger imports packages outside the whitelist: {unexpected} — the ledger must "
        f"stay on stdlib sqlite3 + the existing fd_quant layers"
    )


def test_ledger_imports_no_network_or_llm_client():
    """禁止清单扫描：账本源码里不得出现网络 / LLM / 数据源客户端。"""
    banned = {
        "requests", "httpx", "urllib", "socket", "aiohttp", "sqlalchemy", "psycopg",
        "psycopg2", "pymysql", "openai", "anthropic", "litellm", "transformers", "torch",
        "langchain", "chromadb", "redis", "boto3", "yfinance", "akshare", "tushare",
        "pandas", "numpy",  # 账本不需要数值库：收益是标量算术，别把 BLAS 归约引进来
    }
    hits: list[str] = []
    for py in sorted(LEDGER_DIR.glob("*.py")):
        for lineno, line in enumerate(py.read_text().splitlines(), 1):
            ls = line.strip()
            if not (ls.startswith("import ") or ls.startswith("from ")):
                continue
            for mod in banned:
                if f" {mod}" in ls or ls.startswith(f"import {mod}") or ls.startswith(f"from {mod}"):
                    hits.append(f"{py.name}:{lineno}: {ls}")
    assert not hits, "ledger imported a banned module:\n" + "\n".join(hits)


def test_no_new_third_party_dependency_was_added():
    """pyproject 依赖清单未被本改动扩张（账本只用 stdlib + 既有依赖）。"""
    import tomllib

    data = tomllib.loads((REPO / "pyproject.toml").read_text())
    deps = data["project"]["dependencies"]
    allowed = {"jsonschema", "numpy"}
    for d in deps:
        base = d.split(">=")[0].split("==")[0].split("<")[0].split("[")[0].strip().lower()
        assert base in allowed, f"unexpected dependency {d!r}: the ledger adds none"


def test_ledger_is_importable_without_the_server_extra():
    """账本不依赖 fastmcp：`fd_quant.ledger` 只 import 契约层与 stdlib（工具层才碰 ToolSpec）。

    `ToolSpec` 来自 `fd_quant.compute`（纯计算包，不 import fastmcp），故账本能被
    没装传输层 extras 的环境使用——这一点对"账本可在离线回归里跑"很重要。
    """
    import fd_quant.ledger as ledger
    import fd_quant.ledger.settlement as settlement
    import fd_quant.ledger.store as store

    for mod in (ledger, settlement, store):
        src = Path(mod.__file__).read_text(encoding="utf-8")
        assert "fastmcp" not in src, mod.__name__


def test_ledger_does_not_leak_into_the_compute_package():
    """计算层封闭性回归：`fd_quant/compute/` 内不存在账本或 sqlite 依赖。"""
    banned_terms = ("sqlite3", "ledger", "policy_ledger")
    hits: list[str] = []
    for py in sorted((REPO / "fd_quant" / "compute").glob("*.py")):
        text = py.read_text(encoding="utf-8")
        for term in banned_terms:
            if term in text and py.name != "server.py":
                # 允许注释提及；只在 import 行上判定
                for lineno, line in enumerate(text.splitlines(), 1):
                    ls = line.strip()
                    if (ls.startswith("import ") or ls.startswith("from ")) and term in ls:
                        hits.append(f"{py.name}:{lineno}: {ls}")
    assert not hits, "compute layer imports ledger/sqlite:\n" + "\n".join(hits)


def test_surface_module_keeps_the_dependency_direction_one_way():
    """方向性依赖：`surface` 同时 import compute 与 ledger；反过来不成立。"""
    surface = (REPO / "fd_quant" / "surface.py").read_text(encoding="utf-8")
    assert "from .compute import" in surface
    assert "from .ledger import" in surface
    for py in sorted((REPO / "fd_quant" / "compute").glob("*.py")):
        text = py.read_text(encoding="utf-8")
        assert "from ..ledger" not in text and "from ..surface" not in text, py.name
