"""MCP 参数面收口（seal-ledger-path-and-caller-isolation 任务 2.1 / 2.3）。

三条判据，每条都按「能红」写：

1. 六个账本工具的**参数面**不含 `db_path`——这是 `list_tools()` 读到的、客户端
   实际看到的那份 schema，不是函数签名的推论。
2. 客户端硬塞 `db_path` **不改变落点**（且不在指定处建库）——实测 fastmcp 在
   schema 未声明时仍会透传参数，所以只摘 schema 不够，必须有剥参数的兜底。
3. 进程内关键字参数仍然可用——仓内 34 处测试与嵌入式调用依赖它，
   摘的必须是"暴露面"而不是"能力"。
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from fd_quant.server import PROCESS_ONLY_PARAMS, build_server

LEDGER_TOOLS = (
    "record_decision",
    "settle_decision",
    "settle_portfolio",
    "retrieve_decisions",
    "summarize_pack_performance",
    "list_pending_decisions",
)

PAYLOAD = {
    "schema_version": "1", "as_of": "2026-11-17", "subject": "600519.SS", "side": "hold",
    "conviction": None, "price_target": None, "target_evidence_ids": [],
    "basis": "参数面判据。", "evidence": [],
    "producer": {"pack": "p", "skill": "s"},
}


def _tools() -> dict:
    async def _run():
        app = build_server()
        return {t.name: t for t in await app.list_tools()}
    return asyncio.run(_run())


# ── 2.1 参数面 ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize("name", LEDGER_TOOLS)
def test_db_path_is_not_in_the_mcp_parameter_face(name):
    """`db_path` 不在客户端可见的 inputSchema 里。"""
    props = set(((_tools()[name].parameters or {}).get("properties") or {}))
    assert "db_path" not in props, f"{name} 仍暴露 db_path：{sorted(props)}"


@pytest.mark.parametrize("name", LEDGER_TOOLS)
def test_no_path_shaped_parameter_survives(name):
    """更一般的判据：参数面里没有任何"能改变落点"的字段。

    只钉 `db_path` 一个名字挡不住"下次换名再来"——判据要落在**能力**上：
    参数名里出现 path / dir / file 一律视为可疑。
    """
    props = set(((_tools()[name].parameters or {}).get("properties") or {}))
    suspicious = {p for p in props if any(k in p.lower() for k in ("path", "dir", "file"))}
    assert not suspicious, f"{name} 仍有路径类参数：{sorted(suspicious)}"


def test_process_only_params_is_declared():
    """收口清单本身可读：新增进程内参数必须从这里过，不能各写在函数里。"""
    assert "db_path" in PROCESS_ONLY_PARAMS


# ── 2.3 塞参数不改落点 ───────────────────────────────────────────────────────


def test_stray_db_path_cannot_redirect_storage(tmp_path):
    """客户端硬塞 `db_path`：不报错、不改变落点、不在指定处建库。

    这条是漏洞的可证伪形式——2026-11-17 实测该值曾经**真的**生效，
    账本被建在调用方指定的任意路径且 `mkdir(parents=True)` 建了父目录。
    """
    root = tmp_path / "root"
    evil = tmp_path / "evil" / "ledger.db"

    async def _run():
        import os
        os.environ["FD_QUANT_LEDGER_ROOT"] = str(root)
        try:
            app = build_server()
            return await app.call_tool("record_decision", {
                "as_of": "2026-11-17", "symbol": "600519.SS", "pack": "p", "skill": "s",
                "payload": PAYLOAD, "db_path": str(evil),
            })
        finally:
            os.environ.pop("FD_QUANT_LEDGER_ROOT", None)

    result = asyncio.run(_run())
    assert not getattr(result, "is_error", False), result
    assert not evil.exists() and not evil.parent.exists(), "调用方指定的路径被建出来了"
    assert root.is_dir(), "应落在服务端派生的根目录下"
    assert list(root.rglob("ledger.db")), "库没落在派生位置"


# ── 2.1 进程内入口仍在 ───────────────────────────────────────────────────────


def test_in_process_db_path_still_works(tmp_path):
    """进程内显式传路径仍可用——摘的是暴露面，不是能力。"""
    from fd_quant.ledger import tools as ledger_tools

    db = tmp_path / "in-process.db"
    out = ledger_tools.record_decision(
        as_of="2026-11-17", symbol="600519.SS", pack="p", skill="s",
        payload=PAYLOAD, db_path=str(db),
    )
    assert out["result"]["created"] is True
    assert db.is_file(), "进程内显式路径必须真的生效"
    # 信封仍不泄漏根：给相对范围或文件名。
    assert not out["db_path"].startswith("/"), out["db_path"]
