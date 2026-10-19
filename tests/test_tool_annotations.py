"""工具面协议属性契约（MCP annotations）。

协议层的行为属性决定客户端能否自动批准一次调用：把写库工具标成只读，客户端
会静默执行；把纯计算工具标成可写，客户端会对 46 个无副作用调用逐个弹确认。
两者都是可观测的损失，所以这里把「哪组该是什么」钉成断言。

`fd-quant` 的分组语义：计算面七族是纯函数（离线边界测试已钉住它们不得
import sqlite3/网络/LLM），账本族写本地 SQLite 且**重复提交被拒 ≠ 幂等成功**。
"""

from __future__ import annotations

import pytest

from fd_quant.surface import all_surface_tools, tools_by_group

LEDGER = "decision_ledger"

#: 账本族里真正写库的两个工具；另两个是只读检索/聚合。
LEDGER_WRITERS = {"record_decision", "settle_decision"}
LEDGER_READERS = {"retrieve_decisions", "summarize_pack_performance"}


def _by_name():
    return {spec.name: spec for spec in all_surface_tools()}


def test_every_tool_declares_all_four_hints():
    """四类属性必须齐备——缺一个客户端就无从判断。"""
    missing = []
    for spec in all_surface_tools():
        ann = spec.tool_annotations
        for key in ("readOnlyHint", "destructiveHint", "idempotentHint", "openWorldHint"):
            if key not in ann:
                missing.append(f"{spec.name}.{key}")
    assert missing == [], f"工具属性缺失: {missing}"


def test_compute_tools_are_readonly_and_idempotent():
    """计算面：只读 + 幂等 + 不触达外部世界。"""
    bad = []
    for group, specs in tools_by_group().items():
        if group == LEDGER:
            continue
        for spec in specs:
            ann = spec.tool_annotations
            if not (ann["readOnlyHint"] and ann["idempotentHint"] and not ann["openWorldHint"]):
                bad.append(spec.name)
    assert bad == [], f"计算面工具属性错误: {bad}"


def test_ledger_writers_are_not_readonly_nor_idempotent():
    """账本写入：非只读、非幂等（重复提交被拒是另一种结果，自动重试不安全）。"""
    tools = _by_name()
    for name in LEDGER_WRITERS:
        ann = tools[name].tool_annotations
        assert ann["readOnlyHint"] is False, name
        assert ann["idempotentHint"] is False, name
        # 追加式：不改写既有行，所以不是破坏性
        assert ann["destructiveHint"] is False, name


def test_ledger_readers_are_readonly():
    tools = _by_name()
    for name in LEDGER_READERS:
        ann = tools[name].tool_annotations
        assert ann["readOnlyHint"] is True, name
        assert ann["idempotentHint"] is True, name


def test_writer_and_compute_annotations_differ():
    """协议层的可区分性——这是本契约的目的本身。"""
    tools = _by_name()
    writer = tools["record_decision"].tool_annotations
    compute = tools["summarize_drawdown"].tool_annotations
    assert writer != compute
    assert writer["readOnlyHint"] is not compute["readOnlyHint"]


def test_annotations_are_copied_not_shared():
    """缺省属性必须是每个实例的副本：共享一个 dict 会让改一个工具污染全族。"""
    tools = _by_name()
    a = tools["summarize_drawdown"].tool_annotations
    b = tools["summarize_drawdown"].tool_annotations
    assert a is b  # 同一 spec 上读两次自然是同一对象
    # 不同 spec 之间不共享
    specs = all_surface_tools()
    assert specs[0].tool_annotations is not specs[1].tool_annotations
