"""工作流级知识随服务暴露（mcp-surface-contract 组 7）。

两条要求：

1. **资源可枚举可读**（7.1）：客户端 `list_resources` 能列出，
   `read_resource` 能读到内容，且内容是**多步组合指导**而不是单工具描述的
   复制——判据是含多个编号步骤或分节，且点名多个工具名。
2. **工具描述不含编排内容**（7.2）：描述回答"这个工具怎么用"，编排知识搬到
   资源里。若两边都有，就会各自漂移；而描述会被**每个**调用方读进上下文，
   编排段落的代价按调用次数线性放大。
"""
from __future__ import annotations

import asyncio
import re

import pytest
from fastmcp import Client

from fd_quant import workflows
from fd_quant.server import app

ALL_TOOLS = None


def _tool_names() -> set[str]:
    global ALL_TOOLS
    if ALL_TOOLS is None:
        ALL_TOOLS = {t.name for t in asyncio.run(app._list_tools())}
    return ALL_TOOLS


# ── 7.1 资源可枚举可读，且是多步组合 ──────────────────────────

def test_resources_are_enumerable_and_readable():
    async def go():
        async with Client(app) as c:
            out = {}
            for r in await c.list_resources():
                got = await c.read_resource(r.uri)
                out[str(r.uri)] = "".join(getattr(x, "text", "") for x in got)
            return out

    texts = asyncio.run(go())
    assert set(texts) == set(workflows.RESOURCES), "资源 URI 与声明表不一致"
    for uri, text in texts.items():
        assert len(text) > 400, f"{uri} 内容过短，不像工作流指导"


def test_resource_content_is_multi_step_not_single_tool_docs():
    """工作流资源必须点名多个工具；契约用法类资源不在此列。

    `contract/rating-usage` 讲的是 Rating 契约的边界态，不编排工具——要求它
    点名两个工具会把这条断言变成一个对内容类型的误判。
    """
    for uri, (_title, text) in workflows.RESOURCES.items():
        if "/contract/" in uri:
            continue
        steps = re.findall(r"^(\d)\. \*\*", text, re.M)
        sections = re.findall(r"^## ", text, re.M)
        assert len(steps) >= 2 or len(sections) >= 2, (
            f"{uri} 不像多步组合指导（步骤 {len(steps)} / 分节 {len(sections)}）")
        # 多步组合必然点名多个工具；只提一个说明它其实还是单工具说明
        # 两种写法都要认：正文中既有 `record_decision(as_of, ...)` 也有
        # `summarize_pack_performance` 这种只点名不带参数的写法。
        named = {t for t in _tool_names()
                 if re.search(rf"`{re.escape(t)}`", text)
                 or re.search(rf"\b{re.escape(t)}\(", text)}
        assert len(named) >= 2, f"{uri} 只点名了 {named or '没有'}，不是组合指导"


def test_named_tools_in_resources_actually_exist():
    """资源里点名的工具必须真在工具面上——否则指导会指向不存在的工具。"""
    for uri, (_title, text) in workflows.RESOURCES.items():
        bad = {t for t in re.findall(r"`([a-z_]+)`", text)} - _tool_names()
        # 允许出现字段名（dates / close / ok …），但像工具名的动词下划线串必须是真工具
        bad = {b for b in bad if b.split("_")[0] in _VERB_PREFIXES}
        assert not bad, f"{uri} 点名了不存在的工具：{sorted(bad)}"


def test_resource_uris_are_namespaced_by_service():
    """URI 全局命名空间：不前缀服务名会与其它服务的资源撞名。"""
    for uri in workflows.RESOURCES:
        assert uri.startswith("fd-quant://"), uri


# ── 7.2 工具描述不含工作流编排 ────────────────────────────────

#: 动词前缀：用来把"像工具名的串"与"字段名"分开（dates / close / ok 都是字段名）。
_VERB_PREFIXES = frozenset({
    "check", "validate", "compute", "summarize", "detect", "estimate",
    "build", "record", "settle", "retrieve", "read", "search", "list",
    "get", "rank", "split", "add", "resolve", "seed", "import", "register",
    "propose", "confirm", "consume", "document",
})

#: 编排信号的判据：描述里出现"下一步 / 然后调用 / 接着"这类跨工具指引。
ORCHESTRATION = re.compile(r"(下一步|接下来|然后调用|接着调用|调用顺序|依次调用|工作流)")


def test_no_tool_description_contains_orchestration():
    """工具描述不得含编排内容——那是资源的职责。

    若两边都有，描述与资源会各自漂移；而描述会被每个调用方读进上下文，
    编排段落的代价按调用次数线性放大。
    """
    async def go():
        async with Client(app) as c:
            return {t.name: (t.description or "") for t in await c.list_tools()}

    descs = asyncio.run(go())
    offenders = {n: d for n, d in descs.items() if ORCHESTRATION.search(d)}
    assert not offenders, (
        f"这些工具描述含编排内容，应移到资源里：{sorted(offenders)}")
