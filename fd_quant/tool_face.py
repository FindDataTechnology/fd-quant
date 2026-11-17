"""MCP 参数面收口：把进程内专用参数从工具签名里摘掉（seal-ledger-path-and-caller-isolation 2.1 / 2.3）。

**问题**：fastmcp 从函数签名生成 `inputSchema`，所以 `db_path` 只要还在签名上，
它就在 MCP 参数面里——调用方能把账本建在服务器任意可写位置，而「按 caller 隔离」
变成由调用方自愿（改一下 `db_path` 就绕过）。实测（2026-11-17）确认六个账本工具
全部暴露该参数，且传任意路径真的会在那里建库。

**为什么不在 `tools.py` 删掉这个参数**：仓内 34 处测试与嵌入式调用都以关键字传
`db_path`（隔离临时库），删签名会把它们全推翻，而那些调用是**进程内**的、
本就不经过 MCP。所以要摘的是**暴露面**，不是能力。

**两条实现路径，选了后者**：
1. 注册时改 `Tool.parameters` 的 JSON Schema —— 实测不够：schema 里没有该字段时
   fastmcp 仍把客户端多传的值透传给函数（`stray db_path` 真的生效了）。
2. **包装函数的 `__signature__` + 中间件剥掉多余参数** —— 双保险：签名决定 schema，
   中间件决定「客户端硬塞进来的值一定被丢掉」。缺任何一半都能被绕过。

包装函数用 `*args, **kwargs` + 显式 `__signature__`，而不是 `exec` 生成源码：
签名对象是 fastmcp 唯一读的东西，`exec` 只是把同一份信息写成字符串再解析回来。
"""

from __future__ import annotations

import functools
import inspect
from typing import Any, Callable, Iterable

from fastmcp.server.middleware import Middleware


def hide_params(fn: Callable[..., Any], hidden: Iterable[str]) -> Callable[..., Any]:
    """返回一个签名里不含 `hidden` 的包装函数（调用时这些参数一律取默认值）。

    **调用方给的值被丢弃，不是被拒绝**：拒绝会让「协议层多传一个字段」变成一次
    调用失败，而协议本身允许客户端带额外字段。要保证的是「它不能改变落点」，
    不是「传了就报错」——前者是安全要求，后者只是洁癖。
    """
    hidden_names = tuple(hidden)
    sig = inspect.signature(fn)
    kept = [p for name, p in sig.parameters.items() if name not in hidden_names]

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        for name in hidden_names:
            kwargs.pop(name, None)
        return fn(*args, **kwargs)

    wrapper.__signature__ = sig.replace(parameters=kept)  # type: ignore[attr-defined]
    # `functools.wraps` 复制了 `__annotations__`；签名换掉后注解也必须跟着换，
    # 否则 pydantic 会按「有 db_path 注解、签名里没有」处理并报 `KeyError`。
    wrapper.__annotations__ = {
        k: v for k, v in getattr(fn, "__annotations__", {}).items() if k not in hidden_names
    }
    return wrapper


class StripExtraArgs(Middleware):
    """兜底：客户端在请求体里硬塞的被隐藏参数，进入工具前一律丢弃。

    为什么需要它（而不是只靠签名）：实测 fastmcp 在 schema 未声明某字段时**仍会**
    把它透传给底层函数——于是「schema 里看不见」不等于「传了没用」。两层缺一不可。
    """

    def __init__(self, names: Iterable[str]) -> None:
        self.names = frozenset(names)

    async def on_call_tool(self, context, call_next):  # noqa: ANN001 — fastmcp 的 context 无公开类型
        message = getattr(context, "message", None)
        args = getattr(message, "arguments", None)
        if isinstance(args, dict):
            stray = self.names.intersection(args)
            if stray:
                message.arguments = {k: v for k, v in args.items() if k not in self.names}
        return await call_next(context)
