"""fd-quant 计算面 MCP 服务入口（stdio + HTTP 双传输）。

启动：
    python -m fd_quant.compute.server              # stdio（MCP 默认）
    python -m fd_quant.compute.server --http 8080  # HTTP（streamable-http）

纯计算面：不取数、不读库、不联网、不调 LLM。工具全部来自
`fd_quant.compute.all_tools()`——族的增删只需改族模块，本文件不动。
"""
from __future__ import annotations

import argparse
import sys

from fastmcp import FastMCP

from . import all_tools, missing_families

SERVER_NAME = "fd-quant-compute"


def build_server() -> FastMCP:
    app = FastMCP(
        name=SERVER_NAME,
        instructions=(
            "fd-quant 确定性计算面：技术指标 / 组合统计 / 基本面比率 / 估值算子 / "
            "regime 检测 / 因子统计。所有工具为纯计算——只对传入的序列与参数作答，"
            "不取数、不联网、不调 LLM。每个工具要求 as_of（ISO 日期）并列在输入序列上"
            "执行：含 as_of 之后的数据点会被拒绝并点名该日期。返回统一带 evidence"
            "（六字段：evidence_id/label/value/unit/as_of/source）与输入的起止日期。"
        ),
    )

    missing = missing_families()
    if missing:
        # 不静默：缺族时在 stderr 留痕，免得"工具面少了一族"被当成正常。
        print(
            f"[{SERVER_NAME}] WARNING: families not yet present: {', '.join(missing)}",
            file=sys.stderr,
        )

    for spec in all_tools():
        app.tool(spec.fn, name=spec.name, description=spec.doc)
    return app


app = build_server()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog=SERVER_NAME)
    parser.add_argument("--http", type=int, default=None, metavar="PORT",
                        help="serve streamable-http on PORT instead of stdio")
    parser.add_argument("--host", default="127.0.0.1",
                        help="HTTP bind host (default 127.0.0.1)")
    parser.add_argument("--list-tools", action="store_true",
                        help="print the tool inventory and exit")
    args = parser.parse_args(argv)

    if args.list_tools:
        for spec in all_tools():
            print(f"{spec.group:22s} {spec.name}")
        return 0

    if args.http:
        app.run(transport="http", host=args.host, port=args.http)
        return 0

    app.run()  # stdio
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
