"""输出结构声明（mcp-result-contract）：从 `schemas/outputs.json` 读，随包分发。

**为什么是数据文件而不是 Python 常量**：schema 由 `scripts/gen_output_schemas.py`
从运行时实测生成，是 50 个工具 × 几十个键的形状表；写进代码会让它变成第二个
需要手工维护的真源。

**两份位置**：权威副本在仓根 `schemas/`（供非 Python 消费者与文档引用），
运行时读的是包内副本 `fd_quant/schemas/`——只有包内文件才会随 wheel 分发
（setuptools 的 package-data 不能引用包目录之外的文件）。两份由
`tests/test_output_schemas.py::test_packaged_schemas_match_repo_copy` 钉住一致。

**缺失时的行为**：文件不存在就返回空表并**不**抛错——服务仍能起，但每个工具
都会在 `tools/list` 里显示为未声明输出结构。契约要求"无静默空缺"，故
`selfcheck_surface.py` 会把"声明数 < 工具数"报成失败，而不是让服务拒绝启动。
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

#: 运行时读包内副本（随 wheel 分发）；仓根 `schemas/` 是权威副本，两者必须逐字相同。
SCHEMA_DIR = Path(__file__).resolve().parent / "schemas"
SCHEMA_PATH = SCHEMA_DIR / "outputs.json"

#: 契约真源（payload / target_weights 引用的两份 schema）随包分发，
#: 以便调用方解析 `$ref` 时能就地取到，不必访问外网。
CONTRACT_SCHEMAS = ("rating.schema.json", "target-weights.schema.json")


@lru_cache(maxsize=1)
def load() -> dict[str, Any]:
    """读取声明表（进程内只读一次）。文件缺失返回空表，见模块 docstring。"""
    if not SCHEMA_PATH.exists():
        return {"schema": None, "tools": {}, "undeclared": [], "envelope": []}
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def output_schema_for(tool: str) -> Optional[dict]:
    """某工具的输出结构；未声明返回 None（调用方据此显式标未声明）。"""
    return load()["tools"].get(tool)


def undeclared() -> list[str]:
    """显式登记的未声明工具——"无静默空缺"的可见性载体。"""
    return list(load().get("undeclared") or [])


def contract_schema(name: str) -> Optional[dict]:
    """取契约 schema 原文（rating / target-weights），优先包内副本。"""
    for base in (SCHEMA_DIR, SCHEMA_DIR.parent.parent / "schemas"):
        path = base / name
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    return None
