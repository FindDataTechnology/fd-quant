"""输出结构声明契约（mcp-result-contract）。

三件事必须成立：

1. **齐备**：50 个工具都有 `output_schema`，或显式登记为未声明（本仓为 0 个）。
2. **与真实返回一致**：每条 schema 必须能校验该工具的真实输出。这条最关键——
   fastmcp 3.x 在**调用时**按 `output_schema` 校验返回值，声明写错的后果不是
   "客户端解析失败"而是"工具直接报 Output validation error"。故这里对每个工具
   真跑一次并按 schema 校验。
3. **契约引用可解**：账本工具里的 `payload` / `target_weights` 用 `$ref` 指向
   `schemas/rating.schema.json` 与 `target-weights.schema.json`，这两个 `$ref`
   必须能解析到随包分发的契约文件——否则客户端拿到的是悬空引用。

另外验证"声明确实会被校验"：把某工具的 schema 故意改错后，一次真实调用必须
报错。没有这条，上面三条都可能在一个不生效的校验链上全绿。
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))

from fd_quant.compute import all_tools  # noqa: E402
from fd_quant.surface import all_surface_tools  # noqa: E402
from fd_quant.output_schemas import (  # noqa: E402
    CONTRACT_SCHEMAS, contract_schema, load, output_schema_for, undeclared,
)

HINTS = ("required", "properties", "type")


def _kwargs_for(spec):
    import test_offline_boundary as tob

    return tob._kwargs_for(spec)


@pytest.fixture(scope="module")
def registry():
    from referencing import Registry, Resource
    from referencing.jsonschema import DRAFT202012

    resources = []
    for name in CONTRACT_SCHEMAS:
        s = contract_schema(name)
        assert s is not None, f"契约 schema 缺失：{name}"
        resources.append((s["$id"], Resource.from_contents(s, default_specification=DRAFT202012)))
    return Registry().with_resources(resources)


def test_every_tool_has_a_declared_schema():
    # 上牌面 = 计算面 + 账本面；只拿 compute 比会把账本四工具误判成"多出来"。
    tools = {t.name for t in all_surface_tools()}
    declared = set(load()["tools"])
    assert declared == tools, (
        f"声明与工具面不一致：缺 {sorted(tools - declared)}，多 {sorted(declared - tools)}"
    )
    assert undeclared() == [], f"本仓不应有未声明工具，实为 {undeclared()}"


def test_each_schema_is_itself_valid():
    from jsonschema import Draft202012Validator

    for name, schema in load()["tools"].items():
        for key in HINTS:
            assert key in schema, f"{name} 的 schema 缺 {key}"
        Draft202012Validator.check_schema(schema)


@pytest.mark.parametrize("spec", all_tools(), ids=lambda s: s.name)
def test_schema_accepts_the_tools_real_output(spec, registry):
    """真跑一次，按声明校验——这是"声明与实际一致"的运行时证据。"""
    from jsonschema import Draft202012Validator

    schema = output_schema_for(spec.name)
    assert schema is not None, f"{spec.name} 未声明输出结构"
    kw = _kwargs_for(spec)
    if kw is None:
        pytest.skip(f"{spec.name}: 无声明样本（由合规闸门覆盖）")
    result = spec.fn(**kw)
    errors = list(Draft202012Validator(schema, registry=registry).iter_errors(result))
    assert not errors, (
        f"{spec.name} 的真实返回不符合其声明：{errors[0].message}"
        f"（路径 {'/'.join(str(p) for p in errors[0].absolute_path)}）"
    )


def test_packaged_schemas_match_repo_copy():
    """包内副本必须与仓根权威副本逐字相同。

    运行时读包内（随 wheel 分发），文档与非 Python 消费者读仓根。两份不一致
    意味着"文档里写的形状"与"服务实际校验的形状"不是同一个——而这种偏差
    只在某一方被编辑时出现，不会自己暴露。
    """
    repo_dir = REPO / "schemas"
    pkg_dir = REPO / "fd_quant" / "schemas"
    for f in sorted(repo_dir.glob("*.json")):
        twin = pkg_dir / f.name
        assert twin.exists(), f"包内缺 {f.name}（wheel 里会退化成未声明）"
        assert json.loads(f.read_text()) == json.loads(twin.read_text()), (
            f"{f.name}：仓根与包内副本不一致"
        )
    extra = sorted({p.name for p in pkg_dir.glob("*.json")} - {p.name for p in repo_dir.glob("*.json")})
    assert extra == [], f"包内有仓根没有的 schema：{extra}"


def test_ledger_payload_refs_resolve_to_the_contract_schemas():
    """账本工具的 payload / target_weights 必须引用可解析的契约真源。"""
    rating = contract_schema("rating.schema.json")
    tw = contract_schema("target-weights.schema.json")
    assert rating and tw

    record = output_schema_for("record_decision")
    payload = record["properties"]["result"]["properties"]["payload"]
    assert payload.get("$ref") == rating["$id"], (
        "record_decision.payload 应引用 Rating 契约，"
        f"实为 {payload!r}"
    )

    retrieve = output_schema_for("retrieve_decisions")
    same = retrieve["properties"]["result"]["properties"]["same_symbol"]
    item_props = same["items"]["properties"]
    assert item_props["payload"]["$ref"] == rating["$id"]
    tw_ref = item_props["target_weights"]
    # 可空性必须写在 anyOf 里：$ref 与兄弟键并存时兄弟键按 2020-12 语义被忽略，
    # 把 type 与 $ref 并列会让"可为 null"这条声明形同不存在。
    assert any(r.get("$ref") == tw["$id"] for r in tw_ref["anyOf"]), tw_ref


def test_no_ref_has_ignored_sibling_keywords():
    """`$ref` 不得与其它键并列——否则那些兄弟键会被静默忽略。"""

    def walk(node, path=""):
        if isinstance(node, dict):
            if "$ref" in node and len(node) > 1:
                pytest.fail(f"{path} 的 $ref 与兄弟键并存：{sorted(node)}")
            for k, v in node.items():
                walk(v, f"{path}.{k}")
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, f"{path}[{i}]")

    walk(load()["tools"])


def test_a_wrong_declaration_is_caught_at_call_time():
    """声明确实会被校验：故意写错一条，真实调用必须报错。

    没有这条断言，上面所有"声明与输出一致"都可能在一条不生效的校验链上全绿。
    """
    from fastmcp import Client

    import fd_quant.output_schemas as oschemas
    import fd_quant.server as srv

    real = load()
    broken = json.loads(json.dumps(real))
    # value 实际是 array（warmup 区含 null），声明成 string 必然对不上
    broken["tools"]["compute_sma"]["properties"]["value"] = {"type": "string"}

    oschemas.load.cache_clear()
    original_read = oschemas.SCHEMA_PATH.read_text

    import tempfile

    tmp = Path(tempfile.mkdtemp()) / "outputs.json"
    tmp.write_text(json.dumps(broken, ensure_ascii=False), encoding="utf-8")
    oschemas.SCHEMA_PATH = tmp
    try:
        app = srv.build_server()
        kw = _kwargs_for(next(t for t in all_tools() if t.name == "compute_sma"))

        async def call():
            async with Client(app) as c:
                return await c.call_tool("compute_sma", kw)

        with pytest.raises(Exception) as exc:
            asyncio.run(call())
        assert "Output validation error" in str(exc.value)
    finally:
        oschemas.SCHEMA_PATH = Path(original_read.__self__)  # type: ignore[attr-defined]
        oschemas.load.cache_clear()
