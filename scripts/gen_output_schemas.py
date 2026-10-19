#!/usr/bin/env python3
"""从运行时实测输出生成各工具的输出结构声明（mcp-result-contract task 2.1/2.2）。

**为什么不手写 schema**：fd-quant 的 46 个计算工具产出 40 种不同的载荷形状，
手写必然漂——而 fastmcp 3.x 会在**调用时**按 `output_schema` 校验返回值，
写错的后果不是"客户端解析失败"而是"工具直接报 Output validation error"。
所以唯一可靠的来源是跑一遍真实工具，把观察到的键与类型记下来。

**为什么不直接把观察结果当 schema 用**：观测只能说明"这一次返回了什么"，
说明不了"哪些键一定有、哪些可为 null"。故本生成器做两件事：

1. 用**多组输入**（不同长度、不同种子、含边界样本）跑同一工具，取键集合的交集
   作为 `required`，并集作为 `properties`（差集标注为可选）。
2. 类型取并集（如 `["number","null"]`），因为 `null` 是这些工具表达"该值不存在"
   的正式方式（warmup 区、样本不足、基准缺失），不是异常。

**账本族的 payload 用既有契约**：`record_decision` 的 `payload` 是
`schemas/rating.schema.json`，`target_weights` 是 `target-weights.schema.json`。
这两份已经是真源（契约层测试在跑），故这里**引用**而不是重新推导。

用法：
    python3 scripts/gen_output_schemas.py            # 写 schemas/outputs.json
    python3 scripts/gen_output_schemas.py --check    # 只比对，漂移则退出码 1
退出码：0 一致或已写入 / 1 --check 下发现漂移。
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "schemas" / "outputs.json"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))

#: 计算面的统一外壳（result_envelope 保证）。每个工具都带，故提到公共部分。
ENVELOPE = ("as_of", "input_start", "input_end", "input_length", "evidence")

#: 多组输入：不同长度与种子。长度差异会改变"是否有 warmup 区""样本是否足够"，
#: 正是可选键的来源；只用一组输入会把可选键误标成必填。
VARIANTS = ((400, 301), (120, 911), (700, 77))

#: 快照型工具（输入是映射而非序列）只能用它自己声明的 SAMPLES，无法换种子。
SNAPSHOT_FAMILIES = ("technical", "portfolio", "fundamental", "valuation",
                     "regime", "factor", "validation")


def _json_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return "unknown"


def _element_types(values: list) -> set[str]:
    """数组元素类型（只取一层：本服务的数组元素是标量或对象）。"""
    out = set()
    for v in values:
        if isinstance(v, list):
            for item in v:
                out.add(_json_type(item))
        else:
            out.add(_json_type(v))
    return out


def _kwargs(spec, n: int, seed: int):
    """按名填充一次调用参数（与离线边界测试同一套规则）。"""
    import quantdata as qd

    sig = inspect.signature(spec.fn)
    has_mapping = any(
        any(h in str(p.annotation if isinstance(p.annotation, str)
                     else getattr(p.annotation, "__name__", ""))
            for h in ("Mapping", "dict", "Dict"))
        for p in sig.parameters.values()
        if p.annotation is not inspect.Parameter.empty
    )
    if has_mapping:
        for mod_name in SNAPSHOT_FAMILIES:
            try:
                mod = importlib.import_module(f"fd_quant.compute.{mod_name}")
            except ModuleNotFoundError:
                continue
            if any(s.name == spec.name for s in getattr(mod, "TOOLS", ())):
                sample = getattr(mod, "SAMPLES", {}).get(spec.name)
                return dict(sample) if sample else None
        return None

    d = qd.ohlcv(n, seed=seed)
    kw: dict[str, Any] = {"dates": d["dates"], "as_of": max(d["dates"])}
    for p in sig.parameters:
        if p in ("dates", "as_of"):
            continue
        if p in ("values", "close"):
            kw[p] = d["close"]
        elif p in ("high", "low", "volume", "open"):
            kw[p] = d[p]
        elif p in ("prices", "volatility", "returns", "equity"):
            kw[p] = qd.returns(n, seed=seed + 2)
        elif p in ("benchmark", "benchmark_returns", "factor", "factor_values"):
            kw[p] = qd.returns(n, seed=seed + 4)
        elif p in ("forward_returns", "next_returns"):
            kw[p] = qd.returns(n, seed=seed + 6)
        elif p == "weights":
            kw[p] = [0.4, 0.35, 0.25]
        elif p in ("weights_prior", "prior_weights"):
            kw[p] = [0.3, 0.4, 0.3]
        elif p == "values_prior":
            kw[p] = d["close"]
    return kw


def _observed(spec) -> list[dict]:
    """跑多组输入，返回每次的返回体（快照型工具只有一份样本）。"""
    out = []
    for n, seed in VARIANTS:
        kw = _kwargs(spec, n, seed)
        if kw is None:
            return []
        out.append(spec.fn(**kw))
    return out


def _schema_from(observations: list[dict]) -> dict:
    """把若干次真实返回归纳成 object schema（交集必填、并集可选、类型取并集）。"""
    key_sets = [set(o) for o in observations]
    required = set.intersection(*key_sets)
    every = set.union(*key_sets)
    props: dict[str, Any] = {}
    for key in sorted(every):
        vals = [o[key] for o in observations if key in o]
        types = sorted({_json_type(v) for v in vals})
        prop: dict[str, Any] = {}
        # 单一类型写 type 字符串；含 null 的写数组（JSON Schema 表达可空的标准写法）
        prop["type"] = types[0] if len(types) == 1 else types
        if "array" in types:
            # 元素里的 null **不能丢**：warmup 区与样本不足处的元素就是 null，
            # 这是本服务表达"该位置无数值"的正式方式。丢掉它会让声明与真实
            # 输出不符——而 fastmcp 3.x 会在调用时按声明校验，后果是工具报错。
            elems = _element_types(vals)
            if len(elems) == 1:
                prop["items"] = {"type": elems.pop()}
            elif elems:
                prop["items"] = {"type": sorted(elems)}
        props[key] = prop
    schema: dict[str, Any] = {
        "type": "object",
        "properties": props,
        "required": sorted(required),
        # 本服务的工具是封闭形状：多出来的键意味着实现改了而声明没改，
        # 让 fastmcp 的调用时校验直接拦住，比客户端静默忽略更早暴露。
        "additionalProperties": False,
    }
    return schema


def _ledger_schemas() -> dict[str, dict]:
    """账本族：payload 引用既有契约真源，其余键由实测归纳。"""
    import tempfile

    from fd_quant.ledger.tools import (
        record_decision, retrieve_decisions, settle_decision, summarize_pack_performance,
    )

    rating = json.loads((REPO / "schemas" / "rating.schema.json").read_text(encoding="utf-8"))
    tw = json.loads((REPO / "schemas" / "target-weights.schema.json").read_text(encoding="utf-8"))

    db = str(Path(tempfile.mkdtemp()) / "ledger.db")
    payload = {
        "schema_version": "1", "as_of": "2025-01-02", "subject": "600519.SS", "side": "buy",
        "basis": "schema 生成用的最小合法判定",
        "producer": {"pack": "p", "skill": "s"},
        "evidence": [{"evidence_id": "e1", "label": "l", "value": 1.0, "unit": None,
                      "as_of": "2025-01-02", "source": "src"}],
    }
    rec = record_decision(as_of="2025-01-02", symbol="600519.SS", pack="p", skill="s",
                          payload=payload, db_path=db)
    did = rec["result"]["decision_id"]
    dates = [f"2025-01-{d:02d}" for d in range(2, 12)]
    settle_decision(decision_id=did, symbol_dates=dates, symbol_prices=[100.0 + i for i in range(10)],
                    benchmark_dates=dates, benchmark_prices=[100.0] * 10,
                    as_of="2025-01-20", db_path=db)
    ret = retrieve_decisions(symbol="600519.SS", as_of="2025-03-01", db_path=db)
    stats = summarize_pack_performance(as_of="2025-03-01", db_path=db)

    def with_contract(schema: dict, overrides: dict[str, dict]) -> dict:
        schema = json.loads(json.dumps(schema))
        schema["properties"].update(overrides)
        return schema

    return {
        "record_decision": with_contract(
            _schema_from([rec]),
            # payload 由契约层校验（check_rating），故这里引用契约本身而不是
            # 用一次样本的形状——样本只能说明"这次有哪些键"。
            {"result": _object_with(rec["result"], {"payload": {"$ref": rating["$id"]}})}),
        "settle_decision": _schema_from([{"tool": "settle_decision", "as_of": "x",
                                          "db_path": "x",
                                          "result": _settle_result(db)}]),
        "retrieve_decisions": with_contract(
            _schema_from([ret]),
            {"result": _object_with(ret["result"], {
                "same_symbol": _array_of_item(ret["result"]["same_symbol"], tw, rating),
                "cross_symbol": _array_of_item(ret["result"]["cross_symbol"], tw, rating),
            })}),
        "summarize_pack_performance": _schema_from([stats]),
    }


def _object_with(sample: dict, overrides: dict[str, dict]) -> dict:
    """由一次真实样本造 object schema，并按需覆盖/追加属性。"""
    base = _schema_from([sample])
    base["properties"].update(overrides)
    return base


def _array_of_item(items: list[dict], tw: dict, rating: dict) -> dict:
    """检索结果数组：元素含 payload(Rating) 与 target_weights(TargetWeights)。"""
    if not items:
        return {"type": "array", "items": {"type": "object"}}
    elem = _schema_from(items)
    elem["properties"]["payload"] = {"$ref": rating["$id"]}
    # $ref 与兄弟键并存时按 JSON Schema 2020-12 语义兄弟键被忽略——可空性
    # 必须写进 anyOf，否则"可为 null"这条声明形同不存在。
    elem["properties"]["target_weights"] = {
        "anyOf": [{"$ref": tw["$id"]}, {"type": "null"}],
    }
    return {"type": "array", "items": elem}


def _settle_result(db: str) -> dict:
    """结算的返回体：用一条已结算记录重新结算取幂等分支（形状相同）。"""
    from fd_quant.ledger.store import LedgerStore

    store = LedgerStore(db)
    rows = store.retrieve(symbol="600519.SS", as_of="2025-03-01", n_same=1, m_cross=0)
    return rows["same_symbol"][0]["settlement"] if rows["same_symbol"] else {}


def build() -> dict:
    from fd_quant.compute import all_tools

    tools: dict[str, dict] = {}
    undeclared: list[str] = []
    for spec in all_tools():
        obs = _observed(spec)
        if not obs:
            undeclared.append(spec.name)
            continue
        tools[spec.name] = _schema_from(obs)
    tools.update(_ledger_schemas())
    return {
        "schema": "fd-quant-output-schemas/v1",
        "note": ("由 scripts/gen_output_schemas.py 从运行时实测生成。"
                 "required = 多组输入下键集合的交集；类型取并集（null 是"
                 "「该值不存在」的正式表达）。payload/target_weights 引用契约真源。"),
        "envelope": list(ENVELOPE),
        "tools": dict(sorted(tools.items())),
        # 显式登记未声明的工具：契约要求"无静默空缺"，缺失必须可见。
        "undeclared": sorted(undeclared),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    fresh = build()
    text = json.dumps(fresh, ensure_ascii=False, indent=2) + "\n"

    if args.check:
        if not OUT.exists():
            print(f"✗ 输出结构声明缺失：{OUT}", file=sys.stderr)
            return 1
        cur = json.loads(OUT.read_text(encoding="utf-8"))
        if cur != fresh:
            cur_t, fresh_t = set(cur.get("tools", {})), set(fresh["tools"])
            print("✗ 输出结构与运行时漂移：", file=sys.stderr)
            if fresh_t - cur_t:
                print(f"  新增工具未声明：{sorted(fresh_t - cur_t)}", file=sys.stderr)
            if cur_t - fresh_t:
                print(f"  已删工具仍在声明：{sorted(cur_t - fresh_t)}", file=sys.stderr)
            if not (fresh_t - cur_t) and not (cur_t - fresh_t):
                print("  工具集相同但形状变了（键/类型/必填）", file=sys.stderr)
            return 1
        print(f"✓ 输出结构与运行时一致（{len(fresh['tools'])} 工具，"
              f"未声明 {len(fresh['undeclared'])}）")
        return 0

    OUT.write_text(text, encoding="utf-8")
    print(f"✓ 已写入 {OUT.relative_to(REPO.parent)}：{len(fresh['tools'])} 工具，"
          f"未声明 {len(fresh['undeclared'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
