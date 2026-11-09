"""跨族合规：所有族的每个工具都必须满足的不变量。

这是把 spec 的通用要求（as_of 必填、越界拒绝、evidence 形状、参数与 warmup
声明、NaN 不抛、逐位确定）从"各族自己记得写"变成"漏写就红"的地方。族的
专属语义测试在 `test_<family>.py`；本文件只测**所有工具共有**的部分。

新族只需注册进 `fd_quant.compute`，本文件的 parametrize 会自动覆盖它——
不需要新族作者改这里。
"""
from __future__ import annotations

import inspect
import json
import os
import subprocess
import sys
import textwrap

import pytest

import quantdata as qd
from fd_quant.compute import all_tools, missing_families, tools_by_group
from fd_quant.compute.base import ComputeError, assert_bitequal

ALL = all_tools()
IDS = [f"{t.group}:{t.name}" for t in ALL]


# ── 快照型工具（无日期轴）的声明式接入 ───────────────────────────────────────
#
# 序列型工具能按参数名（dates/values/close/...）猜出一组合法输入；快照型工具
# （输入是 `fundamentals` 这类 field → number 的映射）没有逐点日期轴，按名字猜
# 必然失败。判据取**签名里的类型注解**而不是"有没有 dates 参数"——快照工具常为
# 统一调用形状接受一个被忽略的可选 dates，用 dates 判断会把它们误判成序列型。
# 声明了映射输入的工具必须在族模块里声明 SAMPLES = {工具名: 一组完整合法 kwargs}；
# 没声明就红（不静默跳过，也不让调用处抛 TypeError）。

_FAMILY_MODULES = ("technical", "portfolio", "fundamental", "valuation",
                   "regime", "factor", "validation",
                   # 决策一致性族（add-decision-consistency-and-premise-watch）：
                   # 两个快照型工具，SAMPLES 各自声明在族模块里。
                   "consistency", "premise_watch")

#: 形如 Mapping[...] / dict[str, ...] / Dict[...] 的注解 → 该工具吃快照映射输入。
_MAPPING_HINTS = ("Mapping", "MutableMapping", "dict", "Dict")


def _declares_mapping_input(spec) -> bool:
    """签名里是否有参数被注解为映射（快照型工具的定义性特征）。"""
    for p in inspect.signature(spec.fn).parameters.values():
        if p.annotation is inspect.Parameter.empty:
            continue
        text = p.annotation if isinstance(p.annotation, str) else getattr(
            p.annotation, "__name__", str(p.annotation)
        )
        if any(h in str(text) for h in _MAPPING_HINTS):
            return True
    return False


def _owning_module(spec):
    """spec 所属的族模块（用于读该族声明的 SAMPLES）。"""
    import importlib

    for mod_name in _FAMILY_MODULES:
        try:
            mod = importlib.import_module(f"fd_quant.compute.{mod_name}")
        except ModuleNotFoundError:
            continue
        if any(spec.name == s.name for s in getattr(mod, "TOOLS", ())):
            return mod
    raise AssertionError(f"{spec.name}: not declared by any family module")


def _snapshot_kwargs(spec):
    """快照型工具 → 该族 SAMPLES 里声明的 kwargs；序列型工具 → None（走按名填充）。"""
    if not _declares_mapping_input(spec):
        return None
    samples = getattr(_owning_module(spec), "SAMPLES", {})
    assert spec.name in samples, (
        f"{spec.name}: declares a mapping input (snapshot tool) but its family module "
        f"does not declare SAMPLES[{spec.name!r}] = {{...}} — the cross-family gate "
        f"cannot infer a legal snapshot from parameter names"
    )
    kw = dict(samples[spec.name])
    assert kw, f"{spec.name}: SAMPLES entry must be a non-empty dict"
    assert "as_of" in kw, f"{spec.name}: SAMPLES entry must carry as_of"
    return kw


def _is_snapshot_tool(spec) -> bool:
    return _declares_mapping_input(spec)


def _kwargs_for(spec):
    """按 spec 的签名构造一组确定性合法调用参数。

    以**参数名**决定填什么：`dates`/`high`/`low`/`close`/`volume`/`values`
    这些是序列，`as_of` 是口径，其余标量走默认值（不传）。这样新族只要
    沿用 base 的命名约定就能被自动覆盖。

    快照型工具（签名里声明了 mapping 输入）不走这条路径：直接用族模块 SAMPLES
    里声明的那组 kwargs——它们的输入不是序列，按参数名猜没有意义。
    """
    sample = _snapshot_kwargs(spec)
    if sample is not None:
        return sample
    d = qd.ohlcv(80, seed=101)
    kwargs = {"dates": d["dates"], "as_of": max(d["dates"])}
    sig = inspect.signature(spec.fn)
    for pname in sig.parameters:
        if pname in ("dates", "as_of"):
            continue
        if pname in ("values", "close", "prices"):
            kwargs[pname] = d["close"]
        elif pname in ("high", "low", "volume", "open"):
            kwargs[pname] = d[pname]
        elif pname in ("volatility", "vol", "volatility_series"):
            # regime 族的波动率输入由调用方提供：给一条**确定性的正**波动率序列
            # （绝对值 + 小量），使 Z 分数与跳变比的分母良定义。
            kwargs[pname] = [abs(x) + 0.01 for x in qd.returns(80, seed=137)]
        elif pname in ("returns", "equity", "series", "weights", "values_a", "values_b"):
            kwargs[pname] = qd.returns(80, seed=103)
        elif pname in ("benchmark", "benchmark_returns"):
            kwargs[pname] = qd.returns(80, seed=107)
        elif pname in ("factor", "factor_values"):
            kwargs[pname] = qd.returns(80, seed=109)
        elif pname in ("forward_returns", "next_returns"):
            kwargs[pname] = qd.returns(80, seed=113)
    return kwargs


def test_all_families_present():
    """六族 + 校验族全部落地（缺族则工具面不完整，发布前必须为空）。"""
    assert missing_families() == [], f"families missing: {missing_families()}"


def test_tool_names_are_unique_and_prefixed():
    names = [t.name for t in ALL]
    assert len(names) == len(set(names))
    for n in names:
        assert "_" in n, f"{n}: tool names are verb_noun (e.g. compute_sma)"
        assert n.split("_")[0] in {
            "compute", "summarize", "evaluate", "analyze", "validate",
            "assess", "estimate", "detect", "build", "check", "derive",
            "screen", "size", "score", "rank", "split",
            # `document_*`：把一族工具的默认参数与校准注记回给调用方（regime 族 spec 6.3
            # 要求的"工具描述与返回都带默认值注记"落地工具）。
            "document",
            # 决策账本族（`fd_quant/ledger/tools.py`）的三个动词：账本的动作就是"记一笔 /
            # 结算一笔 / 取出历史"，用 compute/summarize 之类的计算面动词会名不副实。
            # 这是本测试唯一一次扩白名单（改动可见、逐动词列出，不是通配）。
            "record", "settle", "retrieve",
        }, f"{n}: unexpected verb prefix"


def test_every_tool_is_declared_once_with_group():
    for spec in ALL:
        assert spec.group and spec.effective_level in ("commercial", "internal")
        assert spec.doc, f"{spec.name}: empty description"


@pytest.mark.parametrize("spec", ALL, ids=IDS)
def test_description_is_self_sufficient(spec):
    """spec 8.1：描述含输入形状 / 参数默认 / as_of 要求 / 返回形状 / warmup 语义。"""
    doc = spec.doc
    assert len(doc) >= 120, f"{spec.name}: description too thin ({len(doc)} chars)"
    low = doc.lower()
    assert "input:" in low, f"{spec.name}: no Input: section"
    assert "return" in low, f"{spec.name}: no Returns: section"
    assert "as_of" in low, f"{spec.name}: does not mention as_of"


@pytest.mark.parametrize("spec", ALL, ids=IDS)
def test_as_of_is_required_kwarg(spec):
    sig = inspect.signature(spec.fn)
    assert "as_of" in sig.parameters, f"{spec.name}: no as_of parameter"
    assert sig.parameters["as_of"].default is inspect.Parameter.empty, (
        f"{spec.name}: as_of must be required (no default)"
    )


@pytest.mark.parametrize("spec", ALL, ids=IDS)
def test_missing_as_of_is_refused(spec):
    kw = _kwargs_for(spec)
    kw["as_of"] = None
    with pytest.raises(ComputeError, match="as_of"):
        spec.fn(**kw)


@pytest.mark.parametrize("spec", ALL, ids=IDS)
def test_future_data_is_refused_and_named(spec):
    """越界拒绝且点名该日期（不静默截断）。

    断言的必须是**首个**越界日期：那是调用方要剪掉的那一位，报一个更晚的
    日期等于让人多切一刀。
    """
    if _is_snapshot_tool(spec):
        pytest.skip(f"{spec.name}: no date axis that could carry future data "
                    f"(snapshot tool; as_of is still required and validated first)")
    kw = _kwargs_for(spec)
    d = qd.ohlcv(80, seed=127)
    kw["dates"] = d["dates"]
    kw["as_of"] = d["dates"][3]
    first_future = d["dates"][4]
    with pytest.raises(ComputeError) as ei:
        spec.fn(**kw)
    assert first_future in str(ei.value), (
        f"{spec.name}: error did not name the first out-of-range date {first_future}"
    )


@pytest.mark.parametrize("spec", ALL, ids=IDS)
def test_result_envelope_is_complete(spec):
    """每个工具返回：as_of + 输入起止 + evidence（非空、六字段、id 唯一）。"""
    kw = _kwargs_for(spec)
    r = spec.fn(**kw)
    assert isinstance(r, dict)
    assert r["as_of"] == kw["as_of"]
    if _is_snapshot_tool(spec):
        # base.result_envelope 明文允许：标量/快照型传 series=None → 三者为 null
        assert r["input_start"] is None
        assert r["input_end"] is None
        assert r["input_length"] is None
    else:
        assert r["input_start"] == kw["dates"][0]
        assert r["input_end"] == kw["dates"][-1]
        assert r["input_length"] == len(kw["dates"])
    ev = r["evidence"]
    assert isinstance(ev, list) and ev, f"{spec.name}: no evidence"
    ids = [e["evidence_id"] for e in ev]
    assert len(ids) == len(set(ids)), f"{spec.name}: duplicate evidence_id"
    for e in ev:
        assert set(e) == {"evidence_id", "label", "value", "unit", "as_of", "source"}
        assert e["as_of"] == kw["as_of"]
        assert e["source"].startswith("fd-quant:"), e["source"]
        bare = spec.name.split("_", 1)[1]
        assert bare in e["source"], f"{spec.name}: source {e['source']} lacks the tool name"


@pytest.mark.parametrize("spec", ALL, ids=IDS)
def test_bitwise_determinism_same_process(spec):
    kw = _kwargs_for(spec)
    assert_bitequal(spec.fn(**kw), spec.fn(**kw), what=spec.name)


def _numeric_sequences(kw: dict) -> dict:
    """只把数值序列替换为含 null 的版本——`dates` 本身不是数值序列。"""
    out = dict(kw)
    for k, v in kw.items():
        if k == "dates" or not isinstance(v, list) or not v:
            continue
        if all(x is None or isinstance(x, (int, float)) for x in v):
            out[k] = qd.with_nan(v, (40, 41))
    return out


@pytest.mark.parametrize("spec", ALL, ids=IDS)
def test_nan_input_never_raises(spec):
    """含 NaN 的输入不抛未捕获异常（缺失按惯例传播或置 null）。"""
    if _is_snapshot_tool(spec):
        pytest.skip(f"{spec.name}: snapshot tool has no numeric series to inject nulls "
                    f"into; its refusal semantics are covered by the family's own tests")
    d = qd.ohlcv(80, seed=131)
    kw = _kwargs_for(spec)
    kw["dates"] = d["dates"]
    kw["as_of"] = max(d["dates"])
    base_kw = dict(kw)
    for k, v in base_kw.items():
        if k != "dates" and isinstance(v, list) and len(v) == 80:
            kw[k] = v
    kw = _numeric_sequences(kw)
    r = spec.fn(**kw)  # 不抛即可；语义由各族测试断言
    assert isinstance(r, dict) and r["evidence"]


@pytest.mark.parametrize("spec", ALL, ids=IDS)
def test_json_serialisable(spec):
    """返回必须可 JSON 序列化（NaN 会被 json 写成非法字面量，故不能出现）。"""
    r = spec.fn(**_kwargs_for(spec))
    s = json.dumps(r, allow_nan=False)  # allow_nan=False：NaN/Inf 直接报错
    assert s


def test_determinism_survives_thread_env_change():
    """spec 1.2：线程数变化下仍逐位一致（不依赖 BLAS 并行归约）。

    两个子进程用不同 OMP_NUM_THREADS 各算一遍并打印结果的 sha256；不一致
    说明归约路径受并行分块影响。这条覆盖的是"跨进程 + 跨线程配置"，比同进程
    重跑强。
    """
    script = textwrap.dedent(
        """
        import hashlib, json, sys
        sys.path.insert(0, {root!r})
        sys.path.insert(0, {tests!r})
        import quantdata as qd
        from fd_quant.compute import all_tools
        out = {{}}
        d = qd.ohlcv(200, seed=211)
        a = max(d["dates"])
        import importlib
        samples_by_tool = {{}}
        for _mn in {mods!r}:
            try:
                _m = importlib.import_module(f"fd_quant.compute.{{_mn}}")
            except ModuleNotFoundError:
                continue
            for _n, _k in getattr(_m, "SAMPLES", {{}}).items():
                samples_by_tool[_n] = _k
        for spec in all_tools():
            import inspect
            if spec.name in samples_by_tool:
                out[spec.name] = hashlib.sha256(
                    json.dumps(spec.fn(**dict(samples_by_tool[spec.name])),
                               sort_keys=True, allow_nan=False).encode()
                ).hexdigest()
                continue
            kwargs = {{"dates": d["dates"], "as_of": a}}
            for p in inspect.signature(spec.fn).parameters:
                if p in ("dates", "as_of"):
                    continue
                if p in ("values", "close", "prices"):
                    kwargs[p] = d["close"]
                elif p in ("high", "low", "volume", "open"):
                    kwargs[p] = d[p]
                elif p in ("volatility", "vol", "volatility_series"):
                    kwargs[p] = [abs(x) + 0.01 for x in qd.returns(200, seed=239)]
                elif p in ("returns", "equity", "series", "weights"):
                    kwargs[p] = qd.returns(200, seed=223)
                elif p in ("benchmark", "benchmark_returns", "factor", "factor_values"):
                    kwargs[p] = qd.returns(200, seed=227)
                elif p in ("forward_returns", "next_returns"):
                    kwargs[p] = qd.returns(200, seed=229)
            out[spec.name] = hashlib.sha256(
                json.dumps(spec.fn(**kwargs), sort_keys=True, allow_nan=False).encode()
            ).hexdigest()
        print(json.dumps(out, sort_keys=True))
        """
    )
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    # 族清单只保留一份（`_FAMILY_MODULES`）。此前这里另抄了一份字面量，新增族时
    # 忘了同步就会以"detect_conflicts() got an unexpected keyword argument 'dates'"
    # 的形式炸出来——症状离原因很远。
    code = script.format(root=str(root), tests=str(root / "tests"), mods=_FAMILY_MODULES)

    digs = []
    for threads in ("1", "4"):
        env = {**os.environ, "OMP_NUM_THREADS": threads, "OPENBLAS_NUM_THREADS": threads,
               "MKL_NUM_THREADS": threads}
        proc = subprocess.run([sys.executable, "-c", code], capture_output=True,
                              text=True, env=env)
        assert proc.returncode == 0, proc.stderr[-2000:]
        digs.append(json.loads(proc.stdout.strip().splitlines()[-1]))
    assert digs[0] == digs[1], "results differ across OMP_NUM_THREADS=" + " vs ".join(
        sorted(digs[0].keys() - {k for k in digs[0] if digs[0][k] == digs[1].get(k)})
    )


def test_no_network_or_llm_imports_in_compute_layer():
    """spec 9.3：计算层依赖清单只含数值库 + 传输层。

    以源码扫描断言：计算层不得 import 网络/DB/LLM 客户端。
    """
    from pathlib import Path

    banned = {
        "requests", "httpx", "urllib", "urllib.request", "socket", "aiohttp",
        "sqlalchemy", "psycopg", "psycopg2", "sqlite3", "pymysql",
        "openai", "anthropic", "litellm", "transformers", "torch", "langchain",
        "chromadb", "redis", "boto3", "yfinance", "akshare", "tushare",
    }
    root = Path(__file__).resolve().parents[1] / "fd_quant" / "compute"
    hits: list[str] = []
    for py in sorted(root.glob("*.py")):
        for lineno, line in enumerate(py.read_text().splitlines(), 1):
            ls = line.strip()
            if not (ls.startswith("import ") or ls.startswith("from ")):
                continue
            for mod in banned:
                if f" {mod}" in ls or ls.startswith(f"import {mod}") or ls.startswith(f"from {mod}"):
                    hits.append(f"{py.name}:{lineno}: {ls}")
    assert not hits, "compute layer imported a banned module:\n" + "\n".join(hits)


def test_third_party_dependency_closure():
    """依赖清单核查：只有数值库与传输层（无 LLM/框架）。"""
    from pathlib import Path

    import tomllib

    root = Path(__file__).resolve().parents[1]
    data = tomllib.loads((root / "pyproject.toml").read_text())
    deps = data["project"]["dependencies"]
    allowed_prefixes = ("jsonschema", "numpy", "pandas", "scipy", "fastmcp", "mcp",
                        "starlette", "uvicorn", "httpx-sse")
    for d in deps:
        base = d.split(">=")[0].split("==")[0].split("[")[0].strip().lower()
        assert base.startswith(allowed_prefixes), f"unexpected dependency: {d}"


def test_groups_are_populated():
    """每族至少有一个工具（空族意味着 spec 的一条 Requirement 未落地）。"""
    by = tools_by_group()
    for group in (
        "technical_indicators", "portfolio_statistics", "fundamental_ratios",
        "valuation_operators", "regime_detection", "factor_statistics", "validation",
    ):
        assert by.get(group), f"group {group} has no tools"
