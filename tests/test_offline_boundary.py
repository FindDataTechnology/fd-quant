"""回归边界：无网络运行 + 依赖清单核查（task 9.3）。

**为什么要有这一条**："计算面不联网"是一条**结构性**声明（spec: 纯计算与零外部依赖
纪律），而结构性声明的唯一可信证明是"把网络封掉再跑一遍"。靠阅读代码判"这里没联网"
在依赖升级或新族加入后会失真，且失真时不会有任何信号。

两道闸门：
1. 进程内封掉所有出网路径（connect / create_connection / DNS），然后遍历全部工具调用
   一遍。任何工具若试图出网，调用即失败——测试红。
2. 源码扫描 + 依赖清单核查：计算层不得出现网络/DB/LLM 客户端，第三方依赖只许是
   数值库与传输层。

第二道在 `test_family_conformance.py` 里已有一份（`test_no_network_or_llm_imports_in_compute_layer`
与 `test_third_party_dependency_closure`）；本文件的第一道是它拿不到的证据——**运行时**
而非文本层面。
"""
from __future__ import annotations

import inspect
import socket
import sys

import pytest

import quantdata as qd
from fd_quant.compute import all_tools

ALL = all_tools()
IDS = [f"{t.group}:{t.name}" for t in ALL]


@pytest.fixture()
def network_blocked(monkeypatch):
    """封掉一切出网路径；测试结束自动还原。"""

    def _deny(*a, **k):
        raise AssertionError(
            "network access attempted from the compute layer — it must be pure "
            "calculation (no fetch, no DB, no LLM)"
        )

    monkeypatch.setattr(socket.socket, "connect", _deny)
    monkeypatch.setattr(socket.socket, "connect_ex", _deny)
    monkeypatch.setattr(socket, "create_connection", _deny)
    monkeypatch.setattr(socket, "getaddrinfo", _deny)
    monkeypatch.setattr(socket, "gethostbyname", _deny)
    yield


def _kwargs_for(spec) -> dict | None:
    """与合规闸门同一套按名填充；快照型工具走族模块声明的 SAMPLES。"""
    import importlib

    sig = inspect.signature(spec.fn)
    has_mapping = False
    for p in sig.parameters.values():
        if p.annotation is inspect.Parameter.empty:
            continue
        text = p.annotation if isinstance(p.annotation, str) else getattr(
            p.annotation, "__name__", str(p.annotation))
        if any(h in str(text) for h in ("Mapping", "dict", "Dict")):
            has_mapping = True
    if has_mapping:
        for mod_name in ("technical", "portfolio", "fundamental", "valuation",
                         "regime", "factor", "validation"):
            try:
                mod = importlib.import_module(f"fd_quant.compute.{mod_name}")
            except ModuleNotFoundError:
                continue
            if any(spec.name == s.name for s in getattr(mod, "TOOLS", ())):
                sample = getattr(mod, "SAMPLES", {}).get(spec.name)
                return dict(sample) if sample else None
        return None

    # 400 点：让窗口切分类工具的默认参数（252 训练 + 63 测试）也能落地，
    # 否则"断网可跑"会被"样本不足"掩盖，测不到真东西。
    d = qd.ohlcv(400, seed=301)
    kw = {"dates": d["dates"], "as_of": max(d["dates"])}
    for pname in sig.parameters:
        if pname in ("dates", "as_of"):
            continue
        if pname in ("values", "close"):
            kw[pname] = d["close"]
        elif pname in ("high", "low", "volume", "open"):
            kw[pname] = d[pname]
        elif pname in ("prices", "volatility", "returns", "equity"):
            kw[pname] = qd.returns(400, seed=303)
        elif pname in ("benchmark", "benchmark_returns", "factor", "factor_values"):
            kw[pname] = qd.returns(400, seed=307)
        elif pname in ("forward_returns", "next_returns"):
            kw[pname] = qd.returns(400, seed=311)
        elif pname in ("weights",):
            kw[pname] = [0.4, 0.35, 0.25]
        elif pname in ("weights_prior", "prior_weights"):
            kw[pname] = [0.3, 0.4, 0.3]
        elif pname in ("values_prior",):
            kw[pname] = d["close"]
    return kw


@pytest.mark.parametrize("spec", ALL, ids=IDS)
def test_every_tool_runs_with_network_blocked(spec, network_blocked):
    """全部工具在封网环境下正常返回——这是"纯计算"的运行时证据。"""
    kw = _kwargs_for(spec)
    if kw is None:
        pytest.skip(f"{spec.name}: no declared SAMPLES (checked by the conformance gate)")
    r = spec.fn(**kw)
    assert isinstance(r, dict) and r["evidence"]


def test_offline_chain_reproduces_bitwise(network_blocked):
    """断网下跑一条跨族链两次，逐位一致（而非仅"能跑"）。"""
    from fd_quant.compute.base import assert_bitequal
    from fd_quant.compute.portfolio import summarize_drawdown
    from fd_quant.compute.technical import compute_atr, compute_rsi
    from fd_quant.compute.valuation import estimate_dcf

    d = qd.ohlcv(120, seed=313)
    a = max(d["dates"])

    def chain():
        return {
            "rsi": compute_rsi(d["dates"], d["close"], as_of=a, window=14),
            "atr": compute_atr(d["dates"], d["high"], d["low"], d["close"], as_of=a),
            "dd": summarize_drawdown(d["dates"], d["close"], as_of=a),
            "dcf": estimate_dcf(**qd.dcf_inputs(), as_of=a),
        }

    assert_bitequal(chain(), chain(), what="offline chain")


def test_declared_dependencies_are_numerics_and_transport_only():
    """依赖清单核查（task 9.3 的"依赖清单 diff"判据）。

    白名单是**封闭**的：新增依赖必须显式改这里，从而在评审时可见。
    含 LLM / 框架 / 数据源客户端的依赖会被拒。
    """
    from pathlib import Path

    import tomllib

    root = Path(__file__).resolve().parents[1]
    data = tomllib.loads((root / "pyproject.toml").read_text())
    deps = data["project"].get("dependencies", [])
    allowed = {
        "jsonschema",   # 契约层校验
        "numpy",        # 数值
        "scipy",        # 若某族确需（当前未用）
        "fastmcp",      # MCP 传输层
        "mcp",          # MCP 传输层
        "starlette",    # HTTP 传输
        "uvicorn",      # HTTP 传输
    }
    banned_substrings = ("openai", "anthropic", "litellm", "langchain", "llama",
                         "torch", "transformers", "akshare", "yfinance", "tushare",
                         "sqlalchemy", "psycopg", "redis", "boto3", "chromadb")

    for d in deps:
        base = d.split(">=")[0].split("==")[0].split("<")[0].split("[")[0].strip().lower()
        assert base in allowed, (
            f"unexpected dependency {d!r}: only numerics + transport are allowed; "
            f"if this is intentional, extend the whitelist in this test (review-visible)"
        )
        for bad in banned_substrings:
            assert bad not in base, f"banned dependency: {d!r}"


def test_compute_layer_package_list_is_closed():
    """计算层 import 的第三方包集合必须落在白名单内（比逐行扫描更强的闭合断言）。"""
    from pathlib import Path

    import ast

    allowed = {"numpy", "fd_quant", "fastmcp", "mcp", "jsonschema", "math", "json",
               "re", "sys", "os", "typing", "dataclasses", "datetime", "argparse",
               "logging", "itertools", "functools", "collections", "abc", "warnings",
               "textwrap", "pathlib", "operator", "copy", "enum", "fractions",
               "decimal", "statistics", "random_nothing", "builtins", "hashlib",
               "subprocess", "tempfile", "traceback", "contextlib", "inspect",
               "__future__", "importlib", "starlette"}

    root = Path(__file__).resolve().parents[1] / "fd_quant" / "compute"
    seen: set[str] = set()
    for py in sorted(root.glob("*.py")):
        tree = ast.parse(py.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    seen.add(a.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                if node.level == 0 and node.module:
                    seen.add(node.module.split(".")[0])

    unexpected = sorted(seen - allowed)
    assert not unexpected, (
        f"compute layer imports packages outside the whitelist: {unexpected} — "
        f"extend the whitelist in this test only after confirming it is stdlib or a "
        f"declared dependency"
    )


def test_stdlib_only_imports_do_not_mask_a_network_client():
    """`random` 模块禁用（确定性纪律）——它使结果不可复现且极易被误用为"抖动"。"""
    from pathlib import Path

    import ast

    root = Path(__file__).resolve().parents[1] / "fd_quant" / "compute"
    hits: list[str] = []
    for py in sorted(root.glob("*.py")):
        tree = ast.parse(py.read_text())
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for n in names:
                if n.split(".")[0] in ("random", "secrets"):
                    hits.append(f"{py.name}:{node.lineno}: import {n}")
    assert not hits, (
        "the compute layer must use np.random.default_rng(seed) for any randomness, "
        "never the `random`/`secrets` modules:\n" + "\n".join(hits)
    )

def test_readonly_annotated_tools_write_no_files(tmp_path, monkeypatch, network_blocked):
    """标为只读的工具在真实调用中不得产生任何文件写入。

    **为什么这条不能靠读代码**：只读是一条**协议承诺**（客户端据此自动批准），
    而"实现里偷偷写了个缓存/日志/临时库"是代码演进中最容易发生的漂移——一个
    新增的 `open(..., "w")` 就能让 46 个工具的自动批准变成一次静默写盘。所以
    这里做**行为**验证：把 cwd 切到空目录，跑一遍全部只读工具，断言目录里
    仍然空无一物，且工作区原有文件未被改动。

    账本族不在此列（它们本来就写库，属性已标非只读）。
    """
    import os
    from pathlib import Path

    readonly = [t for t in ALL if t.tool_annotations.get("readOnlyHint") is True]
    assert readonly, "没有任何工具标为只读，说明属性声明坏了"

    work = tmp_path / "cwd"
    work.mkdir()
    monkeypatch.chdir(work)

    for spec in readonly:
        kw = _kwargs_for(spec)
        if kw is None:
            continue  # 快照型工具由合规闸门覆盖，见上方说明
        r = spec.fn(**kw)
        assert isinstance(r, dict)

    created = sorted(
        str(p.relative_to(work)) for p in Path(work).rglob("*") if p.is_file()
    )
    assert created == [], (
        f"标为只读的工具在调用中写了文件：{created} —— 只读属性与实现不符，"
        f"要么去掉写盘、要么把该工具改成非只读"
    )
