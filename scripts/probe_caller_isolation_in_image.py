#!/usr/bin/env python3
"""镜像内验签闭环探针（openspec change: seal-ledger-path-and-caller-isolation, task 5.3）。

**为什么不用仓内那份 `.sh`**：`scripts/probe-caller-isolation.sh` 靠 `curl`，而运行片是
`python:3.12-slim`——镜像里没有 curl。探针必须在**镜像自己**的环境里跑得起来，否则
「验过了」验的是宿主机而不是片。这份只依赖标准库（`urllib` + `hmac` + `sqlite3`）。

**它判的不是「实现是否符合规格」，而是「这一片在真环境里六条闭环是否都通」。** 单测
验的是函数；这里验的是「HTTP 传输 → MCP 会话 → 工具 → 派生的库文件」整条链，以及
**被拒的请求一行都没落**。

六条判据（`--mode keyed` 跑 1-5，`--mode unkeyed` 跑 6）：

1. 无头               → 调用成功，身份落 `anonymous` 分库；
2. 好签名 A            → 成功，且记录落 `A` 自己的分库目录；
3. 坏签名             → **被拒**，原因码 `caller_signature_mismatch`，且它那个 symbol 一行都没落；
4. 超窗（+600s）       → 被拒，原因码 `caller_signature_expired`，同样一行都没落；
5. 别客户 B            → 成功，落在**B 自己的**目录（与 A 不同），A 的库里看不到 B 的 symbol；
6. 密钥未配 + 好形头    → 成功（头被忽略），**且没有按该头派生身份**（A 的目录不存在）。

判据取的是「工具调用是否成功 + 原因码 + 盘上哪个目录有哪个 symbol」，不取 HTTP 状态码：
MCP streamable-http 把**工具执行失败**包在 200 响应的 JSON-RPC 结果里（`isError: true`），
所以 HTTP 层恒 200。规格（`specs/ledger-storage-isolation`）要求的是「请求被拒绝并返回
稳定的原因码、不写入任何记录」——这三件事本探针都**直接**断言，比状态码更强。

用法（pod 内）：

    python3 /probe.py --mode keyed   --key-file /key/key --root /data
    python3 /probe.py --mode unkeyed --root /data

退出码：0 = 六条全绿；1 = 有判据红；2 = 跑不起来（连不上 / 参数非法）。
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import sqlite3
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

CALLER_HEADER = "X-Fd-Caller"

#: 每条尝试用**专属 symbol**：被拒的那两条要在扫盘时「找不到」，这是「一行都没落」
#: 的正向证据——只看原因码只能证明"回了错"，证不了"没写"。
SYM_NOHEAD = "600519.SS"
SYM_A = "000001.SZ"
SYM_B = "300750.SZ"
SYM_REJ_BAD = "601318.SS"
SYM_REJ_EXP = "000002.SZ"

CID_A = "probe-a@example.com"
CID_B = "probe-b@example.com"
AS_OF = "2026-11-13"


def _payload(symbol: str) -> dict:
    """一份**合法** Rating（白名单恰好 11 个字段；多一个键整份拒收）。"""
    return {
        "schema_version": "1",
        "as_of": AS_OF,
        "subject": symbol,
        "side": "buy",
        "conviction": 0.6,
        "price_target": None,
        "target_evidence_ids": [],
        "basis": "探针构造：现金流稳定，估值处于历史中位。",
        "time_horizon": "3M",
        "evidence": [],
        "producer": {"pack": "pack-equity-research", "skill": "equity-research-thesis"},
    }


def sign(cid: str, ts: int, key: str) -> str:
    return hmac.new(key.encode(), f"{cid}.{ts}".encode(), hashlib.sha256).hexdigest()


def header_for(cid: str, key: str, *, offset: int = 0) -> str:
    ts = int(time.time()) + offset
    return f"{cid}.{ts}.{sign(cid, ts, key)}"


class Mcp:
    """最小 MCP streamable-http 客户端：一次 initialize + 多次 tools/call。"""

    def __init__(self, base_url: str, timeout: float = 20.0) -> None:
        self.url = base_url.rstrip("/") + "/mcp"
        self.timeout = timeout
        self.session: str | None = None

    # ── 传输 ────────────────────────────────────────────────────────────
    @staticmethod
    def _decode(raw: bytes) -> dict | None:
        """响应可能是 JSON，也可能是 SSE（`data: {...}` 行）。"""
        text = raw.decode("utf-8", "replace").strip()
        if not text:
            return None
        if text.startswith("{"):
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return None
        for chunk in text.split("\n\n"):
            for line in chunk.splitlines():
                if line.startswith("data:"):
                    try:
                        return json.loads(line[5:].strip())
                    except json.JSONDecodeError:
                        continue
        return None

    def post(self, body: dict, *, caller_header: str | None = None) -> tuple[int, dict | None]:
        headers = {
            "content-type": "application/json",
            "accept": "application/json, text/event-stream",
        }
        if self.session:
            headers["mcp-session-id"] = self.session
        # 身份头必须随**每一次** POST 发：会话建立之后的那次 tools/call 才是真正走
        # `resolve_caller()` 的一跳，只在 initialize 上带头等于没测。
        if caller_header is not None:
            headers[CALLER_HEADER] = caller_header
        req = urllib.request.Request(
            self.url, data=json.dumps(body).encode(), headers=headers, method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                sid = resp.headers.get("mcp-session-id")
                if sid:
                    self.session = sid
                return resp.status, self._decode(resp.read())
        except urllib.error.HTTPError as exc:  # 4xx/5xx 也要读正文
            sid = exc.headers.get("mcp-session-id") if exc.headers else None
            if sid:
                self.session = sid
            return exc.code, self._decode(exc.read())

    # ── 协议 ────────────────────────────────────────────────────────────
    def initialize(self) -> tuple[int, dict | None]:
        return self.post(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {},
                    "clientInfo": {"name": "seal-probe", "version": "1"},
                },
            }
        )

    def initialized(self) -> int:
        status, _ = self.post({"jsonrpc": "2.0", "method": "notifications/initialized"})
        return status

    def call(self, name: str, arguments: dict, *, caller_header: str | None = None,
             rid: int = 2) -> dict:
        """返回 `{"status", "error": {code, reason} | None, "result": {...} | None}`。"""
        status, payload = self.post(
            {
                "jsonrpc": "2.0",
                "id": rid,
                "method": "tools/call",
                "params": {"name": name, "arguments": arguments},
            },
            caller_header=caller_header,
        )
        return _classify(status, payload)


def _classify(status: int, payload: dict | None) -> dict:
    """把 JSON-RPC 结果收敛成 `{status, error, result}`；工具失败在 `result.isError` 里。"""
    out = {"status": status, "error": None, "result": None}
    if not isinstance(payload, dict):
        out["error"] = {"code": f"http_{status}", "reason": "no_json_body"}
        return out
    err = payload.get("error")
    if isinstance(err, dict):
        data = err.get("data") or {}
        out["error"] = {
            "code": data.get("code") or err.get("code") or err.get("message"),
            "reason": data.get("reason"),
            "message": err.get("message"),
        }
        return out
    result = payload.get("result") or {}
    if result.get("isError"):
        body = result.get("structuredContent") or {}
        error = body.get("error") if isinstance(body, dict) else None
        if isinstance(error, dict):
            out["error"] = {
                "code": error.get("code"),
                "reason": error.get("reason"),
                "message": error.get("message"),
            }
        else:
            out["error"] = {"code": "isError", "reason": None, "message": str(body)[:200]}
        return out
    out["result"] = result.get("structuredContent") or {}
    return out


# ── 盘面证据 ────────────────────────────────────────────────────────────────


def scan(root: Path) -> dict:
    """扫每个 `*/ledger.db`：目录名 → 该库里的 symbol 集合。

    直接读 sqlite 而不是问工具：被拒的请求连**工具**都没跑到，用工具去数会得到
    「这条 caller 的库里 0 行」——那证不了「别的库里也没混进去」。
    """
    found: dict[str, list[str]] = {}
    if not root.is_dir():
        return {"root": str(root), "exists": False, "dbs": {}}
    for db in sorted(root.glob("*/ledger.db")):
        try:
            con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
            try:
                rows = con.execute("SELECT DISTINCT symbol FROM decisions").fetchall()
            finally:
                con.close()
        except sqlite3.Error as exc:
            found[db.parent.name] = [f"<error:{exc}>"]
            continue
        found[db.parent.name] = sorted(str(r[0]) for r in rows if r[0] is not None)
    return {"root": str(root), "exists": True, "dbs": found}


def dir_of(cid: str) -> str:
    """与服务端同一条派生规则（`safe_segment` + `sha256(原文)[:8]`）——独立实现，
    故意不从 `fd_quant` import：探针要能在**镜像外**跑，也要能验镜像里的**旧片**
    （旧片根本没有这个模块）。"""
    import re

    seg = re.sub(r"[^a-z0-9_-]", "-", cid.lower())[:48] or "anon"
    return f"{seg}-{hashlib.sha256(cid.encode()).hexdigest()[:8]}"


# ── 判据 ───────────────────────────────────────────────────────────────────


class Report:
    def __init__(self) -> None:
        self.rows: list[tuple[str, bool, str]] = []

    def check(self, label: str, ok: bool, detail: str) -> None:
        self.rows.append((label, ok, detail))
        print(f"{'✓' if ok else '✗'} {label}: {detail}")

    @property
    def failed(self) -> int:
        return sum(1 for _, ok, _ in self.rows if not ok)


def _ok_call(res: dict) -> bool:
    return res["error"] is None and res["result"] is not None


def run_keyed(mcp: Mcp, root: Path, key: str) -> Report:
    rep = Report()

    # 1. 无头 → 成功，落 anonymous
    r1 = mcp.call("record_decision", {
        "as_of": AS_OF, "pack": "pack-equity-research", "skill": "equity-research-thesis",
        "symbol": SYM_NOHEAD, "payload": _payload(SYM_NOHEAD),
    }, rid=2)
    rep.check("1 无头 → 成功（回落匿名）", _ok_call(r1),
              f"error={r1['error']}" if r1["error"] else f"db_path={r1['result'].get('db_path')}")

    # 2. 好签名 A
    r2 = mcp.call("record_decision", {
        "as_of": AS_OF, "pack": "pack-equity-research", "skill": "equity-research-thesis",
        "symbol": SYM_A, "payload": _payload(SYM_A),
    }, caller_header=header_for(CID_A, key), rid=3)
    rep.check("2 好签名 A → 成功", _ok_call(r2),
              f"error={r2['error']}" if r2["error"] else f"db_path={r2['result'].get('db_path')}")
    if _ok_call(r2):
        rep.check("2b 落 A 自己的目录", dir_of(CID_A) in str(r2["result"].get("db_path")),
                  f"db_path={r2['result'].get('db_path')} 期望目录={dir_of(CID_A)}")

    # 3. 坏签名（时戳在窗内、签名错）→ 拒
    bad = f"{CID_A}.{int(time.time())}." + "0" * 64
    r3 = mcp.call("record_decision", {
        "as_of": AS_OF, "pack": "pack-equity-research", "skill": "equity-research-thesis",
        "symbol": SYM_REJ_BAD, "payload": _payload(SYM_REJ_BAD),
    }, caller_header=bad, rid=4)
    rep.check("3 坏签名 → 被拒", r3["error"] is not None
              and (r3["error"].get("reason") == "caller_signature_mismatch"),
              f"{r3['error']}")

    # 4. 超窗（+600s，签名正确但过期）→ 拒
    r4 = mcp.call("record_decision", {
        "as_of": AS_OF, "pack": "pack-equity-research", "skill": "equity-research-thesis",
        "symbol": SYM_REJ_EXP, "payload": _payload(SYM_REJ_EXP),
    }, caller_header=header_for(CID_A, key, offset=600), rid=5)
    rep.check("4 超窗 +600s → 被拒", r4["error"] is not None
              and (r4["error"].get("reason") == "caller_signature_expired"),
              f"{r4['error']}")

    # 5. 别客户 B → 成功且落 B 自己目录
    r5 = mcp.call("record_decision", {
        "as_of": AS_OF, "pack": "pack-equity-research", "skill": "equity-research-thesis",
        "symbol": SYM_B, "payload": _payload(SYM_B),
    }, caller_header=header_for(CID_B, key), rid=6)
    rep.check("5 别客户 B → 成功", _ok_call(r5),
              f"error={r5['error']}" if r5["error"] else f"db_path={r5['result'].get('db_path')}")

    # 盘面：谁家有谁，被拒的两条必须哪里都没有
    got = scan(root)
    print(f"· 盘面 {json.dumps(got, ensure_ascii=False)}")
    dbs = got["dbs"]
    rep.check("5b B 落自己的目录（与 A 不同）",
              dir_of(CID_B) in dbs and dir_of(CID_B) != dir_of(CID_A),
              f"目录集={sorted(dbs)}")
    rep.check("5c A 的库里看不到 B 的 symbol",
              SYM_B not in dbs.get(dir_of(CID_A), []),
              f"A 库={dbs.get(dir_of(CID_A))}")
    rep.check("3b 坏签名那条一行都没落", SYM_REJ_BAD not in {s for v in dbs.values() for s in v},
              f"扫到={sorted({s for v in dbs.values() for s in v})}")
    rep.check("4b 超窗那条一行都没落", SYM_REJ_EXP not in {s for v in dbs.values() for s in v},
              f"扫到={sorted({s for v in dbs.values() for s in v})}")
    rep.check("1b 无头落 anonymous 分库",
              any(name.startswith("anonymous-") and SYM_NOHEAD in syms for name, syms in dbs.items()),
              f"库={ {k: v for k, v in dbs.items()} }")
    return rep


def run_unkeyed(mcp: Mcp, root: Path, key: str) -> Report:
    """密钥未配：带一个**格式完全正确**的头（签名也按缺省密钥算对了）必须被忽略。"""
    rep = Report()
    r = mcp.call("record_decision", {
        "as_of": AS_OF, "pack": "pack-equity-research", "skill": "equity-research-thesis",
        "symbol": SYM_A, "payload": _payload(SYM_A),
    }, caller_header=header_for(CID_A, key), rid=2)
    rep.check("6 密钥未配 + 好形头 → 成功（不拒）", _ok_call(r),
              f"error={r['error']}" if r["error"] else f"db_path={r['result'].get('db_path')}")

    got = scan(root)
    print(f"· 盘面 {json.dumps(got, ensure_ascii=False)}")
    dbs = got["dbs"]
    rep.check("6b 头被忽略：没有按它派生身份", dir_of(CID_A) not in dbs,
              f"目录集={sorted(dbs)}")
    rep.check("6c 身份按既有链回落（anonymous）",
              any(name.startswith("anonymous-") for name in dbs), f"目录集={sorted(dbs)}")
    return rep


def run_transcript(mcp: Mcp) -> dict:
    """5.5 用的确定性读数：同一请求在两个片上的**完整返回体**。

    用只读且空库上恒定的 `list_pending_decisions`（不是写接口：写接口的 `decision_id`
    是 uuid4，逐字段对比会永远不等，那是夹具的噪声不是行为的差异）。
    """
    r = mcp.call("list_pending_decisions", {"as_of": AS_OF}, rid=2)
    return {"status": r["status"], "error": r["error"], "result": r["result"]}


def run_write_transcript(mcp: Mcp) -> dict:
    """5.5 的写入侧读数：同一个 `record_decision` 在两个片上的状态码与形态。

    不逐字比正文——`decision_id` 是 uuid4、`db_path` 是两侧唯一被设计改变的字段。
    比的是**状态码 + 键集合 + 非随机字段**,这三样任一对不上就说明行为不是中性的。
    """
    r = mcp.call("record_decision", {
        "as_of": AS_OF, "pack": "pack-equity-research", "skill": "equity-research-thesis",
        "symbol": SYM_A, "payload": _payload(SYM_A),
    }, caller_header=header_for(CID_A, "unused-in-unkeyed-mode"), rid=2)

    def walk(obj, path=""):
        """把随机/环境相关的叶子换成占位符，其余原样留下。"""
        if isinstance(obj, dict):
            return {k: ("<db_path>" if k == "db_path" else
                        "<id>" if k == "decision_id" else
                        "<ts>" if k == "created_at" else
                        walk(v, f"{path}/{k}")) for k, v in obj.items()}
        return obj

    return {
        "status": r["status"],
        "error": r["error"],
        "result_shape": walk(r["result"]),
        "result_keys": sorted((r["result"] or {}).keys()),
        "result_inner_keys": sorted(((r["result"] or {}).get("result") or {}).keys()),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="镜像内验签闭环探针（六条）")
    ap.add_argument("--base-url", default="http://127.0.0.1:8080")
    ap.add_argument("--root", default=None, help="账本根（缺省取 FD_QUANT_LEDGER_DB 的父目录，再缺省 CWD）")
    ap.add_argument("--mode", choices=["keyed", "unkeyed", "transcript", "write-transcript"],
                    default="keyed")
    ap.add_argument("--key", default=None, help="HMAC 密钥（缺省读 --key-file）")
    ap.add_argument("--key-file", default="/key/key")
    ap.add_argument("--json", action="store_true", help="只打一行 JSON（5.5 对照用）")
    args = ap.parse_args(argv)

    import os

    key = args.key
    key_configured = bool(key)
    if not key:
        p = Path(args.key_file)
        if p.is_file():
            key = p.read_text().strip()
            key_configured = bool(key)
    if key_configured and not key:
        print("✗ 密钥文件存在但为空", file=sys.stderr)
        return 2

    root = args.root
    if root is None:
        legacy = (os.environ.get("FD_QUANT_LEDGER_DB") or "").strip()
        root = str(Path(legacy).expanduser().parent) if legacy else os.getcwd()
    root_path = Path(root)

    mcp = Mcp(args.base_url)
    try:
        status, payload = mcp.initialize()
        mcp.initialized()
    except (urllib.error.URLError, OSError) as exc:
        print(f"✗ 跑不起来：连不上 {mcp.url}：{exc}", file=sys.stderr)
        return 2
    if status != 200:
        print(f"✗ 跑不起来：initialize HTTP {status} {payload}", file=sys.stderr)
        return 2

    if args.mode == "transcript":
        print(json.dumps(run_transcript(mcp), ensure_ascii=False, sort_keys=True))
        return 0
    if args.mode == "write-transcript":
        print(json.dumps(run_write_transcript(mcp), ensure_ascii=False, sort_keys=True))
        return 0

    if args.mode == "keyed" and not key_configured:
        print("✗ keyed 模式但没给密钥（--key / --key-file）", file=sys.stderr)
        return 2
    if args.mode == "unkeyed" and key_configured:
        # 「密钥未配」这一条只有在服务端**真的**没配时才成立；探针自己拿着钥去测它，
        # 测的是别的前提（那正是"夹具绕过被验对象"那一族）。
        print("✗ unkeyed 模式下探针侧不应有密钥——请确认服务端也未配", file=sys.stderr)
        return 2

    rep = run_keyed(mcp, root_path, key or "") if args.mode == "keyed" \
        else run_unkeyed(mcp, root_path, key or "probe-only-unused-key")

    print(f"\n=== {sum(1 for _, ok, _ in rep.rows if ok)}/{len(rep.rows)} 通过, {rep.failed} 失败")
    return 0 if rep.failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
