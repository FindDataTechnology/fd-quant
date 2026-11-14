"""清单公开投影的内容卫生：每个字符串都会随 /.well-known 匿名公开。

判据源=converge-line-identity-and-keys 的 specs/line-manifest（2026-11-15 补的
「清单的每个字符串都会匿名公开」那条）；全量体检=`scripts/line_manifest_check.py`。
2026-11-15 本仓曾有 5 段 note（含 Secret 名、内部代码路径、豁免面清单），已移入
docs/line-auth-evidence.md——本测试防它回来。
"""

import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE = REPO_ROOT / "line-manifest.json"

_SANCTIONED_TEXT_PATHS = {
    "schema", "line", "public_base_url", "internal_base_url",
    "identity.issuer", "identity.jwks_uri", "identity.jit_policy",
    "billing.plane", "billing.evidence_url", "billing.evidence_path",
    "auth_tracks[].name", "auth_tracks[].kind", "auth_tracks[].accepts",
    "auth_tracks[].migration_task", "scopes[]",
}

_INTERNAL_PATTERNS = (
    (r"fd-[a-z0-9][a-z0-9-]*-(?:token|key)\b", "Secret 名"),
    (r"\b[A-Z][A-Z0-9_]{2,}(?:TOKEN|SECRET|PASSWORD|PEM)\b", "凭据类环境变量"),
    (r"(?<![\w/.-])(?:[\w.-]+/)+[\w.-]+\.(?:py|js|ts|sh|ya?ml)(?::\d+(?:-\d+)?)?", "内部代码路径"),
    (r"/api/(?:servers|patch-keys|tenants)\b|X-Internal-Token", "内部管理面调用形状"),
    (r"/(?:health|healthz|metrics)\b", "豁免面探针路径"),
)


def _walk_strings(node, path=""):
    if isinstance(node, dict):
        for k in node:
            yield from _walk_strings(node[k], (path + "." + k) if path else k)
    elif isinstance(node, list):
        for i in range(len(node)):
            yield from _walk_strings(node[i], path + "[]")
    elif isinstance(node, str):
        yield path, node


def test_source_manifest_has_no_free_text():
    """清单的每个字符串都会匿名公开 ⇒ 只许产品语义，不许内部细节。

    判据是**键的闭集**而不是长度（20 字符的 note 一样能写出 Secret 名）；
    外加内容扫描（凭据标识 / 内部代码路径 / 管理面形状 / 探针路径）。
    `internal_base_url` 是规格允许写集群内地址的唯一字段，排除在内容扫描外。
    """
    doc = json.loads(SOURCE.read_text(encoding="utf-8"))
    strings = list(_walk_strings(doc))

    stray = [p for p, _ in strings if p not in _SANCTIONED_TEXT_PATHS]
    assert not stray, f"清单里出现自由文本键（会随投影公开）: {stray}；证据请写进 docs/"

    body = json.dumps({k: v for k, v in doc.items() if k != "internal_base_url"},
                      ensure_ascii=False)
    for pattern, what in _INTERNAL_PATTERNS:
        hit = re.search(pattern, body)
        assert hit is None, f"清单里出现{what}: {hit.group(0)!r}"
