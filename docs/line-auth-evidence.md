# 线身份与钥匙轨的生效证据

> 这份文档是**仓内证据**，不属于对外的 `line-manifest.json`。
> 清单的每个字段都会随本线 `/.well-known/line-manifest.json` **匿名公开、可被索引**，
> 所以清单里只留产品语义（2026-11-15 起；键闭集见
> `openspec/changes/converge-line-identity-and-keys/specs/line-manifest/spec.md`）。
> 内部代码路径、Secret 名、环境变量、翻转步骤、实测判据一律写在这里。

## 以下原文 2026-11-15 从清单移出，逐字保留，按原键路径分节

### note

两组共 58 工具：①确定性计算面七族（只对传入序列作答，不取数/不联网/不调 LLM）②结算与台账面（3 个非只读）。`caller` 不是入参而是由调用上下文解析，故本线天然多租户隔离，但没有身份校验。 **8.6 线自有钥匙**：`auth.py` 与 fd-info/fd-text-core/fd-open-data-mcp/fd-cn-report 的门同族；门对 /health(z) 精确豁免，探针不受影响。

### identity.note

线自有共享 bearer 门**已在生产生效**（2026-11-12 翻转，change converge-line-identity-and-keys 8.6）：`MCP_BEARER_TOKEN` 由线级 Secret 注入，注册表该 server 行 `auth_scheme=bearer` + `auth_credential` 为同一把令牌（出口把 `Authorization` 换成它）；匿名直连 401、带令牌 200。同时提供线自描述端点 `/.well-known/line-manifest.json`（公开、免鉴权、零网络，`Cache-Control: public, max-age=300`），**原样回吐**本仓 `line-manifest.json` 的字节；该路径在门的 `EXEMPT_PATHS` 里，所以未持凭据的平台/客户也能读到「这条线认哪些钥匙」。本仓计算面不取数、不联网；账本落本地 SQLite。这把钥匙是**线级**共享钥匙，一把全客户共用，不承担客户级身份（谁在用由网关/注册表记录）；每客户一把需要钥匙库 + 吊销面，属后续 8.x。`jit_policy=off`：无断言兑换端点。

### auth_tracks[].note

线自有共享钥匙（`MCP_BEARER_TOKEN`），**已在运行路径生效**（2026-11-12 翻转，change converge-line-identity-and-keys 8.6）。翻转两步：① 注册表服务器级 backend auth（该 server 行 `auth_scheme=bearer` + `auth_credential` = 同一把令牌，出口由 auth-server 把 `Authorization` 换成它；`PATCH /api/servers/<path>/auth-credential`）→ ② 线侧 Secret `fd-quant-line-token` 注入 `MCP_BEARER_TOKEN` 并滚版（本线镜像 `sha-f2df5a7`）。**实测判据**（历次同形）：匿名直连 **401** `{"error":"unauthorized"}`、假令牌 **401**、真令牌 **200** 且拿到 MCP session；探针路径 /health,/healthz 豁免（不带令牌仍 200）；注册表行 `scheme=bearer enabled=true health=healthy tools=58`；消费者 fd-paper-trading 带同一令牌实测 200、匿名 401。**这正是 fallback 轨**：摘掉平台后，持有该令牌的客户仍能直接消费本线。⚠️ 共享钥匙**不承载客户身份**——客户级隔离/配额仍要靠平台（词元轨，见 `ciyuan-sk` 那条的 migration_task）。

### auth_tracks[].note

未接：本仓不校验凭据，词元钥匙来去都只是「一个不同的指纹」。

### billing.note

本仓零计费代码（grep `sub2api` 无命中）：按调用扣费发生在网关/registry 的词元平面，不在本线代码里。ledger 是**调用台账**（谁在何时跑了哪族计算，含结算窗口计算），不是钱；真实扣费在词元平面。

## 相关

- 清单键闭集与内容判据：`openspec/changes/converge-line-identity-and-keys/specs/line-manifest/spec.md`
- 体检：`python3 scripts/line_manifest_check.py --line fd-quant`（正本 `fd-infra-deploy/tools/`）
