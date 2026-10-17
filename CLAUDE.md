# CLAUDE.md — fd-quant

寻数 · 柏讯量化能力线。**契约先行、计算后置。**

## 本仓现状（2026-10-17）

批 0.2（**两个输出契约** `Rating` / `TargetWeights`）、批 1.1（**确定性计算面** `fd_quant/compute/` 七族 46 工具）与批 1.3（**决策账本** `fd_quant/ledger/`，4 工具）**均已落地**：`1078 passed / 28 skipped`，surface manifest + 双向 selfcheck 通过。工具面合计 **50**（清单 `tool-surface.json`，由 `scripts/gen_surface.py` 生成）。

**尚未做**：部署（无 k8s 清单 / 无 NodePort）+ 网关注册——故 pack 侧暂不能引用本服务的工具名（见 `MCP-REGISTRY.md` §3 的挂账行）。`Dockerfile` 与 `.github/workflows/image.yml` 已就位，注册前置只剩端口分配与台账流程。

```
fd-quant/
├── schemas/            # JSON Schema（权威形状，供 Python 与非 Python 消费者）
│   ├── rating.schema.json
│   └── target-weights.schema.json
├── fd_quant/contracts/ # Python 模型 + 统一校验入口 + 无出处数字检测
│   ├── models.py
│   └── unsourced.py
├── fd_quant/compute/   # 确定性计算面（七族 46 工具；纯计算、不读库，有封闭性断言）
├── fd_quant/ledger/    # 决策账本（追加式决策 + 到期结算 alpha + 点内检索 + 聚合）
│   ├── store.py        # SQLite 存储（幂等键唯一约束 / 修订指针 / 点内过滤）
│   ├── settlement.py   # 结算纯函数（基准映射 / 对齐覆盖 / alpha 算术）
│   ├── tools.py        # MCP 工具组（record/settle/retrieve/summarize）
│   └── errors.py
├── fd_quant/surface.py # 工具面汇总点（compute + ledger；server / manifest 都读它）
└── tests/              # 对拍（schema↔模型）+ 语义 + 往返 + 确定性 + 账本全量
    └── fixtures/       # 合法 6 / 非法 12（非法项标注 __parity__ 归属层）
```

## 开发

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[test]"
.venv/bin/python -m pytest tests/ -q          # 1078 passed / 28 skipped
PYTHONPATH=scripts .venv/bin/python -m pytest tests/ -q -p netblock   # 断网同结果
python scripts/ledger_e2e.py                  # 账本端到端（离线，报告落 openspec/changes/.../reports/）
```

## 硬约束（改动前必读）

1. **两层分工**：JSON Schema 只管形状；跨字段算术（`Σweights ≤ 1`、`cash_floor + Σ`）与引用可解性（悬空 evidence、重复 id、universe 归属）由 `validate()` 入口强制。样例集用 `__parity__: shape|relational` 标注归属；`test_relational_fixtures_are_the_schemas_known_gap` 钉住这个限制。
2. **拒绝而非归一化**：违规权重一律拒绝，绝不在契约边界静默缩放或补值（测试断言输入对象未被修改）。
3. **往返等价**：可选字段（`time_horizon`、`run_id`）未设时**省略**而非写 `null`；省略与显式 null 都合法。
4. **review 不可降级**：`side="review"` 必须无点位无强度，且不存在任何自动改写为 `hold` 的路径。
5. **零外部依赖**：契约层无 LLM、无网络、无 DB、无计算库（`test_validation_is_local_only` 断言）。新增依赖需在此说明理由。

## 消费方与后续

- 计算实现 → `fd-quant-compute-plane`（批 1.1，本仓新增 `fd_quant/compute/`）
- 决策账本 → `fd-quant-decision-ledger`（批 1.3，以契约实例为记录单位）
- 引擎消费 → `fd-paper-trading` 的 `strategy-weight-mode`（接受 `TargetWeights` 实例）
- 套件产出 → `pack-equity-research` / `pack-decision-review` / `pack-ic-debate` 等

## 版本演进

`SUPPORTED_SCHEMA_VERSION = "1"`。升 v2 时：新增 schema 文件（不就地改 v1）+ 在 `models.py` 扩支持集合 + 样例集补两侧对拍用例。**不支持版本被拒绝**是既有行为，不得放宽为静默兼容。

## 已知未决

- 组合层评级形状（当前仅标的级）——若 pack 需要，作为 v2 字段
- `fd-quant` 的 MCP 工具面（计算面落地时按 `mcp-surface-contract` 上牌）
