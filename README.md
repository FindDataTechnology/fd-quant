# fd-quant

寻数 · **柏讯量化能力线**的主仓。**契约先行、计算后置**——两条都已落地：

| 批 | 内容 | 状态 |
|---|---|---|
| 批 0.2 | **两个输出契约**（`Rating` / `TargetWeights`） | 已落地 |
| 批 1.1 | **确定性计算面**（`fd_quant/compute/` 七族 46 工具） | 已落地 |
| 批 1.3 | **决策账本**（`fd_quant/ledger/`，4 工具） | 已落地 |

工具面合计 **50 个**（计算面 46 + 账本 4），权威清单是 `tool-surface.json`（脚本生成，勿手改）。

## 计算面（`fd_quant/compute/`）

纯计算 MCP 服务：**只对传入的序列与参数作答——不取数、不读库、不联网、不调 LLM**。

| 族 | 工具数 | 内容 |
|---|---|---|
| `technical_indicators` | 10 | sma/ema/rsi/macd/bollinger/atr/kdj/obv/cci/dmi（自实现，不依赖 ta-lib） |
| `portfolio_statistics` | 10 | 年化收益与波动、回撤、Sharpe/Sortino/Calmar、IR/TE/Beta、集中度、换手 |
| `fundamental_ratios` | 8 | 盈利/成长/偿债/效率/现金流 + DuPont + 红旗检测 |
| `valuation_operators` | 6 | DCF/DDM/可比倍数/Monte Carlo/敏感性网格/**多方法合成** |
| `regime_detection` | 3 | 慢速（趋势+回撤+波动率 Z）/ 快速（冲击+波动率跳变）/ 默认值文档化 |
| `factor_statistics` | 4 | IC（Pearson+Spearman）/ 分组收益 / 暴露回归 / 因子打分排序 |
| `validation` | 5 | walk-forward 切分 / **无未来函数检查** / 序列口径体检 / 隔离段切分 / 观测充足性 |

**三条贯穿全族的纪律**：

1. **`as_of` 必填**，含 `as_of` 之后的数据点被拒绝并**点名第一个越界日期**（不静默截断）。
2. **同输入逐位一致**——固定归约顺序（Kahan），不用非确定并行归约，不碰全局随机源。
3. **算不出就说算不出**——输入不足返回 `null` + 精确 `reason`（如"needs 260 points, got 120"），
   不返回占位值；多方法分歧超阈值时 `price_target` 为 `null` 并附分歧说明（**扣值不扣结论**）。

```bash
python -m fd_quant.server                      # stdio（上牌入口：计算面 + 账本）
python -m fd_quant.server --http 8080          # HTTP
python -m fd_quant.server --list-tools         # 打印 50 工具清单
python -m fd_quant.compute.server              # 纯计算面入口（不含账本，可跑在只读盘环境）
python -m pytest tests/ -q                     # 1078 passed / 28 skipped
python scripts/selfcheck_surface.py --gen-check # surface 双向断言
python scripts/e2e_chain.py                    # 端到端（真实 A 股数据，见下）
python scripts/ledger_e2e.py                   # 决策账本端到端（离线，全构造序列）
```

**依赖边界**：计算面七族只依赖 `numpy` / 标准库；账本只用 stdlib `sqlite3`；
`fastmcp` 仅属传输层（`pip install -e ".[server]"` 才装），故两层都可在无框架环境离线回归。

## 端到端演练（真实数据）

`scripts/e2e_chain.py` 以**真实 A 股 600519 日线**（来自价格通道 B 的 `daas.db`，
400 个交易日）跑通「口径校验 → 技术指标 → 组合统计 → regime → 估值与合成 → 因子原子」，
每步数字带 evidence，复跑逐位一致。报告落
`openspec/changes/fd-quant-compute-plane/reports/e2e-chain.md`（**脚本生成，勿手改**）。

契约联调另见 `scripts/contract_e2e.py`：四份契约实例（含 **`price_target: null`** 与
**`side: review`** 两个边界）全部过校验。

## 为什么先有契约

三家开源项目（FinRL-X / FinRobot / TradingAgents）里最值得抄的第一条不是算法，而是**输出契约**：

- FinRL-X 的全部价值建立在"策略输出恒为目标权重向量"上——回测与实盘吃同一个输出；
- FinRobot 用"扣值不扣结论"（方法分歧时点位给 null、方向照发）与"每个数字标出处"支撑它的审计；
- TradingAgents 用结构化评级 + 显式 `REVIEW` 态（解析失败不静默降级为 Hold）让判定可被机器消费。

没有统一契约，"很多模式"就无法互相比较、无法被同一回测器与账本消费——做再多也只是一地散文。

## 两个契约

### `Rating` — 判定

```json
{
  "schema_version": "1",
  "as_of": "2026-10-16",
  "subject": "600519.SS",
  "side": "buy",
  "conviction": 0.6,
  "price_target": 1500.0,
  "target_evidence_ids": ["ev-dcf"],
  "basis": "现金流稳定，估值处于历史中位。",
  "time_horizon": "6-12 个月",
  "evidence": [
    {"evidence_id": "ev-dcf", "label": "DCF 估值", "value": 1520.5,
     "unit": "CNY/share", "as_of": "2026-10-16", "source": "fd-quant:dcf"}
  ],
  "producer": {"pack": "pack-equity-research", "skill": "equity-research-thesis"}
}
```

三条关键语义：

| 语义 | 规则 |
|---|---|
| **扣值不扣结论** | `price_target` 可为 `null` 而 `side` 是真实方向（buy/sell/…）——方法分歧、价格通道不可用或输入不足时保留方向判定 |
| **review 是显式态** | `side="review"` 时 `price_target` 与 `conviction` 必须为 `null`；**不存在把它静默改写为 hold 的路径** |
| **数字要出处** | `price_target` 非 null 时，`target_evidence_ids` 必须非空且每个 id 在 `evidence` 中可解 |

### `TargetWeights` — 目标权重

```json
{
  "schema_version": "1",
  "as_of": "2026-10-16",
  "universe": ["600519.SS", "000001.SZ"],
  "weights": {"600519.SS": 0.4, "000001.SZ": 0.3},
  "cash_floor": 0.3,
  "constraints": {"max_weight_per_symbol": 0.4, "max_weight_per_sector": 0.5},
  "evidence": [],
  "producer": {"pack": "pack-adaptive-rotation", "skill": "rotation-portfolio"}
}
```

不变式：权重非负、键 ∈ `universe`、`Σweights ≤ 1`（容差 1e-9）、`cash_floor + Σweights ≤ 1`（同容差）。
**违规一律拒绝，绝不自动归一化**——契约边界不做静默缩放，违规输入由生产者修正。

## 用法

```python
from fd_quant.contracts import Rating, TargetWeights, validate, validate_target_weights

# 统一校验入口：接受 dict / JSON 字符串 / 模型实例，自动判别契约类型
res = validate({"side": "review", "price_target": 120.0, ...})
# → {"valid": False, "errors": [{"path": "price_target", "reason": "must be null when side is 'review' ..."}]}

# 或按类型校验
validate_target_weights({"weights": {"A": 1.2}, ...})

# 构造与序列化（往返逐字段等价）
r = Rating.from_json(raw)
r.to_json()
```

命令行：

```bash
pytest tests/ -q                      # 31 项（对拍 + 语义 + 往返 + 确定性）
python -m json.tool schemas/rating.schema.json   # 形状（供非 Python 消费者）
```

## 两层分工（重要）

契约形状有两份表达，各有明确边界：

| 层 | 表达什么 | 不表达什么 |
|---|---|---|
| **JSON Schema**（`schemas/`） | 字段形状、类型、枚举、必填、未知字段拒绝、`review` 态约束 | **跨字段算术**（`Σweights ≤ 1`、`cash_floor + Σ`）与**引用可解性**（悬空 evidence、重复 id、universe 归属）——JSON Schema 无法表达 |
| **Python 校验**（`fd_quant/contracts/`） | 形状 + 上述全部关系约束（统一入口 `validate()`） | 任何计算（属后续 change） |

对拍测试（`tests/test_contract_parity.py`）钉住两侧一致：形状类违规**两侧都必须拒**；关系类违规由模型层拒（schema 放行是**已知且被测试钉住的**限制，样例集以 `__parity__: shape|relational` 标注归属）。

## 消费方（本批不接入）

| 消费方 | 用什么 |
|---|---|
| `fd-quant-compute-plane`（批 1.1） | 计算算子输出 `evidence[]`；估值合成产出可带 null 点位的结果 |
| `backtest-engine-v2`（批 1.2） | 消费 `TargetWeights`（多标的回测与成本模型） |
| `strategy-weight-mode`（批 3） | 权重模式引擎直接接受契约实例作为静态权重序列 |
| `fd-quant-decision-ledger`（批 1.3） | 以契约实例为账本记录单位（幂等键含 producer 归因） |
| `pack-equity-research` / `pack-decision-review` / `pack-ic-debate` 等套件 | 产出 `Rating`（含 review 与 null 点位路径） |

## 边界

本仓不含：任何计算实现 · 数据库接入 · 网络调用 · LLM 依赖 · 界面。
`tests/test_contract_parity.py::test_validation_is_local_only` 断言契约层不引用 requests/httpx/sqlalchemy/urllib/socket。
