"""工作流级知识（mcp-surface-contract 组 7）：以 MCP 资源随服务暴露。

**为什么是资源而不是塞进工具描述**：工具描述回答"这个工具怎么用"，工作流
知识回答"这几个工具该按什么顺序、以什么口径串起来"。把后者塞进工具描述会
让描述变成一篇文档——而描述会被**每个**调用方读进上下文，代价按调用次数
线性放大；资源则是调用方需要时才拉取一次。

**为什么内容必须跨多步**：单工具描述已经在 `tools/list` 里了，再抄一遍是
重复。这里每一份都写清：步骤顺序、每步之间的**数据形状契约**（上一步的哪个
字段喂给下一步的哪个参数）、以及容易错的地方（口径漂移、as_of 越界）。

资源列表：
- `fd-quant://workflow/analysis-chain`：口径校验 → 指标 → 组合统计 → regime
  → 估值 → 因子原子的完整链路
- `fd-quant://workflow/decision-loop`：决策账本的落账 → 结算 → 复盘闭环
- `fd-quant://contract/rating-usage`：Rating 契约的正确用法与边界态
"""
from __future__ import annotations

ANALYSIS_CHAIN = """# fd-quant 分析链路（多步组合，非单工具说明）

## 顺序与数据形状契约

1. **口径校验先行** —— `check_series_as_of_bounds` / `validate_no_lookahead`
   / `check_observation_sufficiency`
   - 输入：dates + values + as_of
   - 输出里的 `ok` / `sufficient` 为假时**停在这里**，不要往下走：后面所有
     工具都假定序列已通过 as_of 校验，跳过校验得到的是"看起来算完了"的结果。

2. **技术指标** —— `compute_rsi` / `compute_macd` / `compute_atr` …
   - 用第 1 步同一份 `dates` / `close` / `high` / `low` / `volume`
   - `warmup` 为 null 表示该位置无值；不要把它当 0 参与后续运算。

3. **组合统计** —— `summarize_return` / `summarize_volatility` /
   `summarize_drawdown` / `summarize_sharpe`
   - 输入 `values`（价格）或 `returns`；`risk_free` 是**显式入参**，本层不内置
     无风险利率假设。

4. **regime 判断** —— `detect_regime_slow` / `detect_shock_fast`
   - `volatility` 序列由调用方传入（本层不取数）。慢速给 risk_on/neutral/
     risk_off 与逐判据证据，快速给触发与否与降仓建议。

5. **估值与合成** —— `estimate_dcf` / `estimate_ddm` /
   `compute_comparable_multiples` / `estimate_monte_carlo` /
   `build_valuation_synthesis`
   - 多方法分歧超阈值时**扣值不扣结论**：`price_target` 为 null 而方向判定
     仍在。不要因为 price_target 为 null 就判"估值失败"。

6. **因子原子** —— `compute_information_coefficient` /
   `compute_grouped_returns` / `compute_factor_exposure`
   - 因子与收益**都由调用方传入**，本层不内置任何因子定义。

## 三个常见错误

- **as_of 之后的数据点**：会被拒绝并点名该日期，绝不静默截断。若结果看起来
  "少了几天"，先查是不是撞了这条。
- **跨步骤换序列**：第 3 步用了归一化的 returns，第 4 步又传原始价格——
  口径不一致，数字没有可比性。
- **把 null 当 0**：warmup 区与样本不足处的 null 是"该位置无值"，参与运算
  会把收益曲线整体压低。
"""

DECISION_LOOP = """# fd-quant 决策账本闭环（多步组合）

## 顺序

1. **落账** —— `record_decision(as_of, symbol, pack, skill, payload)`
   - `payload` 必须是合法 `Rating`，且 `payload.as_of == as_of`、
     `payload.subject == symbol`、`payload.producer.pack/skill == pack/skill`
   - 幂等键 (caller, as_of, symbol, pack, skill)：重复提交**被拒但返回既有
     decision_id**，总数不变——这是"每个决策只算一次账"的屏障
   - `caller` 不是入参：由调用上下文解析（HTTP 用 Authorization 的 Bearer 指纹，
     stdio 用 FD_QUANT_CALLER，都没有则 anonymous），故无法声称自己是别人。
     记录按调用者隔离：检索与聚合只回自己的
   - 修订用 `revises=<既有 decision_id>` 发**新记录**，被修订记录逐字不改

2. **结算** —— `settle_decision(decision_id, symbol_dates, symbol_prices,
   benchmark_dates, benchmark_prices, as_of)`
   - **序列由调用方传入**：账本不取数、不联网
   - 持有期按交易日计，日历来自传入序列的日期集合
   - 基准：显式 `benchmark_symbol` 优先，否则按后缀映射
     （.SS→000001.SS / .SZ→399001.SZ / .HK→^HSI / .T→^N225 / 无后缀→SPY）
   - 结算窗不足时**拒绝并报缺多少交易日**（error 带
     `missing_trading_days`），不落行——半窗 alpha 比没有 alpha 更坏

3. **点内检索** —— `retrieve_decisions(symbol, as_of)`
   - 只回**结算日 ≤ as_of** 的记录（历史回放看到未来结算是这条纪律要防的 bug）
   - 分两组：同标的 `same_symbol`（含结算结果与 reflection）与跨标的教训
     `cross_symbol`

4. **按 pack 聚合** —— `summarize_pack_performance(as_of, packs?)`
   - 口径：alpha = raw_return − benchmark_return；win_rate = alpha > 0 占比
     （alpha 恰为 0 记负）；零样本的组**显式返回 samples=0** 而不是省略

## 一条纪律

反思只存、只随检索带出，**绝不推送**。把反思注入决策者是消费侧的行为，
账本不主动广播任何东西——四个工具里没有推送/广播/订阅形状的接口。
"""

RATING_USAGE = """# Rating 契约用法与边界态

## 必填与可空

必填：`schema_version` / `as_of` / `subject` / `side` / `basis` / `evidence` /
`producer`。可空：`conviction`、`price_target`、`time_horizon`。

## 两个边界态（不要"修正"它们）

- **`side: review`** —— 显式表示"无法形成判定"（证据不足或输出不可解析）。
  它**不是** hold，也不存在任何把它静默降级为 hold 的路径。看到 review 就该
  如实上报"没判出来"，而不是替模型选一个方向。
- **`price_target: null` 与非 review 的 side 并存** —— 扣值不扣结论：方法分歧、
  价格通道不可用或输入不足时保留方向判定、不给点位。这也不是失败。

## evidence 硬要求

`evidence_spans` 非空是合法分数的必要条件；`quote` 必须是原文逐字子串。
"无证据"应当拒绝打分，不返回无证据的分数。

## 幂等

同 (doc_id, content_hash, prompt_version, model) 返回既有记录。换 prompt 版本
或换模型会产生**新**记录而非覆盖旧记录——两批分数因此可比而不互相污染。
"""

#: URI → (标题, 内容)。URI 用服务名打头，避免与其它服务的资源撞名。
RESOURCES = {
    "fd-quant://workflow/analysis-chain": ("分析链路（多步组合）", ANALYSIS_CHAIN),
    "fd-quant://workflow/decision-loop": ("决策账本闭环（多步组合）", DECISION_LOOP),
    "fd-quant://contract/rating-usage": ("Rating 契约用法与边界态", RATING_USAGE),
}
