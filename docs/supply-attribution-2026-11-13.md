# A 股价格供给归因（2026-11-13）

> 由 `close-first-decision-loop` task 4 产出。**这是一份归因结论，不是修复记录。**
> 每条数字都能用文末的一条 SQL 复现；不含「疑似」「可能」这类无法证伪的措辞。

## 一句话结论

**A 股 `price.close` 不是"断了"，而是从来没在派发范围内**：近 60 天抓取尝试 **0 次**，
而同期基金同类概念被抓取 **14 613 次**、最新日期到当天。指数族则更彻底——**实体在册
（17 个），观测 0 行**。

三路证据（`fd-prod` 语义库实测，2026-11-13）：

| 概念族 | 近 60 天抓取尝试 | 观测行数 | 最新观测日 | 归因 |
|---|---|---|---|---|
| `price.close` / `stock` | **0** | 120 182 | **2026-09-18** | `not_dispatched` |
| `price.close` / `fund` | 14 613 | 3 990 | 2026-11-13 | `producing` |
| `price.close` / `index` | **0** | **0** | — | `not_dispatched` |

对照本身就是结论：**同一张 `fetch_log` 表、同一个抓取器、同一段时间**，基金在跑、
A 股不在跑。所以这不是"抓取器坏了"，也不是"源被封了"，是这条概念族没有进日更派发。

## 为什么这条会静默

三条判据各自看都是绿的：

1. `price_caliber_gate.py` 判据一是「研究 / 交易 / 结算三路是否指向同一个 `source_id`」。
   三路**一致地**指向 `astock_daily`（一张同为陈旧的表）时，该判据通过。
   **口径一致不等于数据可用**——这是本 change 在 spec 里单独立一条判据的原因。
2. 账本 `decisions = 0` / `settlements = 0`，所以"结算失败"这件事也没有留下痕迹。
3. 团队的报数（`report_status` 的 `settled` 槽位）来自 agent 自述，不是账本。

于是"链路没有通电"这件事在生产里没有任何一个信号会红。

## 一个必须避开的错误推论

**`schedules` 表在生产里全表 0 行**（实测 `SELECT count(*) FROM schedules` → `0`）。
库内调度器根本未被使用，抓取由外部派发器驱动。因此：

- 「某概念没有 `schedule` 行」**不能**作为"它没被调度"的证据——那张表对**每个**概念都为空。
- 这也是我在本轮探索中差点用错的判据：先看到"stock 价格概念 0 个 schedule 行"，
  差点写成"调度停了"；查了全表 0 行才纠正。

空表不能证明不存在。根因判定只能落在 `fetch_log`（有没有真的去抓）与
`semantic_observations`（有没有产出）的比值上。

## 影响

- `alpha_vs_benchmark` 结算**结构性不可得**：基准（指数）零观测，基准收益无从计算。
- 若此时执行岗收到权重，第 ⑤ 条不行动条件（价格水位与 `as_of` 差距超阈值）必然触发
  → 零订单。这是 fail-closed 在正确工作，不是 bug。
- 研究侧估值/技术面/敏感性三类章节只能降级为缺口声明（`docs/wire-presets/reports/e2e-yizuo/`
  里 2026-10-17 的实测报告已经如此）。

## 修复方向（本 change 不实施）

`not_dispatched` 意味着要做的是**把这条概念族纳入派发**，不是"重启任务"。而且**模型层已经齐全**，
缺的只是最后一步：

| 层 | 状态（实测） |
|---|---|
| 概念 | `concepts.code='price.close'`、`entity_type='stock'` 在册 |
| 来源绑定 | `concept_bindings` 里有 **4 条已验证绑定**（`provenance='rule-confirmed'`、`reviewed=t`）：`stock_zh_a_hist` / `stock_zh_a_daily` / `stock_zh_a_hist_tx` / `stock_zh_a_minute`，列均为 `收盘` |
| 函数登记 | `manifests/akshare.yaml` 里 `get_stock_zh_a_hist` 标 `category: price-history`、`frequency: daily`、`verified: true` |
| **周期派发** | **缺**——近 60 天 0 次抓取 |

所以方案是给 A 股日线价格登记一条**周期抓取**，落点二选一（按现有实现任选其一，不要两处都加）：

1. `fd-industry-data` 的派发器配置：新增一条 `price.close` / `stock` 的日更条目，引用已绑定的
   `akshare.get_stock_zh_a_hist`（`frequency: daily` 已在 manifest 里，不需要新写 spider）。
2. 若走批量回填路径，则复用 `bulk_ingest_stock_indicators_events.py` 的形状补一段价格族
   （它已处理 `financials.*` / `performance.*`，价格族是同一批白名单 805 家）。

两条都要连带确认：

- **复权口径**：`get_stock_zh_a_hist` 默认 `qfq`，必须与 `astock_daily`（交易路）的
  `adjustment` 与 `adjustment_anchor` 一致，否则"补上派发"会把两套口径同时写进库，
  而 `price_caliber_gate` 判据三（跨源价差）才会红——那时已经晚了。
- **指数族单独一批**：`000300` / `000905` / `000906` / `000016` / `000852` 实体在册、观测 0 行，
  需要价格概念族 + 观测导入。属 `add-price-supply-and-caliber` 的 `index-price-series` 范围，
  与本 change 相邻但不重叠。

补齐后，`decision_loop_probe.py --attribution` 的 `stock` 行应从 `not_dispatched`
翻为 `producing`，且聚合里的 `price_watermark` 缺口应消失。

## 复现命令

探针自带这条归因（只读）：

```bash
FD_OPEN_DATA_DSN=postgresql://... python scripts/decision_loop_probe.py --attribution
```

等价的手工 SQL（在语义库上执行）：

```sql
-- 抓取尝试（近 60 天，按实体类型）
SELECT c.entity_type, COUNT(*)
FROM fetch_log f JOIN concepts c ON c.id = f.concept_id
WHERE c.code = 'price.close' AND f.timestamp > now() - interval '60 days'
GROUP BY 1;

-- 产出观测
SELECT c.entity_type, COUNT(*), MAX(o.date)
FROM semantic_observations o
JOIN concepts c ON c.id = o.concept_id
JOIN entities e ON e.id = o.entity_id AND e.entity_type = o.entity_type
WHERE c.code = 'price.close'
GROUP BY 1;

-- 指数实体在册但无观测
SELECT e.code, e.name_zh, COUNT(o.id)
FROM entities e
LEFT JOIN semantic_observations o ON o.entity_id = e.id AND o.entity_type = e.entity_type
WHERE e.entity_type = 'index'
GROUP BY 1, 2;
```
