# A 股价格供给归因（2026-11-13，**订正版**）

> 由 `close-first-decision-loop` task 4 产出。**这是一份归因结论，不是修复记录。**
> 每条数字都能用文末的 SQL 复现；不含「疑似」「可能」这类无法证伪的措辞。
>
> **本文档订正了同一天早些时候的版本。** 早版本写的是「A 股价格从未纳入派发」——
> 那是错的，错在只用 `fetch_log` 判"有没有派发"。见文末「我为什么判错」。

## 一句话结论

**A 股日线价格有启用中的派发策略，但它从 2026-11-07 起持续失败，且今天有一条卡死的
open run 把所有后续派发挡住**——所以上游观测停在 2026-09-18，而所有闸门都是绿的。

## 三层实测（`fd-prod` 语义库，2026-11-13）

| 层 | 证据 |
|---|---|
| **派发** | `crawl_policies` id=2659 `astock-daily-increment`，`cron '17 18 * * 1-5'`（CST），**`enabled = true`**，`concept_ids = [233,234,235,236,238]` = `price.open/close/high/low/amount`（`stock`，daily） |
| **运行** | `policy_runs` 中 policy 2659：**failed 14 条**、success 1 条（2026-11-09 11:07）、zero_yield 1 条、**running 1 条（id=2383，2026-11-13 10:22 起，已 3.6 小时）** |
| **产出** | `price.close`/`stock` 最新观测 **2026-09-18**；`fetch_log` 近 60 天该概念 0 次（批量脚本路径不写 fetch_log） |

失败原因（`policy_runs.detail`，逐条引用）：

```
2382 → 2026-11-12 12:52  rate limited by eastmoney (IP ban from today's diagnosis);
                          new backoff logic waited 300s x2 before abort would trigger
2371 → 2026-11-12 12:48  manual close: pre-rollout verification job
2369 → 2026-11-12 11:57  job cheap9/crawl-policy-2659-... failed before the runner checked in
2368 → 2026-11-12 11:56  launch failed: script not found: /app/scripts/bulk_ingest_astock_daily.py
```

同一时段对照：**基金**策略（id=3 `fund-nav-daily`、id=5 `fund-etf-lof-ohlcv`）当天
`success`，`price.close`/`fund` 观测到 2026-11-13。抓取器与库都是好的，坏的只有这一条。

## 当前的实际阻塞：卡死的 open run

```
policy_runs id=2383  policy=2659  status=running  started=2026-11-13 10:22  age≈3.6h
今天中午之后 policy 2659 的 run 行数：0
```

派发侧 Job 的 `activeDeadlineSeconds = 10800`（3 小时），所以那个 Job 早在 13:22 就被杀了，
**而 DB 里那行还写着 `running`**——是"行没关"，不是"还在跑"。派发是单飞的
（`policy_trigger_now` 与 cron 都走 single-flight，存在 open run 就跳过），于是：

- **今天 18:17 CST 的计划内执行没有留下任何行**（`policy_runs` 中午之后 0 条）⇒ 被这条
  卡死的 open run 跳过了；
- 跳过**不产生记录、不发告警**，所以下一次有人看表时只会看到"最近一条是 running"。

这就是"环环都绿却拿不到数据"的最内层原因，也是本 change 之前没人能看见的原因。

## 我为什么判错（必须记下来）

早版本用的判据是「窗口内 `fetch_log` 抓取次数 == 0 ⇒ 从未纳入派发」。这条判据的错在于：
**A 股价格走的是 `bulk_ingest_astock_daily.py` 批量脚本路径，它不写 `fetch_log`**
（`fetch_log` 记录的是适配器级抓取）。于是"策略每天在跑但每次失败"和"根本没人派发"
在 fetch_log 里长得一模一样。

这是本项目反复踩的同一个坑的第三种形态：

- 「空表 ⇒ 不存在」（`schedules` 全表 0 行）—— 已在探针里禁止引用；
- 「有行 ⇒ 有深度」（`has-rows-is-not-enough-depth`）—— 已知；
- 现在是「**计数为零 ⇒ 没发生**」——零计数既可能是"没发生"，也可能是"发生的路径不记账"。

已落地防线：归因先看策略层（`crawl_policies` / `policy_runs`），再看抓取层与产出层，
并有一条回归用例把这个误判钉住（`tests/test_decision_loop_probe.py::test_classify_supply`
的第一组参数 + `test_classify_supply_never_says_no_policy_when_policy_exists`）。

## 影响

- `alpha_vs_benchmark` 结算**结构性不可得**：指数族 17 个实体在册、观测 **0 行**，基准收益无从计算。
- 执行岗若收到权重，第 ⑤ 条不行动条件（价格水位与 `as_of` 差距超阈值）必然触发 → 零订单。
  这是 fail-closed 在正确工作。
- 研究侧估值/技术面/敏感性三章只能降级为缺口声明（`docs/wire-presets/reports/e2e-yizuo/`
  2026-10-17 的实测报告已是如此）。

## 修复方向（本 change 不实施）

按优先级，前两条与"补派发"无关——派发本来就有：

1. **关掉卡死的 open run，并让它不可能静默卡住。** 派发侧在 Job 终态（含 `activeDeadline` 被杀）
   时必须把 `policy_runs` 关成终态；单飞跳过时必须留一行 `skipped` 而不是什么都不写。
   否则"一条卡死的行"就等于"这条族永久停更"，且无声。
2. **eastmoney 出口 IP 封禁**：`2372` 的证据显示退避逻辑当天确实跑过（"waited 300s x2"）后
   仍放弃，属封禁不是代码回归。处置在 `docs/ops/astock-daily-ingest-rate-limit.md` 里已有判据。
3. **批量路径补记账**：`bulk_ingest_astock_daily.py` 抓取时不写 `fetch_log`，导致"有没有在抓"
   在抓取层不可见。至少写一条汇总行（源 + 标的数 + 结果），否则下一次仍会误判。
4. **指数族单独一批**：`000300/000905/000906/000016/000852` 已有实体，需要价格概念族 + 观测
   导入；属 `add-price-supply-and-caliber` 的 `index-price-series` 范围。

补齐 1 之后，`decision_loop_probe.py --attribution` 的 `stock` 行应从 `dispatch_blocked`
翻为 `dispatch_failing` 或 `producing`（取决于封禁是否解除）。

## 复现命令

探针自带三层归因（只读）：

```bash
FD_OPEN_DATA_DSN=postgresql://... python scripts/decision_loop_probe.py --attribution
```

手工 SQL：

```sql
-- 层 1：哪些策略在派发这个概念族
SELECT id, name, cron_expr, enabled, concept_ids
FROM crawl_policies
WHERE enabled AND concept_ids && ARRAY['233','234','235','236','238']::text[];

-- 层 2：这些策略跑成什么样（含卡死的 open run）
SELECT id, policy_id, status, started_at, finished_at, detail,
       now() - started_at AS open_age
FROM policy_runs
WHERE policy_id = 2659
ORDER BY started_at DESC LIMIT 10;

-- 层 3：有没有真的产出
SELECT COUNT(*), MAX(o.date)
FROM semantic_observations o
JOIN concepts c ON c.id = o.concept_id
JOIN entities e ON e.id = o.entity_id AND e.entity_type = o.entity_type
WHERE c.code = 'price.close' AND c.entity_type = 'stock';

-- 对照：同一天基金在正常产出
SELECT COUNT(*), MAX(o.date)
FROM semantic_observations o
JOIN concepts c ON c.id = o.concept_id
JOIN entities e ON e.id = o.entity_id AND e.entity_type = o.entity_type
WHERE c.code = 'price.close' AND c.entity_type = 'fund';
```
