# 通道 B 的 A 股/指数供给：有什么、缺什么、怎么补（2026-11-14 实测）

> 本页只回答三个问题：**哪份数据在哪、水位多少**；**为什么观测会落后于日线**；**怎么安全补齐**。
> 全部读数来自 `fd-prod` 实查，命令附在每节末尾，可直接复现。

## 一、两个通道，两份存储，不要互相引用

投研线同时读两份价格，**它们不是同一个库**，历史上因此误判过两次：

| 通道 | 存储 | 谁在用 | 读数（2026-11-14） |
|---|---|---|---|
| **A** | `fd_open_data` 的 `semantic_observations`（14 个 `astock_*` 抓取策略写入） | fd-open-data-mcp、语义检索、账本水位判据 | A 股 `price.close` 停在 **2026-09-18**；指数族 **0 行** |
| **B** | DAAS 的 `daas.db`（`scraw_<code>_daily` 原始日线 + `observations` 物化指标） | 研报取价、结算取数（`settlement_e2e.py`） | 14 个标的原始日线与 `_close` 物化指标均到 **2026-11-06**；指数 `000300`/`000905` 各 **5,304 行** |

**「指数族零观测」只对通道 A 成立。** 这句话 2026-11-10 在
`2026-11-10-deepen-trading-benchmark-and-risk` 的 5.5 已订正过一次（关键词：两个库是不同的存储），
2026-11-13 写投研现状时又按通道 A 的读数下了"`alpha_vs_benchmark` 结构性不可得"的结论——
是错的。通道 B 有完整指数日线，且已注册成指标规则（`000300_close` 等，id 452–461）。

复现：

```bash
kubectl -n fd-prod exec deploy/fd-daas-mcp -c fd-daas-mcp -- python3 -c "
import sqlite3
c=sqlite3.connect('file:/app/fd-daas-mcp/data/daas.db?mode=ro',uri=True)
for t in ('scraw_000300_daily','scraw_000905_daily','scraw_600519_daily'):
    print(t, c.execute('select count(*),max(date) from '+t).fetchone())
"
```

## 二、"观测落后"不是没数据，是物化没跑

原始日线到 11-06，但 `observations` 里的 `_close` 一度只到 **10-16**（14 个标的全部如此）。
`scraw_*_daily` 是手工 seed 进来的（DAAS 的 `schedules` / `tasks` **两张空表**，没有任何定时作业），
而派生指标要另跑一次 `daas_run_indicator` 才会从日线重算并写入 `observations`。

判定性证据（同一时刻的两组对照）：

- 跑之前 `600519_close` = **6,022 行 / 10-16**；跑一次 `daas_run_indicator 600519_close`
  → **6,037 行 / 11-06**（= 原始日线行数）。
- 同期**没跑**的 `000333_close` 仍停在 10-16。
- 补跑全部 70 条 astock 规则（14 标的 × `close`/`ma5`/`ma10`/`ma20`/`rsi14`）后，
  14 个标的的 `_close` 水位**全部**等于其原始日线的 `max(date)`（2026-11-06），零错误。

```bash
# 逐条重算（走工具面，与调用方同一条路径）
kubectl -n fd-prod port-forward svc/fd-daas-mcp 18311:8311 &
# 然后对每个规则名调 MCP tools/call → daas_run_indicator {"name": "<rule>"}
```

## 三、重算是"覆盖写"，安全边界要说清

`upsert_observations()` 用 `ON CONFLICT(...) DO UPDATE SET value=excluded.value`——
**全量重算会覆盖该指标的历史值**。所以不能想当然地"重算一遍没关系"。

实测的安全边界（`600036_close`，先抓 300 个历史日期再重算）：

| 项 | 结果 |
|---|---|
| 重叠日期 | 300 |
| **值发生变化的** | **0** |
| 新增日期 | 15（10-19…11-06） |

结论：**原始日线不变时，重算是值稳定的**（只补新日期）。

但有一条必须一起记住：A 股日线是 **qfq（前复权）**，而前复权**每次抓取都会重锚**。
若某次重新抓取让历史 bar 本身变了，重算就会把这段历史一起改写——那是**正确的传播**，
但它是一次静默的历史重写。所以：

- 重算前先确认 `scraw_configs` 的口径记录没变（`columns_json.source` / `adjust`）；
- 大范围重算前抽一个标的做上面的"重叠日期零变化"比对，作为当次的前置判据；
- 不要用"重算"来掩盖抓取口径的变更——那两件事必须分开报告。

```sql
-- 口径记录（DAAS 侧）
select name, url, columns_json from scraw_configs where name like 'astock_%';
```

## 四、基准表必须是**指数**，不能靠表名推断

同日修掉一条会静默通过的口径事故：`fd-quant/scripts/settlement_e2e.py` 的基准表写的是
`scraw_000001_daily`，而 **000001 在 A 股既是上证综指（`000001.SH`）也是平安银行（`000001.SZ`）**，
那张表装的是后者（11-06 收盘 **11.81**，综指在 4000 上下）。

它之所以能一直"绿"：算术复核逐位一致（只验内部自洽），`benchmark_symbol` 归因行写着
"上证综指"（由后缀映射推出，与真正喂进去的序列无关）。**两条判据都不会红。**

现行护栏（`fd-quant` `b57f358`）：基准表固定 `scraw_000300_daily`、
`assert_index_table()` 去 `scraw_configs.columns_json.source` 核对必须是 `index_zh_a_hist`
（缺口径记录也拒绝，fail-closed）、基准标的显式传 `benchmark_symbol=000300.SH`
（记录标 `mapping=explicit`）。用真实序列跑通：600519 1263.0→1329.0（+5.23%）、
沪深300 4317.25→4170.61（−3.40%）、**alpha +8.62%**，复核逐位一致。

## 五、仍然没解决的（别把上面读成"供给已恢复"）

- **没有任何定时作业在补日线。** DAAS 的 `schedules`/`tasks` 是空的；日线是手工 seed 的，
  所以 11-06 之后会继续停在那里，直到有人再跑一次 seed/refresh。
- **抓价格是平台级红线**（政策冻结后价格停爬）。所以"让它自己往前走"不是工程任务，
  而是**供给决策**：许可数据源，或客户提供序列，或接受停更并把水位如实标出来。
- **通道 A 侧仍空**（指数 0 行、A 股 09-18）。语义库/检索面看到的价格与通道 B 不同源，
  跨通道引用必须显式标注，不能默认"都是最新价"。

**因此本页能兑现的只有一句**：通道 B 里已有的数据可以被正确取用与结算（物化补齐、
基准选对、身份显式）；**它不会自己变新**。
