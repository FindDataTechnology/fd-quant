# 族作者契约（fd-quant 计算面）

写一个新族（`fd_quant/compute/<family>.py`）时，你必须满足下面全部条款。它们不是
建议——`tests/test_family_conformance.py` 会对你的族逐工具断言，漏一条就红。

先读这三份，它们是规格而非参考：

- `fd_quant/compute/base.py` —— 你只能通过它进出的接口（口径、校验、evidence、外壳）。
- `fd_quant/compute/technical.py` —— **照着写**的样板族，已 61 测试全绿。
- `fd_quant/compute/__init__.py` —— `ToolSpec`、`GROUP_LEVELS`、注册中心。

## 1. 文件形态

```python
from . import ToolSpec
from .base import (EvidenceBuilder, ComputeError, check_as_of, check_positive_window,
                   check_series, check_finite, deterministic_reduction,
                   result_envelope, to_none_list)

GROUP = "<group_key>"          # 必须是 GROUP_LEVELS 里的键

def compute_xxx(dates, values, as_of, window=20) -> dict:
    """<自足描述，见 §4>"""
    a = check_as_of(as_of)               # 第一步永远是它
    ...                                   # 参数校验用 check_* 系列
    s = check_series(dates, values, as_of=a, name="values")
    ...                                   # 纯计算
    b = EvidenceBuilder("compute_xxx", a)
    b.add(...)                            # 至少两条
    return result_envelope("compute_xxx", a, s, b, {...})

TOOLS: list[ToolSpec] = [ToolSpec("compute_xxx", GROUP, compute_xxx), ...]
```

## 2. 硬约束（逐条会被断言）

1. **`as_of` 必填 kwarg，无默认值。** 且必须是函数的**第一个校验动作**——
   这样越界错误信息不会先被别的参数错误遮住。
2. **越界拒绝，点名首个越界日期。** 由 `check_series` 保证；不要自己写截断逻辑。
3. **不取数、不联网、不读库、不调 LLM。** 计算层源码里不得出现
   `requests/httpx/urllib/socket/sqlalchemy/psycopg/sqlite3/openai/anthropic/
   torch/transformers/langchain/akshare/yfinance` 等（有自动扫描断言）。
4. **不自实现随机源。** 需要随机（Monte Carlo）时：`seed` 必须可传且**有固定默认值**，
   用 `np.random.default_rng(seed)`（PCG64，跨版本稳定），禁止 `np.random.rand/randn`
   旧式全局接口，禁止 `random` 模块。
5. **归约顺序固定。** 求和用 `deterministic_reduction`（Kahan 从左到右）；或
   `np.cumsum` 顺序扫描。禁止用 `@`/`np.dot`/`np.matmul` 做数值归约（BLAS 分块随
   线程数变化）。回归系数等必须用显式循环或 cumsum 形式的最小二乘。
6. **warmup 一律 `null`**，绝不用 0 或前值填充。函数返回 `warmup` 字典时，
   逐输出线给出"首个非 null 位置之前的 null 个数"。
7. **NaN 局部传播**：窗口内含 NaN → 该点 null；窗口外不受污染。递推类指标自首个
   NaN 起其后全 NaN（不跳过——跳过等于替调用方发明一个数据点）。
8. **返回值必须 JSON 可序列化**（`json.dumps(r, allow_nan=False)` 通过）：
   用 `to_none_list()` 把 ndarray/NaN 换成 list/None。禁止把 np.float64 直接放进返回。
9. **本地不 import 契约层私有函数**（`_is_iso_date` 等），各层各自校验同一形状。
10. **参数与 warmup 回显**：返回含 `params`（实际生效的值，含默认值）与 `warmup`。
11. **evidence ≥ 1 条**；序列型至少"长度 + 首值 + 末值"（用 `_emit_line` 的等价写法，
    但**不要**从 technical.py import 私有函数——那是私有；把需要的 helper 在自己的
    文件里写一份，或直接调 `b.add`）。`evidence.source` 必须含工具名与参数。
12. **工具名 = 函数名**，`动词_名词`，动词限：`compute / summarize / evaluate /
    analyze / validate / assess / estimate / detect / build / check / derive /
    screen / size / score / rank / split / document`。

## 3. 返回外壳

统一用 `result_envelope(tool, as_of, series, builder, payload)`——它会补
`as_of / input_start / input_end / input_length / evidence`。你的 `payload` 里放
业务键（如 `value` / `params` / `warmup` / `ratios` / `windows`）。

标量型工具（如单个最大回撤）传 `series=None`；此时外壳的 start/end 为 None，这
是合法的——`source` 与 `as_of` 仍会带上。若你的工具主角是权益序列，把那个 Series
传进去。

## 4. 描述必须自足（spec 8.1 会被逐工具断言）

docstring 至少 120 字符，且必须含三段（英文小标题，与 technical.py 一致）：

```
Input: dates (ISO, 严格递增), values (...), as_of (ISO, 必填), window (默认 20)。
Returns: value (...), params, warmup, evidence。
```

并在正文写清：**单位**、**warmup 语义**、**NaN 语义**、**返回形状与长度**。
判据是"单独读这段描述能不能正确调用"。

## 5. 测试

新建 `tests/test_<family>.py`。**每个 spec Scenario 至少一个用例**，并且在用例的
docstring 里引用 spec 措辞。除此之外：

- 对拍：关键定义要写一份**独立**的朴素实现（就地写，不调用被测代码内部函数），
  逐点比较。这是唯一能发现"实现与定义一起写错"的方式。
- 边界：空序列、长度不足、全 NaN、单元素、窗口=1、窗口>长度、极端值。
- 拒绝：缺字段/非法参数时报错文本必须**指明缺失的东西**（字段名/参数名/日期）。

运行：

```bash
cd /Users/chengsishi/finddata/fd-quant
.venv/bin/python -m pytest tests/test_<family>.py -q
.venv/bin/python -m pytest tests/test_family_conformance.py -q -k "<group_key>"
```

两条都必须全绿。合规闸门里那两条"族缺失"的失败不算你的——只要你的组不再是
`has no tools`。

## 6. 禁止事项

- 不要修改 `base.py` / `technical.py` / `__init__.py` / `conftest.py` /
  `quantdata.py`。需要新夹具就在你自己的测试文件里构造；确实需要通用夹具时，
  在报告里说明，由主线统一加。
- 不要 `import pandas`。计算层只用 numpy（pandas 的版本语义会把定义漂移带进来）。
- 不要用 `float` 之外的精度技巧（`decimal`/`mpmath`）——契约是 float64。
- 不要在返回值里塞调试信息、原始 ndarray、或未声明的键。
