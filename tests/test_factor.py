"""因子统计族测试：逐 spec Scenario 一个用例 + 定义对拍 + 边界 + 拒绝 + 确定性。

"独立对拍"在各用例内就地写一份朴素实现（只依赖定义、纯 python，不调用被测代码的任何
内部函数）：这是唯一能发现"实现与定义一起写错"的方式。spec 措辞
（`specs/quant-compute-plane/spec.md` 的 `#### Scenario:` 与 `tasks.md` §7.3）逐条引用在
用例 docstring 里。

本族**不内置因子**：用例里的因子值都是就地构造的数列，或取 `quantdata.returns`
（它只是一列确定性数字，没有投资含义）。

被钉住的两条口径（它们是契约，不是实现细节）：
  - **并列名次**：float 精确相等判并列，并列段共享 midrank
    （即 `1 + 比它小的个数 + (与它相等的个数 - 1) / 2`）。
  - **分组余数**：n = g × base + rem 时，**前 rem 组（因子值最低的那几组）各多一个名额**。
"""
from __future__ import annotations

import inspect
import json
import math

import pytest

import quantdata as qd
from fd_quant.compute.base import ComputeError, assert_bitequal
from fd_quant.compute.factor import (
    TOOLS,
    compute_factor_exposure,
    compute_grouped_returns,
    compute_information_coefficient,
    rank_factor_scores,
)

AS_OF_N = 80


# ── 独立朴素实现（只依赖定义，不碰被测代码） ─────────────────────


def naive_pearson(xs, ys):
    """Pearson 相关系数，独立实现（两遍法 + 纯 python 求和）。"""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxy = sxx = syy = 0.0
    for i in range(n):
        dx = xs[i] - mx
        dy = ys[i] - my
        sxy += dx * dy
        sxx += dx * dx
        syy += dy * dy
    return sxy / math.sqrt(sxx * syy)


def naive_midranks(xs):
    """平均名次（1 = 最小），独立实现：逐点数"比我小的有几个、与我相等的有几个"。"""
    return [1.0 + sum(1 for y in xs if y < x) + (sum(1 for y in xs if y == x) - 1) / 2.0
            for x in xs]


def naive_ols(xs, ys):
    """一元 OLS（含截距）：(beta, alpha, t_beta, t_alpha, r_squared, residual_stdev)。

    纯 python 求和 + 显式残差 + 残差自由度 ddof = n - 2，独立于被测代码。
    """
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sxy = syy = 0.0
    for i in range(n):
        sxx += (xs[i] - mx) ** 2
        sxy += (xs[i] - mx) * (ys[i] - my)
        syy += (ys[i] - my) ** 2
    beta = sxy / sxx
    alpha = my - beta * mx
    rss = 0.0
    for i in range(n):
        e = ys[i] - alpha - beta * xs[i]
        rss += e * e
    residual_stdev = math.sqrt(rss / (n - 2))
    se_beta = residual_stdev / math.sqrt(sxx)
    se_alpha = residual_stdev * math.sqrt(1.0 / n + (mx * mx) / sxx)
    return beta, alpha, beta / se_beta, alpha / se_alpha, 1.0 - rss / syy, residual_stdev


def _fixture(n=AS_OF_N, fseed=109, rseed=113):
    """(dates, factor, returns)：两列都只是确定性数字列，无投资含义。"""
    ds = qd.dates(n)
    return ds, qd.returns(n, seed=fseed), qd.returns(n, seed=rseed)


# ── spec Scenario：IC 与分组收益对构造数据 ───────────────────────


def test_ic_pearson_is_one_on_factor_co_ordered_with_forward_returns():
    """spec Scenario:「传入一组与下期收益完全同序的因子值 → IC（Pearson）为 1.0（容差 1e-9）」。

    构造"完全同序"：下期收益 = 3 + 2 × 因子（正仿射），序一致且线性，故 Pearson 恰为 1.0。
    """
    n = 40
    ds = qd.dates(n)
    as_of = qd.of_dates_max(ds)
    factor = [float(i) for i in range(n)]
    fwd = [3.0 + 2.0 * f for f in factor]

    r = compute_information_coefficient(ds, factor, fwd, as_of)

    assert r["ic"] == pytest.approx(1.0, abs=1e-9)
    assert r["n_observations"] == n
    assert r["n_dropped"] == 0
    assert r["method"] == "pearson"
    # 与就地写的独立 Pearson 对拍（防"实现与定义一起写错"）
    assert r["ic"] == pytest.approx(naive_pearson(factor, fwd), abs=1e-12)


def test_ic_reverse_ordering_is_minus_one():
    """反序因子 → IC 为 -1.0：符号必须跟随序的方向（Pearson 与 Spearman 都是）。"""
    n = 30
    ds = qd.dates(n)
    as_of = qd.of_dates_max(ds)
    factor = [float(i) for i in range(n)]
    fwd = factor[::-1]

    for method in ("pearson", "spearman"):
        r = compute_information_coefficient(ds, factor, fwd, as_of, method=method)
        assert r["ic"] == pytest.approx(-1.0, abs=1e-9), method


def test_ic_spearman_is_one_on_rank_co_ordered_but_non_linear_factor():
    """只同序（保序但非线性）时 Spearman 仍为 1.0，Pearson 则小于 1。

    把"同序"与"线性"两件事分开：spec 的 Pearson=1.0 由仿射构造成立，而"仅同序"只有
    Spearman 保证 1.0——两条都要有，否则"Spearman 就是对名次做 Pearson"没有被测到。
    """
    n = 30
    ds = qd.dates(n)
    as_of = qd.of_dates_max(ds)
    factor = [float(i * i) for i in range(n)]            # 严格递增、非线性
    fwd = [math.exp(0.1 * i) for i in range(n)]          # 严格递增、非线性

    r_s = compute_information_coefficient(ds, factor, fwd, as_of, method="spearman")
    r_p = compute_information_coefficient(ds, factor, fwd, as_of)

    assert r_s["ic"] == pytest.approx(1.0, abs=1e-12)
    assert r_p["ic"] == pytest.approx(naive_pearson(factor, fwd), abs=1e-12)
    assert r_p["ic"] < r_s["ic"]


def test_grouped_returns_top_group_beats_bottom_on_ordered_factor():
    """spec Scenario:「最高分组收益大于最低分组」——因子与收益同序时最高组必然跑赢最低组。

    因子 = 0..n-1、收益 = 因子 × 0.001（严格递增）→ 逐组均值严格递增，故
    top_minus_bottom > 0 且 monotonic 为真。
    """
    n = 50
    ds = qd.dates(n)
    as_of = qd.of_dates_max(ds)
    factor = [float(i) for i in range(n)]
    fwd = [0.001 * f for f in factor]

    r = compute_grouped_returns(ds, factor, fwd, as_of, n_groups=5)

    groups = r["groups"]
    assert [g["group"] for g in groups] == [1, 2, 3, 4, 5]        # 1 = 因子最低组
    assert groups[-1]["mean_return"] > groups[0]["mean_return"]
    assert r["top_minus_bottom"] == pytest.approx(
        groups[-1]["mean_return"] - groups[0]["mean_return"], abs=1e-15
    )
    assert r["top_minus_bottom"] > 0.0
    assert r["monotonic"] is True
    assert all(groups[i]["mean_return"] < groups[i + 1]["mean_return"] for i in range(4))


# ── spec Scenario：暴露回归输出统计量 ────────────────────────────


def test_factor_exposure_on_constructed_exact_linear_relation():
    """构造精确线性关系 y = 3 + 2x → beta = 2.0、alpha = 3.0、R² = 1.0（1e-9）、残差标准差 = 0。

    精确线性下残差恒为 0，故残差标准差与两个系数的标准误都为 0：t 值是 0/0，以 null +
    reason 呈现（绝不返回 inf）——这是"退化只拒绝、不产 inf"的落点。
    """
    n = 10
    ds = qd.dates(n)
    as_of = qd.of_dates_max(ds)
    factor = [float(i) for i in range(n)]
    rets = [3.0 + 2.0 * x for x in factor]

    r = compute_factor_exposure(ds, factor, rets, as_of)

    assert r["beta"] == pytest.approx(2.0, abs=1e-9)
    assert r["alpha"] == pytest.approx(3.0, abs=1e-9)
    assert r["r_squared"] == pytest.approx(1.0, abs=1e-9)
    assert r["residual_stdev"] == 0.0
    assert r["n_observations"] == n
    assert r["t_beta"] is None and r["t_alpha"] is None
    assert "perfect fit" in r["reason"]
    json.dumps(r, allow_nan=False)          # null 而非 inf：仍可 JSON 序列化


def test_factor_exposure_r_squared_stays_in_unit_interval():
    """spec Scenario:「返回系数、t 值与 R²，且 R² 在 0..1 内」——随机夹具逐一断言。

    含截距的 OLS 数学上保证 R² ∈ [0, 1]；本族**断言而非夹取**该区间，故越界会在这里响亮
    地失败（若实现改成夹取，本用例会永远变绿、测不到真 bug）。
    """
    for seed in range(25):
        ds, factor, rets = _fixture(AS_OF_N, fseed=1000 + seed, rseed=2000 + seed)
        r = compute_factor_exposure(ds, factor, rets, qd.of_dates_max(ds))

        assert 0.0 <= r["r_squared"] <= 1.0, seed
        assert math.isfinite(r["beta"]) and math.isfinite(r["alpha"])
        assert math.isfinite(r["t_beta"]) and math.isfinite(r["t_alpha"])
        assert r["residual_stdev"] >= 0.0 and math.isfinite(r["residual_stdev"])
        json.dumps(r, allow_nan=False)


def test_factor_exposure_matches_independent_ols():
    """对拍：非平凡实例上与就地写的纯 python OLS 逐值比较（六个统计量全比）。"""
    ds, factor, rets = _fixture(60, fseed=555, rseed=777)
    as_of = qd.of_dates_max(ds)

    r = compute_factor_exposure(ds, factor, rets, as_of)
    b, a, tb, ta, r2, rs = naive_ols(factor, rets)

    assert r["beta"] == pytest.approx(b, abs=1e-12)
    assert r["alpha"] == pytest.approx(a, abs=1e-12)
    assert r["t_beta"] == pytest.approx(tb, abs=1e-9)
    assert r["t_alpha"] == pytest.approx(ta, abs=1e-9)
    assert r["r_squared"] == pytest.approx(r2, abs=1e-12)
    assert r["residual_stdev"] == pytest.approx(rs, abs=1e-12)


def test_factor_exposure_slope_is_invariant_to_adding_a_constant_to_the_factor():
    """把因子整体平移后 beta 与 R² 不变（截距吸收平移）——钉住"回归含截距"这条口径。

    无截距的斜率会随平移漂移，故平移不变性是"含截距"的可判别判据。
    """
    ds, factor, rets = _fixture(40, fseed=31, rseed=37)
    as_of = qd.of_dates_max(ds)

    r1 = compute_factor_exposure(ds, factor, rets, as_of)
    r2 = compute_factor_exposure(ds, [x + 1000.0 for x in factor], rets, as_of)

    assert r1["beta"] == pytest.approx(r2["beta"], abs=1e-9)
    assert r1["r_squared"] == pytest.approx(r2["r_squared"], abs=1e-9)


# ── 并列名次（Spearman 的 tie policy） ───────────────────────────


def test_spearman_tie_rule_is_average_rank_midrank():
    """并列取平均名次（midrank）：三个相等的值占排序后第 3、4、5 位 → 各得 4.0。

    断言**精确名次向量**（不是近似）：并列取法是定义的一部分，必须逐位固定。
    """
    ds = qd.dates(7)
    as_of = qd.of_dates_max(ds)
    factor = [10.0, 20.0, 30.0, 30.0, 30.0, 40.0, 50.0]

    r = rank_factor_scores(ds, factor, as_of)

    assert r["ranks"] == [1.0, 2.0, 4.0, 4.0, 4.0, 6.0, 7.0]
    assert r["n_ties"] == 3
    assert r["n_tie_groups"] == 1
    assert r["ranks"] == pytest.approx(naive_midranks(factor), abs=1e-12)


def test_spearman_tie_rule_with_two_tie_groups():
    """两段并列：10.0 占第 2、3 位 → 各 2.5；30.0 占第 5、6、7 位 → 各 6.0。

    n_ties 是"被并列的观测数"（2 + 3 = 5）、n_tie_groups 是并列段数（2）——两者一起回给
    调用方，因为"3 组各 2 个"与"1 组 6 个"对名次分布的影响不同。
    """
    ds = qd.dates(8)
    as_of = qd.of_dates_max(ds)
    factor = [5.0, 10.0, 10.0, 20.0, 30.0, 30.0, 30.0, 40.0]

    r = rank_factor_scores(ds, factor, as_of)

    assert r["ranks"] == [1.0, 2.5, 2.5, 4.0, 6.0, 6.0, 6.0, 8.0]
    assert r["n_ties"] == 5
    assert r["n_tie_groups"] == 2
    assert r["ranks"] == pytest.approx(naive_midranks(factor), abs=1e-12)


def test_rank_factor_scores_and_spearman_ic_agree_on_ties():
    """`rank_factor_scores` 与 `compute_information_coefficient(method="spearman")` 并列口径一致。

    两者必须共用同一份名次实现（并列规则只允许有一处实现），否则两条路径会在并列处漂移。
    这里逐位比较两条路径给出的因子名次。
    """
    ds = qd.dates(12)
    as_of = qd.of_dates_max(ds)
    factor = [3.0, 3.0, 1.0, 2.0, 5.0, 5.0, 5.0, 4.0, 9.0, 9.0, 7.0, 6.0]
    rets = list(reversed(qd.returns(12, seed=17)))

    ranks_util = rank_factor_scores(ds, factor, as_of)["ranks"]
    ranks_ic = compute_information_coefficient(
        ds, factor, rets, as_of, method="spearman"
    )["ranked_factor"]

    assert ranks_ic == ranks_util
    # 与独立 midrank 实现一致，且名次和恒为 n(n+1)/2（并列只是重分配，不改变总和）
    assert ranks_util == pytest.approx(naive_midranks(factor), abs=1e-12)
    assert sum(ranks_util) == pytest.approx(12 * 13 / 2, abs=1e-12)
    assert min(ranks_util) == 1.0
    assert max(ranks_util) == max(naive_midranks(factor))


def test_spearman_returns_both_ranked_series_for_audit():
    """Spearman 回带两侧名次（供审计），且收益侧名次与独立实现逐点一致。"""
    ds = qd.dates(6)
    as_of = qd.of_dates_max(ds)
    factor = [1.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    rets = [0.5, 0.5, 0.5, 0.1, -0.2, 0.9]

    r = compute_information_coefficient(ds, factor, rets, as_of, method="spearman")

    assert r["ranked_factor"] == pytest.approx(naive_midranks(factor), abs=1e-12)
    assert r["ranked_returns"] == pytest.approx(naive_midranks(rets), abs=1e-12)
    assert r["ic"] == pytest.approx(
        naive_pearson(naive_midranks(factor), naive_midranks(rets)), abs=1e-12
    )
    assert len(r["ranked_factor"]) == r["n_observations"]
    assert r["used_indices"] == list(range(6))


def test_pearson_ic_omits_ranked_series():
    """method="pearson" 时不回带名次键（只在 spearman 下出现，避免无意义的重复载荷）。"""
    ds, factor, rets = _fixture(20, fseed=5, rseed=6)
    r = compute_information_coefficient(ds, factor, rets, qd.of_dates_max(ds))

    assert "ranked_factor" not in r and "ranked_returns" not in r
    assert r["method"] == "pearson"


# ── 逐对删除（pairwise deletion） ────────────────────────────────


def test_pairwise_deletion_drops_only_pairs_with_a_missing_side():
    """逐对删除：任一侧为 null 的那一对整对剔除，其余保留；重叠缺失只算一次。

    索引 2 因子缺失、索引 5 收益缺失、索引 7 两侧同时缺失 → n_dropped = 3、
    n_observations = 7，且 used_indices 恰为剩下的那 7 个。
    """
    ds = qd.dates(10)
    as_of = qd.of_dates_max(ds)
    factor = [1.0, 2.0, None, 4.0, 5.0, 6.0, 7.0, None, 9.0, 11.0]
    fwd = [0.1, -0.2, 0.3, 0.4, -0.1, None, 0.7, None, -0.9, 0.2]

    r = compute_information_coefficient(ds, factor, fwd, as_of)

    assert r["n_observations"] == 7
    assert r["n_dropped"] == 3
    assert r["used_indices"] == [0, 1, 3, 4, 6, 8, 9]
    keep = [i for i in range(10) if factor[i] is not None and fwd[i] is not None]
    assert r["used_indices"] == keep
    # 删除后照常出数（不是整条 null），且删除规模可审计
    assert math.isfinite(r["ic"])


def test_pairwise_deletion_in_grouped_returns_keeps_a_partition():
    """分组在逐对删除后进行：groups 的成员恰是可用观测的一个划分（不重不漏）。

    同时验证 members 是按因子值升序排定的**输入索引**、组大小相差不超过 1。
    """
    n = 23
    ds = qd.dates(n)
    as_of = qd.of_dates_max(ds)
    factor = list(qd.returns(n, seed=11))
    fwd = list(qd.returns(n, seed=13))
    factor[3] = None
    fwd[7] = None
    factor[7] = None            # 与上一条重叠
    fwd[15] = None

    r = compute_grouped_returns(ds, factor, fwd, as_of, n_groups=4)

    used = [i for i in range(n) if factor[i] is not None and fwd[i] is not None]
    members = [i for g in r["groups"] for i in g["members"]]
    assert sorted(members) == used                        # 划分：不重不漏
    assert len(members) == len(set(members))
    assert r["n_observations"] == len(used)
    assert r["n_dropped"] == n - len(used)
    for g in r["groups"]:
        vals = [factor[i] for i in g["members"]]
        assert vals == sorted(vals), "members must be ascending by factor value"
    sizes = [g["n_members"] for g in r["groups"]]
    assert max(sizes) - min(sizes) <= 1
    assert sum(sizes) == len(used)


def test_pairwise_deletion_applies_to_regression_too():
    """回归同样逐对删除、删除数可审计；两列缺失重叠时只算一次。"""
    n = 14
    ds = qd.dates(n)
    as_of = qd.of_dates_max(ds)
    factor = [float(i) for i in range(n)]
    rets = [0.5 * x + 1.0 for x in factor]
    factor[1] = None
    rets[9] = None
    factor[4] = None
    rets[4] = None              # 重叠

    r = compute_factor_exposure(ds, factor, rets, as_of, min_observations=3)

    assert r["n_dropped"] == 3
    assert r["n_observations"] == 11
    assert r["used_indices"] == [0, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13]
    # 剩下的 (x, 0.5x + 1) 仍精确共线
    assert r["beta"] == pytest.approx(0.5, abs=1e-12)
    assert r["alpha"] == pytest.approx(1.0, abs=1e-12)
    assert r["residual_stdev"] == 0.0


def test_rank_factor_scores_leaves_missing_unranked():
    """排名遇缺失：缺失位名次为 null（不占名次位、也不被填值），有限值处名次连续。"""
    ds = qd.dates(6)
    as_of = qd.of_dates_max(ds)
    factor = [4.0, None, 1.0, 3.0, None, 2.0]

    r = rank_factor_scores(ds, factor, as_of)

    assert r["ranks"] == [4.0, None, 1.0, 3.0, None, 2.0]
    assert r["order"] == [2, 5, 3, 0]        # 输入索引，按 (值升序, 输入索引)
    assert r["n_observations"] == 4
    assert r["n_dropped"] == 2
    assert r["warmup"] == {"ranks": 0}


# ── 分组余数规则 ─────────────────────────────────────────────────


def test_group_remainder_goes_to_the_lowest_groups():
    """余数规则：n = g × base + rem 时**前 rem 组（因子值最低的那几组）各多一个名额**。

    n = 7、g = 3 → base = 2、rem = 1 → 组大小 [3, 2, 2]（不是 [2, 2, 3]）。
    "余数进高位组"同样看起来合理，故这条必须被断言，否则规则会随实现漂移。
    """
    n = 7
    ds = qd.dates(n)
    as_of = qd.of_dates_max(ds)
    factor = [float(i) for i in range(n)]
    fwd = [0.01 * i for i in range(n)]

    r = compute_grouped_returns(ds, factor, fwd, as_of, n_groups=3)

    assert [g["n_members"] for g in r["groups"]] == [3, 2, 2]
    assert [g["group"] for g in r["groups"]] == [1, 2, 3]
    assert r["groups"][0]["members"] == [0, 1, 2]      # 最低组拿走那个多出的名额
    assert r["groups"][1]["members"] == [3, 4]
    assert r["groups"][2]["members"] == [5, 6]


def test_group_remainder_of_two_also_goes_to_the_lowest_groups():
    """n = 8、g = 3 → base = 2、rem = 2 → 组大小 [3, 3, 2]（余数只进低位组）。"""
    n = 8
    ds = qd.dates(n)
    as_of = qd.of_dates_max(ds)
    factor = [float(i) for i in range(n)]
    fwd = [0.01 * i for i in range(n)]

    r = compute_grouped_returns(ds, factor, fwd, as_of, n_groups=3)

    assert [g["n_members"] for g in r["groups"]] == [3, 3, 2]


def test_grouped_returns_equal_weights_when_observation_count_is_divisible():
    """整除时组大小全相等，且组均值等于组内成员收益的等权均值（独立实现逐组对拍）。"""
    n = 20
    ds = qd.dates(n)
    as_of = qd.of_dates_max(ds)
    factor = list(qd.returns(n, seed=23))
    fwd = list(qd.returns(n, seed=29))

    r = compute_grouped_returns(ds, factor, fwd, as_of, n_groups=5)

    assert [g["n_members"] for g in r["groups"]] == [4, 4, 4, 4, 4]
    order = sorted(range(n), key=lambda i: (factor[i], i))
    for k, g in enumerate(r["groups"]):
        expected_members = order[k * 4:(k + 1) * 4]
        assert g["members"] == expected_members
        expected_mean = sum(fwd[i] for i in expected_members) / 4     # 独立等权实现
        assert g["mean_return"] == pytest.approx(expected_mean, abs=1e-15)


def test_grouped_returns_monotonic_flag_is_non_decreasing():
    """monotonic 判据是"非递减"（相等不算违反）：因子全并列时各组均值仍随索引递增 → True。"""
    n = 12
    ds = qd.dates(n)
    as_of = qd.of_dates_max(ds)
    factor = [7.0] * n                      # 全并列 → 分组只按输入索引切
    fwd = [0.001 * i for i in range(n)]

    r = compute_grouped_returns(ds, factor, fwd, as_of, n_groups=4)

    assert r["monotonic"] is True
    means = [g["mean_return"] for g in r["groups"]]
    assert means == sorted(means)


# ── 拒绝（报错必须点名缺的东西） ─────────────────────────────────


def test_constant_factor_is_refused_and_names_the_input():
    """常数因子（零方差）→ 拒绝并**点名该列**（不是 NaN，也不是 0）。

    IC 的相关系数分母、回归的斜率都在这里 0/0：spec 要求"拒绝"而不是给一个数。
    """
    ds, _factor, rets = _fixture(20, fseed=3, rseed=4)
    as_of = qd.of_dates_max(ds)
    const = [5.0] * 20

    with pytest.raises(ComputeError, match="factor") as ei:
        compute_information_coefficient(ds, const, rets, as_of)
    assert "constant" in str(ei.value) and "zero variance" in str(ei.value)

    with pytest.raises(ComputeError, match="factor"):
        compute_information_coefficient(ds, const, rets, as_of, method="spearman")

    with pytest.raises(ComputeError, match="factor"):
        compute_factor_exposure(ds, const, rets, as_of)


def test_constant_response_is_refused_and_names_that_column():
    """收益侧零方差同样拒绝并点名那一列（IC 的两侧都查）。"""
    ds, factor, _rets = _fixture(20, fseed=8, rseed=9)
    as_of = qd.of_dates_max(ds)
    const = [0.01] * 20

    with pytest.raises(ComputeError, match="forward_returns"):
        compute_information_coefficient(ds, factor, const, as_of)

    with pytest.raises(ComputeError, match="returns"):
        compute_factor_exposure(ds, factor, const, as_of)


def test_mismatched_lengths_are_refused_and_name_both():
    """长度不一致 → 拒绝并**同时点名两个长度**（调用方据此知道要剪掉多少）。"""
    ds, factor, rets = _fixture(20, fseed=1, rseed=2)
    as_of = qd.of_dates_max(ds)

    with pytest.raises(ComputeError) as ei:
        compute_information_coefficient(ds, factor, rets[:-1], as_of)
    msg = str(ei.value)
    assert "20" in msg and "19" in msg, msg
    assert "forward_returns" in msg

    with pytest.raises(ComputeError) as ei2:
        compute_grouped_returns(ds, factor[:-2], rets, as_of)
    msg2 = str(ei2.value)
    assert "20" in msg2 and "18" in msg2, msg2
    assert "factor" in msg2

    with pytest.raises(ComputeError) as ei3:
        compute_factor_exposure(ds, factor, rets[:-3], as_of)
    msg3 = str(ei3.value)
    assert "20" in msg3 and "17" in msg3, msg3


def test_too_few_observations_is_refused_and_names_the_count():
    """可用观测不足 → 拒绝并**点名该数与下限**（不静默按更少的点算）。"""
    ds = qd.dates(6)
    as_of = qd.of_dates_max(ds)
    factor = [1.0, 2.0, None, 4.0, 5.0, 6.0]
    fwd = [0.1, 0.2, 0.3, None, 0.5, 0.6]

    with pytest.raises(ComputeError) as ei:
        compute_information_coefficient(ds, factor, fwd, as_of)
    msg = str(ei.value)
    assert "4" in msg and "5" in msg, msg          # 可用 4 < 默认下限 5
    assert "min_observations" in msg

    with pytest.raises(ComputeError) as ei2:
        compute_factor_exposure(ds, factor, fwd, as_of, min_observations=9)
    msg2 = str(ei2.value)
    assert "4" in msg2 and "9" in msg2, msg2
    assert "min_observations" in msg2


def test_degrees_of_freedom_not_positive_is_refused():
    """残差自由度 n - 2 ≤ 0 → 拒绝并点名该自由度（残差标准差要除以它，除数为 0 无从报告）。"""
    ds = qd.dates(2)
    as_of = qd.of_dates_max(ds)

    with pytest.raises(ComputeError) as ei:
        compute_factor_exposure(ds, [1.0, 2.0], [0.1, 0.2], as_of, min_observations=2)
    msg = str(ei.value)
    assert "degrees of freedom" in msg and "0" in msg, msg


def test_group_count_out_of_range_is_refused_and_names_both_numbers():
    """n_groups 必须 ≥ 2 且 ≤ 可用观测数：两个方向的违规都拒绝并点名（≥2 才能比高低组）。"""
    ds, factor, rets = _fixture(10, fseed=41, rseed=43)
    as_of = qd.of_dates_max(ds)

    with pytest.raises(ComputeError, match="n_groups"):
        compute_grouped_returns(ds, factor, rets, as_of, n_groups=1)

    with pytest.raises(ComputeError) as ei:
        compute_grouped_returns(ds, factor, rets, as_of, n_groups=11)
    msg = str(ei.value)
    assert "11" in msg and "10" in msg, msg


def test_invalid_method_is_refused_and_names_the_value():
    """非法的 method → 拒绝并点名该值（不静默退回 pearson：那等于替调用方换定义）。"""
    ds, factor, rets = _fixture(20, fseed=51, rseed=53)
    as_of = qd.of_dates_max(ds)

    with pytest.raises(ComputeError) as ei:
        compute_information_coefficient(ds, factor, rets, as_of, method="kendall")
    assert "kendall" in str(ei.value)


def test_rank_factor_scores_refuses_an_all_missing_factor():
    """全是缺失的因子列无值可排 → 拒绝并点名该列（不是返回一列 null）。"""
    ds = qd.dates(5)
    as_of = qd.of_dates_max(ds)

    with pytest.raises(ComputeError, match="factor"):
        rank_factor_scores(ds, [None] * 5, as_of)


def test_missing_as_of_is_refused_on_every_tool():
    """口径先于一切：缺 as_of 一律拒绝，且错误里点名 as_of。"""
    ds, factor, rets = _fixture(20, fseed=61, rseed=67)

    for fn, args in (
        (compute_information_coefficient, (ds, factor, rets)),
        (compute_grouped_returns, (ds, factor, rets)),
        (compute_factor_exposure, (ds, factor, rets)),
        (rank_factor_scores, (ds, factor)),
    ):
        with pytest.raises(ComputeError, match="as_of"):
            fn(*args, as_of=None)


# ── 口径与 as_of（越界拒绝、点名首个日期） ───────────────────────


def test_information_coefficient_refuses_future_data_and_names_the_first_date():
    """越界拒绝且点名**首个**越界日期（不静默截断）——每个工具都查。"""
    d = qd.ohlcv(40, seed=71)
    ds, as_of = d["dates"], d["dates"][10]

    with pytest.raises(ComputeError) as ei:
        compute_information_coefficient(ds, d["close"], d["close"], as_of)
    assert ds[11] in str(ei.value)

    with pytest.raises(ComputeError) as ei2:
        compute_grouped_returns(ds, d["close"], d["close"], as_of)
    assert ds[11] in str(ei2.value)

    with pytest.raises(ComputeError) as ei3:
        compute_factor_exposure(ds, d["close"], d["close"], as_of)
    assert ds[11] in str(ei3.value)

    with pytest.raises(ComputeError) as ei4:
        rank_factor_scores(ds, d["close"], as_of)
    assert ds[11] in str(ei4.value)


def test_unsorted_dates_are_refused():
    """日期非严格递增 → 拒绝并点名那一对（顺序是按位配对的先决条件）。"""
    ds = qd.dates(10)
    ds[4], ds[5] = ds[5], ds[4]
    as_of = qd.of_dates_max(sorted(ds))
    factor = [float(i) for i in range(10)]

    with pytest.raises(ComputeError, match="strictly increasing"):
        rank_factor_scores(ds, factor, as_of)


def test_envelope_declares_input_span_and_evidence():
    """外壳回带输入口径（as_of / 起止 / 长度）与 evidence（非空、六字段、id 唯一、source 带名）。"""
    ds, factor, rets = _fixture(30, fseed=81, rseed=83)
    as_of = qd.of_dates_max(ds)

    r = compute_information_coefficient(ds, factor, rets, as_of)

    assert r["as_of"] == as_of
    assert r["input_start"] == ds[0]
    assert r["input_end"] == ds[-1]
    assert r["input_length"] == 30
    ev = r["evidence"]
    assert ev
    ids = [e["evidence_id"] for e in ev]
    assert len(ids) == len(set(ids))
    for e in ev:
        assert set(e) == {"evidence_id", "label", "value", "unit", "as_of", "source"}
        assert e["as_of"] == as_of
        assert e["source"].startswith("fd-quant:")
        assert "compute_information_coefficient" in e["source"]


def test_params_echo_effective_values_including_defaults():
    """params 回显**实际生效**的值（含默认值）：调用方不必读文档才知道口径。"""
    ds, factor, rets = _fixture(20, fseed=91, rseed=97)
    as_of = qd.of_dates_max(ds)

    r_ic = compute_information_coefficient(ds, factor, rets, as_of)
    # return_caliber 自 deepen-trading-benchmark-and-risk 5.3 起进 params：
    # IC 的口径算不出来，只能由调用方声明；默认 raw（不谎报成超额）。
    assert r_ic["params"] == {"method": "pearson", "min_observations": 5,
                              "return_caliber": "raw"}

    r_ic_s = compute_information_coefficient(ds, factor, rets, as_of, method="Spearman")
    assert r_ic_s["params"]["method"] == "spearman"       # 大小写不敏感，回显小写
    assert r_ic_s["method"] == "spearman"

    assert compute_grouped_returns(ds, factor, rets, as_of)["params"] == {"n_groups": 5}
    assert compute_grouped_returns(ds, factor, rets, as_of, n_groups=3)["params"] == {
        "n_groups": 3
    }
    assert compute_factor_exposure(ds, factor, rets, as_of)["params"] == {"min_observations": 5}
    assert rank_factor_scores(ds, factor, as_of)["params"] == {}


# ── evidence 与确定性 ────────────────────────────────────────────


def _all_calls():
    """每个工具的一组典型合法调用（供 evidence / 确定性 / JSON 用例复用）。"""
    ds, factor, rets = _fixture(40, fseed=101, rseed=103)
    as_of = qd.of_dates_max(ds)
    return [
        (compute_information_coefficient, (ds, factor, rets, as_of), {}),
        (compute_information_coefficient, (ds, factor, rets, as_of), {"method": "spearman"}),
        (compute_grouped_returns, (ds, factor, rets, as_of), {"n_groups": 4}),
        (compute_factor_exposure, (ds, factor, rets, as_of), {}),
        (rank_factor_scores, (ds, factor, as_of), {}),
    ]


def test_every_tool_emits_traceable_evidence():
    """每个工具的 evidence 都非空、id 唯一、六字段齐备，且观测构成（可用/删除）可追溯。"""
    for fn, args, kw in _all_calls():
        r = fn(*args, **kw)
        ev = r["evidence"]
        assert ev, fn.__name__
        ids = [e["evidence_id"] for e in ev]
        assert len(ids) == len(set(ids)), fn.__name__
        labels = " ".join(e["label"] for e in ev)
        for e in ev:
            assert set(e) == {"evidence_id", "label", "value", "unit", "as_of", "source"}
            assert e["as_of"] == r["as_of"]
            assert e["source"].startswith("fd-quant:")
            assert fn.__name__ in e["source"], e["source"]
        assert "usable observations" in labels and "dropped pairs" in labels


def test_bitwise_determinism_on_every_tool():
    """同输入两次调用逐位一致（IEEE 754 精确比较）——每个工具、每条口径。"""
    for fn, args, kw in _all_calls():
        assert_bitequal(fn(*args, **kw), fn(*args, **kw), what=fn.__name__)


def test_json_serialisable_including_degenerate_paths():
    """返回恒可 json.dumps(allow_nan=False)：退化路径给 null，绝不给 NaN/inf。"""
    ds, factor, rets = _fixture(20, fseed=107, rseed=109)
    as_of = qd.of_dates_max(ds)
    exact = [3.0 + 2.0 * x for x in factor]                 # 完美拟合（残差标准差 = 0）

    results = [
        compute_information_coefficient(ds, factor, rets, as_of),
        compute_information_coefficient(ds, factor, rets, as_of, method="spearman"),
        compute_grouped_returns(ds, factor, rets, as_of),
        compute_factor_exposure(ds, factor, rets, as_of),
        compute_factor_exposure(ds, factor, exact, as_of),   # t 值为 null 的退化路径
        rank_factor_scores(ds, factor, as_of),
        rank_factor_scores(ds, [None] + list(factor[1:]), as_of),
    ]
    for r in results:
        assert json.dumps(r, allow_nan=False)
        assert isinstance(r["evidence"], list) and r["evidence"]


def test_nan_input_is_deleted_pairwise_and_never_leaks_nan():
    """NaN 输入 → 逐对删除后照常出数；返回值里不出现 NaN（缺失只以 null 呈现）。"""
    ds, factor, rets = _fixture(30, fseed=113, rseed=127)
    as_of = qd.of_dates_max(ds)
    factor = list(factor)
    rets = list(rets)
    for i in (5, 6, 7):
        factor[i] = None
    for i in (6, 7, 20):
        rets[i] = None
    # 缺失索引并集 = {5, 6, 7, 20} → 4 对删除、26 对可用

    for fn, args, kw in [
        (compute_information_coefficient, (ds, factor, rets, as_of), {}),
        (compute_information_coefficient, (ds, factor, rets, as_of), {"method": "spearman"}),
        (compute_grouped_returns, (ds, factor, rets, as_of), {}),
        (compute_factor_exposure, (ds, factor, rets, as_of), {}),
    ]:
        r = fn(*args, **kw)
        assert json.dumps(r, allow_nan=False)
        assert r["n_dropped"] == 4, fn.__name__
        assert r["n_observations"] == 26, fn.__name__


# ── 无因子库纪律与参数形态 ───────────────────────────────────────


def test_tool_surface_is_only_statistical_atoms():
    """spec:「本能力 SHALL NOT 内置任何因子定义或因子库」——工具面与源码级断言。

    四个工具都是"对调用方给的两列作答"的统计原子；模块里没有任何"由价量算出某个因子"
    的构造入口，也没有因子名到公式的登记表。
    """
    assert [t.name for t in TOOLS] == [
        "compute_information_coefficient",
        "compute_grouped_returns",
        "compute_factor_exposure",
        "rank_factor_scores",
    ]
    assert all(t.group == "factor_statistics" for t in TOOLS)

    import fd_quant.compute.factor as mod

    src = open(mod.__file__, encoding="utf-8").read()
    for banned in ("momentum", "MOMENTUM", "factor_library", "FACTOR_LIBRARY",
                   "builtin_factors", "compute_momentum", "quality_factor"):
        assert banned not in src, f"factor library artifact found: {banned}"


def test_no_global_random_state_is_used():
    """确定性来源核查：模块不用 numpy 全局随机接口、不用 random 模块、不用矩阵归约。

    只扫**可执行代码**（tokenize 掉注释与字符串）：模块 docstring 会把被禁的调用
    名字写出来当反面清单，扫原文会把"禁止的说明"误判成"违规的代码"。
    """
    import io
    import tokenize

    import fd_quant.compute.factor as mod

    code: list[str] = []
    with open(mod.__file__, encoding="utf-8") as fh:
        for tok in tokenize.generate_tokens(fh.readline):
            if tok.type in (tokenize.COMMENT, tokenize.STRING):
                continue                      # 注释与字符串不是代码
            code.append(tok.string)
    src = " ".join(code)
    for banned in ("np.random", "numpy.random", "np.dot", "np.matmul", "np.linalg"):
        assert banned not in src, f"non-deterministic / BLAS reduction found: {banned}"
    # stdlib 随机模块按 import 形态判（单独匹配 "random" 会误伤 numpy 的 default_rng 等名字）
    assert "import random" not in src and "from random" not in src, (
        "the stdlib random module must not be used (pass an explicit seed instead)"
    )


def test_all_tools_take_both_series_from_the_caller():
    """每个工具都由调用方给出两列（无内置默认序列）：序列入参一律无默认值。"""
    for fn, keys in (
        (compute_information_coefficient, ("factor", "forward_returns")),
        (compute_grouped_returns, ("factor", "forward_returns")),
        (compute_factor_exposure, ("factor", "returns")),
        (rank_factor_scores, ("factor",)),
    ):
        p = inspect.signature(fn).parameters
        for key in keys:
            assert key in p, f"{fn.__name__}: no {key} parameter"
            assert p[key].default is inspect.Parameter.empty, (
                f"{fn.__name__}: {key} must not have a default (no bundled factor/returns)"
            )
        assert "as_of" in p
        assert p["as_of"].default is inspect.Parameter.empty


def test_docstrings_are_self_sufficient():
    """spec 8.1 / 族作者契约 §4：描述含 Input: / Returns: 与 as_of、单位、warmup 语义。"""
    for spec in TOOLS:
        doc = spec.doc
        assert len(doc) >= 120, spec.name
        low = doc.lower()
        assert "input:" in low, spec.name
        assert "returns:" in low, spec.name
        assert "as_of" in low, spec.name
        assert "warmup" in low, spec.name
