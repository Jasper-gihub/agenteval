"""统计层：把"看起来涨了 5 个点"变成能不能站得住的判断。

四条来自实战的约束：

1. **配对**。比两个准确率会浪费掉"同一题"的信息。只有 2×2 表里对角线外的
   格子在提供证据，所以检验的对象是不一致对（b vs c），不是两个比例。
2. **先算样本量**。跑完再算 p 值是本末倒置：你可能花 1.15M token
   做了一个从设计上就不可能有结论的实验。
3. **不依赖 scipy**。全部用标准库实现，pip 依赖为空，
   CI 不会因为科学计算包版本问题挂掉。
4. **区分采样噪声与任务难度**。同一题多次采样，方差该归因到"模型随机性"
   还是"这题本身就难"，决定你该加题还是加采样次数。
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from statistics import NormalDist, mean, variance

_NORMAL = NormalDist()


# --------------------------------------------------------------------------
# 配对检验
# --------------------------------------------------------------------------
def mcnemar_exact(b: int, c: int) -> float:
    """精确 McNemar 检验（双侧二项检验），返回 p 值。

    b = A 对 B 错的题数，c = A 错 B 对的题数。都对/都错不参与。

    用精确二项而不是卡方近似：评测集通常是几十到几百题，
    卡方近似在 b+c 较小时偏差明显，而这恰好是最常见的情形。
    """
    if b < 0 or c < 0:
        raise ValueError("b / c 必须是非负整数")
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2**n
    return min(1.0, 2.0 * tail)


@dataclass(frozen=True, slots=True)
class Interval:
    point: float
    low: float
    high: float
    confidence: float = 0.95

    def __str__(self) -> str:
        pct = int(self.confidence * 100)
        return f"{self.point:.3f} [{self.low:.3f}, {self.high:.3f}] ({pct}%)"

    @property
    def width(self) -> float:
        return self.high - self.low


def wilson_interval(
    successes: int, n: int, *, confidence: float = 0.95
) -> Interval:
    """Wilson 得分区间。比 Wald 区间稳：k=0 或 k=n 时不会退化成零宽。"""
    if n <= 0:
        raise ValueError("n 必须为正")
    if not 0 <= successes <= n:
        raise ValueError("successes 必须在 [0, n]")
    z = _NORMAL.inv_cdf(1 - (1 - confidence) / 2)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return Interval(
        point=p,
        low=max(0.0, centre - half),
        high=min(1.0, centre + half),
        confidence=confidence,
    )


def paired_bootstrap(
    differences: list[float],
    *,
    n_boot: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> Interval:
    """配对差异的 bootstrap 区间。

    `differences` 是逐题的差值（例如 0/1 正确率差，或逐题得分差）。
    用重采样而不是正态近似：评测指标常常有界且分布偏斜，
    正态近似给出的区间可能超出 [0,1] 这种不可能的范围。
    """
    if not differences:
        raise ValueError("differences 不能为空")
    rng = random.Random(seed)
    n = len(differences)
    point = mean(differences)
    boots = sorted(
        mean(rng.choices(differences, k=n)) for _ in range(n_boot)
    )
    alpha = (1 - confidence) / 2
    low = boots[int(alpha * n_boot)]
    high = boots[min(n_boot - 1, int((1 - alpha) * n_boot))]
    return Interval(point=point, low=low, high=high, confidence=confidence)


# --------------------------------------------------------------------------
# 样本量规划
# --------------------------------------------------------------------------
def required_sample_size(
    *,
    baseline: float,
    target: float,
    discordance: float,
    alpha: float = 0.05,
    power: float = 0.8,
) -> int:
    """配对设计（McNemar）所需题目数，Connor (1987) 公式。

        n = [z_{α/2}·√π + z_β·√(π − δ²)]² / δ²

    参数
    ----
    baseline / target : 两版的准确率，δ = |target − baseline|
    discordance       : 不一致率 P(一版对、另一版错)，实测取值 0.18–0.30
    alpha / power     : 常规 0.05 / 0.8

    为什么必须传 discordance：题目越"一致"（两版都对或都错），
    同样的 δ 就需要越多题才能被检出。只看 δ 算不出样本量——
    这也是"我跑 20 题准确率涨了 15 个点"这种说法不可信的根本原因。
    """
    delta = abs(target - baseline)
    if delta <= 0:
        raise ValueError("target 与 baseline 不能相同")
    if not 0 < discordance <= 1:
        raise ValueError("discordance 必须在 (0, 1]")
    if delta**2 >= discordance:
        raise ValueError(
            f"δ²={delta**2:.4f} 不小于不一致率 {discordance}，"
            "这组假设不自洽（差异比不一致对还多），请检查输入"
        )
    z_alpha = _NORMAL.inv_cdf(1 - alpha / 2)
    z_beta = _NORMAL.inv_cdf(power)
    term = z_alpha * math.sqrt(discordance) + z_beta * math.sqrt(
        discordance - delta**2
    )
    return math.ceil(term**2 / delta**2)


def detectable_delta(
    n: int,
    *,
    discordance: float,
    alpha: float = 0.05,
    power: float = 0.8,
) -> float:
    """反过来问：给定题目数，最小能检出多大的准确率差异？

    这是跑评测**之前**该问的问题。60 题 + 25% 不一致率下，
    检出下限大约 20 个点——也就是说任何小于 20 点的差异，
    在这个规模上都不可能被证明。
    """
    if n <= 0:
        raise ValueError("n 必须为正")
    if not 0 < discordance <= 1:
        raise ValueError("discordance 必须在 (0, 1]")
    z_alpha = _NORMAL.inv_cdf(1 - alpha / 2)
    z_beta = _NORMAL.inv_cdf(power)
    # 解 n = [zα√π + zβ√(π−δ²)]²/δ² 得 δ：把两边乘开化简成一元二次
    # n δ² = π(zα+zβ)² − 2 zβ δ √(π−δ²)·... 直接二分更稳
    low, high = 1e-6, math.sqrt(discordance) - 1e-9
    for _ in range(200):
        mid = (low + high) / 2
        term = z_alpha * math.sqrt(discordance) + z_beta * math.sqrt(
            discordance - mid**2
        )
        needed = term**2 / mid**2
        if needed > n:
            low = mid
        else:
            high = mid
    return (low + high) / 2


# --------------------------------------------------------------------------
# 方差分解
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class VarianceParts:
    n_tasks: int
    samples_per_task: int
    mean_accuracy: float
    observed_variance: float
    sampling_variance: float
    task_variance: float

    @property
    def task_share(self) -> float:
        """任务难度占总方差的比例（类似 ICC）。"""
        if self.observed_variance <= 0:
            return 0.0
        return self.task_variance / self.observed_variance

    def __str__(self) -> str:
        return (
            f"{self.n_tasks} 题 × {self.samples_per_task} 次采样，"
            f"总方差 {self.observed_variance:.4f} = "
            f"任务难度 {self.task_variance:.4f} + 采样噪声 {self.sampling_variance:.4f}"
            f"（任务占 {self.task_share:.0%}）"
        )


def variance_components(repeated: dict[str, list[bool]]) -> VarianceParts:
    """把多次采样的方差拆成"任务难度"与"采样噪声"。

    输入形如 {"q01": [True, True, False], "q02": [False, False, False], ...}

    判读方式：
    - 任务占大头（>70%）→ 题目之间差别大，**加题**才能提高分辨率
    - 采样噪声占大头 → 同一题跑多次结果都不稳，**加采样次数**更划算，
      同时说明这个配置本身不稳定，指标会漂
    """
    if not repeated:
        raise ValueError("repeated 不能为空")
    lengths = {len(v) for v in repeated.values()}
    if len(lengths) != 1:
        raise ValueError(f"每题的采样次数必须相同，收到 {sorted(lengths)}")
    k = lengths.pop()
    if k < 1:
        raise ValueError("每题至少需要 1 次采样")

    per_task = [mean([1.0 if flag else 0.0 for flag in v]) for v in repeated.values()]
    observed = variance(per_task) if len(per_task) > 1 else 0.0
    # 二项采样方差：每题的 p(1-p)/k，对任务取平均
    sampling = mean(p * (1 - p) / k for p in per_task)
    task_var = max(observed - sampling, 0.0)
    return VarianceParts(
        n_tasks=len(per_task),
        samples_per_task=k,
        mean_accuracy=mean(per_task),
        observed_variance=observed,
        sampling_variance=sampling,
        task_variance=task_var,
    )


# --------------------------------------------------------------------------
# 成本-质量
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Point:
    name: str
    quality: float  # 越高越好（准确率）
    cost: float     # 越低越好（人民币元）

    def __str__(self) -> str:
        return f"{self.name}: quality={self.quality:.3f} cost=¥{self.cost:.4f}"


def pareto_frontier(points: list[Point]) -> list[Point]:
    """返回不被支配的点（质量更高且成本更低者支配）。

    用途：一个准确率高 5 点但贵 10 倍的配置，在这张表上直接出局。
    """
    if not points:
        return []
    frontier: list[Point] = []
    for candidate in points:
        dominated = False
        for other in points:
            if other is candidate:
                continue
            better_quality = other.quality >= candidate.quality
            lower_cost = other.cost <= candidate.cost
            strictly_better = other.quality > candidate.quality or other.cost < candidate.cost
            if better_quality and lower_cost and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier.append(candidate)
    return sorted(frontier, key=lambda p: p.cost)


# --------------------------------------------------------------------------
# 多重比较
# --------------------------------------------------------------------------
def holm_correction(p_values: list[float]) -> list[float]:
    """Holm–Bonferroni 校正。

    评测里会同时比较多个配置 × 多个指标，不校正就是在刷彩票。
    比 BH 更保守，适合"必须不出假阳性"的回归门禁。
    """
    if not p_values:
        return []
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, index in enumerate(order):
        value = min(1.0, p_values[index] * (m - rank))
        running = max(running, value)
        adjusted[index] = running
    return adjusted


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    """BH 校正（控制 FDR）。用于探索性分析，比 Holm 有更多检出。"""
    if not p_values:
        return []
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    running = 1.0
    for rank in range(m - 1, -1, -1):
        index = order[rank]
        value = min(1.0, p_values[index] * m / (rank + 1))
        running = min(running, value)
        adjusted[index] = running
    return adjusted
