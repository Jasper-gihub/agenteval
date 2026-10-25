"""统计层的测试。这层的错误会让所有评测结论失效，所以每条公式都要对住手算。"""

from __future__ import annotations

import math

import pytest

from agenteval.stats import (
    Point,
    benjamini_hochberg,
    detectable_delta,
    holm_correction,
    mcnemar_exact,
    paired_bootstrap,
    pareto_frontier,
    required_sample_size,
    variance_components,
    wilson_interval,
)


# ---------------- McNemar ----------------
def test_mcnemar_no_discordance_is_one():
    assert mcnemar_exact(0, 0) == 1.0


def test_mcnemar_symmetric():
    assert mcnemar_exact(3, 11) == mcnemar_exact(11, 3)


def test_mcnemar_known_value():
    # b=0, c=10：p = 2 * (1/2)^10 = 0.001953125
    assert mcnemar_exact(0, 10) == pytest.approx(2 / 2**10, rel=1e-9)


def test_mcnemar_balanced_is_not_significant():
    assert mcnemar_exact(9, 9) == pytest.approx(1.0, abs=1e-9)


def test_mcnemar_datagent_case():
    """DataAgent 实验 3 的真实数字：flash 独对 7、deepseek 独对 11 → p≈0.48。"""
    assert mcnemar_exact(7, 11) == pytest.approx(0.4807, abs=5e-4)


# ---------------- Wilson 区间 ----------------
def test_wilson_matches_known_value():
    # 29/60 的 Wilson 95% 区间约 [0.363, 0.605]
    interval = wilson_interval(29, 60)
    assert interval.point == pytest.approx(29 / 60)
    assert interval.low == pytest.approx(0.363, abs=5e-3)
    assert interval.high == pytest.approx(0.605, abs=5e-3)


def test_wilson_degenerate_cases_stay_bounded():
    for k in (0, 10):
        interval = wilson_interval(k, 10)
        assert 0.0 <= interval.low <= interval.high <= 1.0
    assert wilson_interval(0, 10).high > 0.0
    assert wilson_interval(10, 10).low < 1.0


def test_wilson_narrows_with_more_samples():
    assert wilson_interval(30, 100).width < wilson_interval(3, 10).width


# ---------------- 样本量 ----------------
def test_required_sample_size_matches_manual_calc():
    """Connor 公式手算：δ=0.1, π=0.25, α=0.05, power=0.8 → n≈194。"""
    n = required_sample_size(baseline=0.48, target=0.58, discordance=0.25)
    assert n == pytest.approx(194, abs=3)


def test_required_sample_size_grows_with_smaller_delta():
    big = required_sample_size(baseline=0.4, target=0.6, discordance=0.25)
    small = required_sample_size(baseline=0.4, target=0.45, discordance=0.25)
    assert small > big


def test_required_sample_size_rejects_inconsistent_assumption():
    with pytest.raises(ValueError):
        required_sample_size(baseline=0.0, target=0.9, discordance=0.2)


def test_detectable_delta_is_inverse_of_required_sample_size():
    """detectable_delta 与 required_sample_size 必须互为反函数。"""
    n = required_sample_size(baseline=0.48, target=0.58, discordance=0.25)
    delta = detectable_delta(n, discordance=0.25)
    assert delta == pytest.approx(0.10, abs=5e-3)


def test_detectable_delta_shows_60_questions_are_useless_for_small_gaps():
    """60 题 + 25% 不一致率，最小可检出差异远大于实际看到的 5-7 个点。"""
    assert detectable_delta(60, discordance=0.25) > 0.15


# ---------------- Bootstrap ----------------
def test_paired_bootstrap_recovers_mean():
    diffs = [1.0] * 12 + [0.0] * 48
    interval = paired_bootstrap(diffs, n_boot=2000)
    assert interval.point == pytest.approx(0.2)
    assert interval.low < interval.point < interval.high


def test_paired_bootstrap_all_same_is_degenerate():
    interval = paired_bootstrap([0.0] * 30, n_boot=500)
    assert interval.low == pytest.approx(0.0)
    assert interval.high == pytest.approx(0.0)


# ---------------- 方差分解 ----------------
def test_variance_components_all_stable_tasks():
    """每题采样结果完全一致 → 采样噪声为 0，方差全归任务难度。"""
    repeated = {"a": [True] * 3, "b": [False] * 3, "c": [True] * 3}
    parts = variance_components(repeated)
    assert parts.sampling_variance == pytest.approx(0.0)
    assert parts.task_share == pytest.approx(1.0)


def test_variance_components_noise_dominated():
    """每题方差很大且任务成功率接近 → 采样噪声占主导。"""
    repeated = {
        "a": [True, False, True, False, True, False],
        "b": [False, True, False, True, False, True],
        "c": [True, False, False, True, False, True],
        "d": [False, True, True, False, True, False],
    }
    parts = variance_components(repeated)
    assert parts.sampling_variance > 0
    assert parts.sampling_variance > parts.task_variance


def test_variance_components_requires_equal_samples():
    with pytest.raises(ValueError):
        variance_components({"a": [True], "b": [True, False]})


# ---------------- 帕累托 ----------------
def test_pareto_frontier_drops_dominated():
    points = [
        Point("flash", 0.48, 0.0),
        Point("flash-v31", 0.50, 0.0),   # 支配 flash
        Point("deepseek", 0.55, 2.66),   # 质量更高但更贵，保留
        Point("expensive-worse", 0.40, 5.0),  # 被 flash 支配
    ]
    names = [p.name for p in pareto_frontier(points)]
    assert "flash" not in names
    assert "expensive-worse" not in names
    assert names == ["flash-v31", "deepseek"]


# ---------------- 多重比较 ----------------
def test_holm_monotone_and_bounded():
    adjusted = holm_correction([0.01, 0.04, 0.03, 0.5])
    assert all(0.0 <= a <= 1.0 for a in adjusted)
    assert adjusted[0] == pytest.approx(0.04)
    assert adjusted[3] == pytest.approx(0.5)


def test_bh_less_conservative_than_holm():
    p_values = [0.01, 0.02, 0.03, 0.2]
    holm = holm_correction(p_values)
    bh = benjamini_hochberg(p_values)
    assert all(b <= h + 1e-12 for b, h in zip(bh, holm))


def test_empty_inputs():
    assert holm_correction([]) == []
    assert benjamini_hochberg([]) == []
