"""门禁测试：误杀比漏放更烦人，所以三条规则都要验。"""

from __future__ import annotations

import pytest

from agenteval.gate import check_gate, main
from agenteval.runner import Record


def make_records(flags: list[bool], *, cost: float = 0.01, prefix: str = "t") -> list[Record]:
    return [
        Record(
            task_id=f"{prefix}{index}",
            sample=0,
            difficulty="easy",
            answer="a",
            correct=flag,
            cost=cost,
            latency_ms=100.0,
        )
        for index, flag in enumerate(flags)
    ]


def test_same_performance_passes():
    baseline = make_records([True] * 15 + [False] * 5)
    candidate = make_records([True] * 15 + [False] * 5)
    result = check_gate(baseline, candidate)
    assert result.passed
    assert result.accuracy_delta == pytest.approx(0.0)


def test_improvement_passes():
    baseline = make_records([True] * 10 + [False] * 10)
    candidate = make_records([True] * 18 + [False] * 2)
    assert check_gate(baseline, candidate).passed


def test_drop_within_threshold_passes():
    baseline = make_records([True] * 13 + [False] * 7)   # 0.65
    candidate = make_records([True] * 12 + [False] * 8)  # 0.60，跌 5 点 > 3 点
    assert not check_gate(baseline, candidate).passed
    # 放宽阈值就通过
    assert check_gate(baseline, candidate, max_accuracy_drop=0.06).passed


def test_large_drop_fails_with_reason():
    baseline = make_records([True] * 18 + [False] * 2)
    candidate = make_records([True] * 10 + [False] * 10)
    result = check_gate(baseline, candidate)
    assert not result.passed
    assert any("准确率下降" in reason for reason in result.reasons)


def test_cost_increase_fails():
    baseline = make_records([True] * 20, cost=0.01)
    candidate = make_records([True] * 20, cost=0.02)  # 翻倍
    result = check_gate(baseline, candidate)
    assert not result.passed
    assert any("成本" in reason for reason in result.reasons)


def test_cost_increase_within_threshold_passes():
    baseline = make_records([True] * 20, cost=0.01)
    candidate = make_records([True] * 20, cost=0.012)  # +20%，阈值 50%
    assert check_gate(baseline, candidate).passed


def test_free_baseline_is_note_not_failure():
    baseline = make_records([True] * 20, cost=0.0)
    candidate = make_records([True] * 20, cost=0.01)
    result = check_gate(baseline, candidate)
    assert result.passed
    assert any("免费" in note for note in result.notes)


def test_no_shared_tasks_fails():
    baseline = make_records([True] * 5, prefix="a")
    candidate = make_records([True] * 5, prefix="b")
    result = check_gate(baseline, candidate)
    assert not result.passed
    assert any("没有共同题目" in reason for reason in result.reasons)


def test_small_sample_only_warns():
    baseline = make_records([True] * 3)
    candidate = make_records([True] * 3)
    result = check_gate(baseline, candidate, min_shared_tasks=10)
    assert result.passed
    assert any("统计意义很弱" in warning for warning in result.warnings)


def test_require_significance_flags_insignificant_gain():
    baseline = make_records([True] * 10 + [False] * 10)
    candidate = make_records([True] * 11 + [False] * 9)
    result = check_gate(baseline, candidate, require_significance=True)
    assert result.passed
    assert any("不显著" in warning for warning in result.warnings)


def test_render_contains_verdict():
    result = check_gate(make_records([True] * 12), make_records([True] * 12))
    text = result.render()
    assert "通过" in text
    assert "共享题目" in text
