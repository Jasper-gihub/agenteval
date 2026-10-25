"""报告生成测试：重点是"该出现的数字都在"，而不是排版好不好看。"""

from __future__ import annotations

import json

import pytest

from agenteval.report import build_report, compare, load_records, per_task_accuracy
from agenteval.runner import Record


def make_records(
    correct_flags: list[bool], *, difficulty: str = "easy", prefix: str = "t"
) -> list[Record]:
    return [
        Record(
            task_id=f"{prefix}{index}",
            sample=0,
            difficulty=difficulty,
            answer="a",
            correct=flag,
            cost=0.01,
            latency_ms=100.0,
        )
        for index, flag in enumerate(correct_flags)
    ]


def test_per_task_accuracy_averages_samples():
    records = [
        Record("t0", 0, "easy", "", True, 0, 0),
        Record("t0", 1, "easy", "", False, 0, 0),
        Record("t1", 0, "easy", "", True, 0, 0),
    ]
    assert per_task_accuracy(records) == {"t0": 0.5, "t1": 1.0}


def test_load_records_roundtrip(tmp_path):
    path = tmp_path / "r.jsonl"
    records = make_records([True, False])
    path.write_text(
        "\n".join(json.dumps(r.__dict__, ensure_ascii=False) if hasattr(r, "__dict__") else
                  json.dumps({"task_id": r.task_id, "sample": r.sample, "difficulty": r.difficulty,
                              "answer": r.answer, "correct": r.correct, "cost": r.cost,
                              "latency_ms": r.latency_ms, "error": r.error}, ensure_ascii=False)
                  for r in records),
        encoding="utf-8",
    )
    loaded = load_records(path)
    assert len(loaded) == 2
    assert loaded[0].task_id == "t0"


def test_load_records_rejects_empty(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text("\n", encoding="utf-8")
    with pytest.raises(ValueError, match="没有记录"):
        load_records(path)


def test_report_has_core_sections():
    text = build_report(make_records([True] * 7 + [False] * 3), label="v2")
    assert "# " in text
    assert "95% 区间" in text
    assert "分难度" in text
    assert "这个规模能检出什么" in text
    assert "0.700" in text


def test_report_includes_comparison_and_pareto():
    baseline = make_records([True] * 5 + [False] * 5)
    candidate = make_records([True] * 8 + [False] * 2)
    text = build_report(candidate, label="v2", baseline=baseline, baseline_label="v1")
    assert "配对对比" in text
    assert "McNemar" in text
    assert "成本-质量" in text
    assert "帕累托" in text


def test_report_marks_insignificant_result():
    baseline = make_records([True] * 6 + [False] * 4)
    candidate = make_records([True] * 7 + [False] * 3)
    text = build_report(candidate, baseline=baseline)
    assert "不显著" in text


def test_report_with_repeat_includes_variance():
    records = [
        Record("t0", 0, "easy", "", True, 0, 0),
        Record("t0", 1, "easy", "", False, 0, 0),
        Record("t1", 0, "hard", "", True, 0, 0),
        Record("t1", 1, "hard", "", True, 0, 0),
    ]
    text = build_report(records)
    assert "方差分解" in text
    assert "采样噪声" in text


def test_compare_counts_discordant_pairs():
    baseline = make_records([True, True, False, False])
    candidate = make_records([False, True, True, False])
    comparison = compare(baseline, candidate)
    assert comparison.b == 1  # 只有 baseline 对
    assert comparison.c == 1  # 只有 candidate 对
    assert comparison.delta == pytest.approx(0.0)
    assert comparison.p_value == pytest.approx(1.0)


def test_compare_requires_shared_tasks():
    with pytest.raises(ValueError, match="没有共同题目"):
        compare(make_records([True], prefix="a"), make_records([True], prefix="b"))
