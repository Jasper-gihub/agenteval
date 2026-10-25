"""执行器测试。用假 agent，不碰网络也不花钱。"""

from __future__ import annotations

import asyncio
import json

import pytest

from agenteval.dataset import Dataset, Task
from agenteval.runner import (
    AgentOutput,
    Record,
    correctness_by_task,
    run_dataset,
    run_sync,
    summarize,
)


def make_dataset(n: int = 4) -> Dataset:
    return Dataset(
        [
            Task(
                id=f"t{index}",
                prompt=f"问题 {index}",
                difficulty="easy" if index % 2 == 0 else "hard",
                gold="答案",
            )
            for index in range(n)
        ]
    )


def always_right(task: Task, output: AgentOutput) -> bool:
    return output.answer == "答案"


def test_runs_every_task():
    records = asyncio.run(
        run_dataset(
            make_dataset(4),
            lambda task: AgentOutput("答案", cost=0.01),
            always_right,
            concurrency=2,
        )
    )
    assert len(records) == 4
    assert all(record.correct for record in records)
    assert {record.task_id for record in records} == {"t0", "t1", "t2", "t3"}


def test_repeat_multiplies_records():
    records = asyncio.run(
        run_dataset(
            make_dataset(3),
            lambda task: AgentOutput("答案"),
            always_right,
            repeat=3,
        )
    )
    assert len(records) == 9
    assert {record.sample for record in records} == {0, 1, 2}


def test_async_agent_supported():
    async def agent(task: Task) -> AgentOutput:
        await asyncio.sleep(0)
        return AgentOutput("答案")

    records = asyncio.run(run_dataset(make_dataset(2), agent, always_right))
    assert all(record.correct for record in records)


def test_agent_exception_is_recorded_not_raised():
    def boom(task: Task) -> AgentOutput:
        raise RuntimeError("模型挂了")

    records = asyncio.run(run_dataset(make_dataset(2), boom, always_right))
    assert len(records) == 2
    assert all(record.error and "模型挂了" in record.error for record in records)
    assert not any(record.correct for record in records)


def test_judge_exception_is_recorded():
    def bad_judge(task: Task, output: AgentOutput) -> bool:
        raise ValueError("判对逻辑炸了")

    records = asyncio.run(
        run_dataset(make_dataset(1), lambda t: AgentOutput("答案"), bad_judge)
    )
    assert records[0].error and "判对逻辑炸了" in records[0].error


def test_writes_jsonl_and_resumes(tmp_path):
    output = tmp_path / "records.jsonl"
    calls: list[str] = []

    def agent(task: Task) -> AgentOutput:
        calls.append(task.id)
        return AgentOutput("答案", cost=0.5)

    first = asyncio.run(
        run_dataset(make_dataset(3), agent, always_right, output=output)
    )
    assert len(first) == 3
    assert output.exists()
    lines = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 3

    calls.clear()
    second = asyncio.run(
        run_dataset(make_dataset(3), agent, always_right, output=output)
    )
    assert calls == [], "续跑时不应重新调用已完成的任务"
    assert len(second) == 3


def test_resume_can_be_disabled(tmp_path):
    output = tmp_path / "records.jsonl"
    asyncio.run(
        run_dataset(
            make_dataset(2), lambda t: AgentOutput("答案"), always_right, output=output
        )
    )
    fresh = asyncio.run(
        run_dataset(
            make_dataset(2),
            lambda t: AgentOutput("答案"),
            always_right,
            output=tmp_path / "other.jsonl",
            resume=False,
        )
    )
    assert len(fresh) == 2


def test_on_record_callback_fires():
    seen: list[Record] = []
    asyncio.run(
        run_dataset(
            make_dataset(2),
            lambda t: AgentOutput("答案"),
            always_right,
            on_record=seen.append,
        )
    )
    assert len(seen) == 2


def test_summarize_by_difficulty_and_cost():
    records = [
        Record("t0", 0, "easy", "答案", True, 0.2, 10.0),
        Record("t1", 0, "hard", "错", False, 0.4, 20.0),
        Record("t2", 0, "easy", "答案", True, 0.2, 10.0),
        Record("t3", 0, "hard", "答案", True, 0.2, 10.0),
    ]
    summary = summarize(records)
    assert summary.n_tasks == 4
    assert summary.accuracy == pytest.approx(0.75)
    assert summary.by_difficulty["easy"] == pytest.approx(1.0)
    assert summary.by_difficulty["hard"] == pytest.approx(0.5)
    assert summary.total_cost == pytest.approx(1.0)
    assert summary.cost_per_task == pytest.approx(0.25)
    assert summary.errors == 0


def test_correctness_by_task_feeds_variance_components():
    records = [
        Record("t0", 0, "easy", "", True, 0, 0),
        Record("t0", 1, "easy", "", True, 0, 0),
        Record("t1", 0, "hard", "", False, 0, 0),
        Record("t1", 1, "hard", "", False, 0, 0),
    ]
    grouped = correctness_by_task(records)
    assert grouped == {"t0": [True, True], "t1": [False, False]}


def test_run_sync_wrapper():
    records = run_sync(dataset=make_dataset(2), agent=lambda t: AgentOutput("答案"), judge=always_right)
    assert len(records) == 2


def test_rejects_bad_options():
    with pytest.raises(ValueError):
        asyncio.run(run_dataset(make_dataset(1), lambda t: AgentOutput("答案"), always_right, repeat=0))
    with pytest.raises(ValueError):
        asyncio.run(
            run_dataset(make_dataset(1), lambda t: AgentOutput("答案"), always_right, concurrency=0)
        )
