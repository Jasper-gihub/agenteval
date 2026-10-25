"""执行器：并发跑任务集，边跑边落盘，支持断点续跑与多次采样。

为什么 runner 值得单独做（而不是在 eval 脚本里写个 for 循环）：

- **断点续跑**：60 题跑 27 分钟，跑到第 50 题网断了不能从头再来。
  每条记录完成即写 jsonl，重启时按 (task_id, sample) 跳过已完成的。
- **多次采样**：方差分解需要同一题跑多次。这是区分"模型随机性"和
  "题目本身难"的唯一办法，也是 AgentEval 与普通 eval 脚本的分界。
- **成本记账**：成本是主指标，不是附属信息。一条准确率高 5 点但贵 10 倍的配置，
  在帕累托前沿上应该直接出局。
- **agent 只接一个 callable**：不绑定 SQL、不绑定 HTTP、不绑定框架。
  同步函数和协程都接受。
"""

from __future__ import annotations

import asyncio
import inspect
import json
import time
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from agenteval.dataset import Dataset, Task


@dataclass(slots=True)
class AgentOutput:
    answer: str
    cost: float = 0.0
    latency_ms: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Record:
    task_id: str
    sample: int
    difficulty: str
    answer: str
    correct: bool
    cost: float
    latency_ms: float
    error: str | None = None


AgentFn = Callable[[Task], AgentOutput | Awaitable[AgentOutput]]
JudgeFn = Callable[[Task, AgentOutput], bool]


async def run_dataset(
    dataset: Dataset,
    agent: AgentFn,
    judge: JudgeFn,
    *,
    repeat: int = 1,
    concurrency: int = 4,
    output: str | Path | None = None,
    resume: bool = True,
    on_record: Callable[[Record], None] | None = None,
) -> list[Record]:
    """跑完整个任务集，返回全部记录（含续跑前已完成的）。"""
    if repeat < 1:
        raise ValueError("repeat 至少为 1")
    if concurrency < 1:
        raise ValueError("concurrency 至少为 1")

    output_path = Path(output) if output else None
    records: list[Record] = []
    done: set[tuple[str, int]] = set()

    if output_path and resume and output_path.exists():
        for line in output_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            records.append(Record(**payload))
            done.add((payload["task_id"], payload["sample"]))

    pending: list[tuple[Task, int]] = [
        (task, sample)
        for task in dataset
        for sample in range(repeat)
        if (task.id, sample) not in done
    ]

    if not pending:
        return records

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        handle = output_path.open("a", encoding="utf-8")
    else:
        handle = None

    semaphore = asyncio.Semaphore(concurrency)
    write_lock = asyncio.Lock()

    async def one(task: Task, sample: int) -> Record:
        async with semaphore:
            started = time.perf_counter()
            error: str | None = None
            output = AgentOutput(answer="")
            # 传副本：同一题会被并发跑多次，直接改原对象会串味
            task_view = replace(task, sample=sample)
            try:
                result = agent(task_view)
                output = await result if inspect.isawaitable(result) else result
            except Exception as exc:  # 单个任务失败不应炸掉整轮评测
                error = f"{type(exc).__name__}: {exc}"
            elapsed = (time.perf_counter() - started) * 1000
            latency = output.latency_ms or elapsed

            correct = False
            if error is None:
                try:
                    correct = bool(judge(task, output))
                except Exception as exc:
                    error = f"judge 失败 {type(exc).__name__}: {exc}"

            record = Record(
                task_id=task.id,
                sample=sample,
                difficulty=task.difficulty,
                answer=output.answer,
                correct=correct,
                cost=output.cost,
                latency_ms=latency,
                error=error,
            )
            if handle is not None:
                async with write_lock:
                    handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
                    handle.flush()
            if on_record is not None:
                on_record(record)
            return record

    try:
        finished = await asyncio.gather(
            *(one(task, sample) for task, sample in pending)
        )
    finally:
        if handle is not None:
            handle.close()

    records.extend(finished)
    return records


# --------------------------------------------------------------------------
# 汇总
# --------------------------------------------------------------------------
@dataclass(slots=True)
class Summary:
    n_records: int
    n_tasks: int
    repeat: int
    accuracy: float
    by_difficulty: dict[str, float]
    total_cost: float
    cost_per_task: float
    total_latency_s: float
    errors: int

    def __str__(self) -> str:
        parts = [f"{k} {v:.3f}" for k, v in self.by_difficulty.items()]
        return (
            f"{self.n_tasks} 题 × {self.repeat} 次 = {self.n_records} 条记录，"
            f"准确率 {self.accuracy:.3f}（{' / '.join(parts)}），"
            f"成本 ¥{self.total_cost:.4f}（每题 ¥{self.cost_per_task:.4f}），"
            f"失败 {self.errors} 条"
        )


def summarize(records: list[Record]) -> Summary:
    if not records:
        raise ValueError("没有记录可汇总")
    task_ids = {record.task_id for record in records}
    accuracy = sum(r.correct for r in records) / len(records)

    grouped: dict[str, list[bool]] = {}
    for record in records:
        grouped.setdefault(record.difficulty, []).append(record.correct)
    by_difficulty = {
        key: sum(values) / len(values) for key, values in sorted(grouped.items())
    }

    total_cost = sum(r.cost for r in records)
    return Summary(
        n_records=len(records),
        n_tasks=len(task_ids),
        repeat=len(records) // max(len(task_ids), 1),
        accuracy=accuracy,
        by_difficulty=by_difficulty,
        total_cost=total_cost,
        cost_per_task=total_cost / len(task_ids),
        total_latency_s=sum(r.latency_ms for r in records) / 1000,
        errors=sum(1 for r in records if r.error),
    )


def correctness_by_task(records: list[Record]) -> dict[str, list[bool]]:
    """按任务聚合多次采样结果——直接喂给 stats.variance_components。"""
    grouped: dict[str, list[bool]] = {}
    for record in sorted(records, key=lambda r: r.sample):
        grouped.setdefault(record.task_id, []).append(record.correct)
    return grouped


def run_sync(**kwargs) -> list[Record]:
    """同步入口，方便在脚本里直接调用。"""
    return asyncio.run(run_dataset(**kwargs))
