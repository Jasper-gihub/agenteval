"""任务集：用 YAML 描述"要评测什么"，不绑定具体 agent 形态。

设计约束来自踩过的坑：
1. **每题必须有可判定的金标准**（gold_sql 或 gold），否则评测没法自动化。
   没写金标准的题会在加载时直接报错，而不是跑完才发现没法判对错。
2. **难度分层是一等公民**。总分掩盖方向相反的效应——
   Litmus 的 v0.3.0 就是 easy −26 点被 medium +13 点抵消，看总分以为"没变化"。
3. **兼容已有格式**：Litmus 的 60 题（`question` / `ref_sql`）能直接读进来，
   不用改文件就能被新工具评测。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

VALID_DIFFICULTIES = ("easy", "medium", "hard", "ambiguous")

_RESERVED_KEYS = {
    "id",
    "prompt",
    "question",
    "difficulty",
    "gold_sql",
    "ref_sql",
    "gold",
}


@dataclass(slots=True)
class Task:
    id: str
    prompt: str
    difficulty: str = "medium"
    gold_sql: str | None = None
    gold: str | None = None
    meta: dict = field(default_factory=dict)
    # runner 在多次采样时会用 replace(task, sample=i) 生成副本传进来。
    # 没有这个字段，agent 就无法让第 3 次采样和第 1 次不同——
    # 那么"多次采样"实际测的是同一个结果，方差分解会永远得出采样噪声为 0。
    sample: int = 0

    def __post_init__(self) -> None:
        self.id = str(self.id).strip()
        self.prompt = str(self.prompt).strip()
        self.difficulty = str(self.difficulty).strip().lower()


class Dataset:
    """一组任务。加载时就完成校验，不留到运行期。"""

    def __init__(self, tasks: list[Task]) -> None:
        self.tasks = list(tasks)
        self._validate()

    # ---------- 校验 ----------
    def _validate(self) -> None:
        if not self.tasks:
            raise ValueError("任务集为空")

        seen: set[str] = set()
        duplicates: set[str] = set()
        for task in self.tasks:
            if task.id in seen:
                duplicates.add(task.id)
            seen.add(task.id)
        if duplicates:
            raise ValueError(f"id 重复：{', '.join(sorted(duplicates))}")

        problems: list[str] = []
        for task in self.tasks:
            if not task.prompt:
                problems.append(f"{task.id}: prompt 为空")
            if task.difficulty not in VALID_DIFFICULTIES:
                problems.append(
                    f"{task.id}: difficulty='{task.difficulty}' 不在 "
                    f"{'/'.join(VALID_DIFFICULTIES)} 里"
                )
            if task.gold_sql is None and task.gold is None:
                problems.append(f"{task.id}: 既没有 gold_sql 也没有 gold，无法判定对错")
        if problems:
            raise ValueError("任务集校验失败：\n  " + "\n  ".join(problems))

    # ---------- 加载 ----------
    @classmethod
    def from_yaml(cls, path: str | Path) -> "Dataset":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raise ValueError(f"{path} 顶层必须是列表，每条一个任务")
        tasks: list[Task] = []
        for item in raw:
            if not isinstance(item, dict):
                raise ValueError(f"{path} 里有非字典条目：{item!r}")
            tasks.append(
                Task(
                    id=item.get("id", ""),
                    # question / ref_sql 是 Litmus 用过的写法，保留兼容
                    prompt=item.get("prompt") or item.get("question") or "",
                    difficulty=item.get("difficulty", "medium"),
                    gold_sql=item.get("gold_sql") or item.get("ref_sql"),
                    gold=item.get("gold"),
                    meta={
                        key: value
                        for key, value in item.items()
                        if key not in _RESERVED_KEYS
                    },
                )
            )
        return cls(tasks)

    # ---------- 查询 ----------
    def __len__(self) -> int:
        return len(self.tasks)

    def __iter__(self):
        return iter(self.tasks)

    def by_difficulty(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for task in self.tasks:
            counts[task.difficulty] = counts.get(task.difficulty, 0) + 1
        return dict(sorted(counts.items()))

    def describe(self) -> str:
        parts = [f"{name} {count}" for name, count in self.by_difficulty().items()]
        return f"{len(self)} 题（{' / '.join(parts)}）"
