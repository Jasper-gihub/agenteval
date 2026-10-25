"""端到端演示：跑两组"配置"→ 生成 jsonl → 出报告 → 过门禁。

全程不联网、不花钱。两个假 agent 的差别只在于对同一题的正确率：
v2 比 v1 高 8 个点——足够让大部分人写下"提升 8 个点"，
但下面会用统计告诉你这个结论站不站得住。

用法：
  python examples/pipeline.py
  python -m agenteval report --records reports/v2.jsonl --baseline reports/v1.jsonl --label v2 --baseline-label v1
  python -m agenteval gate --baseline reports/v1.jsonl --candidate reports/v2.jsonl --require-significance
"""

from __future__ import annotations

import random
from pathlib import Path

from agenteval import Dataset, AgentOutput, run_sync

QUESTIONS = Path(r"D:\Projects\agentlab\dataagent\evals\questions.yaml")
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "reports"

# 两个"配置"：按难度给不同命中率，v2 在 hard 上提升最明显
VERSIONS = {
    "v1": {"easy": 0.85, "medium": 0.45, "hard": 0.25, "ambiguous": 0.15},
    "v2": {"easy": 0.88, "medium": 0.50, "hard": 0.40, "ambiguous": 0.18},
}


def main() -> int:
    dataset = Dataset.from_yaml(QUESTIONS)
    print(f"任务集：{dataset.describe()}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, rates in VERSIONS.items():
        def agent(task, *, table=rates, tag=name) -> AgentOutput:
            rng = random.Random(f"{tag}#{task.id}#{task.sample}")
            return AgentOutput(
                answer="RIGHT" if rng.random() < table[task.difficulty] else "WRONG",
                cost=0.0,
                latency_ms=100.0,
            )

        records = run_sync(
            dataset=dataset,
            agent=agent,
            judge=lambda task, output: output.answer == "RIGHT",
            repeat=2,
            concurrency=8,
            output=OUTPUT_DIR / f"{name}.jsonl",
        )
        hits = sum(record.correct for record in records)
        print(f"  {name}: {hits}/{len(records)} → {OUTPUT_DIR / f'{name}.jsonl'}")

    print("\n下一步：")
    print("  python -m agenteval report --records reports/v2.jsonl --baseline reports/v1.jsonl")
    print("  python -m agenteval gate --baseline reports/v1.jsonl --candidate reports/v2.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
