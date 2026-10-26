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

import argparse
import random
from pathlib import Path

from agenteval import Dataset, AgentOutput, run_sync

HERE = Path(__file__).resolve().parent
# 仓库自带 12 题（CI 与演示用，不依赖外部数据集）
TINY_DATASET = HERE / "tiny_tasks.yaml"
# 本机可指向 Litmus 的真实 60 题
REAL_DATASET = Path(r"D:\Projects\agentlab\dataagent\evals\questions.yaml")
DEFAULT_OUT_DIR = Path(__file__).resolve().parents[1] / "reports"

# 两个"配置"：v2 在每一档都明确更好。
# 差距刻意拉大（约 +30 个点）——12 题的样本量下，小差距会被随机性淹没，
# 那样 CI 会随机变红。真实场景的改进幅度看 replay_dataagent.py 的 60 题结果。
VERSIONS = {
    "v1": {"easy": 0.55, "medium": 0.35, "hard": 0.15, "ambiguous": 0.10},
    "v2": {"easy": 0.95, "medium": 0.75, "hard": 0.55, "ambiguous": 0.35},
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="端到端演示：跑两组配置 → 生成 records.jsonl → 供 report / gate 使用"
    )
    parser.add_argument(
        "--dataset",
        help=f"任务集路径；默认 {TINY_DATASET.name}（12 题），本机可指向 "
        f"Litmus 的 60 题（{REAL_DATASET}）",
    )
    parser.add_argument("--out-dir", help="records 输出目录，默认 reports/")
    args = parser.parse_args(argv)

    dataset_path = Path(args.dataset) if args.dataset else TINY_DATASET
    if not dataset_path.exists():
        print(f"找不到任务集：{dataset_path}")
        return 1
    out_dir = Path(args.out_dir) if args.out_dir else DEFAULT_OUT_DIR

    dataset = Dataset.from_yaml(dataset_path)
    print(f"任务集：{dataset.describe()}（{dataset_path.name}）")

    out_dir.mkdir(parents=True, exist_ok=True)
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
            output=out_dir / f"{name}.jsonl",
        )
        hits = sum(record.correct for record in records)
        print(f"  {name}: {hits}/{len(records)} → {out_dir / f'{name}.jsonl'}")

    print("\n下一步：")
    print(f"  python -m agenteval report --records {out_dir / 'v2.jsonl'} "
          f"--baseline {out_dir / 'v1.jsonl'}")
    print(f"  python -m agenteval gate --baseline {out_dir / 'v1.jsonl'} "
          f"--candidate {out_dir / 'v2.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
