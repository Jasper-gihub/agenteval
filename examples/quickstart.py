"""最小可运行示例：不花钱、不联网，60 题跑 3 次采样。

它演示三件普通 eval 脚本做不到的事：
1. 直接读 DataAgent 的任务集（`question` / `ref_sql` 写法），不用改文件
2. 同一题多次采样 → 才能把方差拆成"题目难度"和"采样噪声"
3. 准确率一律给区间，不给点值

这里的 agent 是假的：按难度用一个确定性伪随机决定答对还是答错，
正确率取的是 DataAgent 实测的四个数值，所以输出可以和真实结果对照着看。

用法：
  python examples/quickstart.py
"""

from __future__ import annotations

import random
from pathlib import Path

from agenteval import (
    AgentOutput,
    Dataset,
    correctness_by_task,
    run_sync,
    summarize,
    variance_components,
    wilson_interval,
)

QUESTIONS = Path(
    r"D:\Projects\agentlab\dataagent\evals\questions.yaml"
)

# DataAgent 实测的逐难度准确率（flash v0.2.0），用来模拟一个"中等能力"agent
ACCURACY = {"easy": 0.93, "medium": 0.47, "hard": 0.33, "ambiguous": 0.20}


def main() -> int:
    if not QUESTIONS.exists():
        print(f"找不到任务集：{QUESTIONS}")
        return 1

    dataset = Dataset.from_yaml(QUESTIONS)
    print(f"任务集：{dataset.describe()}\n")

    def fake_agent(task) -> AgentOutput:
        # 用 (题号, 采样序号) 做种子：同一题不同采样会得到不同结果，
        # 这正是模拟"模型本身的随机性"——方差分解必须有这个才能算出采样噪声。
        rng = random.Random(f"{task.id}#{task.sample}")
        hit = rng.random() < ACCURACY[task.difficulty]
        return AgentOutput(
            answer="RIGHT" if hit else "WRONG",
            cost=0.0,
            latency_ms=120.0,
        )

    def judge(task, output: AgentOutput) -> bool:
        return output.answer == "RIGHT"

    records = run_sync(
        dataset=dataset, agent=fake_agent, judge=judge, repeat=3, concurrency=8
    )

    summary = summarize(records)
    print("== 汇总 ==")
    print(summary)
    print()

    n = summary.n_tasks
    k = round(summary.accuracy * summary.n_records)
    interval = wilson_interval(k, summary.n_records)
    print("== 准确率要给区间 ==")
    print(f"点估计 {interval}  区间宽度 {interval.width:.3f}")
    print()

    print("== 方差分解（同一题 3 次采样）==")
    parts = variance_components(correctness_by_task(records))
    print(parts)
    print(
        "\n判读：任务难度占比高 → 题目之间差别大，加题才提高分辨率；\n"
        "      采样噪声占比高 → 同一题跑多次都不稳，这个配置的指标本身会漂。"
    )
    print()

    print("== 分难度（总分掩盖方向相反的效应）==")
    for difficulty, accuracy in summary.by_difficulty.items():
        print(f"  {difficulty:<10} {accuracy:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
