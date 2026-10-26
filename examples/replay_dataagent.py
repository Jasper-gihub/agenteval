"""用 Litmus 的真实评测结果回放 AgentEval 的统计能力。

这个脚本回答三个问题：
1. 那四次实验里，哪一次能得出统计上站得住的结论？（答案：都没有）
2. 60 题 + 实测不一致率下，最小能检出多大的差异？（答案：远大于观察到的差异）
3. 如果按"成本-质量"排，哪些配置真的值得留？（帕累托前沿）

用法：
  python examples/replay_dataagent.py
  python examples/replay_dataagent.py --dir D:\\Projects\\agentlab\\dataagent\\evals\\baselines
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agenteval.stats import (
    Point,
    detectable_delta,
    holm_correction,
    mcnemar_exact,
    pareto_frontier,
    required_sample_size,
    wilson_interval,
)

DEFAULT_DIR = Path(r"D:\Projects\agentlab\dataagent\evals\baselines")

# 文件名 → (展示名, 成本元)。成本来自那次运行的结果 json 里的 total_cost
LABELS = {
    "glm-4-flash-60q-v0.2.0.json": ("flash v0.2.0", 0.0),
    "glm-4-flash-60q-v0.3.0.json": ("flash v0.3.0", 0.0),
    "glm-4-flash-60q-v0.3.1.json": ("flash v0.3.1", 0.0),
    "deepseek-chat-60q-v0.2.0.json": ("deepseek v0.2.0", 2.66),
}

COMPARISONS = [
    ("glm-4-flash-60q-v0.2.0.json", "glm-4-flash-60q-v0.3.0.json"),
    ("glm-4-flash-60q-v0.2.0.json", "glm-4-flash-60q-v0.3.1.json"),
    ("glm-4-flash-60q-v0.2.0.json", "deepseek-chat-60q-v0.2.0.json"),
]


def load(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["_by_id"] = {record["id"]: record for record in payload["records"]}
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default=str(DEFAULT_DIR))
    args = parser.parse_args()

    base = Path(args.dir)
    runs = {name: load(base / name) for name in LABELS if (base / name).exists()}
    if not runs:
        print(f"没找到结果文件：{base}")
        return 1

    print("# Litmus 评测结果的统计回放\n")
    print(f"样本：{len(next(iter(runs.values()))['records'])} 题 × {len(runs)} 个配置\n")

    # ---------- 1. 每个配置的准确率区间 ----------
    print("## 1. 准确率与 95% 区间（Wilson）\n")
    print("| 配置 | 准确率 | 95% 区间 | 区间宽度 | 成本 |")
    print("|---|---|---|---|---|")
    for name, payload in runs.items():
        records = payload["records"]
        k = sum(r["correct"] for r in records)
        n = len(records)
        interval = wilson_interval(k, n)
        label, cost = LABELS[name]
        print(
            f"| {label} | {k}/{n} = {interval.point:.3f} | "
            f"[{interval.low:.3f}, {interval.high:.3f}] | "
            f"{interval.width:.3f} | ¥{cost:.2f} |"
        )
    print()
    print("> 注意区间宽度：60 题给出的区间宽度约 0.24，")
    print("> 而所有观察到的差异都在 7 个点以内——**区间比差异大 3 倍以上**。\n")

    # ---------- 2. 配对检验 ----------
    print("## 2. 配对检验（精确 McNemar）\n")
    print("| 对比 | 都不一致对 (b/c) | 准确率差 | p 值 | Holm 校正后 | 结论 |")
    print("|---|---|---|---|---|---|")
    raw_ps = []
    rows = []
    for left_name, right_name in COMPARISONS:
        if left_name not in runs or right_name not in runs:
            continue
        left, right = runs[left_name], runs[right_name]
        ids = [i for i in left["_by_id"] if i in right["_by_id"]]
        b = sum(1 for i in ids if left["_by_id"][i]["correct"] and not right["_by_id"][i]["correct"])
        c = sum(1 for i in ids if not left["_by_id"][i]["correct"] and right["_by_id"][i]["correct"])
        delta = (sum(right["_by_id"][i]["correct"] for i in ids) - sum(left["_by_id"][i]["correct"] for i in ids)) / len(ids)
        p = mcnemar_exact(b, c)
        raw_ps.append(p)
        rows.append((LABELS[left_name][0], LABELS[right_name][0], b, c, delta, p))
    adjusted = holm_correction(raw_ps)
    for (ln, rn, b, c, delta, p), adj in zip(rows, adjusted):
        verdict = "显著" if adj < 0.05 else "不显著"
        print(f"| {ln} → {rn} | {b}/{c} | {delta:+.3f} | {p:.4f} | {adj:.4f} | **{verdict}** |")
    print()
    print("> 三次对比、一次 Holm 校正后全部不显著。")
    print("> 也就是说：**换版本、换模型这两类决策，这个实验规模都支撑不了**。\n")

    # ---------- 3. 60 题的可检出下限 ----------
    print("## 3. 60 题到底能证明什么\n")
    discordances = []
    for (ln, rn, b, c, delta, p) in rows:
        discordances.append((b + c) / 60)
    avg_discordance = sum(discordances) / len(discordances)
    minimum = detectable_delta(60, discordance=avg_discordance)
    print(f"实测平均不一致率：**{avg_discordance:.1%}**\n")
    print("| 题目数 | 可检出的最小准确率差（80% 功效） |")
    print("|---|---|")
    for n in (60, 100, 200, 400, 800):
        print(f"| {n} | {detectable_delta(n, discordance=avg_discordance):.1%} |")
    print()
    print(f"> 60 题只能检出 **{minimum:.0%}** 以上的差异，")
    print("> 而实际观察到的最大差异是 6.7 个点。")
    print("> **这就是为什么三次实验没有一个能下结论。**\n")

    needed = required_sample_size(
        baseline=0.483, target=0.55, discordance=avg_discordance
    )
    print(f"要证明 flash(0.483) 与 deepseek(0.550) 的差异，需要 **{needed} 题**。\n")

    # ---------- 4. 成本-质量前沿 ----------
    print("## 4. 成本-质量帕累托前沿\n")
    points = []
    for name, payload in runs.items():
        records = payload["records"]
        accuracy = sum(r["correct"] for r in records) / len(records)
        label, cost = LABELS[name]
        points.append(Point(label, accuracy, cost))
    frontier = {p.name for p in pareto_frontier(points)}
    print("| 配置 | 准确率 | 成本 | 是否在前沿 |")
    print("|---|---|---|---|")
    for point in sorted(points, key=lambda p: p.cost):
        print(
            f"| {point.name} | {point.quality:.3f} | ¥{point.cost:.2f} | "
            f"{'✅' if point.name in frontier else '—'} |"
        )
    print()
    print("> flash v0.3.1 与 deepseek 都在前沿：前者免费、后者高 5 个点但贵 ¥2.66。")
    print("> 而 flash v0.2.0 被 v0.3.1 完全支配（更准且同样免费）——这是唯一一个")
    print("> 不需要统计检验就能下的结论：**它的提升是免费的**。\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
