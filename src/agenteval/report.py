"""把记录变成人能读、能贴进 PR 的报告。

它刻意不画图：图片在 PR 里看不见、在终端里打不开、在 CI 日志里是死链。
一份 Markdown 表格 + 明确的口径说明，比一张好看的热力图更有用。

报告里每一条数字都带口径：
- 准确率给区间，不给点值
- 对比给 2×2 配对表和 p 值，不给"提升了 X%"
- 规模不够时明确写出"这个规模只能检出 Y 以上的差异"
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

from agenteval.runner import Record, correctness_by_task, summarize
from agenteval.stats import (
    Interval,
    Point,
    detectable_delta,
    holm_correction,
    mcnemar_exact,
    paired_bootstrap,
    pareto_frontier,
    variance_components,
    wilson_interval,
)


def load_records(path: str | Path) -> list[Record]:
    """读 runner 写出的 jsonl。"""
    records: list[Record] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(Record(**json.loads(line)))
    if not records:
        raise ValueError(f"{path} 里没有记录")
    return records


def per_task_accuracy(records: list[Record]) -> dict[str, float]:
    """每题的平均正确率（repeat=1 时就是 0/1）。"""
    grouped: dict[str, list[bool]] = {}
    for record in records:
        grouped.setdefault(record.task_id, []).append(record.correct)
    return {key: sum(values) / len(values) for key, values in grouped.items()}


def per_task_majority(records: list[Record], threshold: float = 0.5) -> dict[str, bool]:
    """多数采样是否正确。McNemar 需要二分类，多次采样先投票。"""
    return {
        key: value >= threshold for key, value in per_task_accuracy(records).items()
    }


@dataclass(slots=True)
class Comparison:
    b: int  # 只有 A 对
    c: int  # 只有 B 对
    delta: float
    p_value: float

    @property
    def significant(self) -> bool:
        return self.p_value < 0.05


def compare(left: list[Record], right: list[Record]) -> Comparison:
    a = per_task_majority(left)
    b = per_task_majority(right)
    shared = [key for key in a if key in b]
    if not shared:
        raise ValueError("两组记录没有共同题目，无法配对")
    only_a = sum(1 for key in shared if a[key] and not b[key])
    only_b = sum(1 for key in shared if b[key] and not a[key])
    delta = mean([1.0 if b[key] else 0.0 for key in shared]) - mean(
        [1.0 if a[key] else 0.0 for key in shared]
    )
    return Comparison(
        b=only_a, c=only_b, delta=delta, p_value=mcnemar_exact(only_a, only_b)
    )


def _accuracy_interval(records: list[Record]) -> Interval:
    hits = sum(record.correct for record in records)
    return wilson_interval(hits, len(records))


def _rows(records: list[Record]) -> str:
    summary = summarize(records)
    interval = _accuracy_interval(records)
    lines = [
        f"| 记录数 | {summary.n_records}（{summary.n_tasks} 题 × {summary.repeat} 次） |",
        f"| 准确率 | {interval.point:.3f} |",
        f"| 95% 区间 | [{interval.low:.3f}, {interval.high:.3f}]（宽度 {interval.width:.3f}） |",
        f"| 总成本 | ¥{summary.total_cost:.4f} |",
        f"| 每题成本 | ¥{summary.cost_per_task:.4f} |",
        f"| 总耗时 | {summary.total_latency_s:.0f}s |",
        f"| 失败记录 | {summary.errors} |",
    ]
    return "\n".join(lines)


def build_report(
    records: list[Record],
    *,
    label: str = "candidate",
    baseline: list[Record] | None = None,
    baseline_label: str = "baseline",
    title: str = "Agent 评测报告",
) -> str:
    summary = summarize(records)
    interval = _accuracy_interval(records)
    sections: list[str] = [f"# {title}", ""]

    # ---------- 概览 ----------
    sections += ["## 概览", "", "| 指标 | 值 |", "|---|---|", _rows(records), ""]

    # ---------- 分难度 ----------
    sections += ["## 分难度", "", f"| 难度 | {label} | 题数 |", "|---|---|---|"]
    baseline_summary = summarize(baseline) if baseline else None
    if baseline_summary:
        sections[-2] = f"| 难度 | {baseline_label} | {label} | 题数 |"
        sections[-1] = "|---|---|---|---|"
    for difficulty, accuracy in summary.by_difficulty.items():
        count = sum(1 for record in records if record.difficulty == difficulty)
        if baseline_summary:
            other = baseline_summary.by_difficulty.get(difficulty)
            other_text = f"{other:.3f}" if other is not None else "—"
            sections.append(f"| {difficulty} | {other_text} | {accuracy:.3f} | {count} |")
        else:
            sections.append(f"| {difficulty} | {accuracy:.3f} | {count} |")
    sections += [
        "",
        "> 总分掩盖方向相反的效应：一次改动可能让 easy 掉 26 点、medium 涨 13 点，",
        "> 总分看起来「没变化」。所以分难度必须单列。",
        "",
    ]

    # ---------- 方差分解 ----------
    if summary.repeat > 1:
        parts = variance_components(correctness_by_task(records))
        sections += [
            "## 方差分解",
            "",
            f"`{parts}`",
            "",
            f"- 任务难度占 **{parts.task_share:.0%}**，采样噪声占 {1 - parts.task_share:.0%}",
            "- 任务难度占大头 → 加题才提高分辨率",
            "- 采样噪声占大头 → 这个配置的指标本身会漂，加题没用",
            "",
        ]

    # ---------- 配对对比 ----------
    if baseline is not None:
        comparison = compare(baseline, records)
        sections += [
            "## 配对对比（精确 McNemar）",
            "",
            f"| | {baseline_label} 对、{label} 错 | {baseline_label} 错、{label} 对 |",
            "|---|---|---|",
            f"| 题数 | {comparison.b} | {comparison.c} |",
            "",
            f"- 准确率变化 **{comparison.delta:+.3f}**",
            f"- McNemar p = **{comparison.p_value:.4f}**",
        ]
        if comparison.p_value >= 0.05:
            needed = "更多题目"
            sections += [
                f"- 结论：**不显著**。当前规模不足以支持「变了」的判断，需要{needed}。",
            ]
        else:
            better = label if comparison.c > comparison.b else baseline_label
            sections += [f"- 结论：**显著**，{better} 更好。"]
        sections.append("")

        # ---------- 帕累托 ----------
        points = [
            Point(baseline_label, summarize(baseline).accuracy, summarize(baseline).cost_per_task),
            Point(label, summary.accuracy, summary.cost_per_task),
        ]
        frontier = {point.name for point in pareto_frontier(points)}
        sections += [
            "## 成本-质量",
            "",
            "| 配置 | 准确率 | 每题成本 | 在帕累托前沿 |",
            "|---|---|---|---|",
        ]
        for point in sorted(points, key=lambda p: p.cost):
            mark = "✅" if point.name in frontier else "—"
            sections.append(
                f"| {point.name} | {point.quality:.3f} | ¥{point.cost:.4f} | {mark} |"
            )
        sections += [
            "",
            "> 被支配意味着存在另一个「更准且更便宜」的配置——这种改进不需要 p 值也值得采用。",
            "",
        ]

    # ---------- 可检出下限 ----------
    discordance = 0.25
    observed = 0.0
    if baseline is not None:
        comparison = compare(baseline, records)
        shared = len(per_task_majority(records))
        if shared:
            discordance = (comparison.b + comparison.c) / shared
        observed = abs(comparison.delta)
    minimum = detectable_delta(summary.n_tasks, discordance=max(discordance, 0.01))
    sections += [
        "## 这个规模能检出什么",
        "",
        f"- 实测不一致率 {discordance:.1%}",
        f"- {summary.n_tasks} 题的可检出下限是 **{minimum:.1%}**（80% 功效，α=0.05）",
    ]
    if baseline is not None:
        comparison = compare(baseline, records)
        significant = comparison.p_value < 0.05
        if observed >= minimum:
            sections.append(
                f"- 观察到的差异 {observed:.1%} ≥ 可检出下限 {minimum:.1%}——有足够分辨率"
            )
        elif significant:
            # 事后显著 ≠ 事先有把握：这次刚好越线，换一批采样可能就翻转
            sections.append(
                f"- 观察到的差异 {observed:.1%} 低于可检出下限 {minimum:.1%}："
                "**这次显著，但功效不足**——换一批采样可能翻转，别当成稳固结论"
            )
        else:
            sections.append(
                f"- 观察到的差异 {observed:.1%} 低于可检出下限 {minimum:.1%}——"
                "统计上无法与噪声区分"
            )
    else:
        sections.append(f"- 区间宽度 {_accuracy_interval(records).width:.1%}")
    sections.append("")

    return "\n".join(sections) + "\n"


def write_report(text: str, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return target
