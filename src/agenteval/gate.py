"""CI 门禁：只有"能证明没变差"才放行。

和 report 的分工：report 是给人看的，gate 是给流水线看的——
它只回答一个问题，并且用退出码回答。

门禁的三条规则（都是"宁可漏放，不要误杀"的保守设定）：

1. **准确率跌幅超过阈值 → 失败。** 默认 3 个点。这个值不是拍脑袋：
   60 题的 95% 区间宽 0.245，3 个点远在噪声内，设太严会让每次 PR 都红。
2. **每题成本涨幅超过阈值 → 失败。** 默认 50%。
3. **共享题目太少 → 只警告不失败。** 样本不足时应该提醒"这个结论不可靠"，
   而不是用它去阻断别人的合并。
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean

from agenteval.report import compare, load_records, per_task_accuracy


@dataclass(slots=True)
class GateResult:
    passed: bool
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    accuracy_delta: float = 0.0
    cost_increase: float = 0.0
    shared_tasks: int = 0

    def render(self) -> str:
        lines = ["门禁结果：" + ("通过 ✅" if self.passed else "失败 ❌")]
        lines.append(f"  共享题目 {self.shared_tasks} 道")
        lines.append(f"  准确率变化 {self.accuracy_delta:+.3f}")
        lines.append(f"  每题成本变化 {self.cost_increase:+.1%}")
        for note in self.notes:
            lines.append(f"  · {note}")
        for warning in self.warnings:
            lines.append(f"  ⚠️  {warning}")
        for reason in self.reasons:
            lines.append(f"  ❌ {reason}")
        return "\n".join(lines)


def check_gate(
    baseline,
    candidate,
    *,
    max_accuracy_drop: float = 0.03,
    max_cost_increase: float = 0.5,
    min_shared_tasks: int = 10,
    require_significance: bool = False,
) -> GateResult:
    """比较两组记录，判断能否放行。"""
    if not baseline or not candidate:
        raise ValueError("两组记录都不能为空")

    left = per_task_accuracy(baseline)
    right = per_task_accuracy(candidate)
    shared = [key for key in left if key in right]

    result = GateResult(passed=True, shared_tasks=len(shared))

    if not shared:
        result.passed = False
        result.reasons.append("两组记录没有共同题目，无法比较")
        return result

    delta = mean(right[key] for key in shared) - mean(left[key] for key in shared)
    result.accuracy_delta = delta

    if len(shared) < min_shared_tasks:
        result.warnings.append(
            f"只有 {len(shared)} 道共同题目（< {min_shared_tasks}），"
            "这个结论的统计意义很弱"
        )

    if -delta > max_accuracy_drop:
        result.passed = False
        result.reasons.append(
            f"准确率下降 {-delta:.3f}，超过允许的 {max_accuracy_drop:.3f}"
        )

    baseline_cost = mean(record.cost for record in baseline)
    candidate_cost = mean(record.cost for record in candidate)
    if baseline_cost > 0:
        increase = (candidate_cost - baseline_cost) / baseline_cost
        result.cost_increase = increase
        if increase > max_cost_increase:
            result.passed = False
            result.reasons.append(
                f"每题成本上涨 {increase:.1%}，超过允许的 {max_cost_increase:.1%}"
            )
    elif candidate_cost > 0:
        result.notes.append("基线是免费模型，本次产生了成本（不计入失败）")

    if require_significance:
        comparison = compare(baseline, candidate)
        result.notes.append(
            f"McNemar p = {comparison.p_value:.4f}（"
            f"{'显著' if comparison.p_value < 0.05 else '不显著'}）"
        )
        if comparison.p_value >= 0.05 and delta > 0:
            result.warnings.append(
                "准确率虽然上升，但差异**不显著**——不要以此作为「变好了」的证据"
            )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="agenteval gate", description="比较两组评测记录，回归则返回非零退出码"
    )
    parser.add_argument("--baseline", required=True, help="基线 records.jsonl")
    parser.add_argument("--candidate", required=True, help="本次 records.jsonl")
    parser.add_argument("--max-drop", type=float, default=0.03, help="允许的准确率跌幅")
    parser.add_argument(
        "--max-cost-increase", type=float, default=0.5, help="允许的每题成本涨幅"
    )
    parser.add_argument("--min-tasks", type=int, default=10, help="低于此题目数只警告")
    parser.add_argument(
        "--require-significance",
        action="store_true",
        help="额外报告 McNemar p 值（不改变退出码）",
    )
    args = parser.parse_args(argv)

    result = check_gate(
        load_records(args.baseline),
        load_records(args.candidate),
        max_accuracy_drop=args.max_drop,
        max_cost_increase=args.max_cost_increase,
        min_shared_tasks=args.min_tasks,
        require_significance=args.require_significance,
    )
    print(result.render())
    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())
