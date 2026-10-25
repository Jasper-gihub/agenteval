"""命令行入口：`python -m agenteval <command>`。

两个子命令：
  report  把 records.jsonl 变成 Markdown 报告
  gate    比较两组记录，回归则返回退出码 1（给 CI 用）
"""

from __future__ import annotations

import argparse
import sys

from agenteval import gate as gate_module
from agenteval import report as report_module


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agenteval", description="Agent 评测：先算清楚，再下结论"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_report = sub.add_parser("report", help="生成 Markdown 报告")
    p_report.add_argument("--records", required=True, help="本次 records.jsonl")
    p_report.add_argument("--baseline", help="可选的基线 records.jsonl")
    p_report.add_argument("--label", default="candidate")
    p_report.add_argument("--baseline-label", default="baseline")
    p_report.add_argument("--title", default="Agent 评测报告")
    p_report.add_argument("--out", help="输出路径，默认打印到 stdout")

    p_gate = sub.add_parser("gate", help="回归门禁（CI 用）")
    p_gate.add_argument("--baseline", required=True)
    p_gate.add_argument("--candidate", required=True)
    p_gate.add_argument("--max-drop", type=float, default=0.03)
    p_gate.add_argument("--max-cost-increase", type=float, default=0.5)
    p_gate.add_argument("--min-tasks", type=int, default=10)
    p_gate.add_argument("--require-significance", action="store_true")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "gate":
        return gate_module.main(
            [
                "--baseline",
                args.baseline,
                "--candidate",
                args.candidate,
                "--max-drop",
                str(args.max_drop),
                "--max-cost-increase",
                str(args.max_cost_increase),
                "--min-tasks",
                str(args.min_tasks),
            ]
            + (["--require-significance"] if args.require_significance else [])
        )

    records = report_module.load_records(args.records)
    baseline = report_module.load_records(args.baseline) if args.baseline else None
    text = report_module.build_report(
        records,
        label=args.label,
        baseline=baseline,
        baseline_label=args.baseline_label,
        title=args.title,
    )
    if args.out:
        target = report_module.write_report(text, args.out)
        print(f"报告已写入：{target}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
