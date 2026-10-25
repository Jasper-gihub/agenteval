"""AgentEval：让 agent 评测的结论能过统计这一关。

设计原则见 README。这里只强调一条实现约束：
**不依赖 scipy。** 检验和区间都用标准库实现，这样 pip install 只有一个空依赖，
也不会因为某个科学计算包版本问题让 CI 挂掉。
"""

from agenteval.dataset import Dataset, Task
from agenteval.runner import (
    AgentOutput,
    Record,
    Summary,
    correctness_by_task,
    run_dataset,
    run_sync,
    summarize,
)
from agenteval.stats import (
    mcnemar_exact,
    paired_bootstrap,
    pareto_frontier,
    required_sample_size,
    variance_components,
    wilson_interval,
)

__version__ = "0.1.0"

__all__ = [
    "AgentOutput",
    "Dataset",
    "Record",
    "Summary",
    "Task",
    "correctness_by_task",
    "mcnemar_exact",
    "paired_bootstrap",
    "pareto_frontier",
    "required_sample_size",
    "run_dataset",
    "run_sync",
    "summarize",
    "variance_components",
    "wilson_interval",
]
