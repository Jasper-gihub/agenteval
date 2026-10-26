"""任务集加载与校验。

校验不是可选项：一条没有金标准的题跑完后无法判定对错，
这种错误必须在加载阶段就炸掉，而不是跑完 27 分钟才发现。
"""

from __future__ import annotations

import pytest
import yaml

from agenteval.dataset import Dataset, Task


def write_yaml(tmp_path, items):
    path = tmp_path / "tasks.yaml"
    path.write_text(yaml.safe_dump(items, allow_unicode=True), encoding="utf-8")
    return path


def test_loads_dataagent_format(tmp_path):
    """Litmus 的 question / ref_sql 写法必须能直接读进来。"""
    path = write_yaml(
        tmp_path,
        [
            {
                "id": "q01",
                "difficulty": "easy",
                "question": "一共有多少个订单？",
                "ref_sql": "select count(*) from orders",
            }
        ],
    )
    dataset = Dataset.from_yaml(path)
    task = dataset.tasks[0]
    assert task.id == "q01"
    assert task.prompt == "一共有多少个订单？"
    assert task.gold_sql.startswith("select count(*)")
    assert task.difficulty == "easy"


def test_extra_keys_go_to_meta(tmp_path):
    path = write_yaml(
        tmp_path,
        [{"id": "a", "question": "q", "gold": "42", "note": "口径：排除取消单"}],
    )
    dataset = Dataset.from_yaml(path)
    assert dataset.tasks[0].meta == {"note": "口径：排除取消单"}


def test_rejects_duplicate_ids(tmp_path):
    path = write_yaml(
        tmp_path,
        [{"id": "a", "question": "q1", "gold": "1"}, {"id": "a", "question": "q2", "gold": "2"}],
    )
    with pytest.raises(ValueError, match="id 重复"):
        Dataset.from_yaml(path)


def test_rejects_task_without_gold(tmp_path):
    path = write_yaml(tmp_path, [{"id": "a", "question": "q"}])
    with pytest.raises(ValueError, match="无法判定对错"):
        Dataset.from_yaml(path)


def test_rejects_invalid_difficulty(tmp_path):
    path = write_yaml(tmp_path, [{"id": "a", "question": "q", "gold": "1", "difficulty": "extreme"}])
    with pytest.raises(ValueError, match="difficulty"):
        Dataset.from_yaml(path)


def test_rejects_empty_prompt(tmp_path):
    path = write_yaml(tmp_path, [{"id": "a", "question": "   ", "gold": "1"}])
    with pytest.raises(ValueError, match="prompt 为空"):
        Dataset.from_yaml(path)


def test_rejects_empty_dataset(tmp_path):
    path = write_yaml(tmp_path, [])
    with pytest.raises(ValueError, match="任务集为空"):
        Dataset.from_yaml(path)


def test_by_difficulty_and_describe(tmp_path):
    path = write_yaml(
        tmp_path,
        [
            {"id": "a", "question": "q", "gold": "1", "difficulty": "easy"},
            {"id": "b", "question": "q", "gold": "1", "difficulty": "easy"},
            {"id": "c", "question": "q", "gold": "1", "difficulty": "hard"},
        ],
    )
    dataset = Dataset.from_yaml(path)
    assert dataset.by_difficulty() == {"easy": 2, "hard": 1}
    assert "easy 2" in dataset.describe()


def test_direct_construction_validates_too():
    with pytest.raises(ValueError):
        Dataset([Task(id="a", prompt="q", gold=None, gold_sql=None)])
