from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.reward.execution import compare_outputs, supports_verification, verify_code
from src.reward_functions import accuracy_reward
from src.training.grpo_train import _build_grpo_records


STD_CASES = [
    {"input": "1 2\n", "output": "3"},
    {"input": "5 7\n", "output": "12"},
    {"input": "-2 9\n", "output": "7"},
    {"input": "0 0\n", "output": "0"},
]

# 自动化测试实现：
# OJ standard_input 正确/错误代码验证 ✅
# call_based 顶层函数验证 ✅
# class Solution 验证 ✅
# 异常处理 ✅
# 超时处理 ✅
# unsupported 样本识别 ✅
# GRPO 过滤统计


class DummyTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "\n".join(m["content"] for m in messages) + "\nassistant:"


def test_standard_input_accuracy_reward_does_not_regress():
    good = "a, b = map(int, input().split())\nprint(a + b)\n"
    bad = "a, b = map(int, input().split())\nprint(a - b)\n"

    assert accuracy_reward(good, STD_CASES, timeout=2.0) == pytest.approx(1.0)
    assert accuracy_reward(bad, STD_CASES, timeout=2.0) == pytest.approx(0.25)


def test_call_based_top_level_function():
    cases = [{"input_args": [1, 2], "expected_output": 3, "fn_name": "add"}]
    meta = {"io_mode": "call_based", "fn_name": "add", "test_cases": cases}

    assert verify_code("def add(a, b):\n    return a + b\n", meta, timeout=2.0).pass_rate == pytest.approx(1.0)
    assert verify_code("def add(a, b):\n    return a - b\n", meta, timeout=2.0).pass_rate == pytest.approx(0.0)
    assert accuracy_reward("def add(a, b):\n    return a + b\n", cases, metadata=meta, timeout=2.0) == pytest.approx(1.0)


def test_call_based_class_solution():
    cases = [
        {
            "input_args": [[2, 7, 11, 15], 9],
            "expected_output": [0, 1],
            "fn_name": "twoSum",
        }
    ]
    code = """
class Solution:
    def twoSum(self, nums, target):
        seen = {}
        for i, x in enumerate(nums):
            if target - x in seen:
                return [seen[target - x], i]
            seen[x] = i
"""
    meta = {"io_mode": "call_based", "fn_name": "twoSum", "test_cases": cases}
    result = verify_code(code, meta, timeout=2.0)
    assert result.pass_rate == pytest.approx(1.0)
    assert result.unsupported is False


def test_call_based_errors_and_timeout():
    cases = [{"input_args": [1, 2], "expected_output": 3, "fn_name": "add"}]
    meta = {"io_mode": "call_based", "fn_name": "add", "test_cases": cases}

    assert verify_code("def add(:\n    pass", meta, timeout=2.0).pass_rate == 0.0
    assert verify_code("def add(a, b):\n    raise ValueError('boom')\n", meta, timeout=2.0).pass_rate == 0.0
    assert verify_code("def add(a, b):\n    while True:\n        pass\n", meta, timeout=1.0).pass_rate == 0.0


def test_unsupported_call_based_is_marked_not_failed():
    cases = [{"input_args": [{"node": object()}], "expected_output": 1, "fn_name": "solve"}]
    meta = {"io_mode": "call_based", "fn_name": "solve", "test_cases": cases}
    result = verify_code("def solve(x):\n    return 1\n", meta, timeout=2.0)
    assert result.unsupported is True


def test_comparator_handles_nested_json_values():
    assert compare_outputs({"a": [1, 2.0, None]}, {"a": [1, 2.0 + 1e-7, None]})
    assert not compare_outputs([1, 2], [2, 1])


def test_reward_compatible_support_detection():
    assert supports_verification({"io_mode": "standard_input", "test_cases": STD_CASES}) is True
    assert supports_verification({
        "io_mode": "call_based",
        "fn_name": "add",
        "test_cases": [{"input_args": [1, 2], "expected_output": 3}],
    }) is True
    assert supports_verification({
        "io_mode": "call_based",
        "fn_name": "add",
        "test_cases": [{"input": "[1,2]", "output": "3"}],
    }) is False


def test_grpo_filter_stats_include_call_based(tmp_path: Path):
    records = [
        {
            "messages": [{"role": "user", "content": "stdin add"}],
            "metadata": {"io_mode": "standard_input", "test_cases": STD_CASES, "reward_compatible": True},
        },
        {
            "messages": [{"role": "user", "content": "function add"}],
            "metadata": {
                "io_mode": "call_based",
                "fn_name": "add",
                "test_cases": [
                    {"input_args": [1, 2], "expected_output": 3},
                    {"input_args": [5, 7], "expected_output": 12},
                    {"input_args": [-2, 9], "expected_output": 7},
                    {"input_args": [0, 0], "expected_output": 0},
                ],
                "reward_compatible": True,
            },
        },
        {
            "messages": [{"role": "user", "content": "unsupported"}],
            "metadata": {"io_mode": "call_based", "test_cases": [], "reward_compatible": False},
        },
    ]
    path = tmp_path / "sft.jsonl"
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records), encoding="utf-8")

    built, stats = _build_grpo_records(path, DummyTokenizer())
    assert len(built) == 2
    assert stats["standard_input_reward_compatible"] == 1
    assert stats["call_based_reward_compatible"] == 1
    assert stats["skipped_incompatible"] == 1

    for record in built:
        reward_tests = json.loads(record["test_cases"])
        metadata = json.loads(record["metadata"])
        assert len(reward_tests) == 3
        assert metadata["heldout_test_count"] == 1
        assert "test_cases" not in metadata
