from __future__ import annotations

import pytest

from src.reward.execution import compare_outputs, supports_verification, verify_code

STD_CASES = [
    {"input": "1 2\n", "output": "3"},
    {"input": "5 7\n", "output": "12"},
    {"input": "-2 9\n", "output": "7"},
    {"input": "0 0\n", "output": "0"},
]


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("a, b = map(int, input().split())\nprint(a + b)\n", 1.0),
        ("a, b = map(int, input().split())\nprint(a - b)\n", 0.25),
    ],
)
def test_standard_input_verification(code, expected):
    metadata = {"io_mode": "standard_input", "test_cases": STD_CASES}
    assert verify_code(code, metadata, timeout=2.0).pass_rate == pytest.approx(expected)


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("def add(a, b):\n    return a + b\n", 1.0),
        ("def add(a, b):\n    return a - b\n", 0.0),
        ("def add(:\n    pass", 0.0),
        ("def add(a, b):\n    raise ValueError('boom')\n", 0.0),
    ],
)
def test_call_based_function_verification(code, expected):
    cases = [{"input_args": [1, 2], "expected_output": 3, "fn_name": "add"}]
    metadata = {"io_mode": "call_based", "fn_name": "add", "test_cases": cases}
    assert verify_code(code, metadata, timeout=2.0).pass_rate == pytest.approx(expected)


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
        for index, value in enumerate(nums):
            if target - value in seen:
                return [seen[target - value], index]
            seen[value] = index
"""
    metadata = {"io_mode": "call_based", "fn_name": "twoSum", "test_cases": cases}
    result = verify_code(code, metadata, timeout=2.0)
    assert result.pass_rate == 1.0
    assert result.unsupported is False


def test_unsupported_call_based_is_marked():
    cases = [{"input_args": [{"node": object()}], "expected_output": 1}]
    metadata = {"io_mode": "call_based", "fn_name": "solve", "test_cases": cases}
    assert verify_code("def solve(x):\n    return 1\n", metadata).unsupported is True


def test_comparator_handles_nested_json_values():
    assert compare_outputs({"a": [1, 2.0, None]}, {"a": [1, 2.0 + 1e-7, None]})
    assert not compare_outputs([1, 2], [2, 1])


@pytest.mark.parametrize(
    ("metadata", "supported"),
    [
        ({"io_mode": "standard_input", "test_cases": STD_CASES}, True),
        (
            {
                "io_mode": "call_based",
                "fn_name": "add",
                "test_cases": [{"input_args": [1, 2], "expected_output": 3}],
            },
            True,
        ),
        (
            {
                "io_mode": "call_based",
                "fn_name": "add",
                "test_cases": [{"input": "[1,2]", "output": "3"}],
            },
            False,
        ),
    ],
)
def test_support_detection(metadata, supported):
    assert supports_verification(metadata) is supported
