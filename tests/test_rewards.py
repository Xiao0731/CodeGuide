from __future__ import annotations

import json

import pytest

from src.reward.format import contract_score
from src.reward.grpo import build_reward_functions, completion_text


GOOD_TEACHING = """
## 题意理解
我们需要读取两个整数并输出它们的和。这个例子虽小，但先明确输入输出可以避免把函数题写成标准输入题。

## 关键观察
题目没有隐藏状态，也不需要额外数据结构。因为答案只依赖当前两个整数，所以直接相加就是完整算法。

## 算法步骤
第一步读取两个整数，第二步计算它们的和，第三步输出结果。每一步都只执行常数次操作。

## 正确性
程序输出的值严格等于输入的第一个整数加第二个整数，因此与题目定义的目标完全一致。

```python
a, b = map(int, input().split())
answer = a + b
print(answer)
```

时间复杂度 O(1)，空间复杂度 O(1)。
"""


@pytest.mark.parametrize(
    ("response", "lower", "upper"),
    [
        (GOOD_TEACHING, 0.8, 1.0),
        ("```python\nprint(1)\n```", 0.0, 0.5),
        ("", 0.0, 0.0),
    ],
)
def test_contract_score(response, lower, upper):
    assert lower <= contract_score(response) <= upper


def test_grpo_rewards_execute_standard_input_solution():
    correctness, teaching = build_reward_functions(
        backend="subprocess", container_image=None, timeout=2.0
    )
    cases = [{"input": "1 2\n", "output": "3"}]
    metadata = {"io_mode": "standard_input", "fn_name": None, "starter_code": ""}

    assert correctness(
        [GOOD_TEACHING],
        test_cases=[json.dumps(cases)],
        metadata=[json.dumps(metadata)],
    ) == [1.0]
    assert teaching([GOOD_TEACHING])[0] >= 0.8


def test_grpo_rewards_respect_call_based_interface():
    correctness, _ = build_reward_functions(
        backend="subprocess", container_image=None, timeout=2.0
    )
    cases = [{"input_args": [1, 2], "expected_output": 3}]
    metadata = {"io_mode": "call_based", "fn_name": "add", "starter_code": ""}
    completion = "```python\ndef add(a, b):\n    return a + b\n```"

    assert correctness(
        [completion],
        test_cases=[json.dumps(cases)],
        metadata=[json.dumps(metadata)],
    ) == [1.0]


def test_conversational_completion_is_normalized():
    assert completion_text([{"role": "assistant", "content": "answer"}]) == "answer"
