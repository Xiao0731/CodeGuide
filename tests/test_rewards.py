from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from src.reward.format import contract_score
from src.reward.grpo import (
    RewardWeights,
    build_composite_reward,
    combine_scores,
    completion_text,
    static_validity_score,
)


GOOD_TEACHING = """
## 第一步：理解题意
我们需要读取两个整数并输出它们的和。先明确输入输出合同非常重要，因为标准输入题必须从输入流读取数据，并把唯一答案写到标准输出中，不能误写成只返回结果的函数。

## 第二步：关键观察
这道题没有隐藏状态，也不需要额外的数据结构。因为答案只依赖当前两个整数，所以直接相加就是完整算法；加入循环、搜索或排序只会增加不必要的复杂度。

## 第三步：算法步骤
首先用 input 读取一行并拆成两个整数，然后计算两数之和，最后用 print 输出结果。每一步都只执行常数次操作，因此既容易验证，也不会引入边界条件之外的额外风险。

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
        (GOOD_TEACHING, 0.95, 1.0),
        ("```python\nprint(1)\n```", 0.0, 0.2),
        ("", 0.0, 0.0),
    ],
)
def test_contract_score(response, lower, upper):
    assert lower <= contract_score(response) <= upper


def test_contract_rejects_repeated_short_steps():
    repeated = "\n".join(f"## 第{i}步\n内容太短" for i in range(1, 5))
    assert contract_score(repeated) == 0.0


def test_formal_reward_formula():
    weights = RewardWeights()
    perfect = combine_scores(
        static_validity=1.0, pass_rate=1.0, contract=1.0, weights=weights
    )
    partial = combine_scores(
        static_validity=1.0, pass_rate=0.5, contract=0.8, weights=weights
    )
    assert perfect.total == 1.0
    assert partial.code_reward == 0.4
    assert partial.contract_gate == 0.625
    assert partial.gated_contract == 0.5
    assert partial.total == 0.44


def test_static_validity_checks_interface_without_claiming_correctness():
    standard_meta = {"io_mode": "standard_input"}
    call_meta = {"io_mode": "call_based", "fn_name": "add"}
    assert static_validity_score("x = int(input())\nprint(x)", standard_meta) == 1.0
    assert static_validity_score("def add(a, b):\n    return a - b", call_meta) == 1.0
    assert static_validity_score("def wrong(a, b):\n    return a + b", call_meta) == 0.0
    assert static_validity_score("import subprocess\nsubprocess.run([])", standard_meta) == 0.0


def _reward_config() -> dict:
    return {
        "code_weight": 0.60,
        "contract_weight": 0.40,
        "static_validity_weight": 0.05,
        "partial_pass_weight": 0.70,
        "strict_pass_weight": 0.25,
        "contract_gate_floor": 0.25,
        "execution_backend": "subprocess",
        "timeout": 2.0,
    }


def test_composite_reward_executes_standard_input_solution():
    reward = build_composite_reward(reward_config=_reward_config())
    cases = [{"input": "1 2\n", "output": "3"}]
    metadata = {"io_mode": "standard_input", "fn_name": None, "starter_code": ""}
    assert reward(
        [GOOD_TEACHING],
        test_cases=[json.dumps(cases)],
        metadata=[json.dumps(metadata)],
    ) == [1.0]
    assert reward.last_diagnostics[0]["pass_rate"] == 1.0


def test_composite_reward_respects_call_based_interface():
    reward = build_composite_reward(reward_config=_reward_config())
    cases = [{"input_args": [1, 2], "expected_output": 3}]
    metadata = {"io_mode": "call_based", "fn_name": "add", "starter_code": ""}
    completion = "```python\ndef add(a, b):\n    return a + b\n```"
    score = reward(
        [completion],
        test_cases=[json.dumps(cases)],
        metadata=[json.dumps(metadata)],
    )[0]
    assert score > 0.6
    assert reward.last_diagnostics[0]["strict"] == 1.0


def test_composite_reward_calls_verifier_once_per_completion(monkeypatch):
    calls = []

    def fake_verify(code, metadata, **kwargs):
        calls.append((code, metadata, kwargs))
        return SimpleNamespace(pass_rate=0.5, unsupported=False)

    monkeypatch.setattr("src.reward.grpo.verify_code", fake_verify)
    reward = build_composite_reward(reward_config=_reward_config())
    completion = "```python\nx = int(input())\nprint(x)\n```"
    cases = json.dumps([{"input": "1\n", "output": "1"}])
    metadata = json.dumps({"io_mode": "standard_input"})
    reward([completion, completion], test_cases=[cases, cases], metadata=[metadata, metadata])
    assert len(calls) == 2


def test_conversational_completion_is_normalized():
    assert completion_text([{"role": "assistant", "content": "answer"}]) == "answer"
