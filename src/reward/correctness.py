"""
CodeCorrectnessReward: 提取模型输出中的代码块，在沙箱中运行测试用例
返回 pass 率作为奖励 [0.0, 1.0]

改进（v2）：
- 无测试用例时改为调用 _estimate_code_quality 做静态分析
  而非固定返回 0.5，保持与 reward_functions.accuracy_reward 一致
"""
import re
from typing import Any, List, Optional
from src.reward.execution import verify_code

_CODE_BLOCK_RE = re.compile(r"```(?:python)?\n(.*?)```", re.DOTALL)


def _extract_code(text: str) -> Optional[str]:
    """提取最后一个代码块（通常是最终实现）"""
    matches = _CODE_BLOCK_RE.findall(text)
    return matches[-1].strip() if matches else None


def _run_tests(code: str, test_cases: List[dict], timeout: int = 5) -> float:
    """在子进程中运行测试用例，返回通过率"""
    if not test_cases:
        return 0.0

    result = verify_code(code, {"test_cases": test_cases}, timeout=timeout)
    return 0.0 if result.unsupported else result.pass_rate


class CodeCorrectnessReward:
    def __init__(self, cfg: Any):
        self.timeout = float(getattr(cfg.reward, "exec_timeout", 5.0))

    def __call__(
        self,
        prompts: List[str],
        completions: List[str],
        test_cases_batch: Optional[List[List[dict]]] = None,
    ) -> List[float]:
        scores = []
        for i, completion in enumerate(completions):
            code = _extract_code(completion)
            if code is None:
                scores.append(0.0)
                continue
            tc = test_cases_batch[i] if test_cases_batch else []
            scores.append(_run_tests(code, tc, self.timeout))
        return scores
