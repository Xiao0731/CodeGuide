"""CodeGuide 最小版 GRPO 奖励。

正式梯度只使用两路信号：
1. Code Reward：静态有效性 + 部分测试通过率 + 严格全通过奖励；
2. Format Reward：沿用现有教学格式评分，并由测试通过率门控。

Teaching Reward 只记录诊断值，不进入梯度。该实现有意保持简洁，避免继续扩展项目范围。
"""

from __future__ import annotations

import ast
import json
import statistics
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RewardWeights:
    code: float = 0.60
    contract: float = 0.40
    static_validity: float = 0.05
    partial_pass: float = 0.70
    strict_pass: float = 0.25
    contract_gate_floor: float = 0.25

    def validate(self) -> None:
        if abs(self.code + self.contract - 1.0) > 1e-8:
            raise ValueError("code 与 contract 权重之和必须为 1")
        if abs(self.static_validity + self.partial_pass + self.strict_pass - 1.0) > 1e-8:
            raise ValueError("Code Reward 三个子权重之和必须为 1")
        if not 0.0 <= self.contract_gate_floor <= 1.0:
            raise ValueError("contract_gate_floor 必须位于 [0, 1]")


@dataclass(frozen=True)
class RewardBreakdown:
    total: float
    code: float
    contract: float
    gated_contract: float
    static_validity: float
    pass_rate: float
    strict_pass: float
    contract_gate: float


def _cfg_float(section: Any, name: str, default: float) -> float:
    try:
        value = getattr(section, name)
    except (AttributeError, TypeError):
        value = default
    return float(value)


def weights_from_cfg(cfg: Any) -> RewardWeights:
    reward_cfg = cfg.reward
    weights = RewardWeights(
        code=_cfg_float(reward_cfg, "code_weight", 0.60),
        contract=_cfg_float(reward_cfg, "contract_weight", 0.40),
        static_validity=_cfg_float(reward_cfg, "static_validity_weight", 0.05),
        partial_pass=_cfg_float(reward_cfg, "partial_pass_weight", 0.70),
        strict_pass=_cfg_float(reward_cfg, "strict_pass_weight", 0.25),
        contract_gate_floor=_cfg_float(reward_cfg, "contract_gate_floor", 0.25),
    )
    weights.validate()
    return weights


def _defined_functions(tree: ast.AST) -> set[str]:
    return {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def static_validity_score(code: str, metadata: dict[str, Any]) -> float:
    """只判断代码是否可用，不把 AST 当作算法正确性。"""
    if not code or not code.strip():
        return 0.0
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return 0.0

    # 复用现有安全扫描，危险代码不能靠“语法正确”获得静态奖励。
    try:
        from src.reward_functions import _scan_security

        if _scan_security(code):
            return 0.0
    except ImportError:
        pass

    io_mode = str(metadata.get("io_mode") or "unknown")
    if io_mode == "call_based":
        fn_name = metadata.get("fn_name")
        if not isinstance(fn_name, str) or not fn_name.strip():
            return 0.0
        if fn_name not in _defined_functions(tree):
            return 0.0
    return 1.0


def combine_scores(
    *,
    static_validity: float,
    pass_rate: float,
    contract: float,
    weights: RewardWeights,
) -> RewardBreakdown:
    """计算 0.6 Code + 0.4 门控 Format 的最终奖励。"""
    static_validity = min(max(float(static_validity), 0.0), 1.0)
    pass_rate = min(max(float(pass_rate), 0.0), 1.0)
    contract = min(max(float(contract), 0.0), 1.0)
    strict = 1.0 if pass_rate >= 1.0 - 1e-12 else 0.0

    code = (
        weights.static_validity * static_validity
        + weights.partial_pass * pass_rate
        + weights.strict_pass * strict
    )
    gate = weights.contract_gate_floor + (1.0 - weights.contract_gate_floor) * pass_rate
    gated_contract = contract * gate
    total = weights.code * code + weights.contract * gated_contract

    return RewardBreakdown(
        total=round(total, 6),
        code=round(code, 6),
        contract=round(contract, 6),
        gated_contract=round(gated_contract, 6),
        static_validity=round(static_validity, 6),
        pass_rate=round(pass_rate, 6),
        strict_pass=strict,
        contract_gate=round(gate, 6),
    )


def _parse_json_list(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return value
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        payload = json.loads(value)
    except json.JSONDecodeError:
        return []
    return payload if isinstance(payload, list) else []


def _parse_json_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        payload = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def score_completion(
    *,
    completion: str,
    test_cases: list[dict[str, Any]],
    metadata: dict[str, Any],
    weights: RewardWeights,
    timeout: float,
    execution_backend: str,
    container_image: str | None,
) -> RewardBreakdown:
    from src.data.code_validator import extract_code
    from src.reward_functions import accuracy_reward, format_reward

    code = extract_code(completion) or ""
    static_score = static_validity_score(code, metadata)
    pass_rate = accuracy_reward(
        code,
        test_cases,
        timeout=timeout,
        metadata=metadata,
        execution_backend=execution_backend,
        container_image=container_image,
    )
    contract = format_reward(completion)
    return combine_scores(
        static_validity=static_score,
        pass_rate=pass_rate,
        contract=contract,
        weights=weights,
    )


def make_reward_fn_with_cfg(cfg: Any):
    """生成与 TRL GRPOTrainer 兼容的最小版奖励函数。"""
    weights = weights_from_cfg(cfg)
    timeout = float(getattr(cfg.reward, "exec_timeout", 5.0))
    execution_backend = str(getattr(cfg.reward, "execution_backend", "subprocess"))
    container_image = str(getattr(cfg.reward, "container_image", "")) or None
    num_generations = int(getattr(cfg.grpo, "num_generations", 4))

    try:
        from src.reward.teaching import build_teaching_reward

        teaching_fn = build_teaching_reward(cfg)
    except Exception:
        teaching_fn = None

    def reward_fn(
        prompts: list[str],
        completions: list[str],
        test_cases: list[Any] | None = None,
        metadata: list[Any] | None = None,
        **kwargs: Any,
    ) -> list[float]:
        del kwargs
        if not completions:
            return []

        breakdowns: list[RewardBreakdown] = []
        for index, completion in enumerate(completions):
            cases = _parse_json_list(
                test_cases[index] if test_cases is not None and index < len(test_cases) else []
            )
            meta = _parse_json_dict(
                metadata[index] if metadata is not None and index < len(metadata) else {}
            )
            breakdowns.append(
                score_completion(
                    completion=str(completion),
                    test_cases=cases,
                    metadata=meta,
                    weights=weights,
                    timeout=timeout,
                    execution_backend=execution_backend,
                    container_image=container_image,
                )
            )

        totals = [item.total for item in breakdowns]
        teaching_surface = 0.0
        if teaching_fn is not None:
            try:
                teaching_scores = teaching_fn(prompts, completions)
                teaching_surface = sum(teaching_scores) / len(teaching_scores)
            except Exception:
                teaching_surface = 0.0

        group_vars: list[float] = []
        if num_generations > 1:
            for start in range(0, len(totals), num_generations):
                group = totals[start : start + num_generations]
                if len(group) > 1:
                    mean = sum(group) / len(group)
                    group_vars.append(sum((value - mean) ** 2 for value in group) / len(group))

        metrics = {
            "reward/accuracy": sum(item.code for item in breakdowns) / len(breakdowns),
            "reward/pass_rate": sum(item.pass_rate for item in breakdowns) / len(breakdowns),
            "reward/strict_pass": sum(item.strict_pass for item in breakdowns) / len(breakdowns),
            "reward/static_validity": (
                sum(item.static_validity for item in breakdowns) / len(breakdowns)
            ),
            "reward/format": sum(item.contract for item in breakdowns) / len(breakdowns),
            "reward/gated_format": (
                sum(item.gated_contract for item in breakdowns) / len(breakdowns)
            ),
            "diagnostic/teaching_surface": teaching_surface,
            "reward/total": sum(totals) / len(totals),
            "reward/std": statistics.stdev(totals) if len(totals) > 1 else 0.0,
            "reward/min": min(totals),
            "reward/max": max(totals),
            "_group_vars": group_vars,
        }

        # 复用现有 WandB callback 的缓冲区，不改动大体量训练主文件。
        try:
            from src.training import grpo_train

            grpo_train._reward_buffer.append(metrics)
        except Exception:
            pass
        return totals

    return reward_fn
