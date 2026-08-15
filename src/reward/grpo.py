"""Canonical composite reward for formal TRL GRPO training."""

from __future__ import annotations

import ast
import json
from dataclasses import asdict, dataclass
from typing import Any, Callable, Mapping

from src.data.code_validator import extract_code
from src.reward.execution import verify_code
from src.reward.format import contract_score


@dataclass(frozen=True)
class RewardWeights:
    code: float = 0.60
    contract: float = 0.40
    static_validity: float = 0.05
    partial_pass: float = 0.70
    strict_pass: float = 0.25
    contract_gate_floor: float = 0.25

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RewardWeights":
        weights = cls(
            code=float(value.get("code_weight", 0.60)),
            contract=float(value.get("contract_weight", 0.40)),
            static_validity=float(value.get("static_validity_weight", 0.05)),
            partial_pass=float(value.get("partial_pass_weight", 0.70)),
            strict_pass=float(value.get("strict_pass_weight", 0.25)),
            contract_gate_floor=float(value.get("contract_gate_floor", 0.25)),
        )
        weights.validate()
        return weights

    def validate(self) -> None:
        if abs(self.code + self.contract - 1.0) > 1e-9:
            raise ValueError("code and contract weights must sum to 1")
        if abs(self.static_validity + self.partial_pass + self.strict_pass - 1.0) > 1e-9:
            raise ValueError("code reward component weights must sum to 1")
        if not 0.0 <= self.contract_gate_floor <= 1.0:
            raise ValueError("contract_gate_floor must be in [0, 1]")


@dataclass(frozen=True)
class RewardBreakdown:
    total: float
    code_reward: float
    contract_score: float
    gated_contract: float
    static_validity: float
    pass_rate: float
    strict: float
    contract_gate: float


_DANGEROUS_MODULES = {
    "ctypes",
    "ftplib",
    "http",
    "multiprocessing",
    "requests",
    "socket",
    "subprocess",
    "urllib",
}
_DANGEROUS_CALLS = {"__import__", "breakpoint", "compile", "eval", "exec", "open"}
_DANGEROUS_ATTRIBUTES = {
    "os.popen",
    "os.remove",
    "os.rmdir",
    "os.system",
    "os.unlink",
    "shutil.rmtree",
}


def completion_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return str(value.get("content") or "")
    if isinstance(value, list):
        return "".join(completion_text(item) for item in value)
    return str(value or "")


def _json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        parsed = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _json_list(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return value
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        parsed = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return []
    return parsed if isinstance(parsed, list) else []


def _attribute_name(node: ast.Attribute) -> str:
    parts = [node.attr]
    value: ast.expr = node.value
    while isinstance(value, ast.Attribute):
        parts.append(value.attr)
        value = value.value
    if isinstance(value, ast.Name):
        parts.append(value.id)
    return ".".join(reversed(parts))


def static_validity_score(code: str, metadata: Mapping[str, Any]) -> float:
    """Check syntax, basic safety and interface presence, never semantics."""
    if not code.strip():
        return 0.0
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return 0.0

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            modules = (
                [alias.name for alias in node.names]
                if isinstance(node, ast.Import)
                else [str(node.module or "")]
            )
            if any(module.split(".", 1)[0] in _DANGEROUS_MODULES for module in modules):
                return 0.0
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in _DANGEROUS_CALLS:
                return 0.0
            if isinstance(node.func, ast.Attribute) and _attribute_name(node.func) in _DANGEROUS_ATTRIBUTES:
                return 0.0

    io_mode = str(metadata.get("io_mode") or "")
    if io_mode == "call_based":
        fn_name = metadata.get("fn_name")
        if not isinstance(fn_name, str) or not fn_name.strip():
            return 0.0
        functions = {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        return 1.0 if fn_name in functions else 0.0

    if io_mode == "standard_input":
        call_names = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        has_input = "input" in call_names or "sys.stdin" in code
        has_output = "print" in call_names or "sys.stdout" in code
        return 1.0 if has_input and has_output else 0.0
    return 0.0


def combine_scores(
    *,
    static_validity: float,
    pass_rate: float,
    contract: float,
    weights: RewardWeights,
) -> RewardBreakdown:
    static_validity = min(max(float(static_validity), 0.0), 1.0)
    pass_rate = min(max(float(pass_rate), 0.0), 1.0)
    contract = min(max(float(contract), 0.0), 1.0)
    strict = 1.0 if pass_rate == 1.0 else 0.0
    code_reward = (
        weights.static_validity * static_validity
        + weights.partial_pass * pass_rate
        + weights.strict_pass * strict
    )
    gate = weights.contract_gate_floor + (1.0 - weights.contract_gate_floor) * pass_rate
    gated_contract = contract * gate
    total = weights.code * code_reward + weights.contract * gated_contract
    return RewardBreakdown(
        total=round(total, 6),
        code_reward=round(code_reward, 6),
        contract_score=round(contract, 6),
        gated_contract=round(gated_contract, 6),
        static_validity=round(static_validity, 6),
        pass_rate=round(pass_rate, 6),
        strict=strict,
        contract_gate=round(gate, 6),
    )


def score_completion(
    *,
    completion: str,
    test_cases: list[dict[str, Any]],
    metadata: dict[str, Any],
    weights: RewardWeights,
    backend: str,
    container_image: str | None,
    timeout: float,
) -> RewardBreakdown:
    code = extract_code(
        completion,
        io_mode=metadata.get("io_mode"),
        fn_name=metadata.get("fn_name"),
        starter_code=metadata.get("starter_code"),
    ) or ""
    static_score = static_validity_score(code, metadata)
    pass_rate = 0.0
    if code:
        execution_metadata = dict(metadata)
        execution_metadata["test_cases"] = test_cases
        result = verify_code(
            code,
            execution_metadata,
            timeout=timeout,
            backend=backend,
            container_image=container_image,
        )
        if not result.unsupported:
            pass_rate = float(result.pass_rate)
    return combine_scores(
        static_validity=static_score,
        pass_rate=pass_rate,
        contract=contract_score(completion),
        weights=weights,
    )


def build_composite_reward(
    *,
    reward_config: Mapping[str, Any],
) -> Callable[..., list[float]]:
    """Build the one formal gradient reward; each completion is verified once."""
    weights = RewardWeights.from_mapping(reward_config)
    backend = str(reward_config.get("execution_backend", "subprocess"))
    if backend != "subprocess":
        raise ValueError("formal GRPO training requires execution_backend=subprocess")
    timeout = float(reward_config.get("timeout", 5.0))
    image = reward_config.get("container_image")

    def formal_composite_reward(
        completions: list[Any],
        test_cases: list[Any] | None = None,
        metadata: list[Any] | None = None,
        **_: Any,
    ) -> list[float]:
        breakdowns: list[RewardBreakdown] = []
        for index, raw_completion in enumerate(completions):
            meta = _json_object(metadata[index] if metadata else {})
            cases = _json_list(test_cases[index] if test_cases else [])
            breakdowns.append(
                score_completion(
                    completion=completion_text(raw_completion),
                    test_cases=cases,
                    metadata=meta,
                    weights=weights,
                    backend=backend,
                    container_image=str(image) if image else None,
                    timeout=timeout,
                )
            )
        formal_composite_reward.last_diagnostics = [asdict(item) for item in breakdowns]
        return [item.total for item in breakdowns]

    formal_composite_reward.last_diagnostics = []
    return formal_composite_reward
