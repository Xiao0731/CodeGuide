"""TRL-compatible correctness and teaching-contract reward adapters."""

from __future__ import annotations

import json
from typing import Any, Callable

from src.data.code_validator import extract_code
from src.reward.execution import verify_code
from src.reward.format import contract_score


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
    if isinstance(value, str) and value.strip():
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _json_list(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return value
    if isinstance(value, str) and value.strip():
        parsed = json.loads(value)
        return parsed if isinstance(parsed, list) else []
    return []


def build_reward_functions(
    *, backend: str, container_image: str | None, timeout: float
) -> tuple[Callable[..., list[float]], Callable[..., list[float]]]:
    """Return separate rewards so TRL applies configured reward weights directly."""

    def correctness_reward(
        completions: list[Any],
        test_cases: list[Any] | None = None,
        metadata: list[Any] | None = None,
        **_: Any,
    ) -> list[float]:
        scores: list[float] = []
        for index, completion in enumerate(completions):
            meta = _json_object(metadata[index] if metadata else {})
            meta["test_cases"] = _json_list(test_cases[index] if test_cases else [])
            code = extract_code(
                completion_text(completion),
                io_mode=meta.get("io_mode"),
                fn_name=meta.get("fn_name"),
                starter_code=meta.get("starter_code"),
            )
            if not code:
                scores.append(0.0)
                continue
            result = verify_code(
                code,
                meta,
                timeout=timeout,
                backend=backend,
                container_image=container_image,
            )
            scores.append(float(result.pass_rate) if not result.unsupported else 0.0)
        return scores

    def teaching_contract_reward(completions: list[Any], **_: Any) -> list[float]:
        return [contract_score(completion_text(item)) for item in completions]

    correctness_reward.__name__ = "correctness_reward"
    teaching_contract_reward.__name__ = "teaching_contract_reward"
    return correctness_reward, teaching_contract_reward
