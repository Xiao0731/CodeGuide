"""Deterministic teaching-contract score used by formal GRPO."""

from __future__ import annotations

import re
from typing import Any

_STEP_PATTERNS = (
    re.compile(
        r"(第[一二三四五六七八九十\d]+步[：:])(.+?)"
        r"(?=第[一二三四五六七八九十\d]+步|##|$)",
        re.DOTALL,
    ),
    re.compile(r"(\*\*Step \d+[：:]\*\*)(.+?)(?=\*\*Step \d+|##|$)", re.DOTALL),
    re.compile(r"(#{1,3} .+?\n)(.+?)(?=#{1,3} |$)", re.DOTALL),
    re.compile(r"(\d+\.\s+\*\*.+?\*\*)(.+?)(?=\d+\.\s+\*\*|$)", re.DOTALL),
)
_PYTHON_BLOCK = re.compile(r"```(?:python|python3|py)\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
_COMPLEXITY = re.compile(r"O\([^)]+\)|时间复杂度|空间复杂度|Time complexity", re.IGNORECASE)
_TEACHING_WORDS = re.compile(
    r"因为|所以|因此|注意|关键|核心思路|观察到|我们需要|可以发现|"
    r"暴力|优化|考虑|首先|然后|最后|总结|举例"
)

MIN_STEP_CONTENT = 50
MAX_STEP_SIMILARITY = 0.7


def _bigrams(text: str) -> set[str]:
    compact = re.sub(r"\s+", "", text)
    return {compact[index : index + 2] for index in range(max(0, len(compact) - 1))}


def _jaccard_similarity(left: str, right: str) -> float:
    left_set, right_set = _bigrams(left), _bigrams(right)
    if not left_set or not right_set:
        return 0.0
    return len(left_set & right_set) / len(left_set | right_set)


def _score_steps(text: str) -> float:
    candidates: list[str] = []
    for pattern in _STEP_PATTERNS:
        for match in pattern.finditer(text):
            content = match.group(2).strip()
            if len(content) >= MIN_STEP_CONTENT:
                candidates.append(content)

    unique: list[str] = []
    for candidate in candidates:
        if all(
            _jaccard_similarity(candidate, previous) < MAX_STEP_SIMILARITY
            for previous in unique
        ):
            unique.append(candidate)
    return min(len(unique) / 3.0, 1.0) * 0.40


def _score_code_block(text: str) -> float:
    best_real_lines = 0
    for block in _PYTHON_BLOCK.findall(text):
        real_lines = sum(
            1
            for line in block.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
        best_real_lines = max(best_real_lines, real_lines)
    if best_real_lines >= 3:
        return 0.30
    return 0.10 if best_real_lines else 0.0


def _score_complexity(text: str) -> float:
    return 0.20 if _COMPLEXITY.search(text) else 0.0


def _score_length(text: str) -> float:
    return 0.05 if 150 <= len(text) <= 3000 else 0.0


def _score_teaching_words(text: str) -> float:
    return min(len(_TEACHING_WORDS.findall(text)) / 5.0, 1.0) * 0.05


def contract_score(text: str) -> float:
    """Return the formal structural contract score in [0, 1]."""
    if not isinstance(text, str) or not text.strip():
        return 0.0
    score = (
        _score_steps(text)
        + _score_code_block(text)
        + _score_complexity(text)
        + _score_length(text)
        + _score_teaching_words(text)
    )
    return round(min(score, 1.0), 6)


def _score_format(text: str) -> float:
    """Backward-compatible alias for reports that use the old name."""
    return contract_score(text)


class FormatComplianceReward:
    def __init__(self, cfg: Any = None):
        self.cfg = cfg

    def __call__(self, prompts: list[str], completions: list[str]) -> list[float]:
        del prompts
        return [contract_score(completion) for completion in completions]
