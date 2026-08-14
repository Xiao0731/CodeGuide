"""Deterministic teaching-contract reward used by GRPO and offline evaluation."""

from __future__ import annotations

import re
from typing import Any

_SECTION = re.compile(
    r"(?:^|\n)(?:#{1,3}\s*)?(?:第[一二三四五六七八九十\d]+步|"
    r"题意(?:理解|重述)?|关键(?:观察|思路)|算法(?:步骤|推导)|正确性|复杂度|常见错误)"
    r"[^\n]*\n(.*?)(?=\n(?:#{1,3}\s*)?(?:第[一二三四五六七八九十\d]+步|"
    r"题意(?:理解|重述)?|关键(?:观察|思路)|算法(?:步骤|推导)|正确性|复杂度|常见错误)|\Z)",
    re.DOTALL | re.MULTILINE,
)
_CODE_BLOCK = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
_COMPLEXITY = re.compile(r"(?:时间|空间)?复杂度|O\([^)]+\)", re.IGNORECASE)
_TEACHING_WORDS = re.compile(
    r"因为|所以|因此|注意|关键|核心|观察|为什么|暴力|优化|边界|举例|正确性"
)


def _bigrams(text: str) -> set[str]:
    compact = re.sub(r"\s+", "", text)
    return {compact[index : index + 2] for index in range(max(0, len(compact) - 1))}


def _similarity(left: str, right: str) -> float:
    a, b = _bigrams(left), _bigrams(right)
    return len(a & b) / len(a | b) if a and b else 0.0


def contract_score(text: str) -> float:
    """Score substantive teaching structure, runnable code, and complexity in [0, 1]."""
    if not isinstance(text, str) or not text.strip():
        return 0.0

    sections = [match.group(1).strip() for match in _SECTION.finditer(text)]
    substantive: list[str] = []
    for section in sections:
        if len(section) >= 30 and all(_similarity(section, old) < 0.7 for old in substantive):
            substantive.append(section)
    section_score = min(len(substantive) / 4, 1.0) * 0.4

    code_score = 0.0
    for block in _CODE_BLOCK.findall(text):
        real_lines = [
            line
            for line in block.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        if len(real_lines) >= 3:
            code_score = 0.3
            break
        if real_lines:
            code_score = max(code_score, 0.1)

    complexity_score = 0.2 if _COMPLEXITY.search(text) else 0.0
    teaching_score = min(len(_TEACHING_WORDS.findall(text)) / 5, 1.0) * 0.1
    return round(
        min(section_score + code_score + complexity_score + teaching_score, 1.0), 6
    )


def _score_format(text: str) -> float:
    """Backward-compatible name for older reports and tests."""
    return contract_score(text)


class FormatComplianceReward:
    def __init__(self, cfg: Any = None):
        self.cfg = cfg

    def __call__(self, prompts: list[str], completions: list[str]) -> list[float]:
        del prompts
        return [contract_score(completion) for completion in completions]
