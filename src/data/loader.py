"""
数据集加载与归一化

当前项目默认使用本地 TACO parquet，对外暴露统一的 Problem 结构。
仍保留 code_contests 加载器，便于旧评测脚本兼容。

用法：
    from src.data.loader import load_problems
    problems = load_problems(source="taco", split="train", max_items=10000)
    python -m src.data.loader --source taco --limit 3
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Literal, Optional

logger = logging.getLogger(__name__)

# TACO official data may contain very large integer literals in input_output.
# This is an offline trusted-data loading path, so keep json.loads integer
# semantics instead of using parse_int=str and allow those integers to parse.
if hasattr(sys, "set_int_max_str_digits"):
    sys.set_int_max_str_digits(0)

# ── 难度常量 ────────────────────────────────────────────────

# code_contests 的 difficulty 字段是 int（枚举值）
# 文档：1=UNKNOWN, 2=EASY, 3=MEDIUM, 4=HARD, 5=HARDER, 6=HARDEST
_CC_EASY_MEDIUM = {2, 3}  # EASY / MEDIUM

_DEFAULT_TACO_DATA_ROOT = Path("data/raw/TACO/ALL")


# ── 数据结构 ─────────────────────────────────────────────────

@dataclass
class Problem:
    id: str                          # 全局唯一 ID（由来源+题目 hash 生成）
    source: str                      # "code_contests" | "taco"
    description: str                 # 题目描述（原始文本，已去 HTML）
    difficulty: str                  # "easy" | "medium" | "hard" | ...
    tags: List[str] = field(default_factory=list)
    # 参考答案（来自数据集自带，用于代码校验时构建测试输入）
    reference_solution: Optional[str] = None
    # 原始测试用例（input/output 字符串对）
    public_tests: List[dict] = field(default_factory=list)

    # TACO 原始/规范化字段。保留下来供蒸馏、评测和后续 reward 使用。
    # raw_tags：原始平台标签，TACO从原网站或原始数据源里带来的标签，可能包括题目主题、解法技巧、平台特性
    # tags：TACO 对原始标签进行清洗，归并为36个算法标签
    # skill_types：比tags粗的能力维度，8类核心编程技能标签
    question: str = ""
    solutions: List[Any] = field(default_factory=list)
    # Ordered Python-looking reference candidates. Each item stores at least:
    # {"code": str, "raw_index": int, "priority": int, "reason": str}.
    reference_candidates: List[dict] = field(default_factory=list)
    starter_code: str = ""
    input_output: dict = field(default_factory=dict)
    raw_input_output: str = ""
    raw_tags: List[str] = field(default_factory=list)
    skill_types: List[str] = field(default_factory=list)
    name: str = ""
    url: str = ""
    time_limit: Optional[Any] = None
    memory_limit: Optional[Any] = None
    expected_time_complexity: str = ""
    expected_auxiliary_space: str = ""
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "source": self.source,
            "description": self.description,
            "difficulty": self.difficulty,
            "tags": self.tags,
            "reference_solution": self.reference_solution,
            "public_tests": self.public_tests,
            "question": self.question or self.description,
            "solutions": self.solutions,
            "reference_candidates": self.reference_candidates,
            "starter_code": self.starter_code,
            "input_output": self.input_output,
            "raw_input_output": self.raw_input_output,
            "raw_tags": self.raw_tags,
            "skill_types": self.skill_types,
            "name": self.name,
            "url": self.url,
            "time_limit": self.time_limit,
            "memory_limit": self.memory_limit,
            "Expected Time Complexity": self.expected_time_complexity,
            "Expected Auxiliary Space": self.expected_auxiliary_space,
            "metadata": self.metadata,
        }


# ── 工具函数 ─────────────────────────────────────────────────

def _make_id(source: str, text: str) -> str:
    h = hashlib.md5(text.encode(), usedforsecurity=False).hexdigest()[:10]
    return f"{source}_{h}"


def _strip_html(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"&lt;", "<", text)
    text = re.sub(r"&gt;", ">", text)
    text = re.sub(r"&nbsp;", " ", text)
    return text.strip()


def _pick_python_solution(solutions: dict | list) -> Optional[str]:
    """
    从 code_contests 的 solutions 字段取一段 Python 参考实现。
    solutions 格式为 {"language": [3, 3], "solution": ["code1", "code2"]}
    language 枚举：3 = PYTHON3
    """
    if isinstance(solutions, dict):
        langs = solutions.get("language", [])
        codes = solutions.get("solution", [])
        for lang, code in zip(langs, codes):
            if lang == 3:  # PYTHON3
                return code
    return None


def _parse_json_field(value: Any, default: Any) -> Any:
    """Parse TACO JSON string fields while preserving already-parsed values."""
    if value is None:
        return default
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return default
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return default
    return value


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        parsed = _parse_list_string(value)
        if isinstance(parsed, list):
            return parsed
        if value.strip():
            return [value]
        return []
    return [value]


def _parse_list_string(value: str) -> Any:
    text = value.strip()
    if not text:
        return []

    try:
        # 将形如"['Geometry', 'Sorting']"的字符串转为真正的列表
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        pass

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _normalize_solution_list(value: Any) -> List[str]:
    """Return TACO solutions as plain code strings.

    TACO stores `solutions` as a JSON-encoded list of Python snippets in the
    local parquet files. This helper also accepts already-decoded lists or
    dict-like variants so reference-guided distillation is not tied to one
    parquet serialization detail.
    """
    if isinstance(value, dict):
        value = value.get("solution") or value.get("solutions") or value.get("code") or []

    solutions: List[str] = []
    for item in _as_list(value):
        if isinstance(item, dict):
            item = item.get("code") or item.get("solution") or item.get("text") or ""
        code = str(item or "").strip()
        if code:
            solutions.append(code)
    return solutions


def _looks_like_python_solution(code: str) -> bool:
    """Conservatively reject obvious non-Python snippets before prompting."""
    lowered = code.lower()
    non_python_markers = (
        "#include",
        "using namespace",
        "int main(",
        "public static void main",
        "class main",
        "console.log(",
        "function ",
        "package main",
    )
    if any(marker in lowered for marker in non_python_markers):
        return False

    python_markers = (
        "def ",
        "class Solution",
        "import ",
        "from ",
        "input(",
        "sys.stdin",
        "print(",
        "lambda ",
    )
    return any(marker in code for marker in python_markers)


def _rank_taco_python_solutions(
    raw_solutions: Any,
    *,
    fn_name: str | None = None,
    io_mode: str | None = None,
) -> List[dict]:
    """Return Python-looking TACO solutions ordered by expected compatibility.

    Ordering policy:
    1. call_based: exact top-level `def fn_name`, then `class Solution`, then
       other Python-like snippets.
    2. standard_input: snippets that both read stdin and write stdout, then
       stdin-looking snippets, then other Python-like snippets.
    3. Preserve original order within each priority bucket.
    """
    exact_fn_pattern = (
        re.compile(rf"\bdef\s+{re.escape(fn_name)}\s*\(")
        if fn_name
        else None
    )

    ranked: List[dict] = []
    for raw_index, code in enumerate(_normalize_solution_list(raw_solutions)):
        if not _looks_like_python_solution(code):
            continue
        priority = 10
        reason = "python_like"
        lowered = code.lower()

        if io_mode == "call_based":
            if exact_fn_pattern and exact_fn_pattern.search(code):
                priority = 0
                reason = "exact_fn_name"
            elif "class Solution" in code:
                priority = 1
                reason = "class_solution"
            else:
                priority = 2
                reason = "python_like_call_based"
        elif io_mode == "standard_input":
            reads_stdin = (
                "input(" in code
                or "sys.stdin" in code
                or "stdin.readline" in code
                or ".readline()" in code
            )
            writes_stdout = (
                "print(" in code
                or "sys.stdout" in code
                or "stdout.write" in lowered
            )
            if reads_stdin and writes_stdout:
                priority = 0
                reason = "stdin_stdout_program"
            elif reads_stdin:
                priority = 1
                reason = "stdin_program"
            else:
                priority = 2
                reason = "python_like_standard_input"

        ranked.append({
            "code": code,
            "raw_index": raw_index,
            "priority": priority,
            "reason": reason,
        })

    ranked.sort(key=lambda item: (int(item["priority"]), int(item["raw_index"])))
    for rank, item in enumerate(ranked):
        item["rank"] = rank
    return ranked


def _pick_taco_python_solution(
    raw_solutions: Any,
    *,
    fn_name: str | None = None,
    io_mode: str | None = None,
) -> tuple[Optional[str], Optional[int]]:
    """Pick the first ranked TACO Python reference candidate."""
    candidates = _rank_taco_python_solutions(
        raw_solutions,
        fn_name=fn_name,
        io_mode=io_mode,
    )
    if not candidates:
        return None, None
    return candidates[0]["code"], candidates[0]["rank"]


def _load_dataset(*args, **kwargs):
    from datasets import load_dataset

    return load_dataset(*args, **kwargs)


def _normalize_taco_split(split: str) -> str:
    split = (split or "train").lower()
    if split in {"valid", "validation", "eval", "dev"}:
        return "test"
    if split not in {"train", "test"}:
        raise ValueError(f"TACO 本地 parquet 仅支持 train/test，收到 split={split!r}")
    return split


def _resolve_taco_files(data_root: str | Path | None = None) -> dict[str, str]:
    root = Path(data_root or _DEFAULT_TACO_DATA_ROOT)
    return {
        "train": str(root / "train-*.parquet"),
        "test": str(root / "test-*.parquet"),
    }


def _load_taco_dataset(
    split: str = "train",
    data_root: str | Path | None = None,
):
    """
    优先读取本地 TACO parquet。

    默认路径：
      data/raw/TACO/ALL/train-*.parquet
      data/raw/TACO/ALL/test-*.parquet

    本地文件不存在时直接报错，避免误走远程数据集或旧 dataset script。
    """
    normalized_split = _normalize_taco_split(split)
    files = _resolve_taco_files(data_root)
    root = Path(data_root or _DEFAULT_TACO_DATA_ROOT)

    has_local = any(root.glob("train-*.parquet")) and any(root.glob("test-*.parquet"))
    if not has_local:
        raise FileNotFoundError(
            f"未找到本地 TACO parquet：{root}。"
            "请传入 data_root，或放置 train-*.parquet/test-*.parquet。"
        )

    logger.info("加载本地 TACO parquet：%s", root)
    ds_dict = _load_dataset("parquet", data_files=files)
    return ds_dict[normalized_split]


def _load_taco_dataset_dict(data_root: str | Path | None = None):
    """Load local TACO train/test splits for smoke inspection."""
    files = _resolve_taco_files(data_root)
    root = Path(data_root or _DEFAULT_TACO_DATA_ROOT)
    if not any(root.glob("train-*.parquet")) or not any(root.glob("test-*.parquet")):
        raise FileNotFoundError(f"未找到完整本地 TACO parquet：{root}")
    return _load_dataset("parquet", data_files=files)


def _taco_public_tests(input_output: dict, *, source: str = "") -> List[dict]:
    """
    Convert TACO input_output into a conservative test case list.

    For standard-input tasks, input/output stay as strings and can be consumed by
    the current Accuracy Reward. For call-based tasks, fn_name/input_args and
    expected_output are preserved for later reward/eval support.
    """
    if not isinstance(input_output, dict):
        return []

    inputs = input_output.get("inputs") or []
    outputs = input_output.get("outputs") or []
    fn_name = input_output.get("fn_name")

    if not isinstance(inputs, list) or not isinstance(outputs, list):
        return []

    tests: List[dict] = []
    source_key = str(source or "").lower()
    for inp, out in zip(inputs, outputs):
        expected = out
        # Codewars exports each expected return value wrapped in a one-element
        # list. Unwrap only this source so LeetCode-style list returns remain
        # valid Python list expectations.
        if fn_name and source_key == "codewars" and isinstance(out, list) and len(out) == 1:
            expected = out[0]
        tc = {
            "input": inp if isinstance(inp, str) else json.dumps(inp, ensure_ascii=False),
            "output": out if isinstance(out, str) else json.dumps(out, ensure_ascii=False),
        }
        if fn_name:
            tc["fn_name"] = fn_name
            tc["input_args"] = inp if isinstance(inp, list) else [inp]
            tc["expected_output"] = expected
        tests.append(tc)
    return tests


# ── code_contests 加载器 ─────────────────────────────────────

def _load_code_contests(split: str, max_items: int) -> List[Problem]:
    logger.info("加载 deepmind/code_contests (%s)…", split)
    try:
        ds = _load_dataset("deepmind/code_contests", split=split, trust_remote_code=True)
    except Exception as e:
        logger.error("code_contests 加载失败：%s", e)
        return []

    problems: List[Problem] = []
    for row in ds:
        if len(problems) >= max_items:
            break

        diff_int = row.get("difficulty", 0)
        if diff_int not in _CC_EASY_MEDIUM:
            continue

        desc = _strip_html(row.get("description", ""))
        if len(desc) < 80:  # 过短说明数据有问题
            continue

        ref = _pick_python_solution(row.get("solutions", {}))

        # 公开测试用例
        pub = row.get("public_tests", {})
        tests = [
            {"input": inp, "output": out}
            for inp, out in zip(
                pub.get("input", []), pub.get("output", [])
            )
        ]

        diff_str = "easy" if diff_int == 2 else "medium"
        problems.append(Problem(
            id=_make_id("cc", desc),
            source="code_contests",
            description=desc,
            difficulty=diff_str,
            tags=[],
            reference_solution=ref,
            public_tests=tests,
        ))

    logger.info("code_contests 筛选后：%d 条", len(problems))
    return problems


# ── TACO 加载器 ──────────────────────────────────────────────

def _load_taco(
    split: str,
    max_items: int,
    data_root: str | Path | None = None,
) -> List[Problem]:
    logger.info("加载 TACO (%s)…", split)
    try:
        ds = _load_taco_dataset(split=split, data_root=data_root)
    except Exception as e:
        logger.error("TACO 加载失败：%s", e)
        return []

    problems: List[Problem] = []
    for row in ds:
        if len(problems) >= max_items:
            break

        desc = _strip_html(row.get("question", ""))
        if len(desc) < 80:
            continue

        raw_input_output = row.get("input_output", "")
        input_output = _parse_json_field(raw_input_output, {})
        if not isinstance(input_output, dict):
            input_output = {}
        io_mode = "call_based" if input_output.get("fn_name") else "standard_input"
        fn_name = input_output.get("fn_name")

        raw_solutions = row.get("solutions", "[]")
        solutions = _normalize_solution_list(raw_solutions)
        reference_candidates = _rank_taco_python_solutions(
            raw_solutions,
            fn_name=fn_name if isinstance(fn_name, str) else None,
            io_mode=io_mode,
        )
        ref, selected_reference_index = _pick_taco_python_solution(
            raw_solutions,
            fn_name=fn_name if isinstance(fn_name, str) else None,
            io_mode=io_mode,
        )
        selected_raw_solution_index = (
            reference_candidates[0].get("raw_index")
            if reference_candidates
            else None
        )

        source = str(row.get("source") or "taco")
        tags = [str(t) for t in _as_list(row.get("tags"))]
        raw_tags = [str(t) for t in _as_list(row.get("raw_tags"))]
        skill_types = [str(t) for t in _as_list(row.get("skill_types"))]
        public_tests = _taco_public_tests(input_output, source=source)
        diff = str(row.get("difficulty") or "unknown").lower()

        metadata = {
            "source": source,
            "difficulty": diff,
            "tags": tags,
            "raw_tags": raw_tags,
            "skill_types": skill_types,
            "url": row.get("url") or "",
            "time_limit": row.get("time_limit"),
            "memory_limit": row.get("memory_limit"),
            "Expected Time Complexity": row.get("Expected Time Complexity") or "",
            "Expected Auxiliary Space": row.get("Expected Auxiliary Space") or "",
            "input_output": input_output,
            "test_cases": public_tests,
            "starter_code": row.get("starter_code") or "",
            "name": row.get("name") or "",
            "selected_reference_index": selected_reference_index,
            "selected_raw_solution_index": selected_raw_solution_index,
            "reference_candidate_count": len(reference_candidates),
            "reference_verified": False,
            "reference_pass_rate": None,
        }

        problems.append(Problem(
            id=_make_id("taco", desc),
            source=source,
            description=desc,
            difficulty=diff,
            tags=tags,
            reference_solution=ref,
            public_tests=public_tests,
            question=desc,
            solutions=solutions,
            reference_candidates=reference_candidates,
            starter_code=row.get("starter_code") or "",
            input_output=input_output,
            raw_input_output=raw_input_output if isinstance(raw_input_output, str) else "",
            raw_tags=raw_tags,
            skill_types=skill_types,
            name=row.get("name") or "",
            url=row.get("url") or "",
            time_limit=row.get("time_limit"),
            memory_limit=row.get("memory_limit"),
            expected_time_complexity=row.get("Expected Time Complexity") or "",
            expected_auxiliary_space=row.get("Expected Auxiliary Space") or "",
            metadata=metadata,
        ))

    logger.info("TACO 筛选后：%d 条", len(problems))
    return problems


# ── 公共接口 ─────────────────────────────────────────────────

def load_problems(
    source: Literal["code_contests", "taco", "both"] = "taco",
    split: str = "train",
    max_items: int = 10_000,
    deduplicate: bool = True,
    data_root: str | Path | None = None,
    taco_data_root: str | Path | None = None,
) -> List[Problem]:
    """
    加载并归一化题目数据集。

    Args:
        source:      数据来源；当前项目默认只使用 "taco"
        split:       HF split 名称（code_contests 有 train/valid/test；TACO 只有 train）
        max_items:   每个来源最多加载多少条（过滤后）
        deduplicate: 按题目描述前 200 字去重
        data_root:   TACO 本地 parquet 根目录（推荐）
        taco_data_root: data_root 的别名，便于上层配置命名
    Returns:
        List[Problem]
    """
    problems: List[Problem] = []

    if source in ("code_contests", "both"):
        problems += _load_code_contests(split, max_items)

    if source in ("taco", "both"):
        remaining = max(0, max_items - len(problems))
        if remaining > 0:
            problems += _load_taco(split, remaining, data_root=taco_data_root or data_root)

    if deduplicate:
        seen: set[str] = set()
        deduped: List[Problem] = []
        for p in problems:
            key = p.description[:200]
            if key not in seen:
                seen.add(key)
                deduped.append(p)
        logger.info("去重后：%d → %d 条", len(problems), len(deduped))
        problems = deduped

    return problems[:max_items]


def _inspect_taco(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    ds_dict = _load_taco_dataset_dict(args.taco_data_root)
    print(f"TACO data_root: {Path(args.taco_data_root or _DEFAULT_TACO_DATA_ROOT)}")
    print(f"train size: {len(ds_dict['train'])}")
    print(f"test size: {len(ds_dict['test'])}")

    problems = load_problems(
        source="taco",
        split=args.split,
        max_items=args.limit,
        deduplicate=False,
        taco_data_root=args.taco_data_root,
    )
    print(f"loaded problems ({args.split}, limit={args.limit}): {len(problems)}")

    for idx, problem in enumerate(problems[: args.limit], 1):
        io = problem.input_output
        print(f"\n[{idx}] id={problem.id}")
        print(f"  difficulty: {problem.difficulty}")
        print(f"  tags: {problem.tags[:8]}")
        print(f"  raw_tags: {problem.raw_tags[:8]}")
        print(f"  skill_types: {problem.skill_types[:8]}")
        print(f"  source: {problem.source}")
        print(f"  url: {problem.url}")
        print(f"  time_limit: {problem.time_limit}")
        print(f"  memory_limit: {problem.memory_limit}")
        print(f"  Expected Time Complexity: {problem.expected_time_complexity}")
        print(f"  Expected Auxiliary Space: {problem.expected_auxiliary_space}")
        print(f"  input_output parsed: {isinstance(io, dict) and bool(io)}")
        print(f"  input_output keys: {sorted(io.keys()) if isinstance(io, dict) else []}")
        print(f"  inputs: {len(io.get('inputs', [])) if isinstance(io, dict) else 0}")
        print(f"  outputs: {len(io.get('outputs', [])) if isinstance(io, dict) else 0}")
        print(f"  fn_name: {io.get('fn_name') if isinstance(io, dict) else None}")
        print(f"  public_tests: {len(problem.public_tests)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect local TACO loader")
    parser.add_argument("--source", choices=["taco"], default="taco")
    parser.add_argument("--split", choices=["train", "test", "valid", "validation", "eval"], default="train")
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--taco-data-root", default=str(_DEFAULT_TACO_DATA_ROOT))
    args = parser.parse_args()
    _inspect_taco(args)


if __name__ == "__main__":
    main()
