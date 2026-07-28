#!/usr/bin/env python3
"""
SFT 数据集全量构建（异步并发版）

流程：
  1. 加载 10k 题目池(load_problems)
  2. 断点续传：跳过 out_path 中已有的 ID
  3. 异步并发调用 OpenAI-compatible LLM distillation client(semaphore 控制并发数）
  4. 代码校验：语法检查 + 可选沙箱执行
  5. 组装 ChatML 格式，逐行写入 sft_train.jsonl

输出: data/sft_train.jsonl
每条格式(ChatML):
  {
    "id": str,
    "messages": [
      {"role": "system",    "content": "..."},
      {"role": "user",      "content": "题目描述"},
      {"role": "assistant", "content": "步骤讲解 + 代码"}
    ],
    "metadata": {
      "source": ...,
      "difficulty": ...,
      "tags": [...],
      "skill_types": [...],
      "raw_tags": [...],
      "test_cases": [...],
      "reward_compatible": bool,
      "pass_rate": ...
    }
  }

用法：
    python scripts/build_sft_dataset.py \\
        [--max_items 10000] \\
        [--concurrency 10] \\
        [--max-output-tokens 8192] \\
        [--distill-mode scratch|reference_guided_label|code_explanation] \\
        [--require-reference-solution] \\
        [--require-reward-compatible] \\
        [--seed-examples-per-prompt 1] \\
        [--stratified-difficulties easy medium hard very_hard --per-difficulty 1] \\
        [--run_code] \\
        [--out data/sft_train.jsonl]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Optional

# 确保项目根目录在 sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tqdm.asyncio import tqdm as async_tqdm

from src.data.code_validator import extract_code, validate_syntax
from src.data.loader import Problem, load_problems
from src.data.quality import DataQualityChecker
from src.reward.execution import supports_verification, verify_code

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── 提示词 ───────────────────────────────────────────────────

SFT_SYSTEM = """\
你是 CodeGuide，一位专为 OI/ACM 初学者设计的算法教学助手。
当用户提出一道算法题时，你会：
1. 先用简单的语言理解题意，举例说明
2. 从暴力解出发，逐步优化到最优解
3. 每一步都解释"为什么这样想"，而不只是"怎么做"
4. 最后给出带详细注释的完整 Python 代码

你的讲解应当通俗易懂，适合刚开始学习算法竞赛的初学者。\
"""

SFT_SYSTEM_PROMPT = """\
你是一位经验丰富的 OI/ACM 算法教练，专门面向初学者讲解算法题。
请按照以下格式，用中文逐步拆解解题思路。每一步都要给出推理过程，\
不要跳步骤，让初学者能完全跟上。

## 格式要求

**第一步：理解题意**
- 用自己的话复述题目，明确输入格式、输出格式、数据规模
- 举一个具体的小例子走一遍

**第二步：分析暴力解法**
- 给出最朴素的思路
- 分析其时间/空间复杂度，指出瓶颈在哪里

**第三步：寻找优化方向**
- 暴力的哪个步骤可以加速？
- 给出关键洞察（例如：单调性、数学性质、数据结构加速等）

**第四步：设计最优解**
- 详细推导最优算法的思路
- 给出关键伪代码片段（非完整代码，重点突出核心逻辑）

**第五步：完整 Python 实现**
- 给出可直接运行的 Python 代码
- 每个关键语句都要有注释，解释为什么这样写

**复杂度分析**
- 时间复杂度：O(...)，原因是...
- 空间复杂度：O(...)，原因是...

请严格遵守格式，代码必须放在 ```python ... ``` 代码块中。\
"""

STUDENT_USER_TMPL = """\
请按上述格式讲解以下算法题：

【题目描述】
{description}

【难度】{difficulty}
【标签】{tags}

【判题接口】
- io_mode: {io_mode}
- fn_name: {fn_name}

【starter_code】
```python
{starter_code}
```\
"""

CODE_EXPLANATION_STUDENT_USER_TMPL = """\
请按上述格式讲解以下算法题：

【题目描述】
{description}

【难度】{difficulty}
【标签】{tags}

【判题接口】
- io_mode: {io_mode}
- fn_name: {fn_name}

【starter_code】
```python
{starter_code}
```

【参考代码】
```python
{reference_solution}
```\
"""

SCRATCH_TEACHER_USER_TMPL = """\
请按上述格式讲解以下算法题。

你只能根据题目本身生成教学解答和最终代码。请特别注意接口契约：call_based 题必须严格实现给定 fn_name / starter_code，standard_input 题必须给出完整 stdin/stdout 程序。

【题目描述】
{description}

【难度】{difficulty}
【标签】{tags}

【判题接口】
- io_mode: {io_mode}
- fn_name: {fn_name}

【starter_code】
```python
{starter_code}
```

【测试样例摘要】
{test_summary}

{seed_examples}\
"""

REFERENCE_GUIDED_LABEL_TEACHER_USER_TMPL = """\
请按上述格式讲解以下算法题。

下面的参考解是 teacher-side privileged context：它只用于帮助你生成更可靠的 assistant 标签。最终训练样本的用户输入不会包含参考解，所以你的回答必须像是在直接解题，而不是“解释下面这段代码”。

【题目描述】
{description}

【难度】{difficulty}
【标签】{tags}

【判题接口】
- io_mode: {io_mode}
- fn_name: {fn_name}

【starter_code】
```python
{starter_code}
```

【测试样例摘要】
{test_summary}

{seed_examples}

【参考解代码】
```python
{reference_solution}
```

【重要要求】
1. 不要机械复述参考代码；请提炼算法思想、关键观察、推导过程、常见错误和复杂度。
2. 解题思路必须与参考解的正确算法保持一致，可以为了教学可读性重构变量名和代码结构，但不要改变算法含义。
3. 最终代码必须忠实遵循参考解的核心算法。可以提升可读性、拆分函数、增加注释，但不能把题目改写成相似但不同的常见问题，也不能替换成另一个看起来接近但语义不同的算法。
4. 最终代码必须放在 ```python ... ``` 代码块中。
5. 如果 io_mode == standard_input，最终代码必须是完整 stdin/stdout 程序，可以直接提交到 OJ。
6. 如果 io_mode == call_based，必须严格实现给定 fn_name 或 starter_code 中的函数/类接口，不得自行改名、改参数或改返回格式。
7. 不要在最终代码里写演示测试 main；只保留题目要求的提交代码。
8. 代码要有适量注释，解释关键设计原因，而不是逐行复述语句。\
"""

CODE_EXPLANATION_TEACHER_USER_TMPL = """\
请按上述格式讲解以下算法题，并解释参考代码如何解决它。

注意：这是辅助的 code_explanation 任务，训练样本的用户输入会看到参考代码；它不是主线 SFT 任务。

【题目描述】
{description}

【难度】{difficulty}
【标签】{tags}

【判题接口】
- io_mode: {io_mode}
- fn_name: {fn_name}

【starter_code】
```python
{starter_code}
```

【测试样例摘要】
{test_summary}

{seed_examples}

【参考代码】
```python
{reference_solution}
```

【重要要求】
1. 可以围绕参考代码讲解，但仍要输出完整教学解法，不要只做逐行注释。
2. 最终代码必须放在 ```python ... ``` 代码块中，并保持题目接口不变。
3. 如果 io_mode == standard_input，最终代码必须是完整 stdin/stdout 程序。
4. 如果 io_mode == call_based，必须严格实现给定 fn_name 或 starter_code 中的函数/类接口。\
"""


def _io_metadata(problem: Problem) -> dict:
    """Build the normalized metadata contract consumed by execution verifier."""
    input_output = problem.input_output if isinstance(problem.input_output, dict) else {}
    io_mode = "call_based" if input_output.get("fn_name") else "standard_input"
    return {
        "io_mode": io_mode,
        "fn_name": input_output.get("fn_name"),
        "starter_code": problem.starter_code,
        "test_cases": problem.public_tests or [],
        "tags": list(problem.tags or []),
        "skill_types": list(problem.skill_types or []),
        "raw_tags": list(problem.raw_tags or []),
    }


def _truncate_reference_solution(code: str, max_chars: int) -> str:
    """Keep prompts bounded while preserving the front of the reference code."""
    if max_chars <= 0 or len(code) <= max_chars:
        return code
    return (
        code[:max_chars]
        + "\n\n# [CodeGuide note] Reference solution truncated because it exceeds "
        + f"{max_chars} characters."
    )


def _sample_text(value: Any, max_chars: int = 240) -> str:
    text = json.dumps(value, ensure_ascii=False) if not isinstance(value, str) else value
    text = text.strip().replace("\r\n", "\n")
    return text if len(text) <= max_chars else text[:max_chars] + "..."


def _test_summary(problem: Problem, max_cases: int = 2) -> str:
    """Keep a compact, teacher-only view of tests in the prompt."""
    cases = problem.public_tests or []
    if not cases:
        return "暂无公开测试样例。"

    lines = [f"共有 {len(cases)} 个公开测试样例，下面展示前 {min(max_cases, len(cases))} 个："]
    for idx, case in enumerate(cases[:max_cases], 1):
        inp = case.get("input_args", case.get("input", ""))
        out = case.get("expected_output", case.get("output", ""))
        lines.append(
            f"- 样例 {idx}: input={_sample_text(inp)} | expected={_sample_text(out)}"
        )
    return "\n".join(lines)


def load_seed_examples(seed_dir: str | None) -> dict[str, list[dict[str, Any]]]:
    """Load APPS-style seed teaching examples, grouped by category filename."""
    if not seed_dir:
        return {}
    root = Path(seed_dir)
    if not root.exists():
        logger.warning("seed examples 目录不存在：%s", root)
        return {}

    examples: dict[str, list[dict[str, Any]]] = {}
    for path in root.glob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("读取 seed 文件失败：%s (%s)", path, exc)
            continue
        if not isinstance(payload, list):
            continue
        category = path.stem.lower()
        for item in payload:
            if isinstance(item, dict):
                item_category = str(item.get("category") or category).lower()
                examples.setdefault(item_category, []).append(item)
    return examples


def _problem_seed_categories(problem: Problem) -> list[str]:
    """Map TACO tags/skills to the local APPS seed category names."""
    aliases = {
        "arrays": "array",
        "array": "array",
        "strings": "string",
        "string": "string",
        "hashing": "hash",
        "hash table": "hash",
        "hash": "hash",
        "greedy algorithms": "greedy",
        "greedy": "greedy",
        "graphs": "graph",
        "graph": "graph",
        "dynamic programming": "dp",
        "dp": "dp",
        "depth-first search": "dfs_bfs",
        "breadth-first search": "dfs_bfs",
        "dfs": "dfs_bfs",
        "bfs": "dfs_bfs",
        "binary search": "binary_search",
        "two pointers": "two_pointers",
        "backtracking": "backtracking",
    }
    categories: list[str] = []
    for raw in [*problem.tags, *problem.raw_tags, *problem.skill_types]:
        key = str(raw).strip().lower()
        mapped = aliases.get(key)
        if mapped and mapped not in categories:
            categories.append(mapped)
    return categories


def _format_seed_examples(
    problem: Problem,
    seed_examples: dict[str, list[dict[str, Any]]],
    count: int,
) -> str:
    """Format a few compact teaching exemplars for the teacher prompt only."""
    if count <= 0 or not seed_examples:
        return ""

    selected: list[dict[str, Any]] = []
    for category in _problem_seed_categories(problem):
        for item in seed_examples.get(category, []):
            selected.append(item)
            if len(selected) >= count:
                break
        if len(selected) >= count:
            break

    if not selected:
        return ""

    chunks = ["【教学风格参考 seed examples】"]
    for idx, item in enumerate(selected, 1):
        answer = item.get("teaching_answer") or {}
        plan = answer.get("step_by_step_plan") or []
        mistakes = answer.get("common_mistakes") or []
        chunks.append(
            "\n".join([
                f"示例 {idx}（category={item.get('category', 'unknown')}）:",
                f"- problem: {_sample_text(item.get('problem_statement', ''), 300)}",
                f"- restatement: {_sample_text(answer.get('problem_restatement', ''), 220)}",
                f"- key_observation: {_sample_text(answer.get('key_observation', ''), 260)}",
                f"- plan: {_sample_text(plan, 300)}",
                f"- common_mistakes: {_sample_text(mistakes, 260)}",
            ])
        )
    return "\n\n".join(chunks)


def build_student_user_content(
    problem: Problem,
    *,
    distill_mode: str,
    max_reference_chars: int,
) -> str:
    """Create the user message that will be seen during SFT training."""
    tags_str = ", ".join(problem.tags) if problem.tags else "暂无"
    metadata = _io_metadata(problem)
    template = (
        CODE_EXPLANATION_STUDENT_USER_TMPL
        if distill_mode == "code_explanation"
        else STUDENT_USER_TMPL
    )
    kwargs = {
        "description": problem.description,
        "difficulty": problem.difficulty,
        "tags": tags_str,
        "io_mode": metadata["io_mode"],
        "fn_name": metadata.get("fn_name") or "无",
        "starter_code": problem.starter_code or "",
        "reference_solution": _truncate_reference_solution(
            problem.reference_solution or "",
            max_reference_chars,
        ),
    }
    return template.format(**kwargs)


def build_teacher_user_content(
    problem: Problem,
    *,
    distill_mode: str,
    max_reference_chars: int,
    seed_examples: dict[str, list[dict[str, Any]]] | None = None,
    seed_examples_per_prompt: int = 0,
) -> str:
    """Create the privileged teacher prompt for label distillation."""
    tags_str = ", ".join(problem.tags) if problem.tags else "暂无"
    metadata = _io_metadata(problem)
    seed_text = _format_seed_examples(
        problem,
        seed_examples or {},
        seed_examples_per_prompt,
    )
    common_kwargs = {
        "description": problem.description,
        "difficulty": problem.difficulty,
        "tags": tags_str,
        "io_mode": metadata["io_mode"],
        "fn_name": metadata.get("fn_name") or "无",
        "starter_code": problem.starter_code or "",
        "test_summary": _test_summary(problem),
        "seed_examples": seed_text,
        "reference_solution": _truncate_reference_solution(
            problem.reference_solution or "",
            max_reference_chars,
        ),
    }

    if distill_mode == "reference_guided_label" and problem.reference_solution:
        return REFERENCE_GUIDED_LABEL_TEACHER_USER_TMPL.format(**common_kwargs)
    if distill_mode == "code_explanation" and problem.reference_solution:
        return CODE_EXPLANATION_TEACHER_USER_TMPL.format(**common_kwargs)

    return SCRATCH_TEACHER_USER_TMPL.format(
        description=problem.description,
        difficulty=problem.difficulty,
        tags=tags_str,
        io_mode=metadata["io_mode"],
        fn_name=metadata.get("fn_name") or "无",
        starter_code=problem.starter_code or "",
        test_summary=_test_summary(problem),
        seed_examples=seed_text,
    )


# ── OpenAI-compatible LLM distillation client ─────────────────

async def call_distill_model_async(
    client: object,
    distill_model: str,
    problem: Problem,
    semaphore: asyncio.Semaphore,
    quality_checker: "DataQualityChecker",
    retry: int = 3,
    max_tokens: int = 8192,
    thinking_mode: str = "off",
    distill_mode: str = "scratch",
    max_reference_chars: int = 12000,
    seed_examples: dict[str, list[dict[str, Any]]] | None = None,
    seed_examples_per_prompt: int = 0,
) -> Optional[str]:
    """
    异步调用 OpenAI-compatible 蒸馏模型，semaphore 限制并发，失败时指数退避重试。

    改进（v2）：
    - max_tokens 默认提升到 8192，减少 hard/very_hard 长讲解截断概率
    - 检测 finish_reason == "length" 时自动重试（截断重试）
    - 对返回内容进行质量检测，分数 < 0.6 视为低质量，触发重试
    - 重试时提示词末尾加"请确保完整输出代码块"，引导模型补全
    """
    user_content = build_teacher_user_content(
        problem,
        distill_mode=distill_mode,
        max_reference_chars=max_reference_chars,
        seed_examples=seed_examples,
        seed_examples_per_prompt=seed_examples_per_prompt,
    )
    base_messages = [
        {"role": "system", "content": SFT_SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]

    async with semaphore:
        for attempt in range(retry):
            # 截断重试时在用户消息末尾追加补充指令
            if attempt > 0:
                retry_suffix = "\n\n【重要】请确保输出完整的代码块（```python ... ```），不要中途截断。"
                messages = [
                    base_messages[0],
                    {"role": "user", "content": user_content + retry_suffix},
                ]
            else:
                messages = base_messages

            try:
                request_kwargs = {
                    "model": distill_model,
                    "messages": messages,
                    "temperature": 0.3,
                    "max_tokens": max_tokens,
                }
                # DeepSeek V4 exposes thinking as an OpenAI-compatible
                # extra_body field. It defaults to enabled server-side, so
                # distillation explicitly disables it unless requested.
                if thinking_mode != "omit":
                    request_kwargs["extra_body"] = {
                        "thinking": {
                            "type": "enabled" if thinking_mode == "on" else "disabled"
                        }
                    }
                resp = await client.chat.completions.create(**request_kwargs)
                choice = resp.choices[0]
                content = choice.message.content or ""

                # 检测截断（finish_reason == "length"）
                if choice.finish_reason == "length":
                    logger.warning(
                        "[%s] 触发 max_tokens 截断（attempt %d/%d），重试…",
                        problem.id, attempt + 1, retry,
                    )
                    await asyncio.sleep(2 ** attempt)
                    continue

                # 质量过滤
                quality_score = quality_checker.score(content)
                if quality_score < quality_checker.threshold:
                    logger.warning(
                        "[%s] 质量分 %.2f < %.2f（attempt %d/%d），重试…",
                        problem.id, quality_score, quality_checker.threshold,
                        attempt + 1, retry,
                    )
                    await asyncio.sleep(2 ** attempt)
                    continue

                return content

            except Exception as e:
                wait = 2 ** attempt
                logger.warning(
                    "[%s] 调用失败（%s），%.1fs 后重试…", problem.id, e, wait
                )
                await asyncio.sleep(wait)

    logger.error("[%s] 已重试 %d 次，放弃", problem.id, retry)
    return None


# ── ChatML 组装 ──────────────────────────────────────────────

def to_chatml(
    problem: Problem,
    assistant_content: str,
    pass_rate: float,
    *,
    distill_mode: str,
    max_reference_chars: int,
) -> dict:
    """将题目 + 蒸馏模型输出组装为 ChatML 格式记录。"""
    metadata_for_reward = _io_metadata(problem)
    io_mode = metadata_for_reward["io_mode"]
    fn_name = metadata_for_reward.get("fn_name")
    test_cases = metadata_for_reward["test_cases"]
    reward_compatible = supports_verification(metadata_for_reward)
    reference_guided = distill_mode == "reference_guided_label"

    return {
        "id": problem.id,
        "messages": [
            {"role": "system",    "content": SFT_SYSTEM},
            {
                "role": "user",
                "content": build_student_user_content(
                    problem,
                    distill_mode=distill_mode,
                    max_reference_chars=max_reference_chars,
                ),
            },
            {"role": "assistant", "content": assistant_content},
        ],
        "metadata": {
            "source":            problem.source,
            "difficulty":        problem.difficulty,
            "tags":              list(problem.tags or []),
            "skill_types":       list(problem.skill_types or []),
            "raw_tags":          list(problem.raw_tags or []),
            "url":               problem.url,
            "io_mode":           io_mode,
            "fn_name":           fn_name,
            "starter_code":      problem.starter_code,
            "test_cases":        test_cases,
            "reward_compatible": reward_compatible,
            "pass_rate":         round(pass_rate, 3),
            "distill_mode":      distill_mode,
            "reference_guided":  reference_guided,
            # These are reserved for a future offline reference-verification
            # cache. We intentionally do not store the full reference solution
            # in final training messages.
            "reference_verified": bool(problem.metadata.get("reference_verified", False)),
            "reference_pass_rate": problem.metadata.get("reference_pass_rate"),
            "reference_error": problem.metadata.get("reference_error"),
            "reference_error_type": problem.metadata.get("reference_error_type"),
            "selected_reference_index": problem.metadata.get("selected_reference_index"),
            "selected_raw_solution_index": problem.metadata.get("selected_raw_solution_index"),
            "candidate_count": problem.metadata.get("candidate_count"),
            "attempted_candidates": problem.metadata.get("attempted_candidates"),
        },
    }


# ── 断点续传：加载已完成的 ID ─────────────────────────────────

def load_done_ids(out_path: Path) -> set[str]:
    done: set[str] = set()
    if not out_path.exists():
        return done
    with open(out_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                done.add(record["id"])
            except Exception:
                pass
    return done


def load_reference_cache(cache_path: str | None) -> dict[str, dict[str, Any]]:
    """Load offline TACO reference verification records by problem id."""
    if not cache_path:
        return {}
    path = Path(cache_path)
    if not path.exists():
        logger.warning("reference cache 不存在：%s", path)
        return {}

    cache: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            problem_id = record.get("id")
            if isinstance(problem_id, str):
                cache[problem_id] = record
    logger.info("已读取 reference cache：%d 条 (%s)", len(cache), path)
    return cache


def apply_reference_cache(
    problems: list[Problem],
    cache: dict[str, dict[str, Any]],
) -> None:
    """Merge offline reference verification fields into problem.metadata."""
    if not cache:
        return
    merged = 0
    for problem in problems:
        record = cache.get(problem.id)
        if not record:
            continue
        problem.metadata["reference_verified"] = bool(record.get("reference_verified", False))
        problem.metadata["reference_pass_rate"] = record.get("reference_pass_rate")
        problem.metadata["reference_error"] = record.get("reference_error")
        problem.metadata["reference_error_type"] = record.get("reference_error_type")
        if record.get("selected_reference_index") is not None:
            problem.metadata["selected_reference_index"] = record.get("selected_reference_index")
            try:
                selected_rank = int(record.get("selected_reference_index"))
            except (TypeError, ValueError):
                selected_rank = -1
            for candidate in problem.reference_candidates or []:
                try:
                    candidate_rank = int(candidate.get("rank", -1))
                except (TypeError, ValueError):
                    candidate_rank = -1
                if candidate_rank == selected_rank:
                    problem.reference_solution = str(candidate.get("code") or problem.reference_solution or "")
                    break
        if record.get("selected_raw_solution_index") is not None:
            problem.metadata["selected_raw_solution_index"] = record.get("selected_raw_solution_index")
        problem.metadata["candidate_count"] = record.get("candidate_count")
        problem.metadata["attempted_candidates"] = record.get("attempted_candidates")
        merged += 1
    logger.info("reference cache 命中并合并：%d/%d 条", merged, len(problems))


def _has_verified_reference(problem: Problem, min_pass_rate: float) -> bool:
    if not problem.reference_solution:
        return False
    if not problem.metadata.get("reference_verified"):
        return False
    pass_rate = problem.metadata.get("reference_pass_rate")
    try:
        return float(pass_rate) >= min_pass_rate
    except (TypeError, ValueError):
        return False


def _difficulty_key(problem: Problem) -> str:
    return str(problem.difficulty or "unknown_difficulty").lower()


def _difficulty_distribution(problems: list[Problem]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for problem in problems:
        key = _difficulty_key(problem)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def stratified_sample_by_difficulty(
    problems: list[Problem],
    difficulties: list[str],
    per_difficulty: int,
    seed: int,
) -> list[Problem]:
    """
    Smoke/pilot sampling helper.

    Formal full-scale distillation keeps the loader order and max_items behavior.
    This helper is only activated by --stratified-difficulties so pilot runs can
    cover easy and hard buckets in the same tiny batch.
    """
    if not difficulties:
        return problems
    if per_difficulty <= 0:
        raise ValueError("--per-difficulty 必须为正整数")

    by_difficulty: dict[str, list[Problem]] = {}
    for problem in problems:
        by_difficulty.setdefault(_difficulty_key(problem), []).append(problem)

    rng = random.Random(seed)
    selected: list[Problem] = []
    selected_ids: set[str] = set()
    for raw_diff in difficulties:
        diff = raw_diff.lower()
        pool = by_difficulty.get(diff, [])
        if len(pool) < per_difficulty:
            logger.warning(
                "difficulty=%s 可用样本不足：需要 %d，实际 %d",
                diff, per_difficulty, len(pool),
            )
        take_n = min(per_difficulty, len(pool))
        sampled = rng.sample(pool, take_n) if len(pool) > take_n else list(pool)
        for problem in sampled:
            if problem.id not in selected_ids:
                selected.append(problem)
                selected_ids.add(problem.id)
    return selected


def sample_problems(
    problems: list[Problem],
    sample_size: int,
    seed: int,
) -> list[Problem]:
    """Generic smoke sampling after all cache/compatibility filters."""
    if sample_size <= 0 or len(problems) <= sample_size:
        return problems
    rng = random.Random(seed)
    return rng.sample(problems, sample_size)


# ── 统计计数器（线程安全替代品，asyncio 单线程无锁） ───────────

class Counter:
    def __init__(self):
        self.total = 0
        self.skipped = 0
        self.gpt_failed = 0
        self.low_quality = 0   # 新增：质量过滤丢弃数
        self.no_code = 0
        self.syntax_fail = 0
        self.exec_fail = 0
        self.saved = 0

    def summary(self) -> str:
        return (
            f"总计处理: {self.total}\n"
            f"  ├ 断点跳过:   {self.skipped}\n"
            f"  ├ LLM 请求失败: {self.gpt_failed}\n"
            f"  ├ 质量过滤:   {self.low_quality}\n"
            f"  ├ 无代码块:   {self.no_code}\n"
            f"  ├ 语法错误:   {self.syntax_fail}\n"
            f"  ├ 执行失败:   {self.exec_fail}\n"
            f"  └ 最终写入:   {self.saved}"
        )


def _is_accepted_verification(verification: object) -> bool:
    """Accepted SFT labels must pass every executable test fail-closed."""
    return bool(
        not getattr(verification, "unsupported", True)
        and not getattr(verification, "error", None)
        and int(getattr(verification, "total_cases", 0)) > 0
        and float(getattr(verification, "pass_rate", 0.0)) >= 1.0
    )


# ── 单条异步处理 ─────────────────────────────────────────────

async def process_one(
    client: object,
    distill_model: str,
    problem: Problem,
    semaphore: asyncio.Semaphore,
    quality_checker: DataQualityChecker,
    max_output_tokens: int,
    thinking_mode: str,
    distill_mode: str,
    max_reference_chars: int,
    seed_examples: dict[str, list[dict[str, Any]]],
    seed_examples_per_prompt: int,
    verification_timeout: float,
    run_code: bool,
    execution_backend: str,
    container_image: str,
    counter: Counter,
    out_file,
    lock: asyncio.Lock,
) -> None:
    counter.total += 1

    # 蒸馏模型标注（含截断重试 + 质量过滤）
    cot = await call_distill_model_async(
        client,
        distill_model,
        problem,
        semaphore,
        quality_checker,
        max_tokens=max_output_tokens,
        thinking_mode=thinking_mode,
        distill_mode=distill_mode,
        max_reference_chars=max_reference_chars,
        seed_examples=seed_examples,
        seed_examples_per_prompt=seed_examples_per_prompt,
    )
    if cot is None:
        counter.gpt_failed += 1
        return

    # Code extraction, syntax, and full execution are hard gates for accepted
    # SFT labels. A fluent explanation with failing code is a wrong label.
    code = extract_code(cot)
    if code is None:
        counter.no_code += 1
        return

    syntax_ok, syntax_error = validate_syntax(code)
    if not syntax_ok:
        counter.syntax_fail += 1
        return

    pass_rate = 0.0
    if run_code:
        verification = verify_code(
            code,
            _io_metadata(problem),
            timeout=verification_timeout,
            backend=execution_backend,
            container_image=container_image or None,
        )
        pass_rate = verification.pass_rate
        if not _is_accepted_verification(verification):
            counter.exec_fail += 1
            return

    # 组装 ChatML
    record = to_chatml(
        problem,
        cot,
        pass_rate,
        distill_mode=distill_mode,
        max_reference_chars=max_reference_chars,
    )

    # 原子写入（asyncio 单线程，lock 防止多条记录交错）
    async with lock:
        out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        out_file.flush()
        counter.saved += 1


# ── 主流程 ───────────────────────────────────────────────────

async def async_main(args: argparse.Namespace) -> None:
    # Distillation uses an OpenAI-compatible chat-completions endpoint. Keeping
    # these env vars explicit avoids accidentally sending a large job to the
    # wrong provider or model.
    api_key = os.environ.get("DISTILL_API_KEY")
    base_url = os.environ.get("DISTILL_BASE_URL")
    distill_model = os.environ.get("DISTILL_MODEL")
    if not api_key:
        logger.error("DISTILL_API_KEY 未设置")
        sys.exit(1)
    if not base_url:
        logger.error("DISTILL_BASE_URL 未设置")
        sys.exit(1)
    if not distill_model:
        logger.error("DISTILL_MODEL 未设置")
        sys.exit(1)
    if not args.run_code and not args.allow_unverified_output:
        logger.error(
            "正式输出必须执行验证。若只做 prompt/格式诊断，请同时显式传入 "
            "--no-run-code --allow-unverified-output，并使用独立 smoke 输出路径。"
        )
        sys.exit(1)

    try:
        from openai import AsyncOpenAI
    except ImportError:
        logger.error("缺少 openai SDK：请先安装 openai，或运行 pip install -r requirements.txt")
        sys.exit(1)

    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    logger.info("蒸馏模型：%s (%s)", distill_model, base_url)
    logger.info(
        "生成配置：max_output_tokens=%d, thinking_mode=%s, distill_mode=%s",
        args.max_output_tokens,
        args.thinking_mode,
        args.distill_mode,
    )
    seed_examples = load_seed_examples(args.seed_dir)
    if args.seed_examples_per_prompt > 0:
        logger.info(
            "teacher prompt seed examples：dir=%s, categories=%d, per_prompt=%d",
            args.seed_dir,
            len(seed_examples),
            args.seed_examples_per_prompt,
        )

    # 1. 加载题库
    logger.info("加载题库（source=%s, max=%d）…", args.source, args.max_items)
    all_problems = load_problems(
        source=args.source,
        max_items=args.max_items,
        taco_data_root=args.taco_data_root,
    )
    logger.info("题库大小：%d 条", len(all_problems))
    logger.info("题库 difficulty 分布：%s", _difficulty_distribution(all_problems))
    logger.info(
        "题库参考解可用：%d/%d",
        sum(1 for p in all_problems if p.reference_solution),
        len(all_problems),
    )

    reference_cache = load_reference_cache(args.reference_cache)
    apply_reference_cache(all_problems, reference_cache)

    if args.require_reference_solution:
        before = len(all_problems)
        all_problems = [p for p in all_problems if p.reference_solution]
        logger.info(
            "已启用 --require-reference-solution：过滤 %d → %d 条",
            before,
            len(all_problems),
        )

    if args.require_verified_reference:
        if not reference_cache:
            logger.error("--require-verified-reference 需要提供有效 --reference-cache")
            sys.exit(1)
        before = len(all_problems)
        all_problems = [
            p for p in all_problems
            if _has_verified_reference(p, args.min_reference_pass_rate)
        ]
        logger.info(
            "已启用 --require-verified-reference：过滤 %d → %d 条（min_reference_pass_rate=%.3f）",
            before,
            len(all_problems),
            args.min_reference_pass_rate,
        )

    if args.require_reward_compatible:
        before = len(all_problems)
        all_problems = [p for p in all_problems if supports_verification(_io_metadata(p))]
        logger.info(
            "已启用 --require-reward-compatible：过滤 %d → %d 条",
            before,
            len(all_problems),
        )

    if args.io_mode_filter != "any":
        before = len(all_problems)
        all_problems = [
            p for p in all_problems
            if _io_metadata(p)["io_mode"] == args.io_mode_filter
        ]
        logger.info(
            "已启用 --io-mode-filter=%s：过滤 %d → %d 条",
            args.io_mode_filter,
            before,
            len(all_problems),
        )

    if args.stratified_difficulties:
        all_problems = stratified_sample_by_difficulty(
            all_problems,
            args.stratified_difficulties,
            args.per_difficulty,
            args.stratified_seed,
        )
        logger.info(
            "分层抽样后题库大小：%d 条，difficulty 分布：%s",
            len(all_problems),
            _difficulty_distribution(all_problems),
        )

    if args.sample_size > 0:
        before = len(all_problems)
        all_problems = sample_problems(
            all_problems,
            args.sample_size,
            args.sample_seed,
        )
        logger.info(
            "随机抽样后题库大小：%d → %d 条（sample_seed=%d）",
            before,
            len(all_problems),
            args.sample_seed,
        )

    # 2. 断点续传
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done_ids = load_done_ids(out_path)
    if done_ids:
        logger.info("断点续传：跳过已完成 %d 条", len(done_ids))

    pending = [p for p in all_problems if p.id not in done_ids]
    logger.info("待处理：%d 条，并发数：%d", len(pending), args.concurrency)
    logger.info("待处理 difficulty 分布：%s", _difficulty_distribution(pending))
    logger.info(
        "待处理参考解可用：%d/%d",
        sum(1 for p in pending if p.reference_solution),
        len(pending),
    )

    if not pending:
        logger.info("所有题目已处理完毕！")
        return

    # 3. 异步并发处理
    semaphore = asyncio.Semaphore(args.concurrency)
    lock = asyncio.Lock()
    counter = Counter()
    counter.skipped = len(done_ids)
    quality_checker = DataQualityChecker(threshold=args.quality_threshold)
    logger.info("质量过滤阈值：%.2f（低于此分数的样本触发重试或丢弃）",
                args.quality_threshold)

    with open(out_path, "a", encoding="utf-8") as out_file:
        tasks = [
            process_one(
                client,
                distill_model,
                p,
                semaphore,
                quality_checker,
                args.max_output_tokens,
                args.thinking_mode,
                args.distill_mode,
                args.max_reference_chars,
                seed_examples,
                args.seed_examples_per_prompt,
                args.verification_timeout,
                args.run_code,
                args.execution_backend,
                args.container_image,
                counter,
                out_file,
                lock,
            )
            for p in pending
        ]
        # tqdm 进度条
        for coro in async_tqdm.as_completed(tasks, total=len(tasks), desc="蒸馏进度"):
            await coro

    logger.info("\n%s", counter.summary())
    logger.info("输出文件：%s", out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="构建 SFT 全量数据集（异步并发）")
    parser.add_argument("--max_items",   type=int,  default=10_000,
                        help="最大题目数（默认 10000）")
    parser.add_argument("--source",      type=str,  default="taco",
                        choices=["code_contests", "taco", "both"], help="数据源选择")
    parser.add_argument("--taco-data-root", type=str, default="data/raw/TACO/ALL",
                        help="本地 TACO parquet 根目录（默认 data/raw/TACO/ALL）")
    parser.add_argument("--out",         type=str,  default="data/sft_train.jsonl")
    parser.add_argument("--concurrency", type=int,  default=10,
                        help="并发蒸馏模型请求数（建议 5-20，避免触发限速）")
    parser.add_argument("--max-output-tokens", type=int, default=8192,
                        help="单次蒸馏响应的最大输出 token 数（默认 8192，降低 hard 题截断概率）")
    parser.add_argument("--thinking-mode", choices=["off", "on", "omit"], default="off",
                        help="OpenAI-compatible 额外 thinking 开关：off 默认关闭，on 开启，omit 不传该字段")
    parser.add_argument(
        "--distill-mode",
        choices=["scratch", "reference_guided_label", "code_explanation"],
        default="scratch",
        help=(
            "蒸馏模式：scratch=teacher/student 都不看参考解；"
            "reference_guided_label=teacher 看参考解、student 不看参考解（主线）；"
            "code_explanation=student 也看参考代码（辅助任务）"
        ),
    )
    parser.add_argument("--max-reference-chars", type=int, default=12000,
                        help="teacher prompt 或 code_explanation 用户消息中最多放入多少字符参考解（默认 12000）")
    parser.add_argument("--seed-dir", type=str, default="data/seeds",
                        help="APPS-style seed teaching examples 目录（默认 data/seeds）")
    parser.add_argument("--seed-examples-per-prompt", type=int, default=0,
                        help="每个 teacher prompt 放入多少条同类 seed 教学示例；0 表示不放（默认 0）")
    parser.add_argument("--require-reference-solution", action="store_true",
                        help="仅保留有 TACO 参考解的题；推荐用于 reference_guided_label/code_explanation smoke/pilot")
    parser.add_argument("--reference-cache", type=str, default="",
                        help="离线 reference 验证缓存 JSONL；由 scripts/verify_taco_references.py 生成")
    parser.add_argument("--require-verified-reference", action="store_true",
                        help="仅保留 reference cache 中已验证通过且达到 --min-reference-pass-rate 的题")
    parser.add_argument("--min-reference-pass-rate", type=float, default=1.0,
                        help="--require-verified-reference 的最低 reference_pass_rate（默认 1.0）")
    parser.add_argument("--require-reward-compatible", action="store_true",
                        help="仅保留当前 verifier 支持且有测试用例的题；推荐用于 pass_rate smoke/pilot")
    parser.add_argument("--io-mode-filter", choices=["any", "standard_input", "call_based"], default="any",
                        help="按验证接口类型过滤题目；call_based 专项 smoke 可设为 call_based")
    parser.add_argument("--stratified-difficulties", nargs="+", default=[],
                        help="按 difficulty 分层抽样，仅用于 smoke/pilot，例如 easy medium hard very_hard")
    parser.add_argument("--per-difficulty", type=int, default=1,
                        help="分层抽样时每个 difficulty 抽取多少条（默认 1）")
    parser.add_argument("--stratified-seed", type=int, default=42,
                        help="分层抽样随机种子（默认 42）")
    parser.add_argument("--sample-size", type=int, default=0,
                        help="所有过滤和可选分层后随机抽取 N 条；0 表示不启用")
    parser.add_argument("--sample-seed", type=int, default=42,
                        help="--sample-size 的随机种子（默认 42）")
    parser.add_argument(
        "--run-code",
        "--run_code",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="执行并硬过滤未全量通过测试的 teacher 标签（默认启用）",
    )
    parser.add_argument(
        "--allow-unverified-output",
        action="store_true",
        help="仅供独立 smoke 诊断；允许 --no-run-code 写出未验证标签",
    )
    parser.add_argument(
        "--execution-backend",
        choices=["docker", "subprocess"],
        default="docker",
        help="正式生成必须使用 docker；subprocess 仅供受信任的本地测试",
    )
    parser.add_argument(
        "--container-image",
        default=os.environ.get("CODEGUIDE_EXECUTION_IMAGE", ""),
        help="Docker Python 镜像，必须固定为 name@sha256:...；也可由环境变量提供",
    )
    parser.add_argument("--verification-timeout", type=float, default=8.0,
                        help="--run_code 时单条样本 verifier 超时时间秒数（默认 8.0）")
    parser.add_argument("--quality_threshold", type=float, default=0.6,
                        help="蒸馏数据质量过滤阈值 [0,1]（默认 0.6，低于此值触发重试/丢弃）")
    args = parser.parse_args()

    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
