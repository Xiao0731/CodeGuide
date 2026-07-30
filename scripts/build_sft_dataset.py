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
import ast
import asyncio
import io
import json
import logging
import os
import random
import re
import subprocess
import sys
import time
import tokenize
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


class FatalDistillError(RuntimeError):
    """Stop the batch when continuing would repeat a permanent API error."""


def is_fatal_distill_error(error: BaseException) -> bool:
    status_code = getattr(error, "status_code", None)
    text = str(error).lower()
    return status_code == 402 or "insufficient balance" in text

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

REFERENCE_LOCKED_TEACHER_USER_TMPL = """\
请围绕下面这道算法题和已经执行验证通过的参考解，生成适合初学者的完整中文教学回答。

这是 rejected 样本的可靠性恢复流程。参考解是不可变的代码真值，你只负责把它讲明白，不能重新实现算法。

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

【不可变参考解（左侧行号不属于代码）】
{reference_numbered}

【允许添加注释的安全行号】
{safe_comment_lines}

【输出要求】
1. 回答必须包含：题意重述、关键观察、分步推导、正确性说明、复杂度和常见错误。
2. 所有讲解都必须针对上面的参考解，不得换一种算法，不得虚构另一份实现。
3. 不要重新输出 Python 代码。程序会自动使用原始 reference。
4. 在回答末尾输出一个 ```json 代码块，内容必须是 JSON 数组；每项格式为
   {{"line": 安全行号, "comment": "放在该行前面的中文注释"}}。
5. 只使用“允许添加注释的安全行号”，建议提供 3 到 12 条有教学价值的注释。
6. comment 只能解释该行代码的目的、状态含义或设计原因，不得包含换行。
7. 不得使用 docstring，不得建议修改变量名、表达式、函数、类、参数、导入、缩进结构或入口。
8. call_based 必须保持 fn_name/starter_code 接口；standard_input 必须保持完整 stdin/stdout 程序。
9. 不要写演示测试。\
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


def _safe_comment_line_numbers(code: str) -> list[int]:
    """Return statement starts where a full-line comment can be inserted safely."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    lines: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.stmt) or not hasattr(node, "lineno"):
            continue
        line = int(node.lineno)
        decorators = getattr(node, "decorator_list", None) or []
        if decorators:
            line = min(line, *(int(item.lineno) for item in decorators))
        lines.add(line)
    return sorted(lines)


def _number_reference_lines(code: str) -> str:
    width = max(1, len(str(max(1, len(code.splitlines())))))
    return "\n".join(
        f"{line_no:>{width}} | {line}"
        for line_no, line in enumerate(code.splitlines(), 1)
    )


def build_teacher_user_content(
    problem: Problem,
    *,
    distill_mode: str,
    max_reference_chars: int,
    reference_locked: bool = False,
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

    if reference_locked:
        # Recovery must explain the exact verified implementation. Do not
        # truncate it and then ask the teacher to explain an unseen suffix.
        reference = problem.reference_solution or ""
        safe_lines = _safe_comment_line_numbers(reference)
        common_kwargs["reference_solution"] = reference
        common_kwargs["reference_numbered"] = _number_reference_lines(reference)
        common_kwargs["safe_comment_lines"] = ", ".join(map(str, safe_lines)) or "无"
        return REFERENCE_LOCKED_TEACHER_USER_TMPL.format(**common_kwargs)
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
    retry: int = 1,
    max_tokens: int = 8192,
    thinking_mode: str = "off",
    distill_mode: str = "scratch",
    reference_locked: bool = False,
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
        reference_locked=reference_locked,
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
                        "[%s] 触发 max_tokens 截断（attempt %d/%d）",
                        problem.id, attempt + 1, retry,
                    )
                    if attempt + 1 < retry:
                        await asyncio.sleep(2 ** attempt)
                    continue

                # 质量过滤
                quality_score = quality_checker.score(content)
                if quality_score < quality_checker.threshold:
                    logger.warning(
                        "[%s] 质量分 %.2f < %.2f（attempt %d/%d）",
                        problem.id, quality_score, quality_checker.threshold,
                        attempt + 1, retry,
                    )
                    if attempt + 1 < retry:
                        await asyncio.sleep(2 ** attempt)
                    continue

                return content

            except Exception as e:
                if is_fatal_distill_error(e):
                    logger.error(
                        "[%s] DeepSeek 余额不足，立即熔断当前批次；已落盘记录可断点恢复",
                        problem.id,
                    )
                    raise FatalDistillError(
                        "DeepSeek API insufficient balance"
                    ) from e
                if attempt + 1 < retry:
                    wait = 2 ** attempt
                    logger.warning(
                        "[%s] 调用失败（%s），%.1fs 后重试…",
                        problem.id,
                        e,
                        wait,
                    )
                    await asyncio.sleep(wait)
                else:
                    logger.warning("[%s] 单次调用失败（%s）", problem.id, e)

    logger.error("[%s] 已重试 %d 次，放弃", problem.id, retry)
    return None


# ── Reference-locked recovery helpers ─────────────────────────

_CODE_FENCE_RE = re.compile(
    r"```(?:python3?|py)?\s*\n(.*?)```",
    re.DOTALL | re.IGNORECASE,
)
_JSON_FENCE_RE = re.compile(
    r"```json\s*\n(.*?)```",
    re.DOTALL | re.IGNORECASE,
)
_IGNORED_EQUIVALENCE_TOKENS = {
    tokenize.ENCODING,
    tokenize.COMMENT,
    tokenize.NL,
    tokenize.ENDMARKER,
}


def _executable_token_signature(code: str) -> list[tuple[int, str]] | None:
    """Return all non-comment Python tokens while preserving executable structure."""
    try:
        tokens = tokenize.tokenize(io.BytesIO(code.encode("utf-8")).readline)
        return [
            (
                token.type,
                "\n" if token.type == tokenize.NEWLINE else token.string,
            )
            for token in tokens
            if token.type not in _IGNORED_EQUIVALENCE_TOKENS
        ]
    except (IndentationError, SyntaxError, tokenize.TokenError, UnicodeError):
        return None


def comments_only_equivalent(candidate: str, reference: str) -> bool:
    """Allow comments/physical formatting changes, but no executable token edits."""
    candidate_signature = _executable_token_signature(candidate)
    reference_signature = _executable_token_signature(reference)
    return bool(
        candidate_signature is not None
        and reference_signature is not None
        and candidate_signature == reference_signature
    )


def replace_final_code_block(text: str, code: str) -> str:
    """Replace the final fenced code block, or append one when the teacher omitted it."""
    matches = list(_CODE_FENCE_RE.finditer(text))
    replacement = f"```python\n{code.strip()}\n```"
    if not matches:
        return text.rstrip() + "\n\n" + replacement
    match = matches[-1]
    return text[: match.start()] + replacement + text[match.end() :]


def _extract_comment_plan(text: str) -> list[dict[str, Any]]:
    matches = list(_JSON_FENCE_RE.finditer(text))
    if not matches:
        return []
    try:
        plan = json.loads(matches[-1].group(1))
    except (json.JSONDecodeError, TypeError):
        return []
    return plan if isinstance(plan, list) else []


def _strip_final_json_plan(text: str) -> str:
    matches = list(_JSON_FENCE_RE.finditer(text))
    if not matches:
        return text.rstrip()
    match = matches[-1]
    return (text[: match.start()] + text[match.end() :]).rstrip()


def inject_reference_comments(
    reference: str,
    plan: list[dict[str, Any]],
    *,
    max_comments: int = 12,
) -> tuple[str, int]:
    """Insert model-authored comments only at AST statement boundaries."""
    safe_lines = set(_safe_comment_line_numbers(reference))
    comments_by_line: dict[int, list[str]] = {}
    accepted = 0
    for item in plan:
        if accepted >= max_comments or not isinstance(item, dict):
            break
        try:
            line_no = int(item.get("line"))
        except (TypeError, ValueError):
            continue
        if line_no not in safe_lines:
            continue
        comment = str(item.get("comment") or "").replace("\r", " ").replace("\n", " ")
        comment = re.sub(r"\s+", " ", comment).strip().lstrip("#").strip()
        if not comment:
            continue
        comment = comment[:240]
        comments_by_line.setdefault(line_no, []).append(comment)
        accepted += 1

    if not comments_by_line and safe_lines:
        first_line = min(safe_lines)
        comments_by_line[first_line] = [
            "以下实现保持已验证参考解的算法与接口不变。"
        ]
        accepted = 1

    source_lines = reference.splitlines(keepends=True)
    output: list[str] = []
    for line_no, source_line in enumerate(source_lines, 1):
        indent = source_line[: len(source_line) - len(source_line.lstrip(" \t"))]
        for comment in comments_by_line.get(line_no, []):
            output.append(f"{indent}# {comment}\n")
        output.append(source_line)
    annotated = "".join(output)
    if reference and not source_lines:
        annotated = reference

    # This is a defensive invariant; a comment-plan bug must never alter code.
    if not comments_only_equivalent(annotated, reference):
        return reference, 0
    return annotated, accepted


# ── ChatML 组装 ──────────────────────────────────────────────

def to_chatml(
    problem: Problem,
    assistant_content: str,
    pass_rate: float,
    *,
    distill_mode: str,
    max_reference_chars: int,
    label_strategy: str = "pedagogical_rewrite",
    recovery_metadata: dict[str, Any] | None = None,
) -> dict:
    """将题目 + 蒸馏模型输出组装为 ChatML 格式记录。"""
    metadata_for_reward = _io_metadata(problem)
    io_mode = metadata_for_reward["io_mode"]
    fn_name = metadata_for_reward.get("fn_name")
    test_cases = metadata_for_reward["test_cases"]
    reward_compatible = supports_verification(metadata_for_reward)
    reference_guided = distill_mode == "reference_guided_label"

    record = {
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
            "label_strategy": label_strategy,
        },
    }
    if recovery_metadata:
        record["metadata"].update(recovery_metadata)
    return record


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


def load_done_ids_many(paths: list[Path]) -> set[str]:
    done: set[str] = set()
    for path in paths:
        done.update(load_done_ids(path))
    return done


def load_latest_records(path: Path) -> dict[str, dict[str, Any]]:
    """Load the latest JSONL record for each id, tolerating interrupted tail lines."""
    records: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return records
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                record = json.loads(line)
            except (json.JSONDecodeError, TypeError):
                continue
            problem_id = record.get("id")
            if isinstance(problem_id, str):
                records[problem_id] = record
    return records


def compact_rejected_record(record: dict[str, Any]) -> dict[str, Any]:
    """Keep only fields needed to diagnose and resume a current rejection."""
    metadata = record.get("metadata") or {}
    metadata_keys = (
        "source",
        "difficulty",
        "io_mode",
        "fn_name",
        "distill_mode",
        "reference_guided",
        "reference_verified",
        "reference_pass_rate",
        "selected_reference_index",
        "recovery_attempted",
        "recovery_version",
        "initial_failure",
    )
    return {
        "id": record.get("id"),
        "failure_type": record.get("failure_type"),
        "pass_rate": record.get("pass_rate", 0.0),
        "error": record.get("error"),
        "first_failure": record.get("first_failure"),
        "metadata": {
            key: metadata.get(key)
            for key in metadata_keys
            if key in metadata
        },
    }


class CurrentRejectedStore:
    """Atomic JSONL snapshot containing only currently unresolved problem IDs."""

    def __init__(
        self,
        path: Path,
        records: dict[str, dict[str, Any]],
        accepted_ids: set[str],
    ) -> None:
        self.path = path
        self.records = {
            problem_id: compact_rejected_record(record)
            for problem_id, record in records.items()
            if problem_id not in accepted_ids
        }
        self.flush()

    def reject(self, record: dict[str, Any]) -> None:
        problem_id = str(record.get("id") or "")
        if not problem_id:
            return
        self.records[problem_id] = compact_rejected_record(record)
        self.flush()

    def resolve(self, problem_id: str) -> None:
        if self.records.pop(problem_id, None) is not None:
            self.flush()

    def flush(self) -> None:
        temp_path = self.path.with_suffix(
            f"{self.path.suffix}.{os.getpid()}.{time.time_ns()}.tmp"
        )
        try:
            with temp_path.open("w", encoding="utf-8", newline="\n") as fh:
                for record in self.records.values():
                    fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                fh.flush()
                os.fsync(fh.fileno())

            for attempt in range(6):
                try:
                    os.replace(temp_path, self.path)
                    break
                except PermissionError:
                    if attempt == 5:
                        raise
                    time.sleep(0.05 * (attempt + 1))
        finally:
            temp_path.unlink(missing_ok=True)


def is_recoverable_rejected_record(record: dict[str, Any]) -> bool:
    """Allow external teacher/API failures to resume after service recovery."""
    diagnostic = " ".join(
        str(record.get(key) or "") for key in ("error", "first_failure")
    ).lower()
    if record.get("failure_type") in {"recovery_llm_failed", "recovery_docker_unavailable"}:
        return True
    # Compatibility for records created before Docker connection failures had
    # their own failure type. These are infrastructure failures, not wrong
    # answers, and must remain resumable.
    if (
        "docker_engine" in diagnostic
        or "dockerdesktoplinuxengine" in diagnostic
        or "docker daemon" in diagnostic
        or "error during connect" in diagnostic
    ):
        return True
    metadata = record.get("metadata") or {}
    return (
        not bool(metadata.get("recovery_attempted", False))
        or int(metadata.get("recovery_version", 0) or 0) < 2
    )


def _short_text(value: Any, max_chars: int = 600) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if len(text) <= max_chars else text[:max_chars] + "..."


def classify_verification_failure(result: object) -> str:
    error = str(getattr(result, "error", "") or "").lower()
    first_failure = str(getattr(result, "first_failure", "") or "").lower()
    text = f"{error} {first_failure}"
    if (
        "docker_engine" in text
        or "dockerdesktoplinuxengine" in text
        or "docker daemon" in text
        or "error during connect" in text
    ):
        return "docker_unavailable"
    if getattr(result, "unsupported", False):
        if "container image must be pinned" in text:
            return "docker_unsupported"
        if "no top-level function" in text or "no callable method" in text or "callable method" in text:
            return "interface_mismatch"
        return "unsupported"
    if "timeout" in text or "timed out" in text:
        return "timeout"
    if "traceback" in text or "error:" in text or "exception" in text:
        return "runtime_error"
    return "wrong_answer"


def rejected_record(
    problem: Problem,
    failure_type: str,
    *,
    distill_mode: str,
    pass_rate: float = 0.0,
    error: Any = None,
    first_failure: Any = None,
    assistant_content: str | None = None,
    recovery_attempted: bool = False,
    initial_failure: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata = _io_metadata(problem)
    return {
        "id": problem.id,
        "failure_type": failure_type,
        "pass_rate": round(float(pass_rate or 0.0), 3),
        "error": _short_text(error),
        "first_failure": _short_text(first_failure),
        "assistant_content": assistant_content,
        "metadata": {
            "source": problem.source,
            "difficulty": problem.difficulty,
            "tags": list(problem.tags or []),
            "skill_types": list(problem.skill_types or []),
            "raw_tags": list(problem.raw_tags or []),
            "url": problem.url,
            "io_mode": metadata.get("io_mode"),
            "fn_name": metadata.get("fn_name"),
            "starter_code": metadata.get("starter_code"),
            "test_cases": metadata.get("test_cases") or [],
            "distill_mode": distill_mode,
            "reference_guided": distill_mode == "reference_guided_label",
            "reference_verified": bool(problem.metadata.get("reference_verified", False)),
            "reference_pass_rate": problem.metadata.get("reference_pass_rate"),
            "reference_error": problem.metadata.get("reference_error"),
            "reference_error_type": problem.metadata.get("reference_error_type"),
            "selected_reference_index": problem.metadata.get("selected_reference_index"),
            "selected_raw_solution_index": problem.metadata.get("selected_raw_solution_index"),
            "candidate_count": problem.metadata.get("candidate_count"),
            "attempted_candidates": problem.metadata.get("attempted_candidates"),
            "recovery_attempted": recovery_attempted,
            "recovery_version": 2 if recovery_attempted else 0,
            "initial_failure": initial_failure,
        },
    }


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


def stratified_sample_by_io_and_difficulty(
    problems: list[Problem],
    io_modes: list[str],
    difficulties: list[str],
    per_bucket: int,
    seed: int,
) -> list[Problem]:
    """Sample from each io_mode x difficulty bucket for pilot generation."""
    if not io_modes or not difficulties:
        return problems
    if per_bucket <= 0:
        raise ValueError("--per-io-difficulty must be positive")

    by_bucket: dict[tuple[str, str], list[Problem]] = {}
    for problem in problems:
        io_mode = str(_io_metadata(problem).get("io_mode") or "unknown")
        diff = _difficulty_key(problem)
        by_bucket.setdefault((io_mode, diff), []).append(problem)

    rng = random.Random(seed)
    selected: list[Problem] = []
    selected_ids: set[str] = set()
    for io_mode in io_modes:
        for raw_diff in difficulties:
            diff = raw_diff.lower()
            pool = by_bucket.get((io_mode, diff), [])
            if len(pool) < per_bucket:
                logger.warning(
                    "io_mode=%s difficulty=%s available samples are insufficient: need %d, got %d",
                    io_mode,
                    diff,
                    per_bucket,
                    len(pool),
                )
            take_n = min(per_bucket, len(pool))
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
        self.direct_saved = 0
        self.recovery_attempted = 0
        self.recovery_saved = 0
        self.final_rejected = 0

    def summary(self) -> str:
        return (
            f"总计处理: {self.total}\n"
            f"  ├ 断点跳过:   {self.skipped}\n"
            f"  ├ LLM 请求失败: {self.gpt_failed}\n"
            f"  ├ 质量过滤:   {self.low_quality}\n"
            f"  ├ 无代码块:   {self.no_code}\n"
            f"  ├ 语法错误:   {self.syntax_fail}\n"
            f"  ├ 执行失败:   {self.exec_fail}\n"
            f"  ├ A类直接写入: {self.direct_saved}\n"
            f"  ├ B类恢复尝试: {self.recovery_attempted}\n"
            f"  ├ B类恢复写入: {self.recovery_saved}\n"
            f"  ├ 最终 rejected: {self.final_rejected}\n"
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
    distill_retries: int,
    verification_timeout: float,
    run_code: bool,
    execution_backend: str,
    container_image: str,
    counter: Counter,
    out_file,
    rejected_store: CurrentRejectedStore,
    lock: asyncio.Lock,
    force_reference_locked: bool = False,
    previous_rejection: dict[str, Any] | None = None,
) -> None:
    counter.total += 1

    initial_failure = None
    if previous_rejection:
        initial_failure = {
            "failure_type": previous_rejection.get("failure_type"),
            "pass_rate": previous_rejection.get("pass_rate"),
            "error": previous_rejection.get("error"),
            "first_failure": previous_rejection.get("first_failure"),
        }

    # A class: retain the existing pedagogical rewrite path, but make only the
    # configured number of attempts. Any failure falls through to B class.
    if not force_reference_locked:
        cot = await call_distill_model_async(
            client,
            distill_model,
            problem,
            semaphore,
            quality_checker,
            retry=max(1, distill_retries),
            max_tokens=max_output_tokens,
            thinking_mode=thinking_mode,
            distill_mode=distill_mode,
            max_reference_chars=max_reference_chars,
            seed_examples=seed_examples,
            seed_examples_per_prompt=seed_examples_per_prompt,
        )
        code = extract_code(cot) if cot else None
        verification = None

        if cot is None:
            counter.gpt_failed += 1
            initial_failure = {"failure_type": "llm_failed", "pass_rate": 0.0}
        elif code is None:
            counter.no_code += 1
            initial_failure = {"failure_type": "no_code_block", "pass_rate": 0.0}
        else:
            syntax_ok, syntax_error = validate_syntax(code)
            if not syntax_ok:
                counter.syntax_fail += 1
                initial_failure = {
                    "failure_type": "syntax_error",
                    "pass_rate": 0.0,
                    "error": syntax_error,
                }
            elif run_code:
                verification = verify_code(
                    code,
                    _io_metadata(problem),
                    timeout=verification_timeout,
                    backend=execution_backend,
                    container_image=container_image or None,
                )
                if not _is_accepted_verification(verification):
                    counter.exec_fail += 1
                    initial_failure = {
                        "failure_type": classify_verification_failure(verification),
                        "pass_rate": verification.pass_rate,
                        "error": _short_text(verification.error),
                        "first_failure": _short_text(verification.first_failure),
                    }

        if initial_failure is None and cot is not None:
            pass_rate = verification.pass_rate if verification is not None else 0.0
            record = to_chatml(
                problem,
                cot,
                pass_rate,
                distill_mode=distill_mode,
                max_reference_chars=max_reference_chars,
                label_strategy="pedagogical_rewrite",
            )
            async with lock:
                out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_file.flush()
                rejected_store.resolve(problem.id)
                counter.saved += 1
                counter.direct_saved += 1
            return

    # B class: generate an explanation tied to the immutable verified
    # reference. The teacher's annotated code is used only when removing
    # comments yields exactly the same executable token stream.
    counter.recovery_attempted += 1
    reference = problem.reference_solution or ""
    locked_cot = None
    if reference:
        locked_cot = await call_distill_model_async(
            client,
            distill_model,
            problem,
            semaphore,
            quality_checker,
            retry=1,
            max_tokens=max_output_tokens,
            thinking_mode=thinking_mode,
            distill_mode=distill_mode,
            reference_locked=True,
            max_reference_chars=max_reference_chars,
            seed_examples=seed_examples,
            seed_examples_per_prompt=seed_examples_per_prompt,
        )

    if not reference or locked_cot is None:
        counter.gpt_failed += 1
        failure_type = "no_verified_reference" if not reference else "recovery_llm_failed"
        reject = rejected_record(
            problem,
            failure_type,
            distill_mode=distill_mode,
            assistant_content=locked_cot,
            recovery_attempted=True,
            initial_failure=initial_failure,
        )
        async with lock:
            rejected_store.reject(reject)
            counter.final_rejected += 1
        return

    comment_plan = _extract_comment_plan(locked_cot)
    final_code, inserted_comment_count = inject_reference_comments(
        reference,
        comment_plan,
    )
    comments_preserved = comments_only_equivalent(final_code, reference)
    explanation = _strip_final_json_plan(locked_cot)
    # Remove any teacher-authored Python block as well. The final block is
    # always assembled from the immutable reference plus safe comments.
    teacher_code = extract_code(explanation)
    if teacher_code is not None:
        explanation = replace_final_code_block(explanation, "").replace(
            "```python\n\n```",
            "",
        ).rstrip()
    final_content = explanation + f"\n\n```python\n{final_code.strip()}\n```"

    syntax_ok, syntax_error = validate_syntax(final_code)
    verification = None
    pass_rate = 0.0
    if syntax_ok and run_code:
        verification = verify_code(
            final_code,
            _io_metadata(problem),
            timeout=verification_timeout,
            backend=execution_backend,
            container_image=container_image or None,
        )
        pass_rate = verification.pass_rate

    recovery_ok = syntax_ok and (
        not run_code or (verification is not None and _is_accepted_verification(verification))
    )
    if not recovery_ok:
        counter.exec_fail += int(bool(syntax_ok))
        counter.syntax_fail += int(not syntax_ok)
        failure_type = (
            "recovery_syntax_error"
            if not syntax_ok
            else "recovery_" + classify_verification_failure(verification)
        )
        reject = rejected_record(
            problem,
            failure_type,
            distill_mode=distill_mode,
            pass_rate=pass_rate,
            error=syntax_error if not syntax_ok else verification.error,
            first_failure=None if verification is None else verification.first_failure,
            assistant_content=final_content,
            recovery_attempted=True,
            initial_failure=initial_failure,
        )
        async with lock:
            rejected_store.reject(reject)
            counter.final_rejected += 1
        return

    record = to_chatml(
        problem,
        final_content,
        pass_rate,
        distill_mode=distill_mode,
        max_reference_chars=max_reference_chars,
        label_strategy="reference_locked",
        recovery_metadata={
            "recovery_from_rejected": True,
            "initial_failure": initial_failure,
            "comment_only_equivalent": comments_preserved,
            "reference_code_injected": True,
            "comment_plan_count": inserted_comment_count,
        },
    )
    async with lock:
        out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        out_file.flush()
        rejected_store.resolve(problem.id)
        counter.saved += 1
        counter.recovery_saved += 1


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
    if args.run_code and args.execution_backend == "docker":
        try:
            docker_probe = subprocess.run(
                ["docker", "info", "--format", "{{.ServerVersion}}"],
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            logger.error("Docker verifier 预检失败，未发送任何 API 请求：%s", exc)
            sys.exit(1)
        if docker_probe.returncode != 0:
            diagnostic = (docker_probe.stderr or docker_probe.stdout or "").strip()
            logger.error(
                "Docker verifier 不可用，未发送任何 API 请求。请先启动 Docker Desktop：%s",
                diagnostic,
            )
            sys.exit(1)

    try:
        from openai import AsyncOpenAI
    except ImportError:
        logger.error("缺少 openai SDK：请先安装 openai，或运行 pip install -r requirements.txt")
        sys.exit(1)

    # Business-level retry policy is controlled by --distill-retries. Disable
    # the SDK's hidden transport retries so one attempt means one billed call.
    client = AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
        max_retries=0,
    )
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

    if args.stratified_io_modes:
        all_problems = stratified_sample_by_io_and_difficulty(
            all_problems,
            args.stratified_io_modes,
            args.stratified_difficulties,
            args.per_io_difficulty,
            args.stratified_seed,
        )
        logger.info(
            "io_mode x difficulty stratified sample size: %d, difficulty distribution: %s",
            len(all_problems),
            _difficulty_distribution(all_problems),
        )
    elif args.stratified_difficulties:
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

    # 2. Resume. Accepted ids are immutable. Older rejected ids are recovered
    # directly with the reference-locked path; they never pay for A class again.
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rejected_path = Path(args.rejected_out) if args.rejected_out else out_path.with_name(out_path.stem + "_rejected.jsonl")
    rejected_path.parent.mkdir(parents=True, exist_ok=True)
    accepted_ids = load_done_ids(out_path)
    rejected_records = load_latest_records(rejected_path)
    rejected_store = CurrentRejectedStore(
        rejected_path,
        rejected_records,
        accepted_ids,
    )
    rejected_records = dict(rejected_store.records)
    recoverable_rejected_ids = {
        problem_id
        for problem_id, record in rejected_records.items()
        if problem_id not in accepted_ids
        and is_recoverable_rejected_record(record)
    }
    permanently_rejected_ids = set(rejected_records) - recoverable_rejected_ids
    done_ids = accepted_ids | permanently_rejected_ids
    if done_ids:
        logger.info("断点续传：跳过已完成 %d 条", len(done_ids))
    if recoverable_rejected_ids:
        logger.info(
            "reference-locked recovery：已有 rejected 中 %d 条将直接进入 B 类",
            len(recoverable_rejected_ids),
        )

    pending = [p for p in all_problems if p.id not in done_ids]
    pending.sort(key=lambda problem: problem.id not in recoverable_rejected_ids)
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
        recovery_pending = [
            problem for problem in pending if problem.id in recoverable_rejected_ids
        ]
        fresh_pending = [
            problem for problem in pending if problem.id not in recoverable_rejected_ids
        ]
        phases = [
            ("B类恢复", recovery_pending),
            ("A类生成", fresh_pending),
        ]
        for phase_name, phase_problems in phases:
            if not phase_problems:
                continue
            logger.info("%s阶段开始：%d 条", phase_name, len(phase_problems))
            tasks = [
                process_one(
                    client,
                    distill_model,
                    problem,
                    semaphore,
                    quality_checker,
                    args.max_output_tokens,
                    args.thinking_mode,
                    args.distill_mode,
                    args.max_reference_chars,
                    seed_examples,
                    args.seed_examples_per_prompt,
                    args.distill_retries,
                    args.verification_timeout,
                    args.run_code,
                    args.execution_backend,
                    args.container_image,
                    counter,
                    out_file,
                    rejected_store,
                    lock,
                    problem.id in recoverable_rejected_ids,
                    rejected_records.get(problem.id),
                )
                for problem in phase_problems
            ]
            for coro in async_tqdm.as_completed(
                tasks,
                total=len(tasks),
                desc=phase_name,
            ):
                await coro

    logger.info("\n%s", counter.summary())
    logger.info("输出文件：%s", out_path)
    logger.info("rejected output file: %s", rejected_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="构建 SFT 全量数据集（异步并发）")
    parser.add_argument("--max_items",   type=int,  default=10_000,
                        help="最大题目数（默认 10000）")
    parser.add_argument("--source",      type=str,  default="taco",
                        choices=["code_contests", "taco", "both"], help="数据源选择")
    parser.add_argument("--taco-data-root", type=str, default="data/raw/TACO/ALL",
                        help="本地 TACO parquet 根目录（默认 data/raw/TACO/ALL）")
    parser.add_argument("--out",         type=str,  default="data/sft_train.jsonl")
    parser.add_argument("--rejected-out", type=str, default="",
                        help="Write rejected generations and failure reasons to this JSONL file")
    parser.add_argument("--concurrency", type=int,  default=10,
                        help="并发蒸馏模型请求数（建议 5-20，避免触发限速）")
    parser.add_argument("--max-output-tokens", type=int, default=8192,
                        help="单次蒸馏响应的最大输出 token 数（默认 8192，降低 hard 题截断概率）")
    parser.add_argument(
        "--distill-retries",
        type=int,
        default=1,
        help="A 类生成在截断、低质量或 API 异常时的最大尝试次数；默认 1，失败后转 B 类",
    )
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
    parser.add_argument("--stratified-io-modes", nargs="+", default=[],
                        choices=["standard_input", "call_based"],
                        help="Sample each io_mode x difficulty bucket; use with --stratified-difficulties")
    parser.add_argument("--per-io-difficulty", type=int, default=1,
                        help="Samples per io_mode x difficulty bucket")
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
