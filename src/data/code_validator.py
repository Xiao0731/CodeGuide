"""
代码提取与校验

功能：
1. extract_code()       — 从 LLM 输出中选择最符合题目接口的 Python 代码块
2. validate_syntax()    — ast.parse() 语法检查（快，无副作用）
3. validate_execution() — 沙箱子进程执行 + 测试用例比对（可选）
4. validate()           — 组合校验，返回 ValidationResult
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

# 匹配 ```python ... ``` 或 ``` ... ```（宽松）
_PYTHON_FENCE_RE = re.compile(
    r"^[ \t]*```(?:python3?|py)[ \t]*\r?\n(.*?)^[ \t]*```[ \t]*$",
    re.DOTALL | re.IGNORECASE | re.MULTILINE,
)
_UNTAGGED_FENCE_RE = re.compile(
    r"^[ \t]*```[ \t]*\r?\n(.*?)^[ \t]*```[ \t]*$",
    re.DOTALL | re.MULTILINE,
)

# 执行超时（秒）
_EXEC_TIMEOUT = 8


# ── 数据结构 ─────────────────────────────────────────────────

@dataclass
class ValidationResult:
    ok: bool
    code: Optional[str]      # 提取到的代码，None 表示未找到代码块
    error: Optional[str]     # 失败原因（供调试）
    pass_rate: float = 0.0   # 测试用例通过率（0.0 ~ 1.0）


# ── 代码提取 ─────────────────────────────────────────────────

def _code_candidates(text: str) -> list[tuple[str, bool]]:
    candidates: list[tuple[str, bool]] = []
    lines = text.splitlines()
    index = 0
    while index < len(lines):
        opening = re.fullmatch(r"[ \t]*```([^`]*)[ \t]*", lines[index])
        if not opening:
            index += 1
            continue
        language = opening.group(1).strip().lower()
        end = index + 1
        while end < len(lines) and not re.fullmatch(r"[ \t]*```[ \t]*", lines[end]):
            end += 1
        if end >= len(lines):
            break
        if language in ("", "python", "python3", "py"):
            candidates.append(("\n".join(lines[index + 1:end]).strip(), bool(language)))
        index = end + 1
    return candidates


def _defined_functions(tree: ast.AST) -> tuple[set[str], set[str]]:
    top_level: set[str] = set()
    solution_methods: set[str] = set()
    for node in getattr(tree, "body", []):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            top_level.add(node.name)
        elif isinstance(node, ast.ClassDef) and node.name == "Solution":
            solution_methods.update(
                child.name
                for child in node.body
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
            )
    return top_level, solution_methods


def _is_example_only(tree: ast.AST) -> bool:
    body = getattr(tree, "body", [])
    if not body:
        return True
    meaningful = [node for node in body if not isinstance(node, (ast.Import, ast.ImportFrom))]
    return bool(meaningful) and all(
        isinstance(node, ast.Expr)
        or (
            isinstance(node, ast.Assign)
            and isinstance(node.value, (ast.Constant, ast.List, ast.Tuple, ast.Dict, ast.Set))
        )
        for node in meaningful
    )


def _candidate_score(
    code: str,
    *,
    tagged_python: bool,
    io_mode: Optional[str],
    fn_name: Optional[str],
) -> int:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return -10_000

    score = 5 + int(tagged_python)
    top_level, solution_methods = _defined_functions(tree)
    example_only = _is_example_only(tree)
    if io_mode == "call_based":
        score += 200 * int(bool(fn_name and fn_name in top_level))
        score += 190 * int(bool(fn_name and fn_name in solution_methods))
        score += 30 * int(bool(top_level or solution_methods))
        score -= 150 * int(example_only)
    elif io_mode == "standard_input":
        calls = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        has_input = "input" in calls or bool(re.search(r"\bsys\.stdin\b", code))
        has_output = "print" in calls or bool(re.search(r"\bsys\.stdout\b", code))
        score += 80 * int(has_input) + 80 * int(has_output)
        score += 15 * int(bool(top_level))
        score += 10 * int('__name__' in code and '__main__' in code)
        score -= 100 * int(example_only or (has_output and not has_input and not top_level))
    else:
        score += 10 * int(bool(top_level or solution_methods))
        score -= 10 * int(example_only)
    return score


def extract_code(
    text: str,
    *,
    io_mode: Optional[str] = None,
    fn_name: Optional[str] = None,
    starter_code: Optional[str] = None,
) -> Optional[str]:
    """
    单代码块保持原有行为。多代码块时，根据语法、I/O 完整性和
    call-based 目标函数/`Solution` 方法匹配选择，避免抽到示例调用。
    """
    del starter_code  # Reserved for future contract-aware ranking.
    candidates = _code_candidates(text)
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0][0]
    if io_mode is None and fn_name is None:
        explicit_python = [code for code, tagged in candidates if tagged]
        return (explicit_python or [code for code, _ in candidates])[-1]
    last_code, last_tagged = candidates[-1]
    last_score = _candidate_score(
        last_code, tagged_python=last_tagged, io_mode=io_mode, fn_name=fn_name
    )
    # Preserve the established final-block convention when that block is a
    # syntactically valid, contract-complete solution. Rank alternatives only
    # when the final block is an example, incomplete snippet, or wrong interface.
    if (io_mode == "call_based" and last_score >= 190) or (
        io_mode == "standard_input" and last_score >= 160
    ):
        return last_code
    return max(
        candidates,
        key=lambda item: _candidate_score(
            item[0], tagged_python=item[1], io_mode=io_mode, fn_name=fn_name
        ),
    )[0]


# ── 语法校验 ─────────────────────────────────────────────────

def validate_syntax(code: str) -> tuple[bool, Optional[str]]:
    """
    用 ast.parse 检查 Python 语法。
    返回 (ok, error_msg)。
    """
    try:
        ast.parse(code)
        return True, None
    except SyntaxError as e:
        return False, f"SyntaxError line {e.lineno}: {e.msg}"


# ── 沙箱执行校验 ─────────────────────────────────────────────

def _build_harness(code: str, test_cases: List[dict]) -> str:
    """
    构造测试驱动脚本。
    test_cases 格式：[{"input": "1 2\n", "output": "3\n"}, ...]
    代码必须从 stdin 读取输入、向 stdout 输出结果。
    """
    cases_repr = ascii(test_cases)
    harness = textwrap.dedent(f"""\
# -*- coding: utf-8 -*-
import sys, io

_user_code = {ascii(code)}
_test_cases = {cases_repr}

passed = 0
for tc in _test_cases:
    _in  = tc.get("input", "")
    _exp = tc.get("output", "").strip()
    sys.stdin  = io.StringIO(_in)
    sys.stdout = io.StringIO()
    try:
        exec(compile(_user_code, "<solution>", "exec"), {{"__name__": "__main__"}})
        got = sys.stdout.getvalue().strip()
        if got == _exp:
            passed += 1
    except Exception as exc:
        pass  # 运行时错误算失败

sys.stdout = sys.__stdout__
print(f"{{passed}}/{{len(_test_cases)}}")
""")
    return harness


def validate_execution(
    code: str,
    test_cases: List[dict],
    timeout: int = _EXEC_TIMEOUT,
) -> tuple[bool, float, Optional[str]]:
    """
    在独立子进程中运行代码，比对测试用例输出。

    Args:
        code:       待测代码字符串
        test_cases: [{"input": str, "output": str}, ...]
        timeout:    子进程超时秒数

    Returns:
        (ok, pass_rate, error_msg)
        ok = pass_rate >= 0.5（至少通过一半测试）
    """
    if not test_cases:
        # 无测试用例时只做语法检查
        ok, err = validate_syntax(code)
        return ok, (1.0 if ok else 0.0), err

    harness = _build_harness(code, test_cases)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(harness)
        tmp_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        out = result.stdout.strip()
        # 解析 "passed/total" 格式
        if "/" in out:
            num, denom = out.split("/")
            pass_rate = int(num) / max(int(denom), 1)
        else:
            pass_rate = 0.0
        ok = pass_rate >= 0.5
        err = result.stderr.strip() or None
        return ok, pass_rate, err

    except subprocess.TimeoutExpired:
        return False, 0.0, f"执行超时（>{timeout}s）"
    except Exception as e:
        return False, 0.0, str(e)
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# ── 组合校验入口 ─────────────────────────────────────────────

def validate(
    llm_output: str,
    test_cases: Optional[List[dict]] = None,
    run_code: bool = False,
) -> ValidationResult:
    """
    Validate one distilled response for dataset construction.

    Contract used by scripts/build_sft_dataset.py:
    - always reject missing code blocks and syntax errors;
    - when run_code=True and test_cases exist, execute code and report pass_rate;
    - pass_rate < 0.5 sets ok=False, but build_sft_dataset.py currently keeps
      non-syntax execution failures so the sample can still be audited/filtered
      later using metadata.pass_rate.
    """
    """
    完整校验流程：提取 → 语法 → （可选）执行。

    Args:
        llm_output:  LLM 的完整输出文本
        test_cases:  用于执行校验的测试用例，为空列表或 None 时跳过执行
        run_code:    是否启用沙箱执行（False 时只做语法检查）

    Returns:
        ValidationResult
    """
    # Step 1: 提取代码块
    code = extract_code(llm_output)
    if code is None:
        return ValidationResult(ok=False, code=None, error="未找到代码块")

    # Step 2: 语法检查
    syntax_ok, syntax_err = validate_syntax(code)
    if not syntax_ok:
        return ValidationResult(ok=False, code=code, error=syntax_err)

    # Step 3: 执行校验（可选）
    if run_code and test_cases:
        exec_ok, pass_rate, exec_err = validate_execution(code, test_cases)
        return ValidationResult(
            ok=exec_ok,
            code=code,
            error=exec_err,
            pass_rate=pass_rate,
        )

    return ValidationResult(ok=True, code=code, error=None, pass_rate=1.0)
