#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import codecs
import json
import math
import os
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Any

DEFAULT_SEED_PATHS = [
    Path("data/seeds/seed.json"),
    Path("/mnt/data/seed.json"),
]

COMMON_IMPORTS = textwrap.dedent(
    """
    from __future__ import annotations
    import bisect
    import collections
    import functools
    import heapq
    import itertools
    import json
    import math
    import random
    import statistics
    import string
    from collections import *
    from functools import *
    from heapq import *
    from itertools import *
    from math import *
    from typing import *
    """
)


def infer_seed_path(cli_path: str | None) -> Path:
    if cli_path:
        path = Path(cli_path)
        if not path.exists():
            raise FileNotFoundError(f"Seed file not found: {path}")
        return path
    for path in DEFAULT_SEED_PATHS:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Could not find seed.json. Tried: " + ", ".join(str(p) for p in DEFAULT_SEED_PATHS)
    )


class SandboxedExecutionError(RuntimeError):
    pass


def normalize_code(code: str) -> str:
    """Decode accidental double-escaped code strings like '\\n' -> '\n'."""
    if not code:
        return code
    if "\\n" in code and "\n" not in code:
        try:
            return codecs.decode(code, "unicode_escape")
        except Exception:
            return code.replace("\\n", "\n").replace('\\"', '"').replace("\\'", "'")
    return code


LITERAL_PREFIXES = ('"', "'", '[', '{', '(',)
LITERAL_EXACT = {"true": True, "false": False, "null": None}


def maybe_parse_literal(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    lower = stripped.lower()
    if lower in LITERAL_EXACT:
        return LITERAL_EXACT[lower]
    if stripped.startswith(LITERAL_PREFIXES):
        try:
            return ast.literal_eval(stripped)
        except Exception:
            return value
    return value


def normalize_call_case(case: Any) -> Any:
    if isinstance(case, list):
        return [normalize_call_case(x) for x in case]
    if isinstance(case, tuple):
        return tuple(normalize_call_case(x) for x in case)
    return maybe_parse_literal(case)


def detect_callable_name(code: str) -> str:
    tree = ast.parse(code)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "Solution":
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and not item.name.startswith("_"):
                    return item.name
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith("_"):
            return node.name
    raise ValueError("Could not detect callable entrypoint from annotated_code")


def make_call_based_runner(code: str, args: list[Any], method_name: str) -> str:
    return (
        COMMON_IMPORTS
        + f"\nCODE = {code!r}\nARGS = {args!r}\nMETHOD_NAME = {method_name!r}\n\n"
        + textwrap.dedent("""
ns = {"__name__": "__main__"}
exec(CODE, ns, ns)

if "Solution" in ns:
    obj = ns["Solution"]()
    fn = getattr(obj, METHOD_NAME)
else:
    fn = ns[METHOD_NAME]

result = fn(*ARGS)
print(json.dumps(result))
""")
    )


def make_standard_input_runner(code: str) -> str:
    return (
        COMMON_IMPORTS
        + f"\nCODE = {code!r}\n"
        + textwrap.dedent("""
ns = {"__name__": "__main__"}
exec(CODE, ns, ns)
""")
    )


def run_python_snippet(snippet: str, stdin_data: str, timeout_sec: float) -> tuple[str, str, int]:
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(snippet)
        tmp_path = f.name
    try:
        proc = subprocess.run(
            [sys.executable, tmp_path],
            input=stdin_data,
            text=True,
            capture_output=True,
            timeout=timeout_sec,
        )
        return proc.stdout, proc.stderr, proc.returncode
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def tokens_are_float_like(tokens: list[str]) -> bool:
    if not tokens:
        return False
    seen_floatish = False
    for tok in tokens:
        try:
            float(tok)
        except ValueError:
            return False
        if any(ch in tok.lower() for ch in ".e"):
            seen_floatish = True
    return seen_floatish


def compare_standard_output(actual: str, expected: str, float_tol: float) -> tuple[bool, str]:
    actual_tokens = actual.strip().split()
    expected_tokens = expected.strip().split()
    if tokens_are_float_like(actual_tokens) and tokens_are_float_like(expected_tokens) and len(actual_tokens) == len(expected_tokens):
        for i, (a, e) in enumerate(zip(actual_tokens, expected_tokens)):
            af = float(a)
            ef = float(e)
            if not math.isclose(af, ef, rel_tol=float_tol, abs_tol=float_tol):
                return False, f"float token mismatch at index {i}: actual={af}, expected={ef}"
        return True, "float tokens matched within tolerance"
    if actual_tokens == expected_tokens:
        return True, "whitespace-normalized tokens matched"
    return False, "token mismatch"


def compare_call_based_output(actual_stdout: str, expected_value: Any) -> tuple[bool, str]:
    actual_stdout = actual_stdout.strip()
    if not actual_stdout:
        return False, "no stdout produced"
    try:
        actual_value = json.loads(actual_stdout)
    except json.JSONDecodeError:
        return False, f"runner did not emit valid JSON: {actual_stdout[:200]!r}"
    if actual_value == expected_value:
        return True, "JSON values matched"
    return False, f"value mismatch: actual={actual_value!r}, expected={expected_value!r}"


def validate_one_seed(seed: dict[str, Any], timeout_sec: float, float_tol: float) -> dict[str, Any]:
    train_id = seed.get("train_ID")
    problem_id = seed.get("problem_id")
    mode = seed["io_format"]["mode"]
    code = normalize_code(seed["teaching_answer"].get("annotated_code", ""))
    verifiable = seed.get("quality_tags", {}).get("verifiable_by_tests", True)

    result: dict[str, Any] = {
        "train_ID": train_id,
        "problem_id": problem_id,
        "mode": mode,
        "status": "unknown",
        "message": "",
    }

    if not code.strip():
        result["status"] = "skipped"
        result["message"] = "annotated_code is empty"
        return result

    if mode == "standard_input" and not verifiable:
        result["status"] = "skipped"
        result["message"] = "verifiable_by_tests is false (likely needs special judge / multi-solution)"
        return result

    inputs = seed["io_format"].get("inputs", [])
    outputs = seed["io_format"].get("outputs", [])
    if len(inputs) != len(outputs):
        result["status"] = "error"
        result["message"] = f"input/output count mismatch: {len(inputs)} vs {len(outputs)}"
        return result

    try:
        if mode == "call_based":
            method_name = detect_callable_name(code)
            runner_args_cases = [normalize_call_case(case) for case in inputs]
            expected_cases = [normalize_call_case(case) for case in outputs]
            for idx, (args, expected) in enumerate(zip(runner_args_cases, expected_cases), start=1):
                if not isinstance(args, list):
                    args = [args]
                snippet = make_call_based_runner(code, args, method_name)
                stdout, stderr, rc = run_python_snippet(snippet, "", timeout_sec)
                if rc != 0:
                    raise SandboxedExecutionError(
                        f"case #{idx} crashed with exit code {rc}. stderr={stderr.strip()[:400]}"
                    )
                ok, why = compare_call_based_output(stdout, expected)
                if not ok:
                    result["status"] = "failed"
                    result["message"] = f"case #{idx}: {why}; stdout={stdout.strip()[:200]!r}"
                    return result
            result["status"] = "passed"
            result["message"] = f"all {len(inputs)} call-based cases passed"
            return result

        if mode == "standard_input":
            snippet = make_standard_input_runner(code)
            for idx, (stdin_data, expected) in enumerate(zip(inputs, outputs), start=1):
                stdout, stderr, rc = run_python_snippet(snippet, stdin_data, timeout_sec)
                if rc != 0:
                    raise SandboxedExecutionError(
                        f"case #{idx} crashed with exit code {rc}. stderr={stderr.strip()[:400]}"
                    )
                ok, why = compare_standard_output(stdout, expected, float_tol)
                if not ok:
                    result["status"] = "failed"
                    result["message"] = (
                        f"case #{idx}: {why}; actual={stdout.strip()[:200]!r}; expected={str(expected).strip()[:200]!r}"
                    )
                    return result
            result["status"] = "passed"
            result["message"] = f"all {len(inputs)} standard-input cases passed"
            return result

        result["status"] = "error"
        result["message"] = f"unknown io_format.mode={mode!r}"
        return result
    except subprocess.TimeoutExpired:
        result["status"] = "error"
        result["message"] = f"timed out after {timeout_sec} seconds"
        return result
    except Exception as e:
        result["status"] = "error"
        result["message"] = str(e)
        return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Sandbox-validate annotated_code inside seed.json")
    parser.add_argument("--seed", default=None, help="Path to seed.json (default: data/seeds/seed.json or /mnt/data/seed.json)")
    parser.add_argument("--timeout", type=float, default=2.5, help="Per-testcase timeout in seconds")
    parser.add_argument("--float-tol", type=float, default=1e-6, help="Tolerance for floating-point outputs")
    parser.add_argument("--only", nargs="*", default=[], help="Optional train_ID/problem_id filters, e.g. train_0001 2457")
    parser.add_argument("--write-report", default=None, help="Optional path to write JSON report")
    args = parser.parse_args()

    seed_path = infer_seed_path(args.seed)
    with seed_path.open("r", encoding="utf-8") as f:
        seeds = json.load(f)

    only_set = {str(x) for x in args.only}
    if only_set:
        seeds = [s for s in seeds if str(s.get("train_ID")) in only_set or str(s.get("problem_id")) in only_set]

    results = [validate_one_seed(seed, args.timeout, args.float_tol) for seed in seeds]

    summary = {
        "seed_path": str(seed_path),
        "total": len(results),
        "passed": sum(r["status"] == "passed" for r in results),
        "failed": sum(r["status"] == "failed" for r in results),
        "skipped": sum(r["status"] == "skipped" for r in results),
        "error": sum(r["status"] == "error" for r in results),
        "results": results,
    }

    print(f"Seed file: {summary['seed_path']}")
    print(
        f"Summary => total={summary['total']}, passed={summary['passed']}, failed={summary['failed']}, "
        f"skipped={summary['skipped']}, error={summary['error']}"
    )
    for r in results:
        status = r["status"].upper().ljust(7)
        print(f"[{status}] {r['train_ID']} (problem_id={r['problem_id']}, mode={r['mode']}): {r['message']}")

    if args.write_report:
        out_path = Path(args.write_report)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nJSON report written to: {out_path}")

    return 0 if summary["failed"] == 0 and summary["error"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
