"""Execution-based correctness verification for GRPO rewards.

Callers pass generated code plus normalized metadata, and the verifier selects
either stdin/stdout execution or call-based function execution. Unsupported
call-based shapes are reported as unsupported so data filtering can skip them
instead of training on false failures.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass
class VerificationResult:
    """Uniform result returned by all execution runners."""

    passed_cases: int
    total_cases: int
    pass_rate: float
    first_failure: Optional[str] = None
    error: Optional[str] = None
    unsupported: bool = False
    io_mode: str = "unknown"
    execution_backend: str = "unknown"


def compare_outputs(actual: Any, expected: Any, *, float_tol: float = 1e-6) -> bool:
    """Compare Python values directly so call-based outputs are not stringified."""

    if isinstance(actual, bool) or isinstance(expected, bool):
        return actual is expected
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        if isinstance(actual, float) or isinstance(expected, float):
            return math.isclose(float(actual), float(expected), rel_tol=float_tol, abs_tol=float_tol)
        return actual == expected
    if isinstance(actual, str) or isinstance(expected, str):
        return actual == expected
    if actual is None or expected is None:
        return actual is None and expected is None
    if isinstance(actual, tuple):
        actual = list(actual)
    if isinstance(expected, tuple):
        expected = list(expected)
    if isinstance(actual, list) and isinstance(expected, list):
        return len(actual) == len(expected) and all(
            compare_outputs(a, e, float_tol=float_tol) for a, e in zip(actual, expected)
        )
    if isinstance(actual, dict) and isinstance(expected, dict):
        if set(actual.keys()) != set(expected.keys()):
            return False
        return all(compare_outputs(actual[k], expected[k], float_tol=float_tol) for k in actual)
    return actual == expected


def _is_json_compatible(value: Any) -> bool:
    if value is None or isinstance(value, (str, int, float, bool)):
        return True
    if isinstance(value, list):
        return all(_is_json_compatible(v) for v in value)
    if isinstance(value, dict):
        return all(isinstance(k, str) and _is_json_compatible(v) for k, v in value.items())
    return False


def _has_unsupported_tag(metadata: dict[str, Any]) -> bool:
    tags: list[str] = []
    for key in ("tags", "raw_tags", "skill_types"):
        value = metadata.get(key) or []
        if isinstance(value, str):
            tags.append(value.lower())
        elif isinstance(value, list):
            tags.extend(str(v).lower() for v in value)
    joined = " ".join(tags)
    return any(marker in joined for marker in ("interactive", "special judge", "custom checker"))


def _infer_io_mode(metadata: dict[str, Any], test_cases: list[dict]) -> str:
    io_mode = metadata.get("io_mode")
    if io_mode in {"standard_input", "call_based"}:
        return str(io_mode)
    if metadata.get("fn_name") or any("input_args" in tc for tc in test_cases):
        return "call_based"
    return "standard_input"


def supports_verification(metadata: dict[str, Any]) -> bool:
    """
    Return True only for metadata the current verifier can execute reliably.

    Contract:
    - standard_input cases need {"input": str, "output": str}
    - call_based cases need fn_name and
      {"input_args": list, "expected_output": JSON-compatible value}

    Unsupported shapes are kept out of GRPO rather than scored as failed,
    because custom structures, generators, in-place-only tasks, interactive
    problems, and special judges require different harnesses.
    """
    # 暂不支持链表、树、自定义对象、迭代器/生成器、复杂in-place无返回、special judge/interactive等场景
    test_cases = metadata.get("test_cases") or []
    if not isinstance(test_cases, list) or not test_cases or _has_unsupported_tag(metadata):
        return False

    io_mode = _infer_io_mode(metadata, test_cases)
    if io_mode == "standard_input":
        return all(isinstance(tc, dict) and "input" in tc and "output" in tc for tc in test_cases)

    if io_mode != "call_based":
        return False

    fn_name = metadata.get("fn_name") or next(
        (tc.get("fn_name") for tc in test_cases if isinstance(tc, dict) and tc.get("fn_name")),
        None,
    )
    if not isinstance(fn_name, str) or not fn_name.strip():
        return False

    for tc in test_cases:
        if not isinstance(tc, dict):
            return False
        if "input_args" not in tc or "expected_output" not in tc:
            return False
        if not isinstance(tc["input_args"], list):
            return False
        if not _is_json_compatible(tc["input_args"]) or not _is_json_compatible(tc["expected_output"]):
            return False
    return True


def _parse_harness_result(
    stdout: str,
    stderr: str,
    *,
    backend: str,
) -> VerificationResult:
    out = stdout.strip()
    if not out:
        return VerificationResult(
            0,
            0,
            0.0,
            error=stderr.strip() or "no harness output",
            execution_backend=backend,
        )
    try:
        payload = json.loads(out.splitlines()[-1])
        return VerificationResult(
            passed_cases=int(payload.get("passed_cases", 0)),
            total_cases=int(payload.get("total_cases", 0)),
            pass_rate=float(payload.get("pass_rate", 0.0)),
            first_failure=payload.get("first_failure"),
            error=payload.get("error"),
            unsupported=bool(payload.get("unsupported", False)),
            io_mode=str(payload.get("io_mode", "unknown")),
            execution_backend=backend,
        )
    except Exception as exc:
        return VerificationResult(
            0,
            0,
            0.0,
            error=f"invalid harness output: {exc}",
            execution_backend=backend,
        )


def _run_harness_subprocess(harness_path: Path, timeout: float) -> VerificationResult:
    """Development-only runner. It is not a security boundary."""
    try:
        result = subprocess.run(
            [sys.executable, "-I", str(harness_path)],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=harness_path.parent,
            env={
                "PATH": os.environ.get("PATH", ""),
                "PYTHONIOENCODING": "utf-8",
                "PYTHONDONTWRITEBYTECODE": "1",
            },
        )
        return _parse_harness_result(
            result.stdout,
            result.stderr,
            backend="subprocess",
        )
    except subprocess.TimeoutExpired:
        return VerificationResult(
            0,
            0,
            0.0,
            error=f"timeout after {timeout}s",
            execution_backend="subprocess",
        )
    except Exception as exc:
        return VerificationResult(
            0,
            0,
            0.0,
            error=str(exc),
            execution_backend="subprocess",
        )


def _run_harness_docker(
    harness_path: Path,
    timeout: float,
    *,
    image: str,
) -> VerificationResult:
    """Run a harness with an explicit, restricted Docker execution contract."""
    docker = shutil.which("docker")
    if docker is None:
        return VerificationResult(
            0,
            0,
            0.0,
            error="docker executable not found",
            unsupported=True,
            execution_backend="docker",
        )
    if not image or "@sha256:" not in image:
        return VerificationResult(
            0,
            0,
            0.0,
            error=(
                "container image must be pinned by digest; set "
                "CODEGUIDE_EXECUTION_IMAGE=name@sha256:..."
            ),
            unsupported=True,
            execution_backend="docker",
        )

    command = [
        docker,
        "run",
        "--rm",
        "--network",
        "none",
        "--read-only",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--pids-limit",
        "64",
        "--memory",
        "256m",
        "--cpus",
        "1.0",
        "--user",
        "65534:65534",
        "--tmpfs",
        "/tmp:rw,noexec,nosuid,size=64m",
        "--env",
        "PYTHONDONTWRITEBYTECODE=1",
        "--env",
        "PYTHONIOENCODING=utf-8",
        "--volume",
        f"{harness_path}:/runner.py:ro",
        image,
        "python",
        "-I",
        "/runner.py",
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout + 2.0,
            env={"PATH": os.environ.get("PATH", "")},
        )
        return _parse_harness_result(
            result.stdout,
            result.stderr,
            backend="docker",
        )
    except subprocess.TimeoutExpired:
        return VerificationResult(
            0,
            0,
            0.0,
            error=f"container timeout after {timeout}s",
            execution_backend="docker",
        )
    except Exception as exc:
        return VerificationResult(
            0,
            0,
            0.0,
            error=str(exc),
            execution_backend="docker",
        )


def _run_harness(
    harness: str,
    timeout: float,
    *,
    backend: str,
    container_image: str | None,
) -> VerificationResult:
    if backend not in {"subprocess", "docker"}:
        return VerificationResult(
            0,
            0,
            0.0,
            error=f"unknown execution backend: {backend}",
            unsupported=True,
            execution_backend=backend,
        )

    with tempfile.TemporaryDirectory(prefix="codeguide-exec-") as tmp_dir:
        harness_path = Path(tmp_dir) / "runner.py"
        harness_path.write_text(
            "# -*- coding: utf-8 -*-\n" + harness,
            encoding="utf-8",
        )
        harness_path.chmod(0o444)
        if backend == "docker":
            image = container_image or os.environ.get("CODEGUIDE_EXECUTION_IMAGE", "")
            return _run_harness_docker(
                harness_path,
                timeout,
                image=image,
            )
        return _run_harness_subprocess(harness_path, timeout)


class StandardInputRunner:
    def run(
        self,
        code: str,
        test_cases: list[dict],
        *,
        timeout: float = 5.0,
        backend: str = "subprocess",
        container_image: str | None = None,
    ) -> VerificationResult:
        harness = textwrap.dedent(
            f"""
            import io, json, sys, traceback

            _code = {ascii(code)}
            _cases = {ascii(test_cases)}
            _passed = 0
            _first_failure = None

            for _idx, _tc in enumerate(_cases, 1):
                _inp = _tc.get("input", "")
                _exp = str(_tc.get("output", "")).rstrip("\\n")
                _stdin_bak, _stdout_bak = sys.stdin, sys.stdout
                sys.stdin = io.StringIO(_inp)
                sys.stdout = io.StringIO()
                try:
                    exec(compile(_code, "<solution>", "exec"), {{"__name__": "__main__"}})
                    _got = sys.stdout.getvalue().rstrip("\\n")
                    _ok = (_got == _exp)
                    if not _ok and _first_failure is None:
                        _first_failure = f"case #{{_idx}}: got {{_got!r}}, expected {{_exp!r}}"
                except Exception as _exc:
                    _ok = False
                    if _first_failure is None:
                        _first_failure = f"case #{{_idx}}: {{type(_exc).__name__}}: {{_exc}}"
                finally:
                    sys.stdin = _stdin_bak
                    sys.stdout = _stdout_bak
                if _ok:
                    _passed += 1

            _total = len(_cases)
            print(json.dumps({{
                "passed_cases": _passed,
                "total_cases": _total,
                "pass_rate": (_passed / _total) if _total else 0.0,
                "first_failure": _first_failure,
                "io_mode": "standard_input",
            }}, ensure_ascii=False))
            """
        )
        result = _run_harness(
            harness,
            timeout,
            backend=backend,
            container_image=container_image,
        )
        result.io_mode = "standard_input"
        result.total_cases = result.total_cases or len(test_cases)
        return result


class CallBasedRunner:
    def run(
        self,
        code: str,
        test_cases: list[dict],
        *,
        fn_name: str | None = None,
        starter_code: str | None = None,
        timeout: float = 5.0,
        backend: str = "subprocess",
        container_image: str | None = None,
    ) -> VerificationResult:
        metadata = {"io_mode": "call_based", "fn_name": fn_name, "test_cases": test_cases}
        if not supports_verification(metadata):
            return VerificationResult(
                0, len(test_cases), 0.0, unsupported=True, error="unsupported call_based metadata", io_mode="call_based"
            )

        target_name = fn_name or test_cases[0].get("fn_name")
        harness = textwrap.dedent(
            f"""
            import json, math, traceback

            _code = {ascii(code)}
            _cases = {ascii(test_cases)}
            _fn_name = {ascii(target_name)}

            def _compare(_actual, _expected, _tol=1e-6):
                if isinstance(_actual, bool) or isinstance(_expected, bool):
                    return _actual is _expected
                if isinstance(_actual, (int, float)) and isinstance(_expected, (int, float)):
                    if isinstance(_actual, float) or isinstance(_expected, float):
                        return math.isclose(float(_actual), float(_expected), rel_tol=_tol, abs_tol=_tol)
                    return _actual == _expected
                if isinstance(_actual, str) or isinstance(_expected, str):
                    return _actual == _expected
                if _actual is None or _expected is None:
                    return _actual is None and _expected is None
                if isinstance(_actual, tuple):
                    _actual = list(_actual)
                if isinstance(_expected, tuple):
                    _expected = list(_expected)
                if isinstance(_actual, list) and isinstance(_expected, list):
                    return len(_actual) == len(_expected) and all(_compare(a, e, _tol) for a, e in zip(_actual, _expected))
                if isinstance(_actual, dict) and isinstance(_expected, dict):
                    if set(_actual.keys()) != set(_expected.keys()):
                        return False
                    return all(_compare(_actual[k], _expected[k], _tol) for k in _actual)
                return _actual == _expected

            _ns = {{"__name__": "__codeguide_call_based__"}}
            _passed = 0
            _first_failure = None
            _unsupported = False
            _error = None

            try:
                exec(compile(_code, "<solution>", "exec"), _ns)
            except Exception as _exc:
                _error = f"exec {{type(_exc).__name__}}: {{_exc}}"

            if _error is None:
                for _idx, _tc in enumerate(_cases, 1):
                    try:
                        _args = _tc["input_args"]
                        _expected = _tc["expected_output"]
                        if _fn_name in _ns and callable(_ns[_fn_name]):
                            _actual = _ns[_fn_name](*_args)
                        elif "Solution" in _ns:
                            _obj = _ns["Solution"]()
                            _method = getattr(_obj, _fn_name, None)
                            if not callable(_method):
                                _unsupported = True
                                _error = f"Solution has no callable method {{_fn_name!r}}"
                                break
                            _actual = _method(*_args)
                        else:
                            _unsupported = True
                            _error = f"no top-level function or Solution.{{_fn_name}} found"
                            break

                        if _compare(_actual, _expected):
                            _passed += 1
                        elif _first_failure is None:
                            _first_failure = f"case #{{_idx}}: got {{_actual!r}}, expected {{_expected!r}}"
                    except Exception as _exc:
                        if _first_failure is None:
                            _first_failure = f"case #{{_idx}}: {{type(_exc).__name__}}: {{_exc}}"

            _total = len(_cases)
            print(json.dumps({{
                "passed_cases": _passed,
                "total_cases": _total,
                "pass_rate": (_passed / _total) if _total else 0.0,
                "first_failure": _first_failure,
                "error": _error,
                "unsupported": _unsupported,
                "io_mode": "call_based",
            }}, ensure_ascii=False))
            """
        )
        result = _run_harness(
            harness,
            timeout,
            backend=backend,
            container_image=container_image,
        )
        result.io_mode = "call_based"
        result.total_cases = result.total_cases or len(test_cases)
        return result


def verify_code(
    code: str,
    metadata: dict[str, Any],
    *,
    timeout: float = 5.0,
    backend: str = "subprocess",
    container_image: str | None = None,
) -> VerificationResult:
    test_cases = metadata.get("test_cases") or []
    if not isinstance(test_cases, list) or not test_cases:
        return VerificationResult(0, 0, 0.0, error="missing test cases", unsupported=True)

    io_mode = _infer_io_mode(metadata, test_cases)
    if io_mode == "standard_input":
        return StandardInputRunner().run(
            code,
            test_cases,
            timeout=timeout,
            backend=backend,
            container_image=container_image,
        )
    if io_mode == "call_based":
        return CallBasedRunner().run(
            code,
            test_cases,
            fn_name=metadata.get("fn_name"),
            starter_code=metadata.get("starter_code"),
            timeout=timeout,
            backend=backend,
            container_image=container_image,
        )
    return VerificationResult(0, len(test_cases), 0.0, unsupported=True, error=f"unknown io_mode: {io_mode}")
