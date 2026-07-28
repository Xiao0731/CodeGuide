#!/usr/bin/env python3
"""Run the real Docker verifier contract and emit a secret-free G0 report."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from threading import Thread
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.reward.execution import VerificationResult, verify_code


LABEL_FILTER = "label=codeguide.verifier=true"


def _docker_json(*args: str) -> Any:
    result = subprocess.run(
        ["docker", *args],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    return json.loads(result.stdout)


def _active_container_ids() -> list[str]:
    result = subprocess.run(
        ["docker", "ps", "--filter", LABEL_FILTER, "--format", "{{.ID}}"],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _passed(result: VerificationResult) -> bool:
    return (
        not result.unsupported
        and result.error is None
        and result.total_cases > 0
        and result.pass_rate == 1.0
    )


def _summary(result: VerificationResult) -> dict[str, Any]:
    return {
        "passed_cases": result.passed_cases,
        "total_cases": result.total_cases,
        "pass_rate": result.pass_rate,
        "error": result.error,
        "unsupported": result.unsupported,
        "io_mode": result.io_mode,
        "execution_backend": result.execution_backend,
    }


def _verify_standard(code: str, expected: str, image: str, timeout: float) -> VerificationResult:
    return verify_code(
        code,
        {
            "io_mode": "standard_input",
            "test_cases": [{"input": "", "output": expected}],
        },
        timeout=timeout,
        backend="docker",
        container_image=image,
    )


def _inspect_live_contract(image: str, timeout: float) -> dict[str, Any]:
    holder: dict[str, VerificationResult] = {}
    probe_timeout = timeout + 4

    def run_probe() -> None:
        holder["result"] = _verify_standard(
            "import time\ntime.sleep(3)\n",
            "",
            image,
            probe_timeout,
        )

    thread = Thread(target=run_probe, daemon=True)
    thread.start()
    container_ids: list[str] = []
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        container_ids = _active_container_ids()
        if container_ids:
            break
        time.sleep(0.1)
    if len(container_ids) != 1:
        raise RuntimeError(f"expected one live verifier container, got {container_ids}")

    payload = _docker_json("inspect", container_ids[0])[0]
    host = payload["HostConfig"]
    config = payload["Config"]
    mounts = payload["Mounts"]
    ulimits = {
        item.get("Name"): item
        for item in (host.get("Ulimits") or [])
        if isinstance(item, dict)
    }
    contract = {
        "network_none": host["NetworkMode"] == "none",
        "read_only_root": host["ReadonlyRootfs"] is True,
        "cap_drop_all": "ALL" in (host.get("CapDrop") or []),
        "no_new_privileges": "no-new-privileges" in (host.get("SecurityOpt") or []),
        "pids_limit_64": host.get("PidsLimit") == 64,
        "memory_256m": host.get("Memory") == 256 * 1024 * 1024,
        "memory_swap_disabled": host.get("MemorySwap") == 256 * 1024 * 1024,
        "one_cpu": host.get("NanoCpus") == 1_000_000_000,
        "cpu_ulimit": (
            ulimits.get("cpu", {}).get("Soft") == max(1, math.ceil(probe_timeout))
            and ulimits.get("cpu", {}).get("Hard")
            == max(1, math.ceil(probe_timeout)) + 1
        ),
        "init_enabled": host.get("Init") is True,
        "non_root_user": config.get("User") == "65534:65534",
        "workdir_tmp": config.get("WorkingDir") == "/tmp",
        "tmpfs_restricted": "noexec" in (host.get("Tmpfs", {}).get("/tmp", "")),
        "runner_only_read_only_mount": (
            len(mounts) == 1
            and mounts[0].get("Destination") == "/runner.py"
            and mounts[0].get("RW") is False
        ),
    }
    thread.join(timeout + 6)
    if thread.is_alive() or not _passed(holder["result"]):
        raise RuntimeError(f"live contract probe failed: {holder.get('result')}")
    return contract


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image",
        default=os.environ.get("CODEGUIDE_EXECUTION_IMAGE", ""),
    )
    parser.add_argument("--timeout", type=float, default=2.0)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/g0/docker_verifier_report.json"),
    )
    args = parser.parse_args()

    if "@sha256:" not in args.image:
        raise SystemExit("--image must be pinned by sha256 digest")
    image_info = _docker_json("image", "inspect", args.image)[0]
    before = _active_container_ids()
    if before:
        raise SystemExit(f"stale verifier containers exist before test: {before}")

    standard = verify_code(
        "a, b = map(int, input().split())\nprint(a + b)\n",
        {
            "io_mode": "standard_input",
            "test_cases": [
                {"input": "1 2\n", "output": "3\n"},
                {"input": "-2 9\n", "output": "7\n"},
            ],
        },
        timeout=args.timeout,
        backend="docker",
        container_image=args.image,
    )
    call_based = verify_code(
        "def add(a, b):\n    return a + b\n",
        {
            "io_mode": "call_based",
            "fn_name": "add",
            "test_cases": [
                {"input_args": [1, 2], "expected_output": 3},
                {"input_args": [-2, 9], "expected_output": 7},
            ],
        },
        timeout=args.timeout,
        backend="docker",
        container_image=args.image,
    )
    call_based_solution_class = verify_code(
        "class Solution:\n"
        "    def add(self, a, b):\n"
        "        return a + b\n",
        {
            "io_mode": "call_based",
            "fn_name": "add",
            "test_cases": [
                {"input_args": [1, 2], "expected_output": 3},
                {"input_args": [-2, 9], "expected_output": 7},
            ],
        },
        timeout=args.timeout,
        backend="docker",
        container_image=args.image,
    )
    wrong_answer = verify_code(
        "print('wrong')\n",
        {
            "io_mode": "standard_input",
            "test_cases": [{"input": "", "output": "right\n"}],
        },
        timeout=args.timeout,
        backend="docker",
        container_image=args.image,
    )
    network = _verify_standard(
        "import socket\n"
        "try:\n"
        "    socket.create_connection(('1.1.1.1', 53), timeout=0.5)\n"
        "    print('connected')\n"
        "except OSError:\n"
        "    print('blocked')\n",
        "blocked\n",
        args.image,
        args.timeout,
    )
    filesystem = _verify_standard(
        "import os\n"
        "root_blocked = False\n"
        "try:\n"
        "    open('/codeguide-root-probe', 'w').write('bad')\n"
        "except OSError:\n"
        "    root_blocked = True\n"
        "open('relative-probe.txt', 'w').write('ok')\n"
        "print(os.geteuid(), os.getcwd(), root_blocked, "
        "open('relative-probe.txt').read())\n",
        "65534 /tmp True ok\n",
        args.image,
        args.timeout,
    )
    timeout_result = _verify_standard(
        "while True:\n    pass\n",
        "",
        args.image,
        1.0,
    )
    time.sleep(0.5)
    timeout_cleanup = not _active_container_ids()
    live_contract = _inspect_live_contract(args.image, args.timeout)

    def concurrent_probe(index: int) -> bool:
        result = verify_code(
            f"print({index} + 1)\n",
            {
                "io_mode": "standard_input",
                "test_cases": [{"input": "", "output": f"{index + 1}\n"}],
            },
            timeout=args.timeout,
            backend="docker",
            container_image=args.image,
        )
        return _passed(result)

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        concurrent_results = list(pool.map(concurrent_probe, range(args.concurrency)))
    time.sleep(0.5)
    no_leaks_after_concurrency = not _active_container_ids()

    floating = verify_code(
        "print(1)\n",
        {
            "io_mode": "standard_input",
            "test_cases": [{"input": "", "output": "1\n"}],
        },
        timeout=args.timeout,
        backend="docker",
        container_image="python:3.11.9-slim-bookworm",
    )
    checks = {
        "digest_pinned": "@sha256:" in args.image,
        "standard_input": _passed(standard),
        "call_based_function": _passed(call_based),
        "call_based_solution_class": _passed(call_based_solution_class),
        "wrong_answer_detected": wrong_answer.pass_rate == 0.0,
        "floating_tag_rejected": floating.unsupported is True,
        "network_blocked": _passed(network),
        "non_root_read_only_and_tmp_writable": _passed(filesystem),
        "timeout_detected": (
            timeout_result.error is not None and "timeout" in timeout_result.error
        ),
        "timeout_container_cleaned": timeout_cleanup,
        "runtime_contract": all(live_contract.values()),
        "concurrency": all(concurrent_results),
        "no_leaks_after_concurrency": no_leaks_after_concurrency,
    }
    report = {
        "schema_version": 1,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "image": args.image,
        "image_id": image_info["Id"],
        "platform": image_info["Os"],
        "architecture": image_info["Architecture"],
        "timeout_seconds": args.timeout,
        "concurrency": args.concurrency,
        "checks": checks,
        "runtime_contract": live_contract,
        "results": {
            "standard_input": _summary(standard),
            "call_based_function": _summary(call_based),
            "call_based_solution_class": _summary(call_based_solution_class),
            "wrong_answer": _summary(wrong_answer),
            "network": _summary(network),
            "filesystem": _summary(filesystem),
            "timeout": _summary(timeout_result),
            "floating_tag": _summary(floating),
        },
        "all_passed": all(checks.values()),
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    raise SystemExit(0 if report["all_passed"] else 1)


if __name__ == "__main__":
    main()
