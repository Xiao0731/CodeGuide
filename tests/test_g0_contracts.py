from __future__ import annotations

from pathlib import Path
import subprocess
from types import SimpleNamespace

import yaml

from scripts.build_sft_dataset import _is_accepted_verification
from scripts.evaluate_model import validate_manifest
from scripts.validate_config import validate
from src.reward.execution import verify_code
from src.reward import execution
from src.reward.format import _score_format


ROOT = Path(__file__).resolve().parent.parent


def test_teacher_label_requires_full_execution_pass():
    assert _is_accepted_verification(
        SimpleNamespace(
            unsupported=False,
            error=None,
            total_cases=4,
            pass_rate=1.0,
        )
    )
    assert not _is_accepted_verification(
        SimpleNamespace(
            unsupported=False,
            error=None,
            total_cases=4,
            pass_rate=0.75,
        )
    )
    assert not _is_accepted_verification(
        SimpleNamespace(
            unsupported=True,
            error="unsupported",
            total_cases=0,
            pass_rate=0.0,
        )
    )


def test_docker_backend_fails_closed_without_pinned_image():
    cases = [{"input": "1 2\n", "output": "3"}]
    result = verify_code(
        "a, b = map(int, input().split())\nprint(a + b)\n",
        {"io_mode": "standard_input", "test_cases": cases},
        backend="docker",
        container_image="python:3.11-slim",
    )
    assert result.unsupported is True
    assert result.execution_backend == "docker"
    assert result.pass_rate == 0.0


def test_docker_timeout_forces_unique_container_cleanup(monkeypatch, tmp_path):
    harness = tmp_path / "runner.py"
    harness.write_text("while True: pass\n", encoding="utf-8")
    calls = []

    monkeypatch.setattr(execution.shutil, "which", lambda _: "docker")

    def fake_run(command, **kwargs):
        calls.append(command)
        if command[1] == "run":
            raise subprocess.TimeoutExpired(command, timeout=3.0)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(execution.subprocess, "run", fake_run)
    result = execution._run_harness_docker(
        harness,
        1.0,
        image="python@sha256:" + "a" * 64,
    )

    run_command, cleanup_command = calls
    name = run_command[run_command.index("--name") + 1]
    assert name.startswith("codeguide-verifier-")
    assert ["docker", "rm", "--force", name] == cleanup_command
    assert "--network" in run_command
    assert "none" in run_command
    assert "--read-only" in run_command
    assert "--memory-swap" in run_command
    assert "--workdir" in run_command
    assert result.error == "container timeout after 1.0s"


def test_docker_cpu_limit_exit_is_reported_as_timeout(monkeypatch, tmp_path):
    harness = tmp_path / "runner.py"
    harness.write_text("while True: pass\n", encoding="utf-8")

    monkeypatch.setattr(execution.shutil, "which", lambda _: "docker")

    def fake_run(command, **kwargs):
        if command[1] == "run":
            return subprocess.CompletedProcess(command, 137, "", "")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(execution.subprocess, "run", fake_run)
    result = execution._run_harness_docker(
        harness,
        1.0,
        image="python@sha256:" + "a" * 64,
    )

    assert result.error == "container timeout/resource limit after 1.0s (exit code 137)"


def test_main_config_contract_is_static_valid_but_lengths_unfrozen():
    config = yaml.safe_load(
        (ROOT / "configs/train_config.yaml").read_text(encoding="utf-8")
    )
    errors, warnings = validate(config, allow_unfrozen=True)
    assert errors == []
    assert any("max_seq_length is unfrozen" in item for item in warnings)
    assert config["sft"]["learning_rate"] == 1.0e-4
    assert config["sft"]["num_train_epochs"] == 1.0
    assert config["sft"]["completion_only_loss"] is True
    assert config["sft"]["length_grouped_sampling"] is True


def test_manifest_reader_checks_frozen_smoke_file():
    manifest = validate_manifest(ROOT / "data/manifests/g0_smoke.json")
    assert manifest["record_count"] == 5
    assert manifest["split_name"] == "g0_reference_guided_smoke"


def test_empty_step_headings_do_not_get_full_contract_reward():
    hollow = """
第一步：
占位。
第二步：
占位。
第三步：
占位。
```python
pass
```
"""
    assert _score_format(hollow) < 0.4
