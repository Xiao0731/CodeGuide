"""Small shared helpers for the framework-backed training entry points."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]


def resolve_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_config(path: str | Path, expected_stage: str) -> dict[str, Any]:
    resolved = resolve_path(path)
    assert resolved is not None
    with resolved.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"invalid training config: {resolved}")
    if config.get("schema_version") != "codeguide-training-v2":
        raise ValueError(f"unsupported config schema: {config.get('schema_version')}")
    if config.get("stage") != expected_stage:
        raise ValueError(f"expected stage={expected_stage}, got {config.get('stage')}")
    return config


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected object at {path}:{line_number}")
            records.append(value)
    return records


def select_attention_backend(requested: str) -> str:
    if requested not in {"auto", "flash_attention_2", "sdpa"}:
        raise ValueError(f"unsupported attention backend: {requested}")
    if requested == "sdpa":
        return "sdpa"
    try:
        import flash_attn  # noqa: F401

        return "flash_attention_2"
    except ImportError:
        if requested == "flash_attention_2":
            raise RuntimeError("flash_attention_2 requested but flash-attn is unavailable")
        return "sdpa"


def report_targets(value: Any) -> list[str]:
    return [] if value in (None, "", "none", []) else [str(value)]
