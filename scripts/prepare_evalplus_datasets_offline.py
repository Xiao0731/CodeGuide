#!/usr/bin/env python3
"""通过 GitHub 镜像下载 EvalPlus 官方原始数据，并准备离线覆盖文件。"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_KEYS = {
    "task_id",
    "prompt",
    "contract",
    "canonical_solution",
    "base_input",
    "plus_input",
    "atol",
}

DATASETS = {
    "humaneval": {
        "version": "v0.1.10",
        "expected_tasks": 164,
        "filename": "HumanEvalPlus-v0.1.10.jsonl.gz",
        "official_url": (
            "https://github.com/evalplus/humanevalplus_release/"
            "releases/download/v0.1.10/HumanEvalPlus.jsonl.gz"
        ),
        "override_env": "HUMANEVAL_OVERRIDE_PATH",
    },
    "mbpp": {
        "version": "v0.2.0",
        "expected_tasks": 378,
        "filename": "MbppPlus-v0.2.0.jsonl.gz",
        "official_url": (
            "https://github.com/evalplus/mbppplus_release/"
            "releases/download/v0.2.0/MbppPlus.jsonl.gz"
        ),
        "override_env": "MBPP_OVERRIDE_PATH",
    },
}


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mirrored_url(official_url: str, mirror_prefix: str) -> str:
    prefix = mirror_prefix.strip()
    if not prefix:
        return official_url
    return prefix.rstrip("/") + "/" + official_url


def inspect_dataset(path: Path, expected_tasks: int) -> dict[str, Any]:
    """验证 gzip、JSONL、题数、任务 ID 唯一性和 EvalPlus 必需字段。"""
    rows = 0
    task_ids: set[str] = set()
    missing_examples: list[dict[str, Any]] = []

    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise RuntimeError(f"第 {line_no} 行不是 JSON 对象")
                rows += 1
                task_id = str(payload.get("task_id") or "")
                if not task_id:
                    raise RuntimeError(f"第 {line_no} 行缺少 task_id")
                if task_id in task_ids:
                    raise RuntimeError(f"出现重复 task_id：{task_id}")
                task_ids.add(task_id)

                missing = sorted(REQUIRED_KEYS - set(payload))
                if missing and len(missing_examples) < 5:
                    missing_examples.append(
                        {
                            "line": line_no,
                            "task_id": task_id,
                            "missing": missing,
                        }
                    )
    except (OSError, EOFError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"数据文件损坏或格式错误：{path}: {exc}") from exc

    if rows != expected_tasks:
        raise RuntimeError(
            f"题数不匹配：expected={expected_tasks} actual={rows} path={path}"
        )
    if missing_examples:
        raise RuntimeError(
            "数据不是 EvalPlus 官方完整 schema；缺失字段示例："
            + json.dumps(missing_examples, ensure_ascii=False)
        )

    return {
        "tasks": rows,
        "unique_task_ids": len(task_ids),
        "required_keys": sorted(REQUIRED_KEYS),
    }


def download_file(url: str, destination: Path, retries: int, timeout: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    temporary.unlink(missing_ok=True)

    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": "CodeGuide-EvalPlus-Offline/1.0",
                "Accept": "application/octet-stream",
            },
        )
        try:
            print(f"[下载] 第 {attempt}/{retries} 次：{url}", flush=True)
            with urllib.request.urlopen(request, timeout=timeout) as response:
                with temporary.open("wb") as handle:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        handle.write(chunk)
            if temporary.stat().st_size == 0:
                raise RuntimeError("下载结果为空文件")
            temporary.replace(destination)
            return
        except (urllib.error.URLError, TimeoutError, OSError, RuntimeError) as exc:
            last_error = exc
            temporary.unlink(missing_ok=True)
            if attempt < retries:
                time.sleep(min(5 * attempt, 15))

    raise RuntimeError(f"下载失败：{url}: {last_error}")


def prepare_one(
    name: str,
    output_dir: Path,
    mirror_prefix: str,
    force: bool,
    retries: int,
    timeout: int,
) -> dict[str, Any]:
    spec = DATASETS[name]
    path = output_dir / str(spec["filename"])
    official_url = str(spec["official_url"])
    source_url = mirrored_url(official_url, mirror_prefix)

    validation: dict[str, Any] | None = None
    if path.is_file() and not force:
        try:
            validation = inspect_dataset(path, int(spec["expected_tasks"]))
            print(f"[复用] 已验证现有文件：{path}", flush=True)
        except RuntimeError as exc:
            print(f"[替换] 现有文件无效：{exc}", flush=True)
            path.unlink(missing_ok=True)

    if validation is None:
        download_file(source_url, path, retries, timeout)
        validation = inspect_dataset(path, int(spec["expected_tasks"]))

    return {
        "name": name,
        "version": spec["version"],
        "expected_tasks": spec["expected_tasks"],
        "official_url": official_url,
        "download_url": source_url,
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "override_env": spec["override_env"],
        "schema_validation": validation,
    }


def validate_with_evalplus(entries: dict[str, dict[str, Any]]) -> None:
    # EvalPlus 在模块导入时读取覆盖路径，因此必须先设置环境变量。
    os.environ["HUMANEVAL_OVERRIDE_PATH"] = entries["humaneval"]["path"]
    os.environ["MBPP_OVERRIDE_PATH"] = entries["mbpp"]["path"]

    from evalplus.data import get_human_eval_plus, get_mbpp_plus

    human = get_human_eval_plus()
    mbpp = get_mbpp_plus()
    if len(human) != 164:
        raise RuntimeError(f"EvalPlus HumanEval+ 验收失败：{len(human)}")
    if len(mbpp) != 378:
        raise RuntimeError(f"EvalPlus MBPP+ 验收失败：{len(mbpp)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="data/external/evalplus")
    parser.add_argument(
        "--github-mirror",
        default=os.environ.get("EVALPLUS_GITHUB_MIRROR", "https://gh.llkk.cc/"),
        help="GitHub 镜像前缀；传空字符串可尝试直连官方地址",
    )
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.retries <= 0 or args.timeout <= 0:
        raise ValueError("重试次数和超时时间必须为正数")

    output_dir = resolve_path(args.output_dir)
    entries = {
        name: prepare_one(
            name,
            output_dir,
            args.github_mirror,
            args.force,
            args.retries,
            args.timeout,
        )
        for name in DATASETS
    }
    validate_with_evalplus(entries)

    manifest = {
        "schema_version": "codeguide-evalplus-offline-datasets-v2",
        "source_policy": "EvalPlus 官方 GitHub Release 原始 JSONL，经镜像下载",
        "github_mirror": args.github_mirror,
        "datasets": entries,
        "validation": {
            "humaneval_tasks": 164,
            "mbpp_tasks": 378,
            "required_schema_passed": True,
            "evalplus_loader_passed": True,
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"[完成] EvalPlus 官方离线数据已准备：{manifest_path}")


if __name__ == "__main__":
    main()
