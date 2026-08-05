#!/usr/bin/env python3
"""从 Hugging Face 固定下载 EvalPlus 数据，并生成本地离线覆盖文件。"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

DATASETS = {
    "humaneval": {
        "repo_id": "evalplus/humanevalplus",
        "expected_tasks": 164,
        "filename": "HumanEvalPlus-v0.1.10.jsonl.gz",
        "override_env": "HUMANEVAL_OVERRIDE_PATH",
    },
    "mbpp": {
        "repo_id": "evalplus/mbppplus",
        "expected_tasks": 378,
        "filename": "MbppPlus-v0.2.0.jsonl.gz",
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


def json_value(value: Any) -> Any:
    """把 datasets 返回值递归转换成标准 JSON 可序列化对象。"""
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value


def select_split(dataset_dict: Any) -> tuple[str, Any]:
    if not hasattr(dataset_dict, "keys"):
        return "direct", dataset_dict
    names = list(dataset_dict.keys())
    if not names:
        raise RuntimeError("Hugging Face 数据集没有任何 split")
    split = "test" if "test" in names else names[0]
    return split, dataset_dict[split]


def write_deterministic_jsonl_gz(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as raw:
        # 固定 mtime，确保相同输入生成相同压缩文件哈希。
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
            for row in rows:
                line = json.dumps(
                    json_value(row),
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                compressed.write((line + "\n").encode("utf-8"))
    temporary.replace(path)


def load_one(name: str, output_dir: Path, force: bool) -> dict[str, Any]:
    try:
        from datasets import load_dataset
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError(
            "缺少 datasets 或 huggingface_hub；请先安装项目的数据依赖"
        ) from exc

    spec = DATASETS[name]
    output_path = output_dir / str(spec["filename"])

    api = HfApi(endpoint=os.environ.get("HF_ENDPOINT"))
    info = api.dataset_info(str(spec["repo_id"]))
    revision = str(info.sha)

    if not output_path.is_file() or force:
        loaded = load_dataset(
            str(spec["repo_id"]),
            revision=revision,
        )
        split_name, split = select_split(loaded)
        rows = [dict(row) for row in split]
        if len(rows) != int(spec["expected_tasks"]):
            raise RuntimeError(
                f"{name} 题数不匹配："
                f"expected={spec['expected_tasks']} actual={len(rows)}"
            )
        write_deterministic_jsonl_gz(output_path, rows)
    else:
        split_name = "已存在，未重新下载"

    return {
        "name": name,
        "repo_id": spec["repo_id"],
        "revision": revision,
        "split": split_name,
        "expected_tasks": spec["expected_tasks"],
        "path": str(output_path.resolve()),
        "sha256": sha256_file(output_path),
        "override_env": spec["override_env"],
    }


def validate_with_evalplus(entries: dict[str, dict[str, Any]]) -> None:
    # EvalPlus 在模块导入时读取覆盖路径，所以必须先设置环境变量。
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
    parser.add_argument(
        "--output-dir",
        default="data/external/evalplus",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    output_dir = resolve_path(args.output_dir)
    entries = {
        name: load_one(name, output_dir, args.force)
        for name in DATASETS
    }
    validate_with_evalplus(entries)

    manifest = {
        "schema_version": "codeguide-evalplus-offline-datasets-v1",
        "hf_endpoint": os.environ.get("HF_ENDPOINT", "https://huggingface.co"),
        "datasets": entries,
        "validation": {
            "humaneval_tasks": 164,
            "mbpp_tasks": 378,
            "evalplus_loader_passed": True,
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"[完成] EvalPlus 离线数据已准备：{manifest_path}")


if __name__ == "__main__":
    main()
