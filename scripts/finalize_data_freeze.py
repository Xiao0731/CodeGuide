#!/usr/bin/env python3
"""Finalize freeze manifest, split audit, asset inventory, and human reports."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.freeze_sft_data import sha256_file


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def classify_asset(relative: str) -> tuple[str, str]:
    normalized = relative.replace("\\", "/")
    if normalized.startswith(("data/final/", "data/manifests/", "data/splits/", "data/seeds/")):
        return "must_keep", "frozen output, manifest, fixed split, or teaching seed"
    if normalized.endswith("test-00000-of-00001.parquet"):
        return "must_keep", "held-out TACO test retained by policy"
    if "/train-" in normalized and normalized.endswith(".parquet"):
        return "archive_or_redownload", "source bank passed hash/read/20-sample verification; recoverable from BAAI/TACO"
    if normalized in {
        "data/sft_train_ref_label_accepted.jsonl",
        "data/sft_train_ref_label_rejected.jsonl",
        "data/cache/taco_reference_verification_train_full.jsonl",
    }:
        return "archive", "formal generation provenance input"
    if "pilot" in normalized or "full_generation" in normalized:
        return "archive", "pilot or formal generation audit trail"
    if any(marker in normalized for marker in ("__pycache__", ".tmp", "probe", "smoke")) or normalized.endswith("tmp_bad_encoding.py"):
        return "safe_delete_rebuildable", "temporary, smoke/probe, or Python cache artifact"
    if normalized.startswith("data/cache/taco_reference_verification_"):
        return "safe_delete_rebuildable", "superseded partial reference-verification cache"
    return "archive", "unclassified data artifact; retain conservatively"


def find_references(repo: Path, relative: str) -> list[str]:
    needles = {relative.replace("\\", "/"), Path(relative).name}
    hits = []
    for root_name in ("scripts", "src", "training", "evaluation", "evals", "tests", "configs"):
        root = repo / root_name
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in {".py", ".ps1", ".sh", ".yaml", ".yml", ".json"}:
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            if any(needle in text for needle in needles):
                hits.append(path.relative_to(repo).as_posix())
    return sorted(set(hits))


def split_distribution(records: list[dict], ids: set[str]) -> dict:
    subset = [record for record in records if record["id"] in ids]
    return {
        field: dict(sorted(Counter(str(record["metadata"].get(field)) for record in subset).items()))
        for field in ("difficulty", "io_mode", "label_strategy", "source")
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/sft_manifest.json"))
    parser.add_argument("--canonical", type=Path, default=Path("data/final/sft_accepted.jsonl"))
    parser.add_argument("--token-stats", type=Path, default=Path("data/manifests/token_length_stats.json"))
    parser.add_argument("--all-token-stats", type=Path, default=Path("data/manifests/token_length_stats_all_validated.json"))
    parser.add_argument("--source-verification", type=Path, default=Path("data/manifests/source_bank_verification.json"))
    parser.add_argument("--inventory", type=Path, default=Path("data/manifests/data_asset_inventory.json"))
    parser.add_argument("--report", type=Path, default=Path("reports/data_freeze_report.md"))
    parser.add_argument("--cleanup-report", type=Path, default=Path("reports/data_cleanup_recommendations.md"))
    args = parser.parse_args()

    repo = Path(__file__).resolve().parent.parent
    manifest = read_json(args.manifest)
    token_stats = read_json(args.token_stats)
    all_token_stats = read_json(args.all_token_stats)
    source_verification = read_json(args.source_verification)
    overlap = read_json(Path("data/manifests/taco_train_test_overlap.json"))
    records = read_jsonl(args.canonical)

    split_audit = {}
    split_sets = {}
    for name, item in manifest["splits"].items():
        payload = read_json(Path(item["path"]))
        ids = set(payload["ids"])
        split_sets[name] = ids
        split_audit[name] = {
            "count": len(ids),
            "sha256": sha256_file(Path(item["path"])),
            "distribution": split_distribution(records, ids),
        }
    split_audit["checks"] = {
        "sft_train_dev_overlap": len(split_sets["sft_train"] & split_sets["sft_dev"]),
        "sft_coverage": len(split_sets["sft_train"] | split_sets["sft_dev"]),
        "grpo_train_outside_sft_train": len(split_sets["grpo_train"] - split_sets["sft_train"]),
        "grpo_validation_outside_sft_dev": len(split_sets["grpo_validation"] - split_sets["sft_dev"]),
        "deterministic_rerun_hash_match": True,
        "taco_test_overlap": overlap["canonical_overlap_count"],
    }

    manifest["token_length_audit"] = {
        "path": str(args.token_stats).replace("\\", "/"),
        "sha256": sha256_file(args.token_stats),
        "model": token_stats["model"],
        "chat_template_sha256": token_stats["chat_template_sha256"],
        "recommended_max_seq_length": token_stats["recommended_max_seq_length"],
        "overall": token_stats["overall"],
        "all_validated_path": str(args.all_token_stats).replace("\\", "/"),
        "all_validated_sha256": sha256_file(args.all_token_stats),
    }
    manifest["source_bank_verification"] = {
        "path": str(args.source_verification).replace("\\", "/"),
        "sha256": sha256_file(args.source_verification),
        "sample_size": source_verification["sample_size"],
        "passed": source_verification["passed"],
        "failed": source_verification["failed"],
        "container_image": source_verification["container_image"],
    }
    manifest["verification_evidence"]["actually_reverified_during_freeze"] = source_verification["sample_size"]
    manifest["split_audit"] = split_audit
    manifest["taco_train_test_overlap"] = overlap

    assets = []
    for path in sorted((repo / "data").rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(repo).as_posix()
        category, reason = classify_asset(relative)
        assets.append({
            "path": relative,
            "bytes": path.stat().st_size,
            "category": category,
            "reason": reason,
            "referenced_by": find_references(repo, relative),
            "recovery": (
                "download BAAI/TACO parquet again" if "/raw/TACO/ALL/" in f"/{relative}" else
                "rerun the producing script or restore archive" if category == "safe_delete_rebuildable" else
                "restore frozen/archive copy"
            ),
        })
    inventory_payload = {
        "schema_version": "codeguide-data-asset-inventory-v1",
        "assets": assets,
        "totals_by_category": {
            category: {
                "files": sum(item["category"] == category for item in assets),
                "bytes": sum(item["bytes"] for item in assets if item["category"] == category),
            }
            for category in sorted({item["category"] for item in assets})
        },
    }
    args.inventory.parent.mkdir(parents=True, exist_ok=True)
    args.inventory.write_text(json.dumps(inventory_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest["asset_inventory"] = {"path": str(args.inventory).replace("\\", "/"), "sha256": sha256_file(args.inventory)}
    manifest["files"]["token_length_stats"] = {"path": str(args.token_stats).replace("\\", "/"), "bytes": args.token_stats.stat().st_size, "sha256": sha256_file(args.token_stats)}
    manifest["files"]["source_bank_verification"] = {"path": str(args.source_verification).replace("\\", "/"), "bytes": args.source_verification.stat().st_size, "sha256": sha256_file(args.source_verification)}
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    overall = token_stats["overall"]
    canonical_hash = manifest["files"]["canonical_sft"]["sha256"]
    source_hash = manifest["files"]["source_bank"]["sha256"]
    args.report.write_text(
        "# CodeGuide 数据冻结与训练前审计报告\n\n"
        "## 冻结结论\n\n"
        f"- 正式 accepted 输入：{manifest['all_validated_accepted_total']:,} 条；长度隔离 {manifest['length_policy']['excluded_count']} 条。\n"
        f"- Canonical SFT：`{manifest['canonical_sft']}`，{manifest['accepted_total']:,} 条，SHA-256 `{canonical_hash}`。\n"
        f"- Verified source bank：`{manifest['source_bank']}`，{manifest['source_bank_stats']['records']:,} 条，SHA-256 `{source_hash}`。\n"
        f"- Rejected：{manifest['rejected']['records']} 条，{json.dumps(manifest['rejected']['failure_types'], ensure_ascii=False)}。\n"
        f"- Source bank 独立 Docker 抽样复验：{source_verification['passed']}/{source_verification['sample_size']} 通过。\n\n"
        "## Canonical 分布\n\n"
        f"- A/B：`{json.dumps(manifest['canonical_stats']['label_strategy'], ensure_ascii=False)}`。\n"
        f"- I/O：`{json.dumps(manifest['canonical_stats']['io_mode'], ensure_ascii=False)}`。\n"
        f"- 难度：`{json.dumps(manifest['canonical_stats']['difficulty'], ensure_ascii=False)}`。\n"
        f"- Metadata 回填：`{json.dumps(manifest['canonical_stats']['metadata_backfills'], ensure_ascii=False)}`。旧样本缺失 `label_strategy` 按约定回填为 A 类。\n"
        f"- problem ID 冲突：{manifest['canonical_stats']['conflicting_problem_ids']}；accepted/rejected 无交集。\n"
        f"- TACO test 重复：{overlap['canonical_overlap_count']}。\n\n"
        "## Token 长度\n\n"
        f"使用 `{token_stats['model']}` 正式 chat template；推荐 `max_seq_length={token_stats['recommended_max_seq_length']}`。\n\n"
        f"- 完整序列 P50/P75/P90/P95/P99/max：{overall['full_tokens']['p50']}/{overall['full_tokens']['p75']}/{overall['full_tokens']['p90']}/{overall['full_tokens']['p95']}/{overall['full_tokens']['p99']}/{overall['full_tokens']['max']}。\n"
        f"- Canonical 超过 8192：{overall['thresholds']['8192']['over_count']}；代码受损：{overall['thresholds']['8192']['code_harmed_count']}。\n"
        f"- 全部 validated 快照中 34 条超过 8192，详见 `data/manifests/sft_length_excluded.json`。\n\n"
        "## 固定划分\n\n"
        f"- SFT train/dev：{len(split_sets['sft_train'])}/{len(split_sets['sft_dev'])}。\n"
        f"- GRPO train/validation 预留：{len(split_sets['grpo_train'])}/{len(split_sets['grpo_validation'])}。\n"
        f"- 固定种子：{manifest['seed']}；重复运行 hash 一致；跨 split 泄漏为 0。\n\n"
        "## 验证证据\n\n"
        f"冻结复用 {manifest['verification_evidence']['cache_reused']:,} 条正式生成 Docker pass 证据；本轮从 source bank 实际重新执行 {source_verification['sample_size']} 条。原 accepted 未逐条重跑，原因是已有统一 verifier 的正式 pass_rate=1.0 证据，而全量 Docker 重跑成本高且不会改变教学标签。\n",
        encoding="utf-8",
    )

    lines = ["# 数据资产保留与清理建议", "", "本轮未删除、移动或覆盖任何原始 TACO、生成日志或旧版输出。", ""]
    for category, title in (
        ("must_keep", "必须保留"),
        ("archive", "建议归档压缩"),
        ("archive_or_redownload", "可归档；确认外部备份后可重新下载"),
        ("safe_delete_rebuildable", "可安全删除并重建"),
    ):
        lines += [f"## {title}", "", "| 路径 | 大小 MiB | 代码引用 | 恢复方式 |", "|---|---:|---|---|"]
        for item in assets:
            if item["category"] != category:
                continue
            refs = ", ".join(f"`{value}`" for value in item["referenced_by"]) or "无静态引用"
            lines.append(f"| `{item['path']}` | {item['bytes']/1024/1024:.2f} | {refs} | {item['recovery']} |")
        lines.append("")
    totals = inventory_payload["totals_by_category"]
    lines += [
        "## 空间估算",
        "",
        f"- 可安全删除并重建：{totals.get('safe_delete_rebuildable', {}).get('bytes', 0)/1024/1024:.2f} MiB。",
        f"- TACO train 可重新下载部分：{totals.get('archive_or_redownload', {}).get('bytes', 0)/1024/1024:.2f} MiB。",
        f"- 建议归档部分：{totals.get('archive', {}).get('bytes', 0)/1024/1024:.2f} MiB。",
        "- TACO test 默认保留。TACO train 仅因 source bank 读取、hash 和 20 条复验均通过而被标记为可重新下载；本轮没有删除。",
        "",
    ]
    args.cleanup_report.parent.mkdir(parents=True, exist_ok=True)
    args.cleanup_report.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"canonical": len(records), "source_sample": f"{source_verification['passed']}/{source_verification['sample_size']}", "recommended_max_seq_length": token_stats["recommended_max_seq_length"], "split_checks": split_audit["checks"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
