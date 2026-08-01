#!/usr/bin/env python3
"""Build the canonical SFT snapshot, source bank, deterministic splits, and manifest."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.source_bank_io import iter_source_bank
from src.data.code_validator import extract_code, validate_syntax
from src.data.loader import (
    _make_id,
    _parse_json_field,
    _rank_taco_python_solutions,
    _strip_html,
    _taco_public_tests,
)

SCHEMA_VERSION = "codeguide-sft-freeze-v1"
SEED = 20260728
DOCKER_IMAGE = (
    "python:3.11.9-slim-bookworm@"
    "sha256:8fb099199b9f2d70342674bd9dbccd3ed03a258f26bbd1d556822c6dfc60c317"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
    return records


def load_verification_cache(path: Path) -> dict[str, dict]:
    return {str(record["id"]): record for record in load_jsonl(path)}


def selected_reference(row: dict, cache_record: dict) -> tuple[str, int | None]:
    input_output = _parse_json_field(row.get("input_output", ""), {})
    if not isinstance(input_output, dict):
        input_output = {}
    fn_name = input_output.get("fn_name")
    io_mode = "call_based" if fn_name else "standard_input"
    candidates = _rank_taco_python_solutions(
        row.get("solutions", "[]"),
        fn_name=fn_name if isinstance(fn_name, str) else None,
        io_mode=io_mode,
    )
    selected_rank = int(cache_record.get("selected_reference_index", -1))
    for candidate in candidates:
        if int(candidate.get("rank", -1)) == selected_rank:
            return str(candidate.get("code") or ""), candidate.get("raw_index")
    raise ValueError(
        f"Verified candidate rank {selected_rank} missing for {cache_record.get('id')}"
    )


def iter_taco_rows(root: Path) -> Iterable[tuple[dict, dict]]:
    global_index = 0
    for parquet_path in sorted(root.glob("train-*.parquet")):
        parquet = pq.ParquetFile(parquet_path)
        row_in_file = 0
        for batch in parquet.iter_batches(batch_size=128):
            for row in batch.to_pylist():
                provenance = {
                    "dataset": "BAAI/TACO",
                    "split": "train",
                    "parquet_file": parquet_path.name,
                    "row_in_file": row_in_file,
                    "global_row_index": global_index,
                }
                yield row, provenance
                row_in_file += 1
                global_index += 1


def build_source_bank(
    taco_root: Path,
    cache: dict[str, dict],
    accepted_by_id: dict[str, dict],
    output: Path,
) -> tuple[dict[str, dict], dict]:
    try:
        import zstandard as zstd
    except ImportError as exc:
        raise RuntimeError("Source bank creation requires: pip install zstandard") from exc

    verified_ids = {
        pid
        for pid, record in cache.items()
        if record.get("reference_verified")
        and float(record.get("reference_pass_rate") or 0.0) >= 1.0
    }
    found: dict[str, dict] = {}
    duplicate_rows: Counter[str] = Counter()
    output.parent.mkdir(parents=True, exist_ok=True)

    compressor = zstd.ZstdCompressor(level=10, threads=0)
    with output.open("wb") as raw_out:
        with compressor.stream_writer(raw_out, closefd=False) as compressed:
            for row, provenance in iter_taco_rows(taco_root):
                question_raw = str(row.get("question") or "")
                question = _strip_html(question_raw)
                if len(question) < 80:
                    continue
                problem_id = _make_id("taco", question)
                if problem_id not in verified_ids:
                    continue
                if problem_id in found:
                    duplicate_rows[problem_id] += 1
                    continue

                cache_record = cache[problem_id]
                reference, raw_solution_index = selected_reference(row, cache_record)
                if not reference:
                    raise ValueError(f"Empty verified reference: {problem_id}")
                input_output = _parse_json_field(row.get("input_output", ""), {})
                if not isinstance(input_output, dict):
                    input_output = {}
                source = str(row.get("source") or "taco")
                fn_name = input_output.get("fn_name")
                io_mode = "call_based" if fn_name else "standard_input"
                accepted = accepted_by_id.get(problem_id)
                accepted_meta = (accepted or {}).get("metadata") or {}
                label_strategy = accepted_meta.get("label_strategy")
                if accepted and not label_strategy:
                    label_strategy = "pedagogical_rewrite"
                record = {
                    "schema_version": SCHEMA_VERSION,
                    "problem_id": problem_id,
                    "question": question_raw,
                    "normalized_question": question,
                    "source": source,
                    "difficulty": str(row.get("difficulty") or "unknown").lower(),
                    "tags": [str(x) for x in (row.get("tags") or [])],
                    "raw_tags": [str(x) for x in (row.get("raw_tags") or [])],
                    "skill_types": [str(x) for x in (row.get("skill_types") or [])],
                    "io_mode": io_mode,
                    "fn_name": fn_name,
                    "starter_code": str(row.get("starter_code") or ""),
                    "test_cases": _taco_public_tests(input_output, source=source),
                    "reference_solution": reference,
                    "selected_reference_index": cache_record.get("selected_reference_index"),
                    "selected_raw_solution_index": cache_record.get(
                        "selected_raw_solution_index", raw_solution_index
                    ),
                    "reference_pass_rate": float(cache_record.get("reference_pass_rate") or 0.0),
                    "reference_hash": sha256_text(reference),
                    "reference_verification": {
                        "passed_cases": cache_record.get("passed_cases"),
                        "total_cases": cache_record.get("total_cases"),
                        "attempted_candidates": cache_record.get("attempted_candidates"),
                    },
                    "original_data": {
                        **provenance,
                        "question_hash": sha256_text(question_raw),
                    },
                    "sft_present": accepted is not None,
                    "label_strategy": label_strategy,
                    "code_source": (
                        "verified_reference_with_comments"
                        if label_strategy == "reference_locked"
                        else "teacher_generated_docker_verified"
                        if accepted
                        else None
                    ),
                }
                found[problem_id] = {
                    "source": source,
                    "difficulty": record["difficulty"],
                    "io_mode": io_mode,
                    "fn_name": fn_name,
                    "starter_code": record["starter_code"],
                    "test_case_count": len(record["test_cases"]),
                    "reference_hash": record["reference_hash"],
                    "original_data": record["original_data"],
                }
                compressed.write((json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8"))

    missing = verified_ids - set(found)
    if missing:
        raise ValueError(f"Missing {len(missing)} verified IDs in TACO parquet; first={sorted(missing)[:5]}")
    return found, {
        "records": len(found),
        "duplicate_raw_rows_ignored": sum(duplicate_rows.values()),
        "duplicate_problem_ids": len(duplicate_rows),
    }


def declared_functions(code: str) -> set[str]:
    tree = ast.parse(code)
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names.add(node.name)
    return names


def canonicalize(
    accepted_records: list[dict],
    rejected_ids: set[str],
    source_index: dict[str, dict],
    output: Path,
) -> tuple[list[dict], dict]:
    by_id: dict[str, list[dict]] = defaultdict(list)
    for record in accepted_records:
        by_id[str(record.get("id") or "")].append(record)

    conflicts = {pid: rows for pid, rows in by_id.items() if len(rows) > 1}
    fatal: list[dict] = []
    warnings: Counter[str] = Counter()
    backfills: Counter[str] = Counter()
    canonical: list[dict] = []
    response_lengths: list[int] = []
    code_lengths: list[int] = []
    strategy_counts: Counter[str] = Counter()
    io_counts: Counter[str] = Counter()
    difficulty_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()

    for problem_id in sorted(by_id):
        rows = by_id[problem_id]
        # Formal accepted is append-only; if duplicates ever appear, retain the
        # latest line as the final formal-run version and report the conflict.
        record = rows[-1]
        if problem_id in rejected_ids:
            fatal.append({"id": problem_id, "reason": "accepted_rejected_overlap"})
            continue
        messages = record.get("messages")
        if not isinstance(messages, list) or [m.get("role") for m in messages] != [
            "system",
            "user",
            "assistant",
        ]:
            fatal.append({"id": problem_id, "reason": "invalid_chatml_roles"})
            continue
        assistant = str(messages[2].get("content") or "")
        user = str(messages[1].get("content") or "")
        code = extract_code(assistant)
        if not code:
            fatal.append({"id": problem_id, "reason": "missing_python_code_block"})
            continue
        syntax_ok, syntax_error = validate_syntax(code)
        if not syntax_ok:
            fatal.append({"id": problem_id, "reason": "syntax_error", "detail": syntax_error})
            continue
        # Explanations may legitimately continue with complexity notes after
        # the final code fence. Truncation is indicated by an unclosed fence;
        # code completeness is independently checked by AST parsing above.
        if assistant.count("```") % 2:
            fatal.append({"id": problem_id, "reason": "obvious_truncation"})
            continue
        lowered_user = user.lower()
        if "reference_solution" in lowered_user or "【参考解代码】" in user:
            fatal.append({"id": problem_id, "reason": "reference_leak_marker"})
            continue

        metadata = dict(record.get("metadata") or {})
        if float(metadata.get("pass_rate") or 0.0) < 1.0 or not metadata.get("reference_verified"):
            fatal.append({"id": problem_id, "reason": "missing_final_execution_evidence"})
            continue
        source = source_index.get(problem_id)
        if not source:
            fatal.append({"id": problem_id, "reason": "missing_source_bank_record"})
            continue
        for key in ("source", "difficulty", "io_mode", "fn_name", "starter_code"):
            if metadata.get(key) in (None, "") and source.get(key) not in (None, ""):
                metadata[key] = source[key]
                backfills[key] += 1
        if not metadata.get("label_strategy"):
            metadata["label_strategy"] = "pedagogical_rewrite"
            backfills["label_strategy"] += 1
        if not metadata.get("code_source"):
            metadata["code_source"] = (
                "verified_reference_with_comments"
                if metadata["label_strategy"] == "reference_locked"
                else "teacher_generated_docker_verified"
            )
            backfills["code_source"] += 1

        if metadata.get("io_mode") == "call_based":
            fn_name = str(metadata.get("fn_name") or "")
            if not fn_name or fn_name not in declared_functions(code):
                fatal.append({"id": problem_id, "reason": "call_based_fn_name_mismatch"})
                continue
        elif metadata.get("io_mode") == "standard_input":
            # Execution pass evidence is authoritative; this warning only
            # catches surprising source text for later review.
            if not re.search(r"\b(input|print)\s*\(|sys\.(stdin|stdout)|os\.(read|write)", code):
                warnings["standard_input_no_obvious_io_token"] += 1

        compact_metadata = {
            "source": metadata.get("source"),
            "difficulty": metadata.get("difficulty"),
            "tags": metadata.get("tags") or [],
            "skill_types": metadata.get("skill_types") or [],
            "raw_tags": metadata.get("raw_tags") or [],
            "url": metadata.get("url") or "",
            "io_mode": metadata.get("io_mode"),
            "fn_name": metadata.get("fn_name"),
            "starter_code": metadata.get("starter_code") or "",
            "test_case_count": source["test_case_count"],
            "reward_compatible": bool(metadata.get("reward_compatible")),
            "pass_rate": 1.0,
            "reference_verified": True,
            "reference_pass_rate": float(metadata.get("reference_pass_rate") or 0.0),
            "selected_reference_index": metadata.get("selected_reference_index"),
            "selected_raw_solution_index": metadata.get("selected_raw_solution_index"),
            "label_strategy": metadata["label_strategy"],
            "code_source": metadata["code_source"],
            "source_bank_id": problem_id,
            "reference_hash": source["reference_hash"],
            "verification_evidence": {
                "kind": "formal_generation_docker_pass",
                "pass_rate": 1.0,
                "container_image": DOCKER_IMAGE,
                "reused_for_freeze": True,
            },
            "freeze_schema_version": SCHEMA_VERSION,
        }
        canonical.append({"id": problem_id, "messages": messages, "metadata": compact_metadata})
        response_lengths.append(len(assistant))
        code_lengths.append(len(code))
        strategy_counts[metadata["label_strategy"]] += 1
        io_counts[str(metadata.get("io_mode"))] += 1
        difficulty_counts[str(metadata.get("difficulty"))] += 1
        source_counts[str(metadata.get("source"))] += 1

    if fatal:
        raise ValueError(f"Canonical validation failed for {len(fatal)} records; first={fatal[:10]}")

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for record in canonical:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def basic(values: list[int]) -> dict:
        ordered = sorted(values)
        def q(p: float) -> int:
            return ordered[min(len(ordered) - 1, math.ceil(p * len(ordered)) - 1)]
        return {"min": ordered[0], "p50": q(.5), "p90": q(.9), "p95": q(.95), "p99": q(.99), "max": ordered[-1], "mean": round(sum(ordered)/len(ordered), 2)}

    return canonical, {
        "records": len(canonical),
        "input_duplicate_records": sum(len(v) - 1 for v in conflicts.values()),
        "conflicting_problem_ids": len(conflicts),
        "conflict_rule": "retain last formal accepted record and report conflict",
        "metadata_backfills": dict(backfills),
        "warnings": dict(warnings),
        "label_strategy": dict(strategy_counts),
        "io_mode": dict(io_counts),
        "difficulty": dict(difficulty_counts),
        "source": dict(source_counts),
        "assistant_char_length": basic(response_lengths),
        "code_char_length": basic(code_lengths),
    }


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def training_distributions(records: list[dict]) -> dict:
    return {
        field: dict(sorted(Counter(str(record["metadata"].get(field)) for record in records).items()))
        for field in ("label_strategy", "io_mode", "difficulty", "source")
    }


def stable_order(ids: Iterable[str], seed: int, salt: str) -> list[str]:
    return sorted(ids, key=lambda pid: hashlib.sha256(f"{seed}:{salt}:{pid}".encode()).digest())


def stratified_take(records: list[dict], take: int, seed: int, salt: str) -> list[str]:
    groups: dict[tuple, list[str]] = defaultdict(list)
    for record in records:
        meta = record["metadata"]
        key = (meta.get("difficulty"), meta.get("io_mode"), meta.get("label_strategy"), meta.get("source"))
        groups[key].append(record["id"])
    quotas = {key: len(ids) * take / len(records) for key, ids in groups.items()}
    assigned = {key: math.floor(value) for key, value in quotas.items()}
    remaining = take - sum(assigned.values())
    ranked_groups = sorted(groups, key=lambda key: (-(quotas[key] - assigned[key]), str(key)))
    for key in ranked_groups[:remaining]:
        assigned[key] += 1
    chosen = []
    for key, ids in groups.items():
        chosen.extend(stable_order(ids, seed, f"{salt}:{key}")[: assigned[key]])
    return stable_order(chosen, seed, salt)


def write_id_file(path: Path, ids: list[str], *, seed: int, parent: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema_version": SCHEMA_VERSION, "seed": seed, "parent": parent, "count": len(ids), "ids": ids}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_splits(records: list[dict], split_dir: Path, seed: int) -> dict:
    dev_count = round(len(records) * 0.05)
    dev_ids = stratified_take(records, dev_count, seed, "sft_dev")
    dev_set = set(dev_ids)
    train_records = [record for record in records if record["id"] not in dev_set]
    dev_records = [record for record in records if record["id"] in dev_set]
    train_ids = stable_order((record["id"] for record in train_records), seed, "sft_train")
    grpo_train_ids = stratified_take(train_records, min(900, len(train_records)), seed, "grpo_train")
    grpo_validation_ids = stratified_take(dev_records, min(100, len(dev_records)), seed, "grpo_validation")
    outputs = {
        "sft_train": (split_dir / "sft_train_ids.json", train_ids),
        "sft_dev": (split_dir / "sft_dev_ids.json", dev_ids),
        "grpo_train": (split_dir / "grpo_train_ids.json", grpo_train_ids),
        "grpo_validation": (split_dir / "grpo_validation_ids.json", grpo_validation_ids),
    }
    for name, (path, ids) in outputs.items():
        write_id_file(path, ids, seed=seed, parent="data/final/sft_accepted.jsonl")
    if set(train_ids) & set(dev_ids) or set(grpo_train_ids) & set(dev_ids):
        raise ValueError("Split leakage detected")
    if set(train_ids) | set(dev_ids) != {record["id"] for record in records}:
        raise ValueError("SFT split coverage mismatch")
    if not set(grpo_train_ids) <= set(train_ids) or not set(grpo_validation_ids) <= set(dev_ids):
        raise ValueError("GRPO subset parent mismatch")
    return {name: {"path": str(path).replace("\\", "/"), "count": len(ids), "sha256": sha256_file(path)} for name, (path, ids) in outputs.items()}


def rejected_summary(records: list[dict]) -> dict:
    failures = Counter(str(record.get("failure_type") or "unknown") for record in records)
    return {
        "records": len(records),
        "unique_ids": len({str(record.get("id")) for record in records}),
        "failure_types": dict(sorted(failures.items())),
        "io_mode": dict(sorted(Counter(str((record.get("metadata") or {}).get("io_mode")) for record in records).items())),
        "difficulty": dict(sorted(Counter(str((record.get("metadata") or {}).get("difficulty")) for record in records).items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accepted", type=Path, default=Path("data/sft_train_ref_label_accepted.jsonl"))
    parser.add_argument("--rejected", type=Path, default=Path("data/sft_train_ref_label_rejected.jsonl"))
    parser.add_argument("--reference-cache", type=Path, default=Path("data/cache/taco_reference_verification_train_full.jsonl"))
    parser.add_argument("--taco-root", type=Path, default=Path("data/raw/TACO/ALL"))
    parser.add_argument("--canonical", type=Path, default=Path("data/final/sft_accepted.jsonl"))
    parser.add_argument("--all-validated", type=Path, default=Path("data/final/sft_accepted_all_validated.jsonl"))
    parser.add_argument("--source-bank", type=Path, default=Path("data/final/taco_verified_source_bank.jsonl.zst"))
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/sft_manifest.json"))
    parser.add_argument("--report", type=Path, default=Path("reports/data_freeze_report.md"))
    parser.add_argument("--split-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--token-stats", type=Path, default=Path("data/manifests/token_length_stats_all_validated.json"))
    parser.add_argument("--length-excluded", type=Path, default=Path("data/manifests/sft_length_excluded.json"))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    accepted = load_jsonl(args.accepted)
    rejected = load_jsonl(args.rejected)
    accepted_by_id = {str(record.get("id")): record for record in accepted}
    rejected_ids = {str(record.get("id")) for record in rejected}
    cache = load_verification_cache(args.reference_cache)

    source_index, source_stats = build_source_bank(args.taco_root, cache, accepted_by_id, args.source_bank)
    all_validated, canonical_stats = canonicalize(
        accepted, rejected_ids, source_index, args.all_validated
    )
    canonical = all_validated
    excluded_rows: list[dict] = []
    if args.token_stats.exists():
        token_stats = json.loads(args.token_stats.read_text(encoding="utf-8"))
        if token_stats.get("data_sha256") != sha256_file(args.all_validated):
            raise ValueError(
                "Token stats do not match all-validated SFT snapshot; rerun token audit first"
            )
        threshold = int(token_stats["recommended_max_seq_length"])
        excluded_rows = list(token_stats.get("oversize_samples", {}).get(str(threshold), []))
        excluded_ids = {row["problem_id"] for row in excluded_rows}
        canonical = [record for record in all_validated if record["id"] not in excluded_ids]
    write_jsonl(args.canonical, canonical)
    args.length_excluded.parent.mkdir(parents=True, exist_ok=True)
    args.length_excluded.write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "policy": "exclude samples exceeding recommended max_seq_length; never truncate final code",
                "count": len(excluded_rows),
                "samples": excluded_rows,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    canonical_stats["all_validated_records"] = len(all_validated)
    canonical_stats["length_excluded_records"] = len(excluded_rows)
    canonical_stats["records"] = len(canonical)
    canonical_stats.update(training_distributions(canonical))
    split_stats = build_splits(canonical, args.split_dir, args.seed)
    rejected_stats = rejected_summary(rejected)

    git_commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False).stdout.strip()
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "canonical_sft": str(args.canonical).replace("\\", "/"),
        "source_bank": str(args.source_bank).replace("\\", "/"),
        "accepted_total": len(canonical),
        "all_validated_accepted_total": len(all_validated),
        "canonical_stats": canonical_stats,
        "source_bank_stats": source_stats,
        "rejected": rejected_stats,
        "verifier": {"implementation": "src/reward/execution.py::verify_code", "container_image": DOCKER_IMAGE},
        "verification_evidence": {"cache_reused": len(all_validated), "actually_reverified_during_freeze": 0, "source_bank_sample_report": "data/manifests/source_bank_verification.json"},
        "length_policy": {"stats": str(args.token_stats).replace("\\", "/"), "excluded": str(args.length_excluded).replace("\\", "/"), "excluded_count": len(excluded_rows), "rule": "no right truncation that can damage final code"},
        "data_sources": {"accepted": str(args.accepted).replace("\\", "/"), "rejected": str(args.rejected).replace("\\", "/"), "reference_cache": str(args.reference_cache).replace("\\", "/"), "taco_train": "data/raw/TACO/ALL/train-*.parquet"},
        "dedup_key": "problem_id",
        "conflict_rule": canonical_stats["conflict_rule"],
        "seed": args.seed,
        "git_commit_at_generation": git_commit,
        "splits": split_stats,
        "files": {},
    }
    for name, path in {"canonical_sft": args.canonical, "all_validated_sft": args.all_validated, "source_bank": args.source_bank, "length_excluded": args.length_excluded, "accepted_input": args.accepted, "rejected_input": args.rejected, "reference_cache": args.reference_cache}.items():
        manifest["files"][name] = {"path": str(path).replace("\\", "/"), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        "# CodeGuide 数据冻结报告\n\n"
        f"- Canonical SFT：`{manifest['canonical_sft']}`，{len(canonical):,} 条，SHA-256 `{manifest['files']['canonical_sft']['sha256']}`。\n"
        f"- 全部已验证标签快照：{len(all_validated):,} 条；因长度隔离 {len(excluded_rows)} 条，绝不右截断最终代码。\n"
        f"- Verified source bank：`{manifest['source_bank']}`，{source_stats['records']:,} 条，SHA-256 `{manifest['files']['source_bank']['sha256']}`。\n"
        f"- 标签策略：`{json.dumps(canonical_stats['label_strategy'], ensure_ascii=False)}`。\n"
        f"- I/O：`{json.dumps(canonical_stats['io_mode'], ensure_ascii=False)}`。\n"
        f"- 难度：`{json.dumps(canonical_stats['difficulty'], ensure_ascii=False)}`。\n"
        f"- Rejected：{rejected_stats['records']} 条，`{json.dumps(rejected_stats['failure_types'], ensure_ascii=False)}`。\n"
        f"- Metadata 回填：`{json.dumps(canonical_stats['metadata_backfills'], ensure_ascii=False)}`。\n"
        f"- 冲突：{canonical_stats['conflicting_problem_ids']} 个 problem ID；规则：{canonical_stats['conflict_rule']}。\n"
        f"- 验证口径：复用 {len(canonical):,} 条正式 Docker pass 证据；source bank 抽样复验结果另见 `data/manifests/source_bank_verification.json`。\n"
        f"- Split：`{json.dumps({k:v['count'] for k,v in split_stats.items()}, ensure_ascii=False)}`，固定种子 {args.seed}。\n",
        encoding="utf-8",
    )
    print(json.dumps({"canonical": len(canonical), "source_bank": source_stats["records"], "rejected": rejected_stats["records"], "splits": {k:v["count"] for k,v in split_stats.items()}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
