#!/usr/bin/env python3
"""Audit canonical SFT lengths with the base model's official chat template."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.code_validator import extract_code

THRESHOLDS = (2048, 4096, 6144, 8192)


def percentile(values: list[int], p: float) -> int:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(p * len(ordered)) - 1)]


def stats(values: list[int]) -> dict:
    return {
        "count": len(values),
        "p50": percentile(values, .50),
        "p75": percentile(values, .75),
        "p90": percentile(values, .90),
        "p95": percentile(values, .95),
        "p99": percentile(values, .99),
        "max": max(values),
        "mean": round(sum(values) / len(values), 2),
    }


def summarize_group(rows: list[dict]) -> dict:
    result = {
        "count": len(rows),
        "prompt_tokens": stats([row["prompt_tokens"] for row in rows]),
        "completion_tokens": stats([row["completion_tokens"] for row in rows]),
        "full_tokens": stats([row["full_tokens"] for row in rows]),
        "thresholds": {},
    }
    for threshold in THRESHOLDS:
        over = [row for row in rows if row["full_tokens"] > threshold]
        harmed = [row for row in over if row[f"code_at_{threshold}"] != "preserved"]
        result["thresholds"][str(threshold)] = {
            "over_count": len(over),
            "over_ratio": round(len(over) / len(rows), 6),
            "code_harmed_count": len(harmed),
            "code_harmed_ratio": round(len(harmed) / len(rows), 6),
            "code_missing_count": sum(row[f"code_at_{threshold}"] == "missing" for row in over),
            "code_partial_count": sum(row[f"code_at_{threshold}"] == "partial" for row in over),
        }
    return result


def choose_recommendation(overall: dict) -> int:
    for threshold in (4096, 6144, 8192):
        item = overall["thresholds"][str(threshold)]
        if item["over_ratio"] <= 0.01 and item["code_harmed_ratio"] <= 0.005:
            return threshold
    return 8192


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/final/sft_accepted.jsonl"))
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--output", type=Path, default=Path("data/manifests/token_length_stats.json"))
    parser.add_argument("--report", type=Path, default=Path("reports/token_length_audit.md"))
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        local_files_only=args.local_files_only,
        trust_remote_code=False,
    )
    rows = []
    groups: dict[str, dict[str, list[dict]]] = {
        "label_strategy": defaultdict(list),
        "io_mode": defaultdict(list),
        "difficulty": defaultdict(list),
    }
    with args.data.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            messages = record["messages"]
            prompt_ids = tokenizer.apply_chat_template(
                messages[:2], tokenize=True, add_generation_prompt=True
            )
            full_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
            assistant = str(messages[2].get("content") or "")
            code = extract_code(assistant) or ""
            code_offset = full_text.rfind(code) if code else -1
            if code_offset < 0:
                raise ValueError(f"Cannot locate final code in rendered chat: {record['id']}")
            code_start = len(tokenizer(full_text[:code_offset], add_special_tokens=False)["input_ids"])
            code_end = len(tokenizer(full_text[: code_offset + len(code)], add_special_tokens=False)["input_ids"])
            row = {
                "problem_id": record["id"],
                "prompt_tokens": len(prompt_ids),
                "completion_tokens": max(0, len(full_ids) - len(prompt_ids)),
                "full_tokens": len(full_ids),
                "code_start_token": code_start,
                "code_end_token": code_end,
            }
            for threshold in THRESHOLDS:
                row[f"code_at_{threshold}"] = (
                    "preserved"
                    if threshold >= code_end
                    else "missing"
                    if threshold <= code_start
                    else "partial"
                )
            rows.append(row)
            metadata = record["metadata"]
            for field in groups:
                groups[field][str(metadata.get(field) or "unknown")].append(row)

    overall = summarize_group(rows)
    recommendation = choose_recommendation(overall)
    longest = sorted(rows, key=lambda row: row["full_tokens"], reverse=True)[:20]
    payload = {
        "schema_version": "codeguide-token-length-audit-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "tokenizer_class": tokenizer.__class__.__name__,
        "chat_template_sha256": __import__("hashlib").sha256((tokenizer.chat_template or "").encode()).hexdigest(),
        "data": str(args.data).replace("\\", "/"),
        "data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "overall": overall,
        "groups": {
            field: {value: summarize_group(group_rows) for value, group_rows in sorted(values.items())}
            for field, values in groups.items()
        },
        "longest_samples": longest,
        "oversize_samples": {
            str(threshold): [
                {
                    "problem_id": row["problem_id"],
                    "full_tokens": row["full_tokens"],
                    "code_status": row[f"code_at_{threshold}"],
                }
                for row in rows
                if row["full_tokens"] > threshold
            ]
            for threshold in THRESHOLDS
        },
        "recommended_max_seq_length": recommendation,
        "recommendation_rule": "smallest of 4096/6144/8192 with <=1% overflow and <=0.5% code harm; otherwise 8192",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Qwen2.5-Coder-7B-Instruct Token 长度审计",
        "",
        f"- 数据：`{payload['data']}`，{len(rows):,} 条。",
        f"- Tokenizer：`{args.model}`，正式 `apply_chat_template`。未加载模型权重。",
        f"- 推荐 `max_seq_length={recommendation}`。",
        "",
        "## 总体分位数",
        "",
        "| 范围 | P50 | P75 | P90 | P95 | P99 | Max |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, key in (("Prompt", "prompt_tokens"), ("Assistant/completion", "completion_tokens"), ("完整序列", "full_tokens")):
        item = overall[key]
        lines.append(f"| {label} | {item['p50']} | {item['p75']} | {item['p90']} | {item['p95']} | {item['p99']} | {item['max']} |")
    lines += ["", "## 截断风险", "", "| 阈值 | 超长数量 | 超长比例 | 伤及代码 | 完全移除代码 | 部分代码 |", "|---:|---:|---:|---:|---:|---:|"]
    for threshold in THRESHOLDS:
        item = overall["thresholds"][str(threshold)]
        lines.append(f"| {threshold} | {item['over_count']} | {item['over_ratio']:.2%} | {item['code_harmed_count']} | {item['code_missing_count']} | {item['code_partial_count']} |")
    lines += ["", "## 最长样本", "", "| problem_id | Prompt | Completion | Full |", "|---|---:|---:|---:|"]
    for row in longest[:10]:
        lines.append(f"| `{row['problem_id']}` | {row['prompt_tokens']} | {row['completion_tokens']} | {row['full_tokens']} |")
    lines += ["", "分组的完整机器可读统计见 `data/manifests/token_length_stats.json`。", ""]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"records": len(rows), "overall": overall, "recommended_max_seq_length": recommendation}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
