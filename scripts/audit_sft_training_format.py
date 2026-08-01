#!/usr/bin/env python3
"""Audit the exact assistant-only tokenization used by the SFT trainer."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.training.sft_data import load_canonical, tokenize_assistant_only


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical", default="data/final/sft_accepted.jsonl")
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--max-seq-length", type=int, default=8192)
    parser.add_argument("--output", default="data/manifests/sft_training_format_audit.json")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=args.local_files_only)
    records = load_canonical(ROOT / args.canonical)
    lengths, supervised, failures = [], [], []
    modes: dict[str, int] = {}
    for problem_id, record in records.items():
        try:
            item = tokenize_assistant_only(record, tokenizer, args.max_seq_length)
            labels = item["labels"]
            first = next(index for index, label in enumerate(labels) if label != -100)
            assistant_text = tokenizer.decode(item["input_ids"][first:], skip_special_tokens=False)
            if "```python" not in assistant_text.lower():
                raise ValueError("Python code block is outside supervised region")
            lengths.append(item["length"])
            supervised.append(sum(label != -100 for label in labels))
            mode = record.get("metadata", {}).get("io_mode", "unknown")
            modes[mode] = modes.get(mode, 0) + 1
        except Exception as exc:
            failures.append({"problem_id": problem_id, "error": str(exc)})

    report = {
        "schema_version": "codeguide-sft-training-format-audit-v1",
        "model": args.model,
        "max_seq_length": args.max_seq_length,
        "records": len(records),
        "passed": len(lengths),
        "failed": len(failures),
        "truncated": sum("exceeding" in item["error"] for item in failures),
        "max_length": max(lengths, default=0),
        "supervised_tokens": sum(supervised),
        "total_tokens": sum(lengths),
        "supervised_token_ratio": sum(supervised) / sum(lengths) if lengths else 0,
        "io_modes": modes,
        "failures": failures,
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("records", "passed", "failed", "truncated", "max_length", "supervised_token_ratio")}, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

