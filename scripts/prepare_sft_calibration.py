#!/usr/bin/env python3
"""Create deterministic calibration IDs from the frozen SFT train split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.training.sft_data import ids_sha256, split_records, stratified_sample_ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical", default="data/final/sft_accepted.jsonl")
    parser.add_argument("--train-ids", default="data/splits/sft_train_ids.json")
    parser.add_argument("--dev-ids", default="data/splits/sft_dev_ids.json")
    parser.add_argument("--output", default="data/splits/sft_calibration_500_ids.json")
    parser.add_argument("--sample-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260728)
    args = parser.parse_args()

    train, _ = split_records(
        ROOT / args.canonical, ROOT / args.train_ids, ROOT / args.dev_ids
    )
    selected = stratified_sample_ids(train, args.sample_size, args.seed)
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "codeguide-sft-calibration-v1",
        "seed": args.seed,
        "parent": args.train_ids,
        "count": len(selected),
        "ids_sha256": ids_sha256(selected),
        "ids": selected,
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"count": len(selected), "sha256": ids_sha256(selected), "path": args.output}))


if __name__ == "__main__":
    main()
