#!/usr/bin/env python3
"""从基础配置生成唯一的正式 GRPO 配置。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def count_jsonl(path: Path) -> int:
    count = 0
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            if not isinstance(item, dict):
                raise ValueError(f"{path} 第 {line_no} 行不是 JSON 对象")
            count += 1
    return count


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False, width=1000),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", default="configs/train_config.yaml")
    parser.add_argument("--output", default="configs/grpo_minimal_v1.yaml")
    parser.add_argument(
        "--sft-adapter",
        default=(
            "outputs/sft/qwen25_coder_7b_qlora_8k/"
            "full_lr1e4_seed20260728/checkpoint-200"
        ),
    )
    parser.add_argument(
        "--train-data",
        default="data/splits/grpo_minimal_v1/grpo_train.jsonl",
    )
    parser.add_argument(
        "--eval-data",
        default="data/splits/grpo_minimal_v1/grpo_dev.jsonl",
    )
    parser.add_argument("--seed", type=int, default=20260728)
    args = parser.parse_args()

    base_path = resolve(args.base_config)
    train_path = resolve(args.train_data)
    eval_path = resolve(args.eval_data)
    output_path = resolve(args.output)

    for required in (base_path, train_path, eval_path):
        if not required.is_file():
            raise FileNotFoundError(f"缺少必要文件：{required}")

    train_count = count_jsonl(train_path)
    eval_count = count_jsonl(eval_path)
    if train_count != 6451:
        raise ValueError(f"GRPO train 应为 6451 条，实际 {train_count}")
    if eval_count != 50:
        raise ValueError(f"GRPO dev 应为 50 条，实际 {eval_count}")

    cfg = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError(f"配置格式错误：{base_path}")

    cfg.setdefault("model", {})["max_seq_length"] = 8192

    grpo = cfg.setdefault("grpo", {})
    grpo.update(
        {
            "run_name": "grpo-codeguide-minimal-v1",
            "sft_adapter_path": args.sft_adapter,
            "max_seq_length": 8192,
            "train_data": args.train_data,
            "eval_data": args.eval_data,
            "output_dir": "outputs/grpo/codeguide_minimal_v1",
            "best_model_dir": "outputs/grpo/codeguide_minimal_v1_best",
            "logging_dir": "outputs/grpo/codeguide_minimal_v1_logs",
            "num_generations": 4,
            "max_new_tokens": 1024,
            "max_online_tests": 16,
            "temperature": 0.8,
            "top_p": 0.95,
            "learning_rate": 1.0e-5,
            "num_train_epochs": 1,
            "max_steps": -1,
            "logging_steps": 10,
            "save_steps": 100,
            "normalize_rewards": False,
            "save_best": True,
            "eval_steps": 100,
            "checkpoint_eval_max_samples": 50,
            "merge_model": False,
            "seed": args.seed,
        }
    )

    reward = cfg.setdefault("reward", {})
    reward.update(
        {
            "alpha": 0.60,
            "execution_backend": "subprocess",
            "container_image": "",
            "exec_timeout": 5.0,
            "code_weight": 0.60,
            "contract_weight": 0.40,
            "static_validity_weight": 0.05,
            "partial_pass_weight": 0.70,
            "strict_pass_weight": 0.25,
            "contract_gate_floor": 0.25,
            "teaching_completeness": 0.0,
            "code_correctness": 0.60,
            "format_compliance": 0.40,
            "teaching_mode": "local",
        }
    )

    cfg.setdefault("curriculum", {})["enabled"] = True
    write_yaml(output_path, cfg)

    print(f"[完成] 正式 GRPO 配置：{output_path}")
    print(f"[数据] train={train_count}，dev={eval_count}")
    print(f"[起点] {args.sft_adapter}")
    print("[奖励] 0.6 Code + 0.4 门控 Format；subprocess 在线执行")
    print("[训练] easy -> medium -> hard，1 epoch，不做人为数据截断")
    print("[保存] final LoRA + dev Pass@1 best LoRA；不自动合并模型")


if __name__ == "__main__":
    main()
