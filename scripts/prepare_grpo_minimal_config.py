#!/usr/bin/env python3
"""从现有 train_config.yaml 生成最小版 GRPO 配置。"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


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
    args = parser.parse_args()

    base_path = resolve(args.base_config)
    output_path = resolve(args.output)
    cfg = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError(f"配置格式错误：{base_path}")

    cfg.setdefault("model", {})["max_seq_length"] = 8192

    grpo = cfg.setdefault("grpo", {})
    grpo.update(
        {
            "sft_adapter_path": args.sft_adapter,
            "max_seq_length": 8192,
            "train_data": args.train_data,
            "eval_data": args.eval_data,
            "output_dir": "outputs/grpo/codeguide_minimal_v1",
            "best_model_dir": "outputs/grpo/codeguide_minimal_v1_best",
            "merged_dir": "outputs/grpo/codeguide_minimal_v1_merged",
            "logging_dir": "outputs/grpo/codeguide_minimal_v1_logs",
            "num_generations": 4,
            "max_new_tokens": 1024,
            "temperature": 0.8,
            "top_p": 0.95,
            "learning_rate": 1.0e-5,
            "num_train_epochs": 1,
            "normalize_rewards": False,
            "save_best": True,
            "eval_steps": 100,
            "checkpoint_eval_max_samples": 50,
            "seed": 20260728,
        }
    )

    reward = cfg.setdefault("reward", {})
    reward.update(
        {
            # alpha 仅用于兼容旧日志；实际计算由 minimal_grpo_reward.py 完成。
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

    curriculum = cfg.setdefault("curriculum", {})
    curriculum["enabled"] = True

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False, width=1000),
        encoding="utf-8",
    )
    print(f"[完成] 最小版 GRPO 配置：{output_path}")
    print(f"[起点] {args.sft_adapter}")
    print("[奖励] 0.6 Code + 0.4 门控 Format；Teaching 仅监控")
    print("[执行] 训练阶段 subprocess；最终评测仍使用本地 Docker")


if __name__ == "__main__":
    main()
