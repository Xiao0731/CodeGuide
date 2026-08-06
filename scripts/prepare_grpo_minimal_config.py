#!/usr/bin/env python3
"""从现有 train_config.yaml 生成正式版与 20-step smoke GRPO 配置。"""

from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False, width=1000),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", default="configs/train_config.yaml")
    parser.add_argument("--output", default="configs/grpo_minimal_v1.yaml")
    parser.add_argument("--smoke-output", default="configs/grpo_smoke_v1.yaml")
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
    parser.add_argument(
        "--smoke-train-data",
        default="data/splits/grpo_minimal_v1/grpo_smoke_train.jsonl",
    )
    parser.add_argument("--smoke-samples", type=int, default=160)
    args = parser.parse_args()

    base_path = resolve(args.base_config)
    output_path = resolve(args.output)
    smoke_output_path = resolve(args.smoke_output)
    train_path = resolve(args.train_data)
    smoke_train_path = resolve(args.smoke_train_data)

    cfg = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError(f"配置格式错误：{base_path}")
    if not train_path.exists():
        raise FileNotFoundError(f"请先冻结 GRPO 数据：{train_path}")

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
            "eval_steps": 10,
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
    write_yaml(output_path, cfg)

    lines = train_path.read_text(encoding="utf-8").splitlines(keepends=True)
    if len(lines) < args.smoke_samples:
        raise RuntimeError(
            f"smoke 样本不足：需要 {args.smoke_samples}，实际 {len(lines)}"
        )
    smoke_train_path.parent.mkdir(parents=True, exist_ok=True)
    smoke_train_path.write_text(
        "".join(lines[: args.smoke_samples]),
        encoding="utf-8",
        newline="\n",
    )

    smoke_cfg = copy.deepcopy(cfg)
    smoke_grpo = smoke_cfg["grpo"]
    smoke_grpo.update(
        {
            "train_data": args.smoke_train_data,
            "output_dir": "outputs/grpo/codeguide_smoke_v1",
            "best_model_dir": "outputs/grpo/codeguide_smoke_v1_best",
            "merged_dir": "outputs/grpo/codeguide_smoke_v1_merged",
            "logging_dir": "outputs/grpo/codeguide_smoke_v1_logs",
            "max_new_tokens": 512,
            "logging_steps": 1,
            "save_steps": 20,
            "save_best": False,
            "num_train_epochs": 1,
        }
    )
    smoke_cfg.setdefault("curriculum", {})["enabled"] = False
    write_yaml(smoke_output_path, smoke_cfg)

    batch = int(smoke_grpo["per_device_train_batch_size"])
    accumulation = int(smoke_grpo["gradient_accumulation_steps"])
    expected_steps = math.ceil(args.smoke_samples / (batch * accumulation))

    print(f"[完成] 正式 GRPO 配置：{output_path}")
    print(f"[完成] Smoke GRPO 配置：{smoke_output_path}")
    print(f"[完成] Smoke 数据：{smoke_train_path}（{args.smoke_samples} 条，约 {expected_steps} step）")
    print(f"[起点] {args.sft_adapter}")
    print("[奖励] 0.6 Code + 0.4 门控 Format；Teaching 仅监控")
    print("[执行] 训练阶段 subprocess；最终评测仍使用本地 Docker")


if __name__ == "__main__":
    main()
