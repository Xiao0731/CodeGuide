#!/usr/bin/env python3
"""Validate cross-stage CodeGuide configuration contracts without a GPU."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml


ALLOWED_LENGTHS = {4096, 6144, 8192}


def validate(config: dict, *, allow_unfrozen: bool = False) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    model = config.get("model", {})
    lora = config.get("lora", {})
    sft = config.get("sft", {})
    grpo = config.get("grpo", {})
    reward = config.get("reward", {})

    if model.get("name") != "Qwen/Qwen2.5-Coder-7B-Instruct":
        errors.append("formal backbone must be Qwen/Qwen2.5-Coder-7B-Instruct")
    if (lora.get("r"), lora.get("lora_alpha")) != (
        sft.get("lora_r"),
        sft.get("lora_alpha"),
    ):
        errors.append("SFT and GRPO LoRA rank/alpha must match")
    if list(lora.get("target_modules", [])) != [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]:
        errors.append("LoRA target_modules do not match the frozen contract")

    for key, value in (
        ("sft.max_seq_length", sft.get("max_seq_length")),
        ("grpo.max_seq_length", grpo.get("max_seq_length")),
    ):
        if value not in ALLOWED_LENGTHS:
            message = f"{key} is unfrozen; expected one of {sorted(ALLOWED_LENGTHS)}"
            (warnings if allow_unfrozen and value is None else errors).append(message)

    if grpo.get("normalize_rewards") is not False:
        errors.append("main experiment must keep grpo.normalize_rewards=false")
    if grpo.get("train_data") == grpo.get("eval_data"):
        errors.append("grpo train_data and eval_data must be disjoint files")
    if float(sft.get("learning_rate", 0.0)) != 1.0e-4:
        errors.append("formal SFT learning_rate must be 1.0e-4")
    if float(sft.get("num_train_epochs", 0.0)) != 1.0:
        errors.append("formal SFT must start with exactly 1.0 epoch")
    if sft.get("completion_only_loss") is not True:
        errors.append("formal SFT must use completion_only_loss=true")
    if sft.get("length_grouped_sampling") is not True:
        errors.append("formal SFT must use length_grouped_sampling=true")

    weights = [
        float(reward.get("teaching_completeness", 0.0)),
        float(reward.get("code_correctness", 0.0)),
        float(reward.get("format_compliance", 0.0)),
    ]
    if abs(sum(weights) - 1.0) > 1e-9:
        errors.append(f"reward weights must sum to 1.0, got {sum(weights):.6f}")
    if reward.get("teaching_completeness") not in (0, 0.0):
        errors.append("TeachingCritic is not admitted; teaching gradient weight must be 0")
    if reward.get("execution_backend") != "docker":
        errors.append("formal execution backend must be docker")

    image = str(reward.get("container_image") or "")
    if "${oc.env:" in image:
        warnings.append("container image is resolved from environment at runtime")
    elif "@sha256:" not in image:
        errors.append("container_image must be pinned by digest")

    return errors, warnings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/train_config.yaml"))
    parser.add_argument("--allow-unfrozen", action="store_true")
    args = parser.parse_args()

    payload = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    errors, warnings = validate(payload, allow_unfrozen=args.allow_unfrozen)
    for item in warnings:
        print(f"WARNING: {item}")
    for item in errors:
        print(f"ERROR: {item}")
    if errors:
        sys.exit(1)
    print("configuration contract: PASS")


if __name__ == "__main__":
    main()
