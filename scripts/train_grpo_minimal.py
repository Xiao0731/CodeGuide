#!/usr/bin/env python3
"""CodeGuide 最小正式版 GRPO 训练入口。

约束：
- 从选定的 SFT step-200 adapter 热启动；
- 训练阶段使用受限 subprocess 执行奖励；
- 在线 Reward 每题只取固定数量测试，避免超大 TACO 测试集拖垮训练；
- TACO-515 不进入本入口，独立 GRPO dev 只用于 checkpoint 选择；
- Curriculum 各阶段使用独立 checkpoint 目录，并保持全局日志 step 单调；
- 默认只保存 LoRA adapter，不在云端 smoke/主训练后自动合并 14GB 模型。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.reward.minimal_grpo_reward import make_reward_fn_with_cfg
from src.training import grpo_train

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def _parse_json_list(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return value
    if not isinstance(value, str) or not value.strip():
        return []
    payload = json.loads(value)
    return payload if isinstance(payload, list) else []


def _parse_json_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str) or not value.strip():
        return {}
    payload = json.loads(value)
    return payload if isinstance(payload, dict) else {}


def _cap_online_tests(
    records: list[dict[str, Any]],
    *,
    max_online_tests: int,
) -> tuple[list[dict[str, Any]], dict[str, float | int]]:
    """按 prompt+测试内容稳定抽样，限制每个 completion 的在线测试数量。"""
    if max_online_tests <= 0:
        counts = [len(_parse_json_list(item.get("test_cases"))) for item in records]
        return records, {
            "samples": len(records),
            "before_total": sum(counts),
            "after_total": sum(counts),
            "before_max": max(counts, default=0),
            "after_max": max(counts, default=0),
        }

    capped: list[dict[str, Any]] = []
    before_counts: list[int] = []
    after_counts: list[int] = []

    for record in records:
        tests = _parse_json_list(record.get("test_cases"))
        before_counts.append(len(tests))
        prompt = str(record.get("prompt") or "")
        if len(tests) > max_online_tests:
            order = sorted(
                range(len(tests)),
                key=lambda index: hashlib.sha256(
                    (
                        f"{prompt}\0{index}\0"
                        + json.dumps(
                            tests[index],
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                    ).encode("utf-8")
                ).hexdigest(),
            )
            chosen = sorted(order[:max_online_tests])
            selected = [tests[index] for index in chosen]
        else:
            selected = tests

        metadata = _parse_json_dict(record.get("metadata"))
        metadata["reward_tests_before_online_cap"] = len(tests)
        metadata["online_test_count"] = len(selected)
        metadata["online_test_cap"] = max_online_tests

        item = dict(record)
        item["test_cases"] = json.dumps(selected, ensure_ascii=False)
        item["metadata"] = json.dumps(metadata, ensure_ascii=False)
        capped.append(item)
        after_counts.append(len(selected))

    return capped, {
        "samples": len(capped),
        "before_total": sum(before_counts),
        "after_total": sum(after_counts),
        "before_max": max(before_counts, default=0),
        "after_max": max(after_counts, default=0),
        "before_avg": (
            round(sum(before_counts) / len(before_counts), 3)
            if before_counts
            else 0.0
        ),
        "after_avg": (
            round(sum(after_counts) / len(after_counts), 3)
            if after_counts
            else 0.0
        ),
    }


def _build_dataset(cfg: Any, tokenizer: Any, Dataset: Any, difficulty: str | None = None):
    data_path = Path(str(cfg.grpo.train_data))
    records, stats = grpo_train._build_grpo_records(
        data_path,
        tokenizer,
        difficulty=difficulty,
    )
    label = "GRPO 数据集" if difficulty is None else f"Curriculum [{difficulty}]"
    grpo_train._log_grpo_filter_stats(label, stats)
    if not records:
        raise ValueError(f"{label} 为空")

    max_online_tests = int(getattr(cfg.grpo, "max_online_tests", 16))
    records, cap_stats = _cap_online_tests(
        records,
        max_online_tests=max_online_tests,
    )
    logger.info(
        "%s 在线测试裁剪：cap=%d，平均 %.3f→%.3f，最大 %d→%d",
        label,
        max_online_tests,
        cap_stats.get("before_avg", 0.0),
        cap_stats.get("after_avg", 0.0),
        cap_stats.get("before_max", 0),
        cap_stats.get("after_max", 0),
    )
    return Dataset.from_list(records)


def _load_eval_problems(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"冻结 GRPO dev 不存在：{path}")

    problems: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            obj = json.loads(line)
            metadata = obj.get("metadata")
            if not isinstance(metadata, dict):
                raise ValueError(f"GRPO dev 第 {line_no} 行缺少 metadata")
            heldout_tests = metadata.get("heldout_tests") or []
            if not isinstance(heldout_tests, list) or not heldout_tests:
                raise ValueError(f"GRPO dev 第 {line_no} 行缺少 heldout_tests")
            description = next(
                (
                    str(message.get("content") or "")
                    for message in obj.get("messages", [])
                    if isinstance(message, dict) and message.get("role") == "user"
                ),
                "",
            )
            if not description:
                raise ValueError(f"GRPO dev 第 {line_no} 行缺少 user prompt")
            problems.append(
                {
                    "description": description,
                    "heldout_tests": heldout_tests,
                    "metadata": {
                        key: value
                        for key, value in metadata.items()
                        if key not in {"test_cases", "heldout_tests"}
                    },
                }
            )
    if not problems:
        raise ValueError(f"GRPO dev 为空：{path}")
    return problems


def _effective_steps(n_samples: int, cfg: Any, epochs: int = 1) -> int:
    micro_batch = int(cfg.grpo.per_device_train_batch_size)
    accumulation = int(cfg.grpo.gradient_accumulation_steps)
    return math.ceil(n_samples / (micro_batch * accumulation)) * epochs


def _make_grpo_config(
    *,
    cfg: Any,
    GRPOConfig: Any,
    wandb: Any,
    max_new_tokens: int,
    num_epochs: int,
    output_dir: Path,
    max_steps: int,
):
    grpo = cfg.grpo
    kwargs: dict[str, Any] = {
        "num_generations": int(grpo.num_generations),
        "max_new_tokens": int(max_new_tokens),
        "temperature": float(grpo.temperature),
        "top_p": float(grpo.top_p),
        "kl_coef": float(grpo.kl_coef),
        "max_prompt_length": int(grpo.max_prompt_length),
        "learning_rate": float(grpo.learning_rate),
        "lr_scheduler_type": str(grpo.lr_scheduler_type),
        "warmup_ratio": float(grpo.warmup_ratio),
        "weight_decay": float(grpo.weight_decay),
        "max_grad_norm": float(grpo.max_grad_norm),
        "per_device_train_batch_size": int(grpo.per_device_train_batch_size),
        "gradient_accumulation_steps": int(grpo.gradient_accumulation_steps),
        "num_train_epochs": int(num_epochs),
        "max_steps": int(max_steps),
        "bf16": bool(grpo.bf16),
        "fp16": bool(grpo.fp16),
        "gradient_checkpointing": bool(grpo.gradient_checkpointing),
        "output_dir": str(output_dir),
        "logging_dir": str(output_dir / "logs"),
        "logging_steps": int(grpo.logging_steps),
        "save_strategy": "steps",
        "save_steps": int(grpo.save_steps),
        "save_total_limit": 3,
        "seed": int(grpo.seed),
        "report_to": "wandb",
        "run_name": wandb.run.name,
        "dataloader_num_workers": 2,
        "remove_unused_columns": False,
    }
    return GRPOConfig(**kwargs)


def _offset_callbacks(
    *,
    offset: int,
    reward_callback: Any,
    best_callback: Any | None,
    TrainerCallback: Any,
) -> list[Any]:
    class RewardAdapter(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            original = state.global_step
            state.global_step = original + offset
            try:
                reward_callback.on_log(
                    args,
                    state,
                    control,
                    logs=logs,
                    **kwargs,
                )
            finally:
                state.global_step = original

    callbacks: list[Any] = [RewardAdapter()]

    if best_callback is not None:
        class BestAdapter(TrainerCallback):
            def on_step_end(self, args, state, control, **kwargs):
                original = state.global_step
                state.global_step = original + offset
                try:
                    best_callback.on_step_end(args, state, control, **kwargs)
                finally:
                    state.global_step = original

        callbacks.append(BestAdapter())
    return callbacks


def _train_one_stage(
    *,
    model: Any,
    tokenizer: Any,
    dataset: Any,
    reward_fn: Any,
    cfg: Any,
    GRPOConfig: Any,
    GRPOTrainer: Any,
    TrainerCallback: Any,
    wandb: Any,
    reward_callback: Any,
    best_callback: Any | None,
    output_dir: Path,
    max_new_tokens: int,
    epochs: int,
    max_steps: int,
    step_offset: int,
    resume_from_checkpoint: str | None,
    metric_prefix: str,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    training_args = _make_grpo_config(
        cfg=cfg,
        GRPOConfig=GRPOConfig,
        wandb=wandb,
        max_new_tokens=max_new_tokens,
        num_epochs=epochs,
        output_dir=output_dir,
        max_steps=max_steps,
    )
    callbacks = _offset_callbacks(
        offset=step_offset,
        reward_callback=reward_callback,
        best_callback=best_callback,
        TrainerCallback=TrainerCallback,
    )
    expected = max_steps if max_steps > 0 else _effective_steps(len(dataset), cfg, epochs)
    rollout_count = (
        expected
        * int(cfg.grpo.per_device_train_batch_size)
        * int(cfg.grpo.gradient_accumulation_steps)
        * int(cfg.grpo.num_generations)
    )
    logger.info(
        "%s：samples=%d，预计 optimizer steps=%d，预计 rollouts=%d，output=%s",
        metric_prefix,
        len(dataset),
        expected,
        rollout_count,
        output_dir,
    )

    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        reward_funcs=reward_fn,
        tokenizer=tokenizer,
        callbacks=callbacks,
    )
    result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.log_metrics(metric_prefix, result.metrics)
    trainer.save_metrics(metric_prefix, result.metrics)

    adapter_dir = output_dir / "adapter_final"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    logger.info("%s 阶段 adapter 已保存：%s", metric_prefix, adapter_dir)
    return int(trainer.state.global_step)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/grpo_minimal_v1.yaml")
    parser.add_argument("--resume_from_checkpoint", default=None)
    cli = parser.parse_args()

    try:
        import wandb
        from datasets import Dataset
        from omegaconf import OmegaConf
        from transformers import TrainerCallback
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as exc:
        raise RuntimeError(f"缺少 GRPO 训练依赖：{exc}") from exc

    cfg_path = Path(cli.config)
    if not cfg_path.is_file():
        raise FileNotFoundError(f"配置不存在：{cfg_path}")
    cfg = OmegaConf.load(cfg_path)
    grpo = cfg.grpo

    if int(grpo.max_seq_length) not in {4096, 6144, 8192}:
        raise ValueError("grpo.max_seq_length 必须冻结为 4096/6144/8192")
    if int(cfg.lora.r) != int(cfg.sft.lora_r):
        raise ValueError("SFT 与 GRPO LoRA rank 不一致")
    if int(cfg.lora.lora_alpha) != int(cfg.sft.lora_alpha):
        raise ValueError("SFT 与 GRPO LoRA alpha 不一致")

    run_name = str(getattr(grpo, "run_name", "grpo-codeguide-minimal-v1"))
    wandb.init(
        project="codeguide-grpo",
        name=run_name,
        tags=["grpo", "qlora", "qwen2.5-coder", "execution-reward"],
        config=OmegaConf.to_container(grpo, resolve=True),
    )

    model, tokenizer = grpo_train.load_policy_model(cfg)
    reward_fn = make_reward_fn_with_cfg(cfg)
    reward_callback = grpo_train.RewardLoggingCallback(
        log_example_every_n_steps=int(getattr(grpo, "example_log_steps", 100)),
        collapse_var_threshold=float(getattr(grpo, "collapse_var_threshold", 0.01)),
        collapse_warn_steps=int(getattr(grpo, "collapse_warn_steps", 20)),
    )

    best_callback = None
    if bool(getattr(grpo, "save_best", False)):
        eval_path = Path(str(grpo.eval_data))
        eval_problems = _load_eval_problems(eval_path)
        best_callback = grpo_train.BestCheckpointCallback(
            model,
            tokenizer,
            eval_problems,
            cfg,
        )
        logger.info("GRPO dev 已加载：%d 题，仅用于 checkpoint 选择", len(eval_problems))

    output_root = Path(str(grpo.output_dir))
    output_root.mkdir(parents=True, exist_ok=True)
    global_step = 0

    curriculum_cfg = getattr(cfg, "curriculum", None)
    use_curriculum = (
        curriculum_cfg is not None
        and bool(getattr(curriculum_cfg, "enabled", False))
    )

    if use_curriculum:
        if int(getattr(grpo, "max_steps", -1)) > 0:
            raise ValueError("Curriculum 正式训练不能同时设置 grpo.max_steps>0")
        stages = list(curriculum_cfg.stages)
        logger.info(
            "启用 Curriculum：%s",
            [str(stage.difficulty) for stage in stages],
        )
        for stage_index, stage in enumerate(stages, 1):
            difficulty = str(stage.difficulty)
            dataset = _build_dataset(cfg, tokenizer, Dataset, difficulty=difficulty)
            stage_dir = output_root / f"stage_{stage_index}_{difficulty}"
            completed = _train_one_stage(
                model=model,
                tokenizer=tokenizer,
                dataset=dataset,
                reward_fn=reward_fn,
                cfg=cfg,
                GRPOConfig=GRPOConfig,
                GRPOTrainer=GRPOTrainer,
                TrainerCallback=TrainerCallback,
                wandb=wandb,
                reward_callback=reward_callback,
                best_callback=best_callback,
                output_dir=stage_dir,
                max_new_tokens=int(
                    getattr(stage, "max_new_tokens", grpo.max_new_tokens)
                ),
                epochs=int(getattr(stage, "epochs", 1)),
                max_steps=-1,
                step_offset=global_step,
                resume_from_checkpoint=(
                    cli.resume_from_checkpoint if stage_index == 1 else None
                ),
                metric_prefix=f"train_{difficulty}",
            )
            global_step += completed
    else:
        dataset = _build_dataset(cfg, tokenizer, Dataset)
        global_step = _train_one_stage(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            reward_fn=reward_fn,
            cfg=cfg,
            GRPOConfig=GRPOConfig,
            GRPOTrainer=GRPOTrainer,
            TrainerCallback=TrainerCallback,
            wandb=wandb,
            reward_callback=reward_callback,
            best_callback=best_callback,
            output_dir=output_root / "train",
            max_new_tokens=int(grpo.max_new_tokens),
            epochs=int(grpo.num_train_epochs),
            max_steps=int(getattr(grpo, "max_steps", -1)),
            step_offset=0,
            resume_from_checkpoint=cli.resume_from_checkpoint,
            metric_prefix="train",
        )

    final_adapter = output_root / "final_adapter"
    final_adapter.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(final_adapter))
    tokenizer.save_pretrained(str(final_adapter))
    logger.info("最终 GRPO adapter：%s", final_adapter)

    if bool(getattr(grpo, "merge_model", False)):
        merged_dir = Path(str(grpo.merged_dir))
        logger.info("开始合并 LoRA：%s", merged_dir)
        model.save_pretrained_merged(
            str(merged_dir),
            tokenizer,
            save_method="merged_16bit",
        )
    else:
        logger.info("merge_model=false：跳过完整模型合并，避免额外 14GB 写盘")

    if best_callback is not None:
        if best_callback.best_step >= 0:
            logger.info(
                "最佳 GRPO checkpoint：step=%d，dev Pass@1=%.4f，目录=%s",
                best_callback.best_step,
                best_callback.best_pass1,
                best_callback.best_dir,
            )
        else:
            logger.warning("训练期间尚未触发 dev 评估；最终评测先使用 final_adapter")

    wandb.log({"train/final_global_step": global_step})
    wandb.finish()
    logger.info("GRPO 训练完成，总 optimizer steps=%d", global_step)


if __name__ == "__main__":
    main()
