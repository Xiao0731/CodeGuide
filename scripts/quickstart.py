#!/usr/bin/env python3
"""CodeGuide 一站式快速复现入口。

只负责编排仓库现有训练/评测实现，不复制训练器、奖励或验证器逻辑。

常用命令：
    python scripts/quickstart.py check
    python scripts/quickstart.py sft
    python scripts/quickstart.py grpo --sft-adapter /path/to/checkpoint-200
    python scripts/quickstart.py eval --sft-adapter /path/to/checkpoint-200 --grpo-adapter /path/to/grpo/best
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_TACO_IMAGE = (
    "python:3.11.9-slim-bookworm@sha256:"
    "8fb099199b9f2d70342674bd9dbccd3ed03a258f26bbd1d556822c6dfc60c317"
)


def run(command: list[str]) -> None:
    print(f"\n[CodeGuide] $ {' '.join(command)}\n", flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def py(*args: str) -> list[str]:
    return [sys.executable, *args]


def accelerate(config: str, script: str, *args: str) -> list[str]:
    return [
        sys.executable,
        "-m",
        "accelerate.commands.launch",
        "--config_file",
        config,
        script,
        *args,
    ]


def cmd_check(_: argparse.Namespace) -> None:
    run(py("scripts/train_sft.py", "--validate-only", "--mode", "full"))
    run(py("scripts/train_grpo.py", "--validate-only"))


def cmd_sft(args: argparse.Namespace) -> None:
    accelerate_config = (
        "configs/accelerate/dual_gpu_deepspeed.yaml"
        if args.deepspeed
        else "configs/accelerate/dual_gpu.yaml"
    )
    run(
        accelerate(
            accelerate_config,
            "scripts/train_sft.py",
            "--config",
            "configs/sft.yaml",
            "--mode",
            "full",
            "--learning-rate",
            str(args.learning_rate),
        )
    )


def cmd_grpo(args: argparse.Namespace) -> None:
    run(
        accelerate(
            "configs/accelerate/dual_gpu.yaml",
            "scripts/train_grpo.py",
            "--config",
            "configs/grpo.yaml",
            "--sft-adapter-path",
            args.sft_adapter,
        )
    )


def taco_generate(variant: str, adapter: str | None = None) -> None:
    command = py(
        "scripts/evaluate_sft_matrix.py",
        "--stage",
        "generate",
        "--variant",
        variant,
    )
    if adapter:
        command += ["--adapter-path", adapter]
    run(command)


def taco_verify(variant: str, image: str) -> None:
    run(
        py(
            "scripts/evaluate_sft_matrix.py",
            "--stage",
            "verify",
            "--variant",
            variant,
            "--container-image",
            image,
        )
    )


def eval_taco(args: argparse.Namespace) -> None:
    image = (
        args.container_image
        or os.environ.get("CODEGUIDE_DOCKER_IMAGE")
        or DEFAULT_TACO_IMAGE
    )
    run(py("scripts/evaluate_sft_matrix.py", "--stage", "prepare"))
    taco_generate("base")
    taco_generate("sft_best", args.sft_adapter)
    taco_generate("grpo_best", args.grpo_adapter)
    taco_verify("base", image)
    taco_verify("sft_best", image)
    taco_verify("grpo_best", image)
    run(py("scripts/evaluate_sft_matrix.py", "--stage", "summarize"))


def _load_evalplus_config() -> tuple[Path, dict[str, Any]]:
    import yaml

    config_path = ROOT / "configs/eval/evalplus_code_capability_v1.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise RuntimeError(f"invalid config: {config_path}")
    if config.get("evaluation_module") != "code_capability":
        raise RuntimeError("unexpected EvalPlus configuration")
    return config_path, copy.deepcopy(config)


def evalplus_generate_adapter(variant: str, adapter_value: str) -> None:
    """复用现有 EvalPlus 模块函数，为任意 adapter 生成两套 benchmark 样本。"""
    import torch
    import scripts.generate_evalplus_code_capability as impl

    config_path, config = _load_evalplus_config()
    adapter_path = Path(adapter_value)
    if not adapter_path.is_absolute():
        adapter_path = ROOT / adapter_path
    adapter = impl.adapter_fingerprint(adapter_path)

    seed = int(config["generation"]["seed"])
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for EvalPlus generation")

    output_root = impl.resolve_path(config["output"]["root"])
    assert output_root is not None
    output_root.mkdir(parents=True, exist_ok=True)

    model, tokenizer = impl.load_model(config=config, adapter_path=adapter_path)
    impl.write_manifest(
        config_path=config_path,
        config=config,
        variant=variant,
        adapter=adapter,
        model=model,
        tokenizer=tokenizer,
        output_root=output_root,
    )
    try:
        for dataset_name in ("humaneval", "mbpp"):
            impl.generate_dataset(
                dataset_name=dataset_name,
                config=config,
                variant=variant,
                model=model,
                tokenizer=tokenizer,
                output_root=output_root,
            )
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()


def evalplus_generate_base() -> None:
    run(
        py(
            "scripts/generate_evalplus_code_capability.py",
            "--variant",
            "base",
            "--dataset",
            "humaneval",
            "--dataset",
            "mbpp",
        )
    )


def evalplus_score(variant: str) -> None:
    root = ROOT / "outputs/eval/evalplus_code_capability_v1/samples"
    for dataset in ("humaneval", "mbpp"):
        sample_file = root / dataset / f"{variant}.jsonl"
        run(
            py(
                "-m",
                "evalplus.evaluate",
                "--dataset",
                dataset,
                "--samples",
                str(sample_file),
            )
        )


def eval_evalplus(args: argparse.Namespace) -> None:
    evalplus_generate_base()
    evalplus_generate_adapter("sft_best", args.sft_adapter)
    evalplus_generate_adapter("grpo_best", args.grpo_adapter)
    for variant in ("base", "sft_best", "grpo_best"):
        evalplus_score(variant)


def cmd_eval(args: argparse.Namespace) -> None:
    if not args.skip_taco:
        eval_taco(args)
    if not args.skip_evalplus:
        eval_evalplus(args)


def add_eval_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--sft-adapter", required=True)
    parser.add_argument("--grpo-adapter", required=True)
    parser.add_argument(
        "--container-image",
        help="TACO Docker 镜像；默认使用仓库冻结的 digest，也可用 CODEGUIDE_DOCKER_IMAGE 覆盖",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("check", help="只检查冻结数据与正式配置，不加载 7B 模型")
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("sft", help="运行正式双卡 QLoRA SFT")
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--deepspeed", action="store_true", help="改用已有 ZeRO-2 配置")
    p.set_defaults(func=cmd_sft)

    p = sub.add_parser("grpo", help="从选定 SFT adapter 热启动正式 GRPO")
    p.add_argument("--sft-adapter", required=True)
    p.set_defaults(func=cmd_grpo)

    p = sub.add_parser("eval", help="一键运行 TACO-515 与 EvalPlus")
    add_eval_args(p)
    p.add_argument("--skip-taco", action="store_true")
    p.add_argument("--skip-evalplus", action="store_true")
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("eval-taco", help="只运行 TACO-515")
    add_eval_args(p)
    p.set_defaults(func=eval_taco)

    p = sub.add_parser("eval-evalplus", help="只运行 HumanEval(+)/MBPP(+)")
    add_eval_args(p)
    p.set_defaults(func=eval_evalplus)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
