from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.training.common import load_config
from src.training.grpo_train import validation_summary as grpo_summary
from src.training.train_sft import validation_summary as sft_summary

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("relative", "stage"),
    [("configs/sft.yaml", "sft"), ("configs/grpo.yaml", "grpo")],
)
def test_training_configs_use_one_schema(relative, stage):
    config = load_config(ROOT / relative, stage)
    assert config["schema_version"] == "codeguide-training-v2"
    assert config["model"]["name_or_path"] == "Qwen/Qwen2.5-Coder-7B-Instruct"
    assert config["quantization"]["quant_type"] == "nf4"


@pytest.mark.parametrize(
    ("mode", "train_count", "eval_count"),
    [("calibration", 500, 100), ("full", 9791, 515)],
)
def test_sft_modes_share_the_same_entrypoint(mode, train_count, eval_count):
    summary = sft_summary(load_config(ROOT / "configs/sft.yaml", "sft"), mode)
    assert summary["train"] == train_count
    assert summary["eval"] == eval_count
    assert summary["framework"] == "trl.SFTTrainer"


def test_grpo_config_resolves_frozen_data_without_loading_model():
    summary = grpo_summary(load_config(ROOT / "configs/grpo.yaml", "grpo"))
    assert summary["train"] == 6451
    assert summary["eval"] == 50
    assert summary["framework"] == "trl.GRPOTrainer"


@pytest.mark.parametrize(
    ("filename", "distributed_type"),
    [("dual_gpu.yaml", "MULTI_GPU"), ("dual_gpu_deepspeed.yaml", "DEEPSPEED")],
)
def test_accelerate_config_is_dual_gpu(filename, distributed_type):
    config = yaml.safe_load(
        (ROOT / "configs/accelerate" / filename).read_text(encoding="utf-8")
    )
    assert config["distributed_type"] == distributed_type
    assert config["num_processes"] == 2
    assert config["mixed_precision"] == "bf16"


def test_grpo_generation_batch_is_compatible_with_dual_gpu():
    config = load_config(ROOT / "configs/grpo.yaml", "grpo")
    global_batch = config["training"]["per_device_train_batch_size"] * 2
    assert global_batch % config["generation"]["num_generations"] == 0
    assert config["training"]["loss_type"] in {"grpo", "bnpo", "dr_grpo"}
    assert isinstance(config["training"]["scale_rewards"], bool)


def test_dependencies_have_one_canonical_file():
    requirement_files = sorted(ROOT.glob("requirements*.txt"))
    assert requirement_files == [ROOT / "requirements.txt"]
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    for framework in ("trl", "peft", "accelerate", "deepspeed", "flash-attn"):
        assert framework in requirements
    assert "unsloth" not in requirements.lower()
