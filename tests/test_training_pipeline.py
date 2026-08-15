from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.training.common import load_config
from src.training.grpo_train import partition_curriculum
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
    assert summary["dev"] == 50
    assert summary["taco515_final_only"] == 515
    assert summary["pairwise_overlap"] == {
        "train_dev": 0,
        "train_taco515": 0,
        "dev_taco515": 0,
    }
    assert summary["framework"] == "trl.GRPOTrainer"
    assert summary["checkpoint_selection"] == "grpo_dev"
    assert summary["reward_function"] == "formal_composite_reward"
    assert summary["execution_backend"] == "subprocess"
    assert summary["loss_type"] == "grpo"
    assert summary["scale_rewards"] is False


def test_grpo_validation_reports_configured_sft_adapter(tmp_path):
    config = load_config(ROOT / "configs/grpo.yaml", "grpo")
    adapter = tmp_path / "best_sft_adapter"
    adapter.mkdir()
    config["model"]["sft_adapter_path"] = str(adapter)
    summary = grpo_summary(config)
    assert summary["sft_adapter"] == str(adapter.resolve())
    assert summary["sft_adapter_configured"] is True
    assert summary["sft_adapter_exists"] is True


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
    effective_batch = (
        config["training"]["per_device_train_batch_size"]
        * 2
        * config["training"]["gradient_accumulation_steps"]
    )
    assert effective_batch == 16
    assert effective_batch % config["generation"]["num_generations"] == 0
    eval_batch = config["training"]["per_device_eval_batch_size"] * 2
    assert eval_batch == config["generation"]["num_generations"]
    assert config["training"]["loss_type"] == "grpo"
    assert config["training"]["scale_rewards"] is False


def test_formal_curriculum_is_ordered_and_exhaustive():
    config = load_config(ROOT / "configs/grpo.yaml", "grpo")
    expected = [("easy", 3228, 512), ("medium", 1735, 768), ("hard", 1488, 1024)]
    stages = config["curriculum"]["stages"]
    assert [
        (stage["difficulty"], stage["expected_count"], stage["max_completion_length"])
        for stage in stages
    ] == expected
    records = [
        {"problem_id": f"{difficulty}-{index}", "difficulty": difficulty}
        for difficulty, count, _ in expected
        for index in range(count)
    ]
    partitioned = partition_curriculum(records, config["curriculum"])
    assert [len(items) for _, items in partitioned] == [3228, 1735, 1488]
    assert sum(len(items) for _, items in partitioned) == 6451


def test_dependencies_have_one_canonical_file():
    requirement_files = sorted(ROOT.glob("requirements*.txt"))
    assert requirement_files == [ROOT / "requirements.txt"]
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    for framework in ("trl", "peft", "accelerate", "deepspeed", "flash-attn"):
        assert framework in requirements
    assert "unsloth" not in requirements.lower()
    assert "trl==0.22.2" in requirements
    assert "transformers==4.55.4" in requirements
