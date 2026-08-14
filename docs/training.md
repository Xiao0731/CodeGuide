# Training

## Framework boundary

- TRL owns the SFT and GRPO loops.
- Accelerate owns process launch and device placement.
- PEFT and bitsandbytes own LoRA and NF4 quantization.
- Transformers selects FlashAttention 2 when installed, otherwise SDPA.
- DeepSpeed is optional and injected through Accelerate configuration.
- CodeGuide only owns frozen data contracts, assistant-only labels, code execution and teaching-contract rewards.

## SFT calibration and full runs

Both experiments use `scripts/train_sft.py`:

```bash
accelerate launch --config_file configs/accelerate/dual_gpu.yaml \
  scripts/train_sft.py --mode calibration --max-steps 3

accelerate launch --config_file configs/accelerate/dual_gpu.yaml \
  scripts/train_sft.py --mode full
```

The configuration is `configs/sft.yaml`. CLI values only override run-specific fields such as output directory, model path, learning rate and maximum steps.
The frozen dataset is tokenized before it reaches TRL. Its `labels` already mask every
non-assistant token with `-100`, so `SFTTrainer` is configured to skip dataset preparation
and preserve the audited completion-only loss contract.

## GRPO

`configs/grpo.yaml` points to the frozen SFT adapter and GRPO train/eval data. The trainer receives two separate rewards so TRL applies the configured `[0.9, 0.1]` weights:

1. correctness: verifier test pass rate;
2. teaching contract: substantive sections, code block and complexity analysis.

```bash
export CODEGUIDE_EXECUTION_IMAGE='python:3.11.9-slim-bookworm@sha256:...'
accelerate launch --config_file configs/accelerate/dual_gpu.yaml \
  scripts/train_grpo.py --config configs/grpo.yaml
```

Use `execution_backend: subprocess` only for trusted local smoke tests. Formal reward execution uses the pinned Docker backend.

## Optional DeepSpeed

The default launcher uses ordinary two-GPU Accelerate. ZeRO-2 is an optional launch-time
choice and does not require another training implementation:

```bash
accelerate launch --config_file configs/accelerate/dual_gpu_deepspeed.yaml \
  scripts/train_sft.py --config configs/sft.yaml --mode full
```
