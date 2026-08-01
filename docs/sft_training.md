# QLoRA SFT 运行说明

正式输入是 `data/final/sft_accepted.jsonl`，必须先校验其 SHA256 为
`08ef448f4be6b6b34ee2b6b7af5748827feeba0a0f36cc393350374671c86a1b`。
该大文件不进入 Git，需要与仓库代码一起单独同步到云服务器。

## 本地检查

```bash
python scripts/prepare_sft_calibration.py
python scripts/audit_sft_training_format.py --local-files-only
python -m src.training.train_sft --validate-only --mode calibration
pytest -q tests/test_sft_data.py
```

本地检查只读取 tokenizer，不下载7B权重。训练数据使用固定 train/dev ID，训练入口不会重新随机切分。

## 双 RTX 4090 云端

在已安装 CUDA PyTorch 的环境中安装：

```bash
pip install -r requirements-sft.txt
huggingface-cli login  # 或通过 HF_TOKEN 提供访问权限
bash scripts/preflight_sft_dual_4090.sh
bash scripts/run_sft_calibration_dual_4090.sh
```

当前云镜像若使用 PyTorch CUDA 13，bitsandbytes 必须为0.48.0以上；项目锁定
`bitsandbytes>=0.48.2,<0.49`。preflight 会真实执行一次 NF4 前后向与
PagedAdamW8bit step，不能仅凭 Python 包可导入判定兼容。

若 FlashAttention 2 不可用，`attention_backend: auto` 会显式记录并回退到 PyTorch SDPA。
训练使用普通 DDP，每个进程绑定一张 GPU，不使用 `device_map="auto"`。分布式入口固定为
`python -m torch.distributed.run`，确保继承当前激活虚拟环境；不能直接调用 PATH 中可能来自
base conda 的 `torchrun`。输出仅包含 LoRA adapter、checkpoint、tokenizer/config 和 run manifest。

恢复校准训练：

```bash
bash scripts/run_sft_calibration_dual_4090.sh \
  --resume-from-checkpoint outputs/sft/qwen25_coder_7b_qlora_8k/calibration_seed20260728/checkpoint-25
```

校准完成后执行固定 Base/Adapter 对照（需要 source bank 和 Docker verifier）：

```bash
export CODEGUIDE_EXECUTION_IMAGE='python:3.11.9-slim-bookworm@sha256:8fb099199b9f2d70342674bd9dbccd3ed03a258f26bbd1d556822c6dfc60c317'
bash scripts/eval_sft_calibration.sh \
  outputs/sft/qwen25_coder_7b_qlora_8k/calibration_seed20260728/adapter
```

正式全量入口已经准备，但只有500条校准达到验收门槛后才可执行：

```bash
python -m torch.distributed.run --standalone --nproc_per_node=2 -m src.training.train_sft \
  --config configs/sft/qwen25_coder_7b_qlora_8k.yaml --mode full \
  --output-dir outputs/sft/qwen25_coder_7b_qlora_8k/full_seed20260728
```

不要在当前本地 RTX 4060 环境执行上述训练命令，也不要降低8K长度或删除长样本来规避 OOM。
