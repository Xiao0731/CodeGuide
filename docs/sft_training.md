# QLoRA SFT 运行说明

正式输入是 `data/final/sft_accepted.jsonl`，必须先校验其 SHA256 为
`08ef448f4be6b6b34ee2b6b7af5748827feeba0a0f36cc393350374671c86a1b`。
该大文件不进入 Git，需要与仓库代码一起单独同步到云服务器。

## 本地检查

```bash
python scripts/audit_sft_training_format.py --local-files-only
python -m src.training.train_sft --validate-only --mode full
pytest -q tests/test_sft_data.py
```

本地检查只读取 tokenizer，不下载7B权重。训练数据使用固定 train/dev ID，训练入口不会重新随机切分。

## 双 RTX 4090 云端

在已安装 CUDA PyTorch 的环境中安装：

```bash
pip install -r requirements-sft.txt
huggingface-cli login  # 或通过 HF_TOKEN 提供访问权限
bash scripts/preflight_sft_dual_4090.sh
bash scripts/run_sft_full_dual_4090.sh 2>&1 | tee logs/sft_full.log
```

当前云镜像若使用 PyTorch CUDA 13，bitsandbytes 必须为0.48.0以上；项目锁定
`bitsandbytes>=0.48.2,<0.49`。preflight 会真实执行一次 NF4 前后向与
PagedAdamW8bit step，不能仅凭 Python 包可导入判定兼容。

若 FlashAttention 2 不可用，`attention_backend: auto` 会显式记录并回退到 PyTorch SDPA。
训练使用普通 DDP，每个进程绑定一张 GPU，不使用 `device_map="auto"`。分布式入口固定为
`python -m torch.distributed.run`，确保继承当前激活虚拟环境；不能直接调用 PATH 中可能来自
base conda 的 `torchrun`。输出仅包含 LoRA adapter、checkpoint、tokenizer/config 和 run manifest。

正式全量入口：

```bash
python -m torch.distributed.run --standalone --nproc_per_node=2 -m src.training.train_sft \
  --config configs/sft/qwen25_coder_7b_qlora_8k.yaml --mode full \
  --output-dir outputs/sft/qwen25_coder_7b_qlora_8k/full_seed20260728
```

不要在当前本地 RTX 4060 环境执行上述训练命令，也不要降低8K长度或删除长样本来规避 OOM。
