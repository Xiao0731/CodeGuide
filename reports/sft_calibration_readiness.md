# QLoRA SFT 500条校准准备报告

## 当前结论

本地训练前管线已落地并通过数据与 tokenization 验收；当前机器不是双 RTX 4090，未下载完整7B模型，也未执行真实500条校准。当前停止点是“等待云端执行 calibration”，不能宣称正式 SFT 门槛已经通过。

## 冻结输入

- canonical：`data/final/sft_accepted.jsonl`
- SHA256：`08ef448f4be6b6b34ee2b6b7af5748827feeba0a0f36cc393350374671c86a1b`
- train/dev：9,791 / 515
- max sequence length：8,192
- calibration：500条，仅来自固定 train
- calibration IDs 内容 SHA256：`6d6975dd2938257150ab7b297d7d39d5ef5c55481163ee7e0011ee07eccd4a11`

校准集分布：A/B=337/163，standard-input/call-based=381/119；覆盖 easy、medium、medium_hard、hard、very_hard、unknown 和全部主要来源。

## Completion-only loss 证据

使用训练入口共用的 `tokenize_assistant_only()` 和 Qwen2.5-Coder-7B-Instruct 正式 chat template，对全部10,306条重新审计：

- 通过：10,306
- 失败：0
- 超长/截断：0
- 最大长度：8,173
- supervised token ratio：77.5844%
- prompt token：全部 label=-100
- assistant 教学讲解与 Python 代码块：位于监督区
- padding：动态补齐且 label=-100

机器可读证据：`data/manifests/sft_training_format_audit.json`。

## 正式初始配置

- Base：Qwen/Qwen2.5-Coder-7B-Instruct
- 4-bit NF4 + double quant，bf16 compute
- LoRA r=32, alpha=64, dropout=0.05
- target：q/k/v/o/gate/up/down projections
- 每卡 batch=1，gradient accumulation=8，双卡有效 batch=16
- paged AdamW 8-bit，LR=2e-4，cosine，warmup=0.03
- gradient checkpointing，`use_reentrant=false`
- 首次校准1 epoch，不 packing，动态 padding，按长度分组
- attention backend：优先 FlashAttention 2，不可用时显式记录并回退 SDPA

## 已通过与待验证

已通过：canonical hash、固定 split、500条确定性抽样、配置解析、assistant-only mask、全量零截断、动态 padding 纯逻辑、CLI help/validate-only、Shell语法。

必须在双4090云端验证：CUDA与依赖、两 rank GPU 绑定、实际4-bit模型加载、只有LoRA参数可训练、8K forward/backward、显存峰值、吞吐、loss/grad、checkpoint恢复、adapter重载推理、Base/Adapter Docker执行对照。

