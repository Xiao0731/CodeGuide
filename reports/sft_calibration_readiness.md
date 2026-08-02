# QLoRA SFT 500条校准执行报告

## 当前结论

500 条真实双 RTX 4090 QLoRA 校准、adapter 独立重载以及固定 40 条 dev 的 Base/Adapter Docker 对照已完成。校准训练无 OOM/NaN，Adapter 在 4096-token 无截断条件下同时提升了教学结构完整率和严格代码 Pass@1。

## 云端校准结果

- 模型：Qwen/Qwen2.5-Coder-7B-Instruct，4-bit NF4 QLoRA，LoRA r=32/alpha=64，8K，Liger fused linear CE。
- 环境：双 RTX 4090 24 GiB，PyTorch 2.9.0+cu130，bitsandbytes 0.48.2。
- 数据：500 train / 100 dev，固定 seed=20260728。
- 训练：32/32 optimizer steps，runtime 580.62 s，平均 train loss 0.7215665，step 25 eval loss 0.6630970，无 OOM/NaN。
- adapter：保存、独立重载和真实生成通过。

## 4096-token 固定 dev 对照

| 指标 | Base | Adapter |
|---|---:|---:|
| Docker Pass@1 | 4/40 (10.0%) | 7/40 (17.5%) |
| 教学模板完整 | 0/40 | 21/40 |
| 存在 Python 代码块 | 40/40 | 40/40 |
| 接口匹配 | 40/40 | 40/40 |
| 平均生成 tokens | 841.6 | 1856.55 |
| 撞上生成上限 | 0 | 0 |
| 未闭合代码围栏 | 0 | 0 |

Adapter 相对 Base 严格 Pass@1 净提升 7.5 个百分点，教学模板完整率净提升 52.5 个百分点。Adapter 剩余 33 条失败为 30 条 wrong answer 和 3 条 runtime/timeout。本轮已排除 2048 截断偏差，因此剩余失败按小规模校准后的能力不足记录。

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
