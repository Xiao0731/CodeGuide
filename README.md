# CodeGuide

CodeGuide 是一个面向 OI/ACM 初孊者的算法教孊模型后训练项目。训练目标同时包含两项硬要求：生成可执行的正确代码，以及给出题意、观察、推导、复杂度和常见错误组成的完整教学回答。

## 训练流程

```text
TACO 原题与多候选 Python reference
  -> Docker verifier 筛选 reference
  -> DeepSeek teacher 生成 reference-guided 教学标签
  -> 冻结 canonical SFT 与 train/dev split
  -> TRL SFTTrainer + PEFT NF4 QLoRA
  -> TRL GRPOTrainer + correctness/teaching-contract reward
  -> TACO / EvalPlus 统一评测
```

项目不实现通用训练框架。分布式、混合精度、量化、LoRA 和训练循环分别使用 Accelerate、Transformers、bitsandbytes、PEFT 与 TRL；FlashAttention 和 DeepSpeed 通过标准配置启用。

## 目录

```text
configs/                 训练、Accelerate、DeepSpeed 与评测配置
data/final/              冻结 SFT 与 verified source bank
data/splits/             固定 SFT/GRPO split
scripts/                 少量可直接执行的实验入口
src/data/                TACO、ChatML 与 source bank 数据逻辑
src/reward/              Docker verifier 和 GRPO 两路 reward
src/training/            TRL SFT/GRPO 训练实现
tests/                   参数化合同测试
outputs/                 不入 Git 的模型输出与正式评测结果
```

## 环境

所有 Python 依赖统一在 `requirements.txt`。CUDA 服务器建议先安装与驱动匹配的 PyTorch，再安装项目依赖；`flash-attn` 应在已有 PyTorch 的环境中构建。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip packaging ninja
pip install --no-build-isolation -r requirements.txt
```

国内服务器可先设置：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

## 快速复现

仓库提供统一的 `scripts/quickstart.py` 窗口，只负责编排已有训练、评测和验证入口，不复制底层逻辑。

```bash
# 1. 只检查数据与正式配置
python scripts/quickstart.py check

# 2. 正式 SFT
python scripts/quickstart.py sft

# 3. 从选定 SFT checkpoint 热启动 GRPO
python scripts/quickstart.py grpo \
  --sft-adapter /path/to/sft/checkpoint-200

# 4. 一键评测 Base / SFT / GRPO
python scripts/quickstart.py eval \
  --sft-adapter /path/to/sft/checkpoint-200 \
  --grpo-adapter /path/to/grpo/best_adapter
```

`eval` 会依次完成 TACO-515 的生成、Docker 严格验证与汇总，并运行 HumanEval(+)/MBPP(+)。若只想跑其中一部分，可使用 `eval-taco`、`eval-evalplus`，或给 `eval` 传入 `--skip-taco` / `--skip-evalplus`。

SFT 默认复现 `LR=1e-4` 主轨迹；GRPO 必须显式提供选定的 SFT adapter。完整实验协议、历史 checkpoint 口径和底层分阶段命令仍保留在各脚本与项目记录中，快速入口不改变这些实验语义。

## 核心命令

仅检查冻结数据和配置，不加载模型：

```bash
python scripts/train_sft.py --validate-only --mode full
python scripts/train_grpo.py --validate-only
```

双卡 SFT：

```bash
mkdir -p logs
accelerate launch --config_file configs/accelerate/dual_gpu.yaml \
  scripts/train_sft.py --config configs/sft.yaml --mode full \
  2>&1 | tee logs/sft.log
```

从 SFT adapter 热启动 GRPO：

```bash
accelerate launch --config_file configs/accelerate/dual_gpu.yaml \
  scripts/train_grpo.py --config configs/grpo.yaml \
  --sft-adapter-path /path/to/best_sft_adapter \
  2>&1 | tee logs/grpo.log
```

若实验需要 DeepSpeed，改用已经接入 ZeRO-2 的启动配置，训练代码无需改动：

```bash
accelerate launch --config_file configs/accelerate/dual_gpu_deepspeed.yaml \
  scripts/train_sft.py --config configs/sft.yaml --mode full
```

## 数据与验证合同

- canonical SFT：`data/final/sft_accepted.jsonl`
- SFT split：9,791 train / 515 dev
- assistant-only loss，不允许静默截断最终代码
- GRPO 从 SFT adapter 热启动，不允许从 base 冷启动
- correctness reward 复用 `src.reward.execution.verify_code`
- call-based 与 standard-input 使用同一 verifier
- 正式 Docker 镜像必须固定 digest
- reference 只属于 teacher-side privileged context，不进入 student user message

## 项目记录

- `GPT_PROJECT_CONTEXT.md`：实现与技术上下文
- `EXPERIMENT_LOG.md`：实验结果、故障与处理
- `DECISION_LOG.md`：关键技术决策
- `CLAIMS_MATRIX.md`：可公开结论与证据边界
- `CodeGuide_后训练项目实施与验收规划书_v1.2.md`：主计划
