# CodeGuide（工作名）

面向 OI/ACM 初学者的算法教学后训练系统。目标不是让模型机械套用“第一步、第二步”的题解模板，而是在代码正确这一硬门槛上，逐步构建题意拆解、关键观察、算法步骤、复杂度分析、常见错误和可运行代码的统一输出能力。

> 当前处于 **SFT checkpoint 选优与 GRPO 前验收阶段**。正式 8K QLoRA SFT、adapter 保存与独立重载已经完成；GRPO 尚未开始，教学质量也尚未完成冻结盲评。

## 研究路线

```text
TACO 原题
  → 多候选 Python reference 解析与执行验证
  → reference 仅作为 teacher 私有上下文
  → teacher 教学标签生成
  → 生成代码全测试通过硬过滤
  → Qwen2.5-Coder-7B-Instruct NF4 QLoRA SFT
  → checkpoint 轨迹与冻结执行评测
  → 外源代码能力 + 教学能力双模块验收
  → 从唯一最佳 Mixed SFT adapter 热启动 GRPO
  → Base / SFT / GRPO 统一评测
```

项目正式基座为 `Qwen/Qwen2.5-Coder-7B-Instruct`。正式 SFT 使用 8K 序列、4-bit NF4 QLoRA、双 RTX 4090、assistant-only loss 和 Liger fused linear cross-entropy。

## 当前已完成

- TACO train 去重后 24,237 题的多候选 reference 解析；
- 10,415 道题获得当前 verifier 下全测试通过的 reference；
- 10,340 条正式 accepted 教学标签，接纳率 99.28%；
- 10,306 条 canonical SFT 数据与固定 `9,791/515` train/dev 划分；
- tokenizer 全量长度审计并冻结 `max_seq_length=8192`；
- 双 RTX 4090 完成 612/612 optimizer steps 的正式 QLoRA SFT；
- adapter 保存、独立重载和确定性生成闭环；
- Base、多个 SFT checkpoint 的 TACO-100 轨迹评测；
- Base、step50、step20、step200 的冻结 TACO-515 Docker 严格评测；
- Standard/Call 双 LoRA 专家训练与模式纯净探针流程；
- HumanEval(+)/MBPP(+) 外源代码能力评测协议与云端生成脚本。

## 当前核心结果

冻结 TACO-515 包含 393 道 `standard_input` 和 122 道 `call_based`。在当前显式接口元数据、确定性生成和本地 Docker 严格执行口径下：

| 模型 | Overall | Standard | Call |
|---|---:|---:|---:|
| Base | 80/515 = 15.53% | 17/393 = 4.33% | 63/122 = 51.64% |
| SFT step20 | 113/515 = 21.94% | 49/393 = 12.47% | 64/122 = 52.46% |
| SFT step200 | 128/515 = 24.85% | 71/393 = 18.07% | 57/122 = 46.72% |

step200 相对 Base 增加 48 道严格通过题，Overall 提升 9.32 个百分点，相对提升约 60%；`standard_input` 提升到 Base 的约 4.18 倍。与此同时，step200 的 `call_based` 出现下降，因此最终 SFT checkpoint 仍需结合 HumanEval(+)/MBPP(+) 与教学能力基线选择，不能只看 Overall。

## 当前证据边界

已经可以表述：

- 正式 QLoRA SFT 的训练、保存与重载闭环已完成；
- 在冻结 TACO-515 和当前提示协议下，SFT 严格 Pass@1 相对 Base 有明显提升；
- 不同 I/O 模式存在不同 checkpoint 偏好，step20 是重要的能力保持型对照；
- 双专家用于估计模式拆分上限，不是最终部署架构。

目前仍不能表述：

- step200 已经是最终 `SFT_CHAMPION`；
- 文本 I/O 路由器达到 100% 有效准确率；
- 模型能只看自然题面完美判断判题接口；
- HumanEval(+)/MBPP(+) 外源能力已经保持或提升；
- SFT 已证明教学质量显著提升；
- 双专家已经优于 Mixed；
- GRPO 已经训练完成或有效。

当前文本路由器的 100% 结果已被审计为标签泄漏：SFT user prompt 中存在显式 `io_mode`、`fn_name` 等人工元数据。该负结果只影响路由实验，不否定 SFT checkpoint 的代码生成结果，但要求后续增加独立 clean-interface 评测。

详细记录见：

- [功能声明矩阵](CLAIMS_MATRIX.md)
- [实验复盘](EXPERIMENT_LOG.md)
- [关键决策](DECISION_LOG.md)
- [2026-08-03 至 2026-08-05 阶段复盘](docs/2026-08-03_至_2026-08-05_SFT选优与GRPO前阶段复盘.md)

任何 README 功能描述都以 `CLAIMS_MATRIX.md` 的证据状态为准。

## 评测分为两个模块

### 代码能力

- TACO 严格 Pass@1 与平均测试通过率；
- Standard/Call 分模式结果；
- 接口匹配、语法失败、截断和生成长度；
- HumanEval、HumanEval+、MBPP、MBPP+ 外源保持性。

### 教学能力

- 代码块与五个教学栏目合同；
- 章节顺序、非空、冗余和截断；
- 题意与关键观察的技术正确性；
- 讲解与代码一致性；
- 复杂度、常见错误、初学者友好度和简洁性；
- 全样本综合教学效用与双方代码都通过时的纯教学对照。

本项目最终目标始终是教学模型。代码能力是教学可信度的硬门槛，不能用教学格式替代；同样也不能只报告代码分数而忽略项目的核心教学目标。

## 正式入口

```text
scripts/build_sft_dataset.py
python -m src.training.train_sft
scripts/train_grpo.py
scripts/evaluate_sft_matrix.py
```

`scripts/train_sft.py` 和 `scripts/train_grpo.py` 只保留为兼容入口，正式实现分别位于
`src/training/train_sft.py` 和 `src/training/grpo_train.py`。脚本用途与主入口见
`scripts/README.md`。

## 数据与执行合同

正式标签生成中，以下任一情况都会拒绝写入 accepted SFT：

- 没有代码块；
- Python 语法错误；
- verifier 不支持；
- 执行报错或超时；
- 任一可用测试未通过。

正式执行使用固定 digest 的受限 Docker 后端：

```bash
export CODEGUIDE_EXECUTION_IMAGE='python:3.11.9-slim-bookworm@sha256:8fb099199b9f2d70342674bd9dbccd3ed03a258f26bbd1d556822c6dfc60c317'
python scripts/validate_docker_verifier.py \
  --image "$CODEGUIDE_EXECUTION_IMAGE"
```

该合同已在 Windows 10 + Docker Desktop 实测，证据位于 `artifacts/g0/docker_verifier_report.json`。这不等同于第三方安全审计，也不应简称为通用“安全沙箱”。

## GRPO 进入门槛

正式 GRPO 前必须完成：

```text
唯一 Mixed SFT_CHAMPION 冻结
→ EvalPlus 外源代码能力结果
→ SFT 教学客观格式基线
→ 固定样本教学盲评
→ reward 与 split 冻结
→ 16～32 题、20～50 steps 的 GRPO smoke
```

正式 GRPO 仍从单一 Mixed SFT adapter 暖启动。Standard/Call 专家仅用于 Oracle 模式解耦消融，不作为最终训练或部署主线。

TeachingCritic 或可靠教学奖励完成准入前，执行正确性和接口合同仍是正式梯度的主线。旧的本地 teaching heuristic 只用于表面诊断，不得直接宣称真实教学质量。

## 版本控制约定

- commit message 使用中文；
- 新增代码注释优先使用中文；
- 变量名、函数名、配置键、第三方 API 和原始错误信息保留英文；
- 每次关键实验、负结果、故障与决策同步写入 Markdown 复盘文档；
- checkpoint、正式数据、全量缓存、密钥和大日志不得进入 Git。

## 项目归属与发布

本项目按独立实现管理；TACO、Qwen、DeepSeek API、EvalPlus 与训练框架属于公共资产。公开发布和简历描述只陈述本项目实际完成的数据、训练、奖励与评测闭环，不使用外部项目 README 指标替代可复现实验。
