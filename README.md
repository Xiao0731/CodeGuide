# CodeGuide（工作名）

面向 OI/ACM 初学者的算法教学后训练系统。目标不是让模型套一个“第一步、第二步”的题解模板，而是在代码正确这一硬门槛上，逐步构建题解讲解、学生错误诊断和分级提示能力。

> 当前处于 **G0：仓库可信性修复**。仓库尚未交付正式 SFT/GRPO
> adapter，也没有资格声称教学能力提升或全面优于同类项目。

## 研究路线

```text
TACO 原题
  → 多候选 Python reference 解析与执行验证
  → reference 仅作为 teacher 私有上下文
  → teacher 教学标签生成
  → 生成代码全测试通过硬过滤
  → Qwen2.5-Coder-7B-Instruct NF4 QLoRA SFT
  → 从最佳 SFT adapter 热启动 GRPO
  → 代码正确性 + 输出合同 + 准入后的教学评分
  → ExplainBench / TutorBench / 独立代码冻结集
```

项目的正式基座固定为 `Qwen/Qwen2.5-Coder-7B-Instruct`。SFT 与
GRPO 使用同一套 LoRA 结构；序列长度须根据 tokenizer 全量分布审计后，
从 4096、6144、8192 中冻结，不能沿用旧配置猜一个值。

## 当前证据边界

当前代码快照已经覆盖：

- TACO 本地加载、多候选 reference 解析和离线验证脚本；
- `standard_input` 与受支持的 `call_based` 执行合同；
- `scratch`、`reference_guided_label`、`code_explanation` 三种标签模式；
- teacher 生成代码的语法检查与全测试通过硬门槛；
- QLoRA SFT 和 GRPO 训练入口；
- GRPO online/held-out tests 的确定性拆分；
- 独立 dev 上的严格 Pass@1 checkpoint 合同；
- CLI 与 Gradio 推理入口。

尚未完成：

- 约 10K accepted 教学数据的正式生成；
- tokenizer 全量长度审计与正式配置冻结；
- 代理模型 SFT/GRPO dry-run；
- 正式 7B SFT、GRPO 和最终 adapter；
- TeachingCritic 训练与准入；
- ExplainBench/TutorBench 冻结评测；
- 受控训练消融与外部竞争基线比较。

详细证据见 [功能声明矩阵](CLAIMS_MATRIX.md)。任何 README 功能描述都以该矩阵的状态为准。

## 正式入口

```text
scripts/build_sft_dataset.py
scripts/train_sft.py
scripts/train_grpo.py
scripts/evaluate_model.py
```

`scripts/train_grpo.py` 是唯一公开 GRPO 命令；实际实现位于
`src/training/grpo_train.py`。`scripts/data_generate/` 下的 APPS
脚本属于早期探索路径，不是当前正式数据主线。

## 本地 G0 检查

当前无需 GPU 的检查：

```bash
python -m compileall -q .
python -m pytest -q
python scripts/check_environment.py --json artifacts/g0/environment.json
python scripts/validate_config.py --config configs/train_config.yaml
```

正式训练环境安装必须使用经过目标 GPU 验证的 `requirements.lock.txt`。
当前未验证完成前，不能用浮动的 `requirements.txt` 直接开启五天云端窗口。

## 数据生成安全合同

正式标签生成默认执行 teacher 给出的代码。以下任一情况都会拒绝写入
accepted SFT：

- 没有代码块；
- Python 语法错误；
- verifier 不支持；
- 执行报错或超时；
- 任一可用测试未通过。

正式执行使用 Docker 后端，并要求：

```bash
export CODEGUIDE_EXECUTION_IMAGE='python:3.11.9-slim-bookworm@sha256:8fb099199b9f2d70342674bd9dbccd3ed03a258f26bbd1d556822c6dfc60c317'
python scripts/validate_docker_verifier.py \
  --image "$CODEGUIDE_EXECUTION_IMAGE"
```

镜像必须固定到 digest。该受限容器合同已在 Windows 10 + Docker Desktop
环境完成本机实测，证据见 `artifacts/g0/docker_verifier_report.json`。
这不等同于第三方安全审计，也不应简称为“安全沙箱”。本地 `subprocess`
后端仅供受信任的单元测试。

## 训练口径

TeachingCritic 准入前，GRPO 主配置只使用：

```text
0.9 × execution correctness + 0.1 × output contract
```

旧的本地 teaching heuristic 只统计表面结构、连接词和注释密度，不能判断
易错点是否真实、学生错误诊断是否准确、提示是否越级，因此只作为诊断指标，
不进入梯度。

主配置关闭额外 batch Z-score。组内奖励方差很低时只记录
`zero_advantage_ratio`；没有文本、代码和 AST 多样性证据时，不把它包装成
“Generation Collapse”。

## 项目归属与发布

本项目按独立实现管理；TACO、Qwen、DeepSeek API 与训练框架属于公共资产。
当前审查快照不含 Git 历史，所以公开发布前仍须完成来源、许可证和提交历史
审计。详见 [PROVENANCE.md](PROVENANCE.md)。

正式公开前还需要更换当前可能与外部项目混淆的工作名。简历与面试开场应
描述自己实际实现的数据、训练、奖励和评测闭环，不围绕任何博主项目叙述。
