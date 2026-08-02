# CodeGuide 实验与故障复盘记录

该文件记录后训练项目各阶段的实验对比、异常证据、根因和应对手段，作为论文复现与面试讲解材料。项目架构和当前实现详见 `GPT_PROJECT_CONTEXT.md`，关键取舍详见 `DECISION_LOG.md`。

## EXP-001：TACO reference-guided SFT 全量标签生成

**日期**：2026-08-01
**输入**：10,415 条 reference 离线验证完全通过的 TACO 候选。  
**教师模式**：`reference_guided_label`；教师可见 reference，student user 不可见 reference。  
**验证**：语法检查加统一 Docker `verify_code()` 执行硬门槛。

### 结果

| 指标 | 数量/比例 |
|---|---:|
| verified 候选 | 10,415 |
| accepted | 10,340 |
| unresolved rejected | 75 |
| 接纳率 | 99.28% |
| standard-input accepted | 7,889 |
| call-based accepted | 2,451 |
| student reference 字段泄漏 | 0 |
| accepted pass rate < 1.0 | 0 |

按难度的 accepted / 候选通过率：easy 5,041/5,067（99.49%）、medium 1,061/1,070（99.16%）、medium_hard 1,464/1,472（99.46%）、hard 961/969（99.17%）、very_hard 446/460（96.96%）、unknown 1,367/1,377（99.27%）。

### 剩余失败

| 原因 | 数量 | 处置 |
|---|---:|---|
| 教师响应失败或 max-token 截断 | 40 | 可恢复，但不阻塞当前数据版本 |
| Docker slim 镜像缺少 numpy | 28 | 环境依赖隔离，不误记为算法错误 |
| 执行超时 | 7 | 保留诊断，后续单独复核 |

### 关键故障与修复

1. 曾有 410 条被标为 `recovery_wrong_answer`，抽检后确认全部是 Docker daemon 未启动，测试实际未执行，`pass_rate=0.0` 只是失败默认值。
2. 增加 Docker API 调用前预检，并将 Docker 连接失败分类为可恢复的 `docker_unavailable`；恢复后本轮成功写入 493 条。
3. rejected 改为只保存当前未解决 ID 的紧凑快照，accepted 成功后立即从 rejected 移除，最终两者无重叠。

### 数据版本

- accepted：`data/sft_train_ref_label_accepted.jsonl`
- accepted SHA-256：`CBFA65F00AD635654431004D8C20AD4BCB1D64EF6CFE7DB984331DB5F9E7042D`
- rejected：`data/sft_train_ref_label_rejected.jsonl`
- rejected SHA-256：`AE9149D9FED92EB235F118DB3DFEC949E069271FD37ED3CC7BDAE4BFB350B594`

### 结论

SFT 标签生成阶段通过。10,340 条执行完全通过的教学样本足以进入数据冻结、训练/验证划分和 SFT 训练；75 条隔离失败不应继续阻塞主线。
# EXP-002：SFT 数据冻结、长度审计与固定划分

**日期**：2026-08-01
**基座 tokenizer**：`Qwen/Qwen2.5-Coder-7B-Instruct`  
**正式输入**：10,340 条 accepted，75 条 unresolved rejected。

## 结果

- canonical 10,306 条；34 条超过 8K 的极端长样本隔离，不截断代码。
- canonical SHA256：`08ef448f4be6b6b34ee2b6b7af5748827feeba0a0f36cc393350374671c86a1b`。
- 长度 P50/P95/P99/max：2,603 / 5,062 / 6,728 / 8,173 tokens。
- 4096 截断 1,350 条（13.10%），8192 截断 0 条，因此选择 8192。
- source bank 10,415 条，SHA256：`b74eae83bb538f1a1fb3af24425da8f5f14305dc89a291cf37498e7f590ebda5`；Docker 抽样复验 20/20。
- 固定 SFT train/dev=9,791/515；重复划分 hash 完全一致，train/dev 与 TACO test 均无泄漏。

## 风险与应对

- 14 条 standard-input 没有明显 I/O 关键词，但具有执行通过证据，保留并记录警告。
- 1,360 条难度为 unknown，训练可用，难度消融时单独报告。
- 旧代码提取器可能把非 Python 代码围栏识别成最终代码；已改为优先提取最后一个显式 Python 围栏并加入回归测试。
# EXP-003：SFT训练格式全量校验与500条校准集冻结

**日期**：2026-08-01
**环境**：本地 tokenizer-only；未加载7B模型、未训练。

- 训练共用格式审计：10,306/10,306通过，截断0，最大8,173 tokens。
- supervised token ratio：77.5844%；prompt和动态padding labels均为-100，assistant教学内容与代码均受监督。
- calibration：500条，重复生成文件 hash 一致；ID内容 SHA256 `6d6975dd2938257150ab7b297d7d39d5ef5c55481163ee7e0011ee07eccd4a11`。
- 本地单元测试：7 passed；配置 `validate-only` 得到 train=500、eval=100。
- 首次发现并修复：冻结 split 是带 `ids` 字段的 manifest 对象，不是裸数组；loader 现同时支持两种格式并加入回归测试。
- 未完成项：双4090真实QLoRA校准及Base/Adapter执行对照，状态为等待云端。
# EXP-004：云端 CUDA 13 / bitsandbytes 兼容故障

**日期**：2026-08-01

- 云端：Python 3.12.4、PyTorch 2.9.0+cu130、双 RTX 4090、bitsandbytes 0.47.0。
- 现象：缺少 `libbitsandbytes_cuda130.so`，但旧 preflight 继续完成并打印 passed。
- 根因：项目依赖上限 `<0.48` 排除了首个正式支持CUDA 13的bitsandbytes版本；包导入过程内部记录异常但未让外层脚本失败。
- 修复：依赖改为 `>=0.48.2,<0.49`，preflight 增加真实 NF4 backward 与8-bit optimizer step。
- 未计为训练结果：本次尚未加载7B或执行校准 optimizer step。
# EXP-005：云端 torchrun 解释器漂移

**日期**：2026-08-01

- preflight 原生算子通过：bitsandbytes 0.48.2、NF4和PagedAdamW8bit正常。
- 校准启动失败：`/opt/conda/bin/torchrun` 拉起 base Python，两个rank均报 `ModuleNotFoundError: transformers`。
- 根因：使用 `venv --system-site-packages` 时 `.venv/bin` 未必生成独立 torchrun console script，PATH继续命中conda base入口。
- 修复：改用 `.venv` 当前 `python -m torch.distributed.run`，并让双rank预检验证 transformers 与解释器路径。
- 训练状态：尚未下载完整模型、未发生 optimizer step，本次不计为校准实验。
# EXP-006：双 4090 首次 8K QLoRA 校准 OOM

**日期**：2026-08-01
**环境**：双 RTX 4090 24 GiB；Python 3.12.4；PyTorch 2.9.0+cu130；Transformers 4.53.3；bitsandbytes 0.48.2；SDPA。

- preflight、双 rank 绑定、canonical hash、固定 500/100 数据、NF4 backward 和 PagedAdamW8bit step 均通过。
- 4 个模型分片下载完成，两个 rank 均成功加载 4-bit Qwen2.5-Coder-7B-Instruct。
- 在训练进度 `0/32` 的首个 batch 失败：rank 0 在 cross entropy 申请 4.32 GiB 时 OOM，rank 1 在 backward 申请 3.73 GiB 时 OOM。
- 已完成 optimizer steps：0；adapter：未生成；因此该运行不计为校准实验结果。
- 处理：接入 `liger-kernel==0.8.0` 的 fused linear cross-entropy，开启 `expandable_segments:True`，并增加云端真实 kernel backward 预检。
- 待复验：先运行 1 step 长样本探针，记录峰值显存、loss 与参数更新；通过后再运行完整 32 steps。
# EXP-007：双 4090、500 条 8K QLoRA 校准训练

**日期**：2026-08-02

- 配置：Qwen2.5-Coder-7B-Instruct；双 RTX 4090；4-bit NF4；LoRA r=32/alpha=64；8K；Liger fused linear CE；500 train / 100 dev。
- 结果：32/32 optimizer steps；runtime 580.62 秒；train loss 0.7215665；第 25 step eval loss 0.6630970。
- 吞吐：0.861 train samples/s，0.055 optimizer steps/s；约 18.14 秒/step（含一次约 29.9 秒 dev 评估）。
- 稳定性：所有记录的 loss 与 grad norm 有限，无 OOM、timeout、NCCL rank failure；首步 0 learning rate 来自单 step warmup，第二步达到 2e-4，随后按 cosine 衰减。
- 状态：训练执行通过；adapter 重载生成尚待云端执行。
# EXP-008：SFT calibration adapter 重载 smoke

**日期**：2026-08-02

- adapter：`outputs/sft/qwen25_coder_7b_qlora_8k/calibration_seed20260728/adapter`。
- 结果：`adapter_reloaded=true`；题目 `taco_616bc08bca`；生成 64 tokens；completion 非空。
- 输出特征：以中文“理解题意”分步教学结构开头，并准确引用题目函数 `between(a, b)`。
- 结论：adapter 保存、独立重载与真实推理通过；尚未完成 Base/Adapter 固定 dev 质量对照。
# EXP-009：SFT calibration 跨环境评测准备

**日期**：2026-08-02

- 云端检查：`taco_verified_source_bank.jsonl.zst` 不存在，`docker` 命令不存在。
- 处理：将原一体化评测改为云端生成、本地 Docker 验证；不在训练节点安装 Docker，也不上传 source bank。
- 生成契约：固定 dev seed 20260728，默认 40 题，Base/Adapter 均 `do_sample=false`、最大 2048 tokens，逐题 flush 并支持断点续跑。
- 本地验证契约：固定镜像 digest，分别输出 generation、verification 和汇总 comparison report。
- 状态：代码与离线合约测试完成；等待云端生成产物。
- 回归检查额外发现旧评测入口导入了不存在的 `iter_jsonl`；已改为 source bank 模块真实导出的 `iter_source_bank`，避免本地验证启动即失败。

# EXP-010：40 条 Base/Adapter Docker 对照首轮

**日期**：2026-08-02

- Base：代码块 40/40、接口匹配 40/40、严格 Pass@1 4/40（10%）、教学模板完整 0/40。
- Adapter：代码块 40/40、接口匹配 40/40、严格 Pass@1 7/40（17.5%）、教学模板完整 18/40。
- 配对变化：共同通过 3 题；Base 独有通过 1 题；Adapter 新救回 4 题，净提升 3 题，即 +7.5 个百分点。
- 长度偏差：Base 平均 841.6 tokens、无 2048 撞限；Adapter 平均 1727.3 tokens，8/40 撞限且这 8 条全部未通过，其中 4 条代码围栏未闭合。
- 判定：训练方向有效，但 2048 completion budget 对教学化 Adapter 构成非对称截断。full SFT 暂不启动；先仅重生成 8 条撞限 Adapter，提升上限至 4096，其余 72 份自然结束回答全部复用。

# EXP-011：4096-token 截断恢复后 Base/Adapter 对照

**日期**：2026-08-02

- 样本：固定 40 条 dev，seed=20260728，Base/Adapter 共享同一 selection，最大生成 4096 tokens。
- Base：Docker 严格 Pass@1=4/40（10.0%），教学模板完整=0/40，代码块和接口匹配均为 40/40。
- Adapter：Docker 严格 Pass@1=7/40（17.5%），教学模板完整=21/40，代码块和接口匹配均为 40/40。
- Adapter 相对 Base 代码正确率净提升 7.5 个百分点（4 题到 7 题），教学结构完整率提升 52.5 个百分点。
- 生成长度：Base 平均 841.6，Adapter 平均 1856.55；两者均无撞限、无未闭合代码围栏。
- Adapter 剩余失败为 wrong_answer 30 条、runtime_or_timeout 3 条。由于截断已消失，这些失败主要归因于 500 条、1 epoch 校准后的算法能力与运行时鲁棒性，不再归因于 completion budget。
- 结论：校准实验的质量方向为正，且无代码提取或接口退化；17.5% 是小样本校准模型的诊断结果，不是全量 SFT 的最终指标。
