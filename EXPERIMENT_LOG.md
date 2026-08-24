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
- 33 条失败细分：题意/核心算法 19，边界/状态 5，I/O 4，代码块提取 2，实现异常 2，资源超限 1。详见 `reports/sft_calibration_failure_analysis_4096.md`。

# EXP-012：多代码块提取器离线重放

- 不重新生成，直接重放已保存 40 题 Base/Adapter 输出。
- Base 保持 4/40；Adapter 从 7/40 升至 8/40。
- Adapter 变更提取的题为 `taco_c9bd8772b5` 和 `taco_ee517dace9`；前者解除 NameError 但仅通过 1/15，后者恢复为 100% 通过。
- 原报告与新报告分开保留，差异见 `reports/sft_calibration_extractor_v2_replay.json`。

# EXP-013：full SFT 首次启动的 NCCL SHM 失败

- 两个 rank 均完成 7B checkpoint 加载，失败点为 DDP `_verify_param_shape_across_processes`。
- NCCL 报错为无法 attach `/dev/shm/nccl-*`，属于云容器 IPC/共享内存限制，不是 OOM 或数据错误。
- 处理：双卡入口默认设置 `NCCL_SHM_DISABLE=1`，待云端原配置重启复验。

# EXP-014：full SFT step 25 dev 评估 OOM

- NCCL 修复通过，full SFT 运行到 25/612 steps，loss/grad norm 全部有限。
- 首次 full-dev 评估在 rank 0 申请 4.54 GiB 完整 logits 时 OOM；非训练 forward/backward OOM。
- 修复：`TrainingArguments(prediction_loss_only=True)`，评估仅聚合 loss，不改变 515 条 dev、eval 间隔或训练超参数。

# EXP-015：prediction_loss_only 配置未生效的二次 OOM

- full SFT 再次完成 25/612 steps，训练 loss 从 0.8963 降至 0.5932，随后在首次 dev 评估申请 4.54 GiB 并 OOM。
- traceback 明确显示评估仍调用 `compute_loss(return_outputs=True)`，所以 EXP-014 的配置级修复在当前 Transformers 4.53.3 + Liger 0.8.0 环境中无效。
- 改为显式 loss-only `prediction_step`，从运行路径上禁止评估保留词表维度 logits；待云端同一 full 入口越过 step 25 复验。
- 二次复验仍 OOM，且自定义 `prediction_step` 已实际命中；官方 Liger 0.8.0 `lce_forward` 显示 eval mode 不会自动启用 `skip_logits`。现进一步显式传入 `skip_logits=True`，待第三次云端复验。

# EXP-016：9,791 条 full SFT 正式训练

- 配置：Qwen2.5-Coder-7B-Instruct，双 RTX 4090，4-bit NF4，LoRA r=32/alpha=64，8K，completion-only loss，1 epoch。
- 结果：612/612 optimizer steps；runtime 15,032.6429 秒；train loss 0.5693142565；吞吐 0.651 samples/s、0.041 steps/s。
- dev：末段 515 条评估 `eval_loss=0.5323174596`，runtime 182.406 秒，2.823 samples/s；评估后继续训练至 epoch 1.0。
- 稳定性：无 OOM、NaN、NCCL rank failure；末段 grad norm 约 0.10--0.20，学习率按 cosine 正常衰减至接近 0。
- 状态：训练完成，等待 adapter/manifest 文件检查和独立重载生成；尚未完成 full Adapter 的 Docker Pass@1 质量评测。
- 产物验收：adapter 文件齐全，权重约 309 MiB；独立重载对 `taco_616bc08bca` 生成 64 tokens，输出正确进入题意理解结构，`adapter_reloaded=true`。
- 下一步：复用固定 40 题、seed 20260728、4096-token 确定性生成协议，生成 Base/full Adapter 回答后在本地 Docker 离线验证。

# EXP-017：full Adapter 固定 40 题 Docker 对照

- Base：Pass@1 4/40，模板完整 0/40，代码块/接口匹配 40/40，无撞限。
- full Adapter：Pass@1 4/40，模板完整 9/40，代码块 39/40，接口匹配 40/40；wrong answer 35，runtime/timeout 1。
- 通过集合：共同 3 题；full 独有 `taco_cc94dfa6d5`；Base 独有 `taco_6709566aed`。因此算法正确率净提升 0。
- 长度：full 平均 2,187.6 tokens，最大 4,096；`taco_24ea55cd42`、`taco_4b5c54b35a` 撞限且未输出代码。
- 唯一资源失败：`taco_3bd853140f` 在固定 Docker 5 秒 CPU 限制下超时。高部分通过样本包括 `taco_b967c24088` 0.9913、`taco_ecffe57a6b` 0.9444，属于边界错误而非整体接口失败。
- 结论：full SFT 工程闭环通过，但固定 40 题未证明算法能力提升；进入 GRPO 或发布前应把“教学结构提升”和“代码正确率”分开判断。

# EXP-018：calibration/full 数据构成与顺序归因复查

- calibration 500：A/B=337/163；只与原始 accepted 前 1,300 条重合 65 条，不是“早期纯 A 数据训练”。
- canonical 前 1,300：A/B=858/442；其余 9,006：A/B=6,034/2,972，比例近似。原始 accepted 前 1,300 才是 1,300/0。
- full train 的 ID 已按固定哈希重排，训练还启用 DDP 随机 sampler 和长度分桶；原始前 1,300 条不会集中在 epoch 前段。
- loss 在约 0.53--0.58 平台与监督交叉熵收敛、后段 cosine 学习率下降相容，不是数据顺序切换的证据。
- calibration/full 均用峰值 LR 2e-4，但更新步数为 32 对 612。full Pass@1 回落可能涉及累计更新过强、教学目标压过算法能力、标签风格异质和 40 题方差，需用受控消融区分。

# EXP-019：full Adapter 生成长度与正确率

- 40 题中 21 条生成 >=2048 tokens，Pass@1=0/21，平均测试通过率 0.1410；19 条 <2048，Pass@1=4/19，平均测试通过率 0.4910。
- >=1536 tokens 的 27 条已无完全通过样本；真正达到 4096 上限的 2 条均无 Python 代码块。
- 该结果证明“长回答是风险信号”，但不证明 4096 上限诱导冗长：greedy 解码看不到 `max_new_tokens`，不同上限共享相同前缀。
- calibration 的 2048 与 4096 轮严格 Pass@1 未提升，排除“仅提高 generation budget 即可恢复算法正确率”。

# EXP-020：外部 CodeGuide-LLM 项目证据审查

- 输入：用户提供 `CodeGuide-LLM-main.zip` 与 README。
- 可验证产物：无 SFT JSONL、模型权重、训练日志、W&B 导出、blind eval 输出或 ablation report；`evals/results` 为空。
- 数据正确性风险：TACO tests 为空；默认只做语法检查；execution failure 不一定丢弃；通过一半测试即可 `ok`；不支持 call-based。
- 训练风险：GPT-4o 最多生成 3072 tokens，但 SFT 序列上限 2048 且无超长隔离，代码在回答末尾；3 epochs、LR 2e-4 的效果没有日志。
- 评测口径：ablation 是 reward-function rescoring，不是模型训练消融；blind eval 是未附结果的可执行框架。README 的性能叙述不能视为复现实验数据。
- 对比结论：无法证明外部 GPT-4o 蒸馏更优；当前项目暴露出的 4/40 结果虽不理想，但具备固定题集、原始 generation、Docker verification 和完整训练日志，证据等级更高。

# EXP-021：外部 CodeGuide-LLM 实现级对照

- 审查对象：`D:\Downloads\CodeGuide-LLM-main.zip`，SHA256 `B5F21B9EFEAEAF2DADCA624BF2BB5BB9796CE63603C0BCC97F491E6830C4FA01`；仅解压到临时目录，未写入当前代码树。
- QLoRA/warm-start：外部 SFT 为 4-bit 新 LoRA；warm-start 是 GRPO 加载 SFT adapter。当前项目正式 SFT 同样完成 NF4 QLoRA，GRPO 热启动入口存在但正式 7B GRPO 尚未运行。
- 外部主训练 reward：accuracy 0.6 + format 0.4；teaching 是监控项。README 的“三路奖励”组件存在，但未接入主 GRPO 梯度。
- 外部 LocalTeachingReward：结构关键词、TTR、注释率、连接词四维启发式；包内没有 API 对齐结果。当前项目仅保留为 surface diagnostic，权重为 0。
- 外部 reward normalization：对 combined reward 整批 Z-score；不是三路分别归一化，且 TRL 后续仍做组相对归一化。当前项目默认关闭。
- 外部 collapse/curriculum/best-checkpoint：代码均存在；curriculum 默认关闭，best checkpoint 只评前 20 个 public-test 样本且用平均 pass rate。当前项目已把 checkpoint 口径改为独立 heldout tests + strict Pass@1，但三个功能都尚未进入正式 GRPO 实测。
- 外部静态运行检查：`python -m compileall` 发现 3 个用户入口脚本存在未转义中文引号语法错误；`tests/test_rewards.py` 为 28 passed / 1 failed，失败原因是实现已把无测试评分改为 AST 代理分而测试仍断言 1.0。
- SFT 对照：外部 GPT-4o scratch prompt 不含 reference/tests/interface，TACO tests 为空，默认不执行代码；当前 DeepSeek reference-guided 流程以 10,415 个 verified source 为母库，A/B 两路都以统一 verifier 为接纳硬门槛。
- Benchmark 归因：HumanEval 88.4% 来自 Qwen2.5-Coder-7B-Instruct 官方技术报告，不是外部项目训练结果；本项目也不得直接占用该数字作为训练收益。
- 值得迁移并实测的设计：独立验证集 best checkpoint、zero-advantage 监控、curriculum 消融、Bootstrap 教学评测。暂不迁移：无测试 AST correctness、整批 reward Z-score、未经对齐的 teaching heuristic 入梯度。

# EXP-022：仓库清理与正式入口回归

- 删除 34 个旧/重复已跟踪文件，并清除约 2.874 GiB 可恢复中间资产；没有删除 canonical SFT、source bank、TACO test、GRPO 正式数据或最终评测输出。
- SFT 训练实现从“两套入口”收敛为 `src.training.train_sft`，`scripts/train_sft.py` 只保留兼容转发。
- 旧 2048-token 校准输出删除，4096-token calibration/full 原始 generation 与 verification 保留并移入统一 `outputs/eval/`。
- 测试首次暴露 0 字节 `summarize_evalplus_code_capability.py` 与旧 G0 smoke manifest 测试；两者对应实验已有冻结结果，故删除失效入口/过期测试，不恢复中间 smoke 依赖。
- 回归：compileall 通过，完整 pytest 85 passed；SFT/GRPO CLI help 均可启动。

# EXP-023：统一 TRL/Accelerate 训练入口回归

- 范围：只重构训练与奖励基础设施，不改 canonical SFT、固定 split、teacher 标签或既有正式评测结果。
- SFT：同一 `scripts/train_sft.py` 以 `--mode calibration/full` 解析为 500/100 与 9,791/515；预计算 `labels=-100` 掩码继续承担 assistant-only loss，TRL 不再二次准备数据。
- GRPO：冻结 manifest 解析为 6,451 train / 50 eval；从配置中的 full SFT adapter 热启动。该初次重构错误地将正式奖励简化成 correctness 与 contract 两路，已由 EXP-024 纠正。
- 兼容性修复：该初次重构错误地迁移到 TRL 0.19.1 和 `dr_grpo`；这改变了 2026-08-15 正式运行语义，已由 EXP-024 恢复为 TRL 0.22.2、`loss_type=grpo`、`scale_rewards=false`。
- 清理结果：训练/数据/评测命令均由配置驱动；约 13,700 行旧脚本和重复实现被移除。正式实验结果保留，旧占位矩阵与可重建压缩包删除。
- 验证边界：本地执行 compileall、pytest、SFT 两模式 validate-only、GRPO validate-only 和配置检查；本轮没有 GPU、Docker 奖励或模型训练运行。

# EXP-024：恢复 2026-08-15 正式 GRPO 语义

- 目的：保留 TRL/Accelerate/PEFT/bitsandbytes 新架构，只纠正重构期间发生的实验配置漂移。
- 数据实测：train `6,451`（easy `3,228`、medium `1,735`、hard `1,488`），dev `50`，TACO final `515`；三组交集均为 0。
- curriculum：固定 easy -> medium -> hard，每阶段 1 epoch，completion 上限依次为 512/768/1024；入口不允许关闭或合并阶段。
- 训练：Qwen2.5-Coder-7B-Instruct + 用户指定最佳 SFT adapter；TRL 0.22.2，`loss_type=grpo`、`scale_rewards=false`，generations 4、temperature .8、top-p .95、LR `1e-5`、beta .05、batch/device 1、accumulation 8。
- reward：单个 canonical composite callable 对每条 completion 只执行一次 verifier，并按冻结公式返回 total；pass-rate、static、code、contract 仅记录诊断。training verifier 为 subprocess，Docker 不进入在线训练。
- 验证：compileall、完整 pytest 和 `train_grpo.py --validate-only` 均通过；没有启动模型训练或覆盖任何历史结果。

# EXP-025：Blind Teaching Evaluation 流水线合同验证

- 范围：实现 Base/SFT/GRPO 同题生成、DeepSeek/Qwen 双盲 pairwise + absolute scoring 和 Markdown 聚合；不修改训练代码，不运行模型或调用 Judge API。
- 数据：默认从冻结 SFT dev/TACO-515 ID 池分层抽取 50 条；离线 validate 确认 canonical 与 selection IDs 可读取，参考 assistant 标签不进入生成消息或 Judge prompt。
- 盲评：每个模型对的 A/B 位置按种子平衡；测试确认 10 条时每组正反各 5 条。winner 映射回真实模型后再计算胜率和 Judge disagreement。
- 成本：50 条完整评测为 `50 × 3 pairs × 2 judges = 300` 个成功 judgment；五维绝对分与 pairwise winner 共用同一次请求，API/schema 失败重试会增加实际请求数。
- 验证边界：新增离线测试覆盖 reference 剥离、盲序、加权分重算和 disagreement；真实教学提升结论必须等两位 Judge 完成后才能填写。

# EXP-026：冻结 TACO-515 三阶段回答导入

- 日期：2026-08-24
- 输入：Base、`mixed_lr2e4_step050`、`grpo_best` 三份既有 generation，各 515 条。
- 一致性：三者唯一 ID 均为 515，交集 515，空回答 0，协议均为 `compact-code-first-taco515-selected-v1`。
- Blind50：沿用 `sft_dev_ids.json` 与 seed 20260728 分层抽取 50 条；三阶段各导入 50 条。
- 结果：`tests/test_sft_eval_protocol.py` 9 passed；`--stage validate` 和 `--stage import` 均通过。
- 成本：本步骤没有模型推理、Docker 执行或 Judge API 请求。后续双 Judge 预计 `50×3×2=300` 个成功 judgment。
- 边界：现有回答共享 compact-code-first prompt，因此可做公平的阶段相对比较；不能解释为换用原始长教学 prompt 后的新一轮生成。
- Judge 更正：第二 Judge 从误记的豆包更正为 Qwen3.8 Max；选择最高能力通用模型而非 Coder 专项模型，关闭 thinking，并使用结构化 JSON 输出。本次仅修改配置和调用语义，尚未产生 API 请求。

# EXP-027：Blind Judge 断点落盘故障

- 现象：Judge 运行到 142/300 后，Windows `os.replace(results.json.tmp, results.json)` 返回 `PermissionError: WinError 5`。
- 证据：崩溃后主文件 JSON 合法且含 142 条 judgment；遗留 tmp JSON 合法且含 143 条。DeepSeek/Qwen API 与 judgment schema 不是根因。
- 修复：唯一临时文件、flush/fsync、8 次指数退避原子替换；测试注入一次 PermissionError 后第二次替换成功且无残留临时文件。
- 恢复：结果按已存在的 judge/pair 跳过，不会重跑已落盘的 142 条；崩溃时仍在途但未落盘的少量请求可能重新计费。
