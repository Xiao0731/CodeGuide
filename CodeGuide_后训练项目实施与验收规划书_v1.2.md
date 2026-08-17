# CodeGuide-LLM 后训练项目实施与验收规划书

> 版本：v1.2
> 制定日期：2026-07-27
> 适用对象：后续参与 CodeGuide-LLM 开发、数据构造、训练、评测与文档维护的 Codex/开发者
> 项目定位：面向算法竞赛初学者的“题目 → 分步讲解 → 正确代码”代码推理后训练项目
> 核心路线：TACO 10,415 条高可信候选题 → DeepSeek-V4-Flash reference-guided 教学标签与题目级教学 rubric → Qwen2.5-Coder-7B-Instruct QLoRA SFT → 沿用 QLoRA adapter 的可验证奖励 GRPO → ExplainBench + TutorBench + 代码冻结集评测

### v1.2 纠偏与超越基线说明

本版本继承 v1.1 对 v1.0 的四项强制修订：

1. 不再把正式数据规模擅自缩减到 3,200 条；必须对 10,415 条 verified 候选题全部尝试生成，目标获得 10,000 条最终 accepted 教学数据。
2. 正式基座固定为 `Qwen/Qwen2.5-Coder-7B-Instruct`；`Qwen/Qwen3-8B` 仅保留为可选的低成本推理基线，不再作为云端训练前的二选一阻塞项。
3. 不预设 SFT `max_seq_length=4096`；必须先用最终基座 tokenizer 完成原题和 pilot 标签的分桶统计，再从 4096、6144、8192 中冻结训练上限。
4. QLoRA 从来没有被删除。SFT 与 GRPO 都必须明确使用“4-bit NF4 基座 + LoRA adapter”的参数高效训练方案；GRPO 从最佳 SFT adapter 热启动。

同时新增第五项总约束：

5. 不能把“功能比同类项目多”当成“项目更强”。必须以审计后的外部公开同类项目为竞争基线，分别在数据可信度、奖励抗攻击性、受控训练分支和冻结模型评测上完成可复算对照。未通过第 19 节“超越基线 Gate”时，只能称为独立实现但尚未证实胜出，不得称为全面优于外部基线。

---

## 2026-07-29 execution amendment: A/B label generation

The production label pipeline keeps successful pedagogical rewrites as Class
A and never regenerates existing accepted records. Class A receives one API
request. Any failure enters Class B, while historical rejected records enter
Class B directly.

Class B uses one teacher call for the teaching explanation and a JSON
line-comment plan. It does not request a complete Python rewrite. The pipeline
inserts comments into the verified TACO reference at AST-safe statement
boundaries, proves executable-token equality after removing comments, and
then runs syntax and Docker execution verification.

Recovery state is versioned to prevent repeated API consumption. Production
concurrency is configurable and the current local default is 20 API requests.

Operational amendment: after confirming an account-level
`deepseek-v4-flash` API concurrency limit of 2,500, the production teacher
request concurrency is set to 1,000. Docker verification remains locally
serialized by the current synchronous verifier call. This setting accelerates
generation but can consume API budget rapidly.

The API client must set `max_retries=0`; retries are controlled only by the
pipeline. With `distill_retries=1`, a failed Class-A request proceeds to the
single Class-B recovery path instead of being retried invisibly by the SDK.

Observed production evidence invalidated the 1,000-concurrency local setting:
1,999 HTTP 200 responses were accompanied by 454 client timeouts and only
about 1,478 accepted labels. Because Docker verification synchronously blocks
the API event loop, the local production default is reduced to 20. The 2,500
DeepSeek account limit is treated only as a remote service ceiling.

External teacher failures such as HTTP 402, transport interruption, or an
unavailable API are represented by `recovery_llm_failed` and remain retryable
after service restoration. They resume directly in Class B. Syntax,
interface, execution, and wrong-answer recovery failures remain terminal for
the current recovery version.

HTTP 402/`Insufficient Balance` is a fatal batch condition. On first
observation the pipeline cancels remaining tasks and preserves the JSONL
checkpoint instead of exhausting the queue with guaranteed failed requests.

The rejected JSONL is a current-state snapshot, not an append-only history.
It contains exactly one compact record per unresolved problem ID. Startup
removes IDs already present in accepted, and successful recovery atomically
removes the ID from rejected. Full historical events remain in runtime logs.

## 0. 本规划书的效力

本文件是后续实现的**项目总约束与阶段验收依据**，优先级高于仓库内陈旧 README、旧配置、Notebook 和未经验证的功能说明。

### 0.1 项目归属与对外叙事

本项目是独立设计、独立实现的算法教学后训练项目，不是任何博主仓库的 fork，也不以复用其代码、数据或 checkpoint 为前提。第 2.5 节和第 19 节中的公开同类项目只承担**内部竞争基线**角色，用于检查我们的设计是否有足够含金量。

对外必须准确区分：

- 自主实现：TACO 数据解析、reference 多候选验证、reference-guided 蒸馏、执行反馈修复、教学 rubric、TutorBench、reward、SFT/GRPO 编排、评测与报告；
- 公共资产：TACO 数据集、Qwen 基座、DeepSeek API、TRL、PEFT、bitsandbytes、Unsloth、vLLM、EvalPlus 等；
- 外部同类工作：只在相关工作或竞争对照中出现，不得写成代码来源。

简历、README 和面试开场禁止使用：

> “基于博主 CodeGuide-LLM 二次开发/改进……”

应使用：

> “独立设计并实现面向 OI/ACM 初学者的算法教学后训练系统……”

如果被追问同类项目，应如实回答：

> “公开社区存在目标相近的项目，但本项目未复用其代码、数据和模型；我们独立完成了数据构造、训练与评测，并将同类方案作为外部竞争基线做受控比较。”

正式公开前必须完成：

1. git 历史与文件来源审计；
2. 依赖、数据集和模型许可证/引用清单；
3. README 与外部项目的文字和目录相似性检查，避免沿用其宣传文案；
4. 将当前工作名 `CodeGuide-LLM` 更换为可区分的独立项目名，避免面试官误判为同名复刻；
5. 保留 `PROVENANCE.md`，记录哪些代码自主实现、哪些是公共依赖、是否使用 AI 辅助生成。

后续 Codex 必须遵守：

1. 未达到上一阶段验收门槛，不得进入下一阶段。
2. 不得因为代码已经存在，就声称功能已经“跑通”或“有效”。
3. 所有能力必须区分：
   - **已实现**：代码存在并通过静态检查；
   - **已跑通**：在声明环境中完成端到端运行；
   - **已验证**：在冻结数据和明确指标下取得可复现结果。
4. 不得静默改变基座模型、数据划分、奖励权重、评测集或主要超参数。
5. 一切会消耗双 4090 云端时间的工作，都必须先通过本地前置门槛。
6. 云服务器的五天是**纯训练与快速评测窗口**，不是开发、排错、生成数据或临时选型窗口。
7. 如果 GRPO 未能在五天内可靠完成，应交付可信的 SFT 项目，不得用离线奖励重打分伪装成 GRPO 实验。

---

## 1. 项目目标、研究问题与非目标

### 1.1 最终目标

训练一个能够针对 OI/ACM 风格算法题，输出以下内容的开源模型：

1. 准确理解题意与输入输出契约；
2. 提炼关键观察；
3. 分步推导算法；
4. 给出简洁的正确性说明；
5. 分析时间、空间复杂度；
6. 输出满足 `standard_input` 或 `call_based` 接口的可执行 Python 代码；
7. 提醒初学者常见错误。

模型的第一目标是**正确**，第二目标才是**会教**。语言流畅但算法错误的回答不得被视为优质教学答案。

### 1.2 需要回答的四个研究问题

| 编号 | 研究问题 | 最低证据 |
|---|---|---|
| RQ1 | verified reference 作为 teacher 私有上下文，是否能提高蒸馏标签正确率？ | scratch 与 reference-guided 在同一批题上的生成通过率、置信区间和错误类型 |
| RQ2 | 约 10K 教学数据上的 QLoRA SFT 是否让基座模型更稳定地产生结构化教学答案，同时提高或保持代码正确性？ | Base 与 SFT 在独立冻结集、教学集、EvalPlus 上的对比 |
| RQ3 | GRPO 是否进一步提高严格代码通过率，且不破坏教学质量与通用代码能力？ | SFT 与 SFT+GRPO 的冻结集对比、退化检查和统计区间 |
| RQ4 | 相比外部公开同类项目的实际实现，我们独立设计的 reference-guided 数据、教学奖励和 GRPO 方案是否在受控条件下更强？ | 外部基线式与本项目式奖励的对抗集比较；同一 SFT checkpoint 分叉的等预算 GRPO 对照；ExplainBench/TutorBench 配对评测 |

### 1.3 明确的非目标

本轮不追求：

- 训练 30B/80B 级模型；
- 完成全参数微调；
- 支持所有编程语言；
- 把静态格式启发式包装成真正的教学 Reward Model；
- 为追求正好 10,000 条而放宽代码正确性、接口、泄漏或截断门槛；
- 同时比较大量基座、优化器和 RL 算法；
- 宣称算法创新或 SOTA；
- 用训练集表现选择最佳 checkpoint；
- 在宿主机直接执行来源不可信的模型代码。
- 把一次性结构化题解直接等同于交互式教学；
- 因为 teacher 模型更新，就声称蒸馏质量天然高于 GPT-4o；
- 用 README 中的功能数量代替真实训练与冻结评测结果。

项目含金量来自“高可信数据—执行验证—SFT—在线 RL—冻结评测”的闭环，而不是参数规模。

---

## 2. 当前事实基线

### 2.1 已完成且可作为后续输入的成果

1. TACO train 去重后共 24,237 题。
2. 18,733 题存在 Python-like 参考解候选。
3. 10,415 题至少有一个参考解通过当前 verifier 全部测试。
4. 多候选回退额外救回 960 题。
5. verified reference 按 I/O 模式统计：
   - `standard_input`：7,952 / 21,013；
   - `call_based`：2,463 / 3,224。
6. 已建立 `scratch`、`reference_guided_label`、`code_explanation` 三种蒸馏模式。
7. 主线已确定为 `reference_guided_label`：
   - teacher 可见 verified reference；
   - student/user 消息不可见 reference；
   - 最终任务仍是“看题解题并教学”，不是“看代码解释代码”。
8. smoke 结果：
   - scratch：0/4 通过；
   - reference-guided 分层：4/5 通过；
   - reference-guided call-based：5/5 通过。

这些 smoke 只能证明方向值得扩大，不能宣称稳定通过率为 90% 或 100%。

### 2.2 数据源名称纠偏

必须区分项目的历史设计与当前已经实现的数据链路：

- **历史设计**：APPS 题库 + 50 条人工 seed + teacher 蒸馏；
- **当前已完成验证的主链路**：TACO 24,237 题 → Python reference 解析与执行验证 → 10,415 条 verified 候选；
- 标准 APPS benchmark 本身为 10,000 道题，不能把 TACO 的 `24,237 → 10,415` 统计写成 APPS；
- 50 条 seed 的目录和字段可沿用 APPS 风格，但这不改变正式原题来自 TACO 的事实。

如果后续决定真正切回 APPS，必须重新加载 APPS、重新统计题量、重新验证 reference 和重新生成 manifest；不得直接复用 TACO 的数字。APPS 原论文：[Measuring Coding Challenge Competence With APPS](https://arxiv.org/abs/2105.09938)。

### 2.3 当前不得继承为“已完成”的旧功能

以下内容在完成修复和实验前，一律视为**未闭环**：

- GRPO 训练入口；
- Teaching Reward 进入梯度；
- 新版 anti-hacking Format Reward 进入正式训练；
- Batch Z-score 改进有效；
- “本地沙箱”安全；
- 最优 checkpoint 按验证集 Pass@1 选择；
- 双盲评测；
- reward 消融提高模型能力；
- 2048 token 不会截断；
- 仓库开箱即复现。

### 2.4 已知 P0 问题

1. `train_grpo.py`、部分 Demo 脚本存在语法或入口问题。
2. 旧云平台 setup 脚本混入 Notebook `!pip/!git` 语法。
3. GRPO 当前实质只优化 accuracy + 旧 format。
4. Teaching Reward 仅记录日志，未进入返回奖励。
5. 新版 Format Reward 没有接入正式入口。
6. 自定义 Batch Z-score 的理论说明与实际实现不匹配。
7. `run_code` 失败的 teacher 标签仍可能被写入 SFT。
8. verifier 是隔离子进程，不是真正安全沙箱。
9. checkpoint callback 可能使用训练题，并把平均测试通过率误称为 Pass@1。
10. SFT `max_seq_length=2048` 存在截断代码尾部的高风险。

### 2.5 外部公开同类项目审计基线

该项目不是本项目的上游代码来源，只作为外部竞争基线。比较时必须以其仓库实际代码为准，而不是只按 README 的“12 项改进”宣传口径。当前审计得到的基线是：

| 维度 | 外部同类项目实际情况 | 本项目独立方案 | 当前判断 |
|---|---|---|---|
| 基座与 QLoRA | Qwen2.5-Coder-7B-Instruct；NF4 QLoRA | 同一基座与 QLoRA，完整冻结 revision、长度和 adapter 结构 | 持平；不能当成个人创新 |
| 训练数据广度 | code_contests + TACO，README 声称最多生成 10K | 主线聚焦 TACO 10,415 verified 候选；外部基准负责泛化评测 | 外部项目来源更广；本项目以可信度换广度 |
| 蒸馏标签 | GPT-4o + 格式/长度型 DataQualityChecker | verified reference 作为 teacher 私有上下文；执行反馈修复；最终代码 100% 通过可用测试 | 本项目已具备明显设计优势，仍需 300～500 题 pilot 证实 |
| 正确性执行 | 受限子进程；无测试时使用 AST 静态估分 | 容器隔离；只有真实测试题进入 GRPO；部分通过率 + 全通过 bonus；reward/held-out tests 拆分 | 本项目目标更强 |
| Format Reward | README 声称“正文长度 + Jaccard 防 hacking”，正式入口实际导入旧版 reward | 低权重 contract reward；正式入口唯一；对抗单测与梯度接入测试 | 完成接线与测试后更强 |
| Teaching Reward | LocalTeachingReward 依赖表面词、结构和注释；只监控，不进入梯度 | 使用题目、教学 rubric 和回答的轻量教学评分器；对抗验证后进入梯度 | 当前最大未闭环差距 |
| Reward 缩放 | 合并总奖励后做 batch Z-score，理论解释不成立 | TRL 原生多奖励与权重；主实验关闭额外缩放；缩放只作消融 | 本项目设计更严谨 |
| Curriculum | 按 easy→medium→hard 静态分阶段 | 按 SFT 的每题可学习度与组内有效优势动态采样，再逐步扩展到 hard | 必须训练验证后才能判优 |
| 坍缩监控 | 把 reward 方差低直接称为 generation collapse | 分离 `zero_advantage_ratio` 与文本/代码/AST 多样性 | 本项目指标定义更准确 |
| Best Checkpoint | 训练数据抽样 20 题；平均测试通过率误称 Pass@1；生成上限 512 | 独立冻结 dev；严格 Pass@1；完整 completion；代码与教学双门控 | 本项目目标明显更强 |
| Ablation | 对同一批文本换公式重打分，不是训练消融 | 同一 SFT adapter 分叉、等 prompt/步数/seed 的真实 GRPO 对照 | 完成后更强 |
| 教学评测 | 单评委匿名随机顺序；主要评 clarity/coherence/beginner-friendly | ExplainBench + TutorBench；技术正确性硬门控；自动与人工分开报告 | 完成后更强 |
| 仓库复现 | 多入口、语法/API/依赖错位 | 唯一入口、lock、proxy/full dry-run、run manifest | G0 通过后更强 |
| Demo 与最终模型 | README 已提供 CLI/Web UI 路径 | 目前尚未交付最终 adapter 与可验证 Demo | 当前外部项目展示完成度更强 |

因此当前不能写“我们已经全面优于外部同类项目”。准确结论是：

1. 已完成的数据解析、reference 验证与 reference-guided 方向，技术含量已经高于外部基线的表面质量过滤；
2. 规划中的安全执行、reward、checkpoint 与评测设计更严谨，但在实际跑通前都只是设计；
3. 外部项目在 README 完整度、Demo 路径和“看起来像成品”方面暂时领先；
4. 本项目要在核心主题上真正胜出，必须证明模型不只会生成结构化题解，还能针对学生状态诊断错误并选择合适提示。

---

## 3. 不可变资源约束

### 3.1 本地环境

按当前信息记为：

- GPU：RTX 4060；
- 内存：32 GB；
- CPU：以实际机器信息为准；
- 用途：开发、单元测试、API 数据生成、CPU 验证、数据分析、量化推理、微型 dry-run；
- 禁止承担：正式 7B/8B GRPO、全量多 checkpoint 评测。

如果“32CPU”实际指 32 核 CPU，而不是 32 GB 内存，只更新环境清单，不改变任务分工。

### 3.2 云端环境

- CPU：32 vCPU；
- GPU：2 × RTX 4090 24 GB，总显存 48 GB，但不能简单视为一张 48 GB 显卡；
- 可用时长：连续 5 天，按 120 小时硬上限管理；
- 必须预留：最后 12 小时只做最终评测、导出、校验与备份；
- 正式训练可支配时间上限：108 小时。

### 3.3 云端资源原则

1. 7B/8B QLoRA 应能在单张 4090 上训练。
2. SFT 阶段优先：
   - GPU0：训练；
   - GPU1：基线/验证集推理或并行评测；
   - 只有在多卡方案已提前验证时，才使用 DDP/FSDP。
3. GRPO 阶段优先：
   - GPU0：训练模型；
   - GPU1：独立 vLLM generation server；
   - 若 server 模式不稳定，回退到单卡/colocate，不能在生产窗口临时研究复杂并行。
4. 多 GPU 不等于显存自动相加。任何模型只有在单卡训练方案或已验证的切分方案存在时，才可进入候选。

TRL 官方支持将 vLLM 作为独立 server 放在与训练器不同的 GPU 上，这正适合双 4090 资源，但必须避免训练器与 server 使用同一张卡导致 NCCL 冲突：[TRL GRPO 文档](https://huggingface.co/docs/trl/en/grpo_trainer)。

---

## 4. 基座模型决策

### 4.1 正式基座

本轮正式基座固定为：

```yaml
model: Qwen/Qwen2.5-Coder-7B-Instruct
role: Base / QLoRA-SFT / QLoRA-GRPO 的共同底座
```

选择理由：

1. CodeGuide 的硬约束首先是代码生成、代码推理、修复与接口正确性。Qwen2.5-Coder 是代码专项系列，官方说明其预训练包含源码、文本—代码对齐和合成数据等 5.5T tokens，并专门增强代码生成、推理和修复能力。
2. `Instruct` 版本已经具备稳定的对话模板和指令跟随，更适合“题意—推导—复杂度—代码—错误提醒”的教学回答；在只有五天云端窗口的情况下，没有必要从 Base 权重重新学习基本对话行为。
3. 当前仓库的配置、Unsloth 加载、LoRA target、训练脚本、推理与 Demo 均围绕该模型实现，沿用它能减少与研究问题无关的迁移变量。
4. 7.61B 参数适合单张 24 GB 4090 上进行 NF4 QLoRA；第二张 4090 可留给 vLLM 生成、基线推理和评测。
5. 项目的创新点应放在高可信数据、执行验证、QLoRA SFT、可验证奖励 GRPO 与评测闭环，而不是把“模型更新一代”当成贡献。

官方资料：[Qwen2.5-Coder-7B-Instruct 模型卡](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct)、[Qwen2.5-Coder 官方介绍](https://qwenlm.github.io/blog/qwen2.5-coder-family/)。

### 4.2 Qwen3-8B 的位置

`Qwen/Qwen3-8B` 是更新的通用稠密模型，官方强调思考/非思考切换、通用推理、指令跟随、多语言和代码能力；但它不是 Qwen3-Coder，也没有直接证据表明它在本项目的可执行算法题和接口约束上优于 Qwen2.5-Coder-7B-Instruct。

因此：

- 不再把 Qwen3-8B 设为正式主选；
- 不要求为它生成第二套 SFT 数据或进行完整训练；
- 若本地时间允许，可在同一 60 题冻结集上做一次 4-bit **推理基线**；
- 该比较不构成 G4 阻塞项，也不得消耗正式云端主训练时间；
- 后续若项目闭环完成，可把 Qwen3-8B QLoRA 复现实验列为扩展工作。

官方资料：[Qwen3-8B 模型卡](https://huggingface.co/Qwen/Qwen3-8B)。

### 4.3 冻结规则

1. `configs/model_frozen.yaml` 必须记录精确 model revision、tokenizer revision、chat template hash 与量化配置。
2. 本地 RTX 4060 使用该模型的 4-bit 版本完成加载、tokenizer 审计、单 batch SFT 与推理 smoke；GRPO 代码路径可先由 0.5B～1.5B 代理模型验证。
3. 云端 H2～H6 完成正式 7B 模型的 SFT 单 batch 与 GRPO 单 step。
4. 若该模型因环境问题在 H6 前无法 dry-run，应先修环境或回退稳定依赖，不得静默更换为 Qwen3-8B。

---

## 5. SFT 数据构造方案

### 5.1 Teacher 模型

固定为：

```yaml
provider: deepseek
model: deepseek-v4-flash
```

DeepSeek 官方当前 API 标识确为 `deepseek-v4-flash`，支持思考与非思考模式；V4 Flash 为 284B 总参数、13B 激活参数：[DeepSeek V4 发布说明](https://api-docs.deepseek.com/news/news260424/)、[DeepSeek 模型与价格](https://api-docs.deepseek.com/quick_start/pricing/)。

### 5.2 Teacher 调用策略

#### 首次生成

- 模式：非思考；
- 温度：0.2；
- `top_p`：0.9；
- 最大输出：按 tokenizer 审计结果控制，目标 1,000～1,600 completion tokens；
- 并发：初始 16，稳定后最大 32；
- 返回：只返回最终教学答案，不把 `reasoning_content` 写入 SFT；
- 每次调用记录 request id、token usage、时间、参数、prompt hash。

#### 修复生成

当首次生成出现以下任一情况时进入修复：

- 语法错误；
- 接口错误；
- 运行时错误；
- 测试未全过；
- 代码块截断；
- 明显偏离 reference 算法；
- reference 泄漏；
- 教学结构缺失。

修复策略：

1. 第一次修复：非思考模式，提供错误类型、失败用例摘要，不暴露隐藏评测输出的完整答案。
2. 第二次修复：思考模式，提供原回答、verified reference 和最小必要错误反馈。
3. 两次修复后仍失败：进入 `rejected.jsonl`，不得进入 SFT。

### 5.3 50 条 APPS 风格 seed 的角色

当前 50 条 seed 沿用早期 APPS 路线的教学字段与分类，但正式原题主线已经是 TACO。seed 只用于：

- 约束教学风格；
- 展示字段深度；
- 控制讲解结构；
- 为不同 I/O 类型提供少量 few-shot。

seed 不得：

- 被误写为 10,415 条正式原题的数据来源；
- 替代 TACO verified reference；
- 混入最终评测；
- 在 prompt 中携带与当前题目相似到可能泄漏答案的样例；
- 每题堆叠过多 few-shot 增加 API token。

默认每次调用只放 1 条同 I/O 模式 seed；必要时最多 2 条。

### 5.4 Teacher 私有 prompt 输入

```text
题面
+ difficulty / io_mode
+ fn_name / starter_code（如有）
+ verified reference solution
+ 当前题目的测试契约摘要
+ 1 条教学 seed
+ 输出规范
```

最终 student 数据：

```text
user: 题面 + 必要接口信息
assistant: 教学式答案
```

最终 `messages.user` 和公开数据不得出现 reference。

除生成最终教学答案外，teacher 还必须为每道 accepted 题输出一份不进入 student prompt 的结构化教学 rubric：

```text
key_concepts              该题必须理解的关键概念
reasoning_steps           有依赖顺序的核心推导步骤
proof_invariants          正确性解释必须覆盖的不变量/论证点
complexity_target         正确复杂度及其依据
common_mistakes           真实易错点、错误后果与纠正方式
hint_ladder:
  H1                      只帮助重新理解题意
  H2                      暗示关键观察，不给算法全貌
  H3                      给出算法骨架，不给完整实现
  H4                      接近完整解法，用于多次提示后仍卡住
forbidden_early_reveal    第一轮提示不应直接泄漏的内容
```

rubric 由 verified reference、测试契约与失败样本共同支撑；不能只让 teacher 凭空罗列“注意边界、注意复杂度”之类的通用套话。正式教学评测使用的 rubric 必须人工抽查，且不得作为 student 输入泄漏答案。

### 5.5 输出结构

教学答案应自然覆盖：

1. 题意理解；
2. 关键观察；
3. 解法推导；
4. 正确性说明；
5. 复杂度分析；
6. Python 参考实现；
7. 常见错误。

允许合并相邻小节，不把“标题数量”当成教学质量本身。代码必须位于靠后但不能因 token 上限被截断。

本项目必须区分两种任务：

1. `full_explanation`：用户明确要求完整题解，模型可以最终给出代码；
2. `next_hint`：用户提供自己的进度或错误，只要求下一步帮助，模型不得无条件倾倒完整答案。

当前约 10K 主数据全部保留 `full_explanation`。另外从中分层抽取 500 题，每题构造 2 个不同学生状态，形成约 1,000 条多轮 `next_hint` 辅助 SFT 样本。学生状态至少覆盖：

- 读错题意或输入输出；
- 只想到会超时的暴力解；
- 卡在关键观察；
- 状态/递推式设计错误；
- 边界或下标错误；
- 复杂度判断错误；
- 代码与思路不一致。

这 1,000 条辅助样本不替代 10K full-explanation 数据，也不得混入最终 TutorBench。

### 5.6 数据规模

按“先 pilot、再全量构造”执行。3,200 条不再是正式数据上限。

#### Pilot

- 尝试生成：300～500 题；
- 分层：5 个 difficulty × 2 个 io_mode，尽量每格 30；
- 稀疏 cell 不重复采样，记录实际数量；
- 目标：确认 teacher 首次正确率、修复收益、token 长度和失败类型。

#### 正式 SFT

- 候选池：全部 10,415 条 verified TACO 题；
- 生成尝试：10,415 / 10,415，不得为了省 API 调用只抽 3K；
- 最终目标：10,000 条 accepted 教学数据；
- 目标划分：
  - SFT train：9,500；
  - SFT dev：500；
  - Tutor-SFT 辅助训练：另从 SFT train 的 500 道题构造约 1,000 条 `next_hint` 对话；
- 所有 accepted 样本代码必须通过全部可用测试；
- Tutor-SFT 的学生错误、目标教学动作和提示层级必须与 rubric 一致；它不要求每条回答都输出代码，反而要检查是否过早泄漏完整答案；
- SFT dev 只能用于长度、训练状态和 checkpoint 选择，不得用于梯度更新；
- 若两轮修复后 accepted 少于 10,000，先做失败类型分析与定向再生成；不得把错误代码、reference 泄漏、接口错误或截断样本放进 SFT 凑数；
- 如果 10,415 条候选的最终可用上限客观低于 10,000，应报告真实 accepted 数量，并通过重新验证额外 TACO reference 或新增独立 APPS 训练题扩充；扩充来源必须单独标记，不能伪装成原候选池。

#### GRPO 题目池

- 800～1,000 个训练 prompt，从 `SFT train` 内按 difficulty 与 I/O 模式分层抽取；
- 100 个 GRPO validation prompt，从 500 条 `SFT dev` 中预先冻结；它们不参与 SFT 或 GRPO 梯度更新；
- GRPO train 与 SFT train 允许重叠，因为它是同一模型的顺序后训练；这种重叠必须在报告中披露；
- GRPO validation 与 SFT train、GRPO train 不重叠，但允许作为 SFT dev 的预声明子集；
- 每题必须有足够测试支持 reward/held-out 拆分；
- 测试少于 4 个的题默认不进入 GRPO。

### 5.7 数据划分约束

1. 先按 canonical problem id 去重，再划分。
2. 检测题面近重复、模板重复和同题变体。
3. SFT train 与 GRPO train 可按第 5.6 节定义重叠；GRPO validation 可来自 SFT dev；最终测试集必须来自独立的 TACO test/外部基准，不能从 10,415 条 TACO train 候选中抽取。
4. 数据划分一旦冻结，写入：
   - `data/splits/sft_train_ids.txt`
   - `data/splits/sft_dev_ids.txt`
   - `data/splits/grpo_train_ids.txt`
   - `data/splits/grpo_dev_ids.txt`
   - `data/splits/final_test_ids.txt`
5. 每个文件记录 SHA256；后续不得按实验结果移动样本。

### 5.8 数据字段

每条私有构造记录至少包含：

```text
problem_id
source / source_split
difficulty
io_mode
task_type
prompt
student_state（仅 next_hint）
starter_code / fn_name
reference_hash
reference_verified
selected_reference_index
teacher_model
teacher_mode
teacher_parameters
prompt_template_version
seed_ids
assistant_answer
teaching_rubric_version
key_concepts / reasoning_steps / common_mistakes / hint_ladder
repair_count
verification_status
pass_rate
failure_type
prompt_tokens / completion_tokens / total_tokens
base_model_token_count
git_commit
created_at
```

训练 JSONL 可删除 reference 正文，但必须保留 reference hash 和构造版本。

### 5.9 数据验收门槛

| 指标 | Pilot 门槛 | 正式数据门槛 |
|---|---:|---:|
| 最终 accepted 代码全测试通过率 | 100% | 100% |
| 首次生成严格通过率 | ≥75% | 只统计，不放宽 accepted |
| 最多两次修复后的可用率 | ≥90% | ≥90% |
| 语法合规率 | 100% | 100% |
| 接口合规率 | 100% | 100% |
| reference 泄漏 | 0 | 0 |
| 代码块截断 | 0 | 0 |
| tokenizer 截断 | 0 | 0 |
| 人工解释—代码一致率 | ≥90%（抽 50） | ≥95%（抽 100） |
| rubric 题目特异性与正确率 | ≥90%（抽 50） | ≥95%（抽 100） |
| next_hint 首轮完整答案泄漏率 | ≤10%（抽 50） | ≤5%（抽 100） |

如果 Pilot 不达标，不得继续生成全部 10,415 条候选数据。

### 5.10 API 成本预算

正式预算必须按 10,415 次首次生成和实际修复率计算，不再沿用 3,200 条的个位数美元估算。预算脚本应在 pilot 后使用真实的输入、输出 token 均值与 P95 重算：

```text
预计成本
= 首次生成输入成本
+ 首次生成输出成本
+ 修复调用成本
+ 10% 失败重试/网络冗余
```

项目预算设为：

- pilot 后预测成本超过预算时先汇报，不得静默缩减题量；
- 预警线与硬上限必须由当日官方价格和 pilot token 统计生成；
- 超过硬上限必须暂停并检查异常长输出、重复调用和断点续传失效。

价格可能变化，正式运行前只更新预算表，不改变 teacher 模型。

---

## 6. 代码验证与安全执行

### 6.1 正确性口径

- `strict_pass = 1`：该题全部测试通过；
- `test_pass_rate`：通过测试数 / 总测试数；
- 真正的 `Pass@1`：所有题 `strict_pass` 的平均；
- 不得把 `test_pass_rate` 的平均称作 Pass@1。

### 6.2 I/O 模式

必须分别支持并分别统计：

- `standard_input`；
- `call_based`。

`call_based` 必须检查：

- `fn_name`；
- starter contract；
- 类名/方法名；
- 参数与返回值；
- 不得把标准输入程序误判为函数式答案。

### 6.3 执行隔离

正式验证和 GRPO reward 执行模型代码时，至少使用：

- Docker/等价容器；
- 非 root 用户；
- `--network none`；
- 只读根文件系统；
- 独立临时可写目录；
- CPU 时间限制；
- 内存限制；
- PID/进程数限制；
- 单题 wall-clock timeout；
- 禁止挂载 API key、项目源码和宿主敏感目录。

在未满足以上条件前，文档只能称其为“受限子进程执行器”，不得称“安全沙箱”。

### 6.4 测试拆分

GRPO 题目若有至少 4 个测试：

- 约 70% 作为 online reward tests；
- 约 30% 作为 held-out tests；
- 固定随机种子；
- checkpoint 选择与最终报告使用 held-out tests。

测试过少或存在特殊判题的题：

- 可用于 SFT；
- 默认不用于 GRPO；
- 不得为了凑数量重复测试。

---

## 7. 仓库修复与环境冻结

### 7.1 唯一正式入口

最终只保留或明确指定：

```text
scripts/build_sft_dataset.py
scripts/train_sft.py
scripts/train_grpo.py
scripts/evaluate_model.py
scripts/run_final_eval.py
```

旧入口可保留在 `legacy/`，但 README 不得同时给出多个互相冲突的“主入口”。

### 7.2 环境锁定

正式训练前必须产出：

- `requirements.lock.txt` 或 `uv.lock`；
- Python 版本；
- CUDA、驱动、PyTorch、Transformers、TRL、PEFT、bitsandbytes、vLLM 版本；
- `pip freeze`；
- `nvidia-smi`；
- 容器镜像 tag/digest；
- 环境自检脚本。

禁止使用只有下限的浮动依赖，例如：

```text
trl>=...
transformers>=...
```

### 7.3 本地代理 dry-run

在 RTX 4060 上使用 0.5B～1.5B 代理模型完成：

1. SFT 单 batch；
2. 保存与恢复 adapter；
3. 推理生成；
4. reward 三路输入输出；
5. GRPO 2 prompts × 4 generations × 1 step；
6. checkpoint callback；
7. 最终评测 CLI；
8. 中断后断点续训。

代理模型只验证代码路径，不作为主实验结果。

### 7.4 云端正式模型 dry-run

云端 H0～H6 内必须完成：

- 正式基座 SFT 单 batch；
- 正式基座 GRPO 单 step；
- vLLM server 与 trainer GPU 隔离；
- reward 容器并发执行；
- checkpoint 写盘和读取；
- 日志写盘。

任一失败超过 2 小时未定位，立即触发降级，不得无限排错。

### 7.5 G0 验收

```text
python -m compileall 通过
全部单元测试通过
数据 smoke 通过
代理模型 SFT 单 batch 通过
代理模型 GRPO 单 step 通过
正式评测 CLI 能读取冻结 manifest
无 API key 进入日志或仓库
```

---

## 8. SFT 训练设计

### 8.1 输入格式

优先使用 TRL 支持的 conversational prompt-completion 格式，并只对 completion 计算 loss。TRL 官方 SFTTrainer 支持 conversational 与 prompt-completion 数据：[TRL SFT 文档](https://huggingface.co/docs/trl/en/sft_trainer)。

要求：

- prompt token 不参与 loss；
- assistant completion 完整参与 loss；
- chat template 由 tokenizer 官方模板生成；
- 不手写重复 BOS/EOS；
- 不把 teacher 私有 reference 拼入训练文本。
- `full_explanation` 与 `next_hint` 使用明确的任务指令；后者必须包含 student state；
- 按 task_type 分别统计 loss、长度和泄漏率，避免 10K 完整题解把约 1K 提示对话的行为完全淹没；
- 可以使用 task-balanced sampler，但不能简单复制同一 Tutor-SFT 样本制造虚假数据量。

### 8.2 初始训练配置

QLoRA 是本项目 SFT 的正式训练方法，不是被删除的附属优化。其含义是：

```text
冻结的 Qwen2.5-Coder-7B-Instruct 基座
→ 以 4-bit NF4 量化加载，计算时反量化为 BF16
→ 只训练注入各线性层的 LoRA adapter
```

主配置固定为：

```yaml
method: QLoRA
quantization: NF4
bnb_4bit_compute_dtype: bfloat16
double_quant: true
max_seq_length: TO_BE_FROZEN_AFTER_TOKEN_AUDIT
completion_only_loss: true
lora_r: 16
lora_alpha: 32
lora_dropout: 0.05
learning_rate: 1.0e-4
epochs: 1.0
warmup_ratio: 0.05
lr_scheduler: cosine
weight_decay: 0.01
gradient_checkpointing: true
length_grouped_sampling: true
seed: 42
```

LoRA target 初始覆盖：

```text
q_proj, k_proj, v_proj, o_proj,
gate_proj, up_proj, down_proj
```

如果最终选择的模型层命名不同，应通过模型结构自动解析并记录，不能静默漏掉模块。

LoRA rank 的约束：

- 正式主实验使用 `r=16, alpha=32`，与当前实际 SFT 入口一致；
- 仓库全局配置中遗留的 `r=64, alpha=128` 不得继续与 SFT 配置并存，必须删除或标为 legacy；
- 若 500 条 calibration 明确显示欠拟合，可在本地/云端早期仅比较一次 `r=16` 与 `r=32`；若无明确收益，保持 r=16；
- 一旦 SFT adapter rank 冻结，GRPO 必须使用同一 adapter 结构，不能在热启动时声称改成另一 rank。

### 8.3 长度策略

`4096` 只是候选值，正式统计前不得写成默认结论。长度审计分三次执行：

#### A. 生成前：10,415 道原题审计

用 `Qwen2.5-Coder-7B-Instruct` tokenizer 对全部 student prompt 统计：

- 题面 token；
- 接口信息 token；
- chat template 后的 prompt token；
- P50/P75/P90/P95/P99/max；
- 分桶数量与占比：
  - `≤2048`；
  - `2049–4096`；
  - `4097–6144`；
  - `6145–8192`；
  - `>8192`。

teacher 私有 prompt 另行统计，因为它还包含 reference 与 seed，但不得把 teacher prompt 长度误当成 SFT 输入长度。

#### B. Pilot 后：完整 SFT 样本审计

对 300～500 条 accepted pilot 统计：

- `prompt_tokens`；
- `completion_tokens`；
- `total_tokens`；
- 代码块起止 token 位置；
- 代码块是否闭合；
- 在 2048、4096、6144、8192 四个上限下会被截断的样本数；
- 截断是否落在代码、复杂度或常见错误部分。

#### C. 全量后：最终冻结

对全部 accepted 样本重复统计并输出 `reports/token_length_report.json` 与 `.md`。按照以下规则选择：

1. 全部样本 `≤4096`：冻结 4096；
2. 否则全部样本 `≤6144`：冻结 6144；
3. 否则全部样本 `≤8192`：冻结 8192；
4. `>8192` 的样本先让 teacher 压缩冗余讲解但保留完整题意、算法与代码，再重新审计；
5. 仍 `>8192` 的极端长尾进入 `rejected/overlength.jsonl`，不得截断题面或代码后强行训练。

训练必须使用动态 padding 与按长度分桶/采样，避免因为少量长样本让所有 batch 都按 8192 计算。最终要求：

- 纳入训练的样本截断率为 0；
- completion 代码块完整率为 100%；
- 任何降低上限的决定都必须给出各区间数量，不能凭显存直觉拍板。

### 8.4 训练轮次

主实验只允许：

1. 500 条数据的 SFT calibration；
2. 9,500 条数据的正式 SFT，先运行 1 epoch；
3. 只有 dev 指标仍持续改善且云端时间允许时，才续训到最多 1.5 epoch；
4. 最多一次有明确原因的修正重跑。

不得在五天窗口内进行无计划超参数搜索。

### 8.5 checkpoint 选择

每个 checkpoint 在冻结 SFT dev 上评测：

- 严格 Pass@1；
- 平均测试通过率；
- 格式/接口合规；
- 教学结构；
- 生成截断率。

主排序：

1. 严格 Pass@1；
2. 若并列，教学 pairwise 得分；
3. 若仍并列，选择更早、更短输出的 checkpoint。

不得使用训练 loss 最低或最后一个 epoch 自动作为最佳模型。

### 8.6 SFT 阶段验收

SFT 必须至少满足：

- 相对 Base，教学结构合规率显著提升；
- 严格 Pass@1 不下降超过 2 个百分点；
- 接口合规率不下降；
- EvalPlus 通用代码能力不出现明显灾难性退化；
- completion 截断率为 0；
- 训练过程无 NaN/Inf；
- best checkpoint 可独立加载推理。

若代码正确率明显下降，停止 GRPO，先检查标签质量、completion loss mask、长度和学习率。

---

## 9. GRPO 训练设计

### 9.1 GRPO 的真实目标

GRPO 的核心目标定义为：

> 在保留 SFT 教学风格的前提下，通过可执行测试提高代码正确性与接口稳定性。

除非 Teaching Reward 通过专门验证并进入梯度，否则不得声称“GRPO 提高了教学能力”。可以声称：

> SFT 学习教学表达，GRPO 优化可验证正确性，并监控教学质量不退化。

### 9.2 Reward 组成

#### 必选：Execution Reward

建议归一化到 `[0, 1]`：

```text
R_exec = 0.8 × test_pass_rate + 0.2 × I(all_tests_pass)
```

硬性修正：

- 语法错误：0；
- 接口不匹配：0；
- 超时：0；
- 危险/不允许操作：0；
- 无代码：0。

#### 必选：Contract/Format Reward

只承担低权重约束：

- 包含可提取代码块；
- 代码与接口模式一致；
- 有必要的算法、复杂度说明；
- 没有模板重复、标题堆砌、异常超长；
- 没有 reference 泄漏。

不得仅按标题数量给高分。

#### 核心超越项：Teaching Reward

外部基线的 `LocalTeachingReward` 主要依赖结构词、连接词、注释和表面多样性，而且只记录日志，不进入梯度。我们的 Teaching Reward 必须真正读取：

```text
problem + task_type + student_state（如有）
+ private teaching rubric
+ completion
```

它不能只读 completion。否则无法判断答非所问、易错点是否属于当前题目、提示是否与学生当前进度匹配。

优先实现一个轻量 `TeachingCritic`：

1. 从 accepted full-explanation、Base/SFT 生成和人工构造的缺陷回答中建立 2,000～4,000 个 preference pairs；
2. 缺陷类型必须包括：删去关键步骤、交换其他题讲解、错误但语言漂亮、错误复杂度、解释—代码不一致、模板重复、注释堆砌、过早给完整答案、提示过弱和提示越级；
3. DeepSeek-V4-Flash 可辅助标注，但至少 200 对由人工复核，API 失败不能记为负例；
4. 使用 0.5B 级 critic 先在本地 QLoRA 训练；若达不到准入门槛，再评估 1.5B，不得直接在云端临时试模型；
5. critic 输出连续分数，并分别保留 `explain_score` 与 `hint_score`，不能把两类任务混成一个不可解释总分。

`full_explanation` 的 critic 重点判断：

- 关键概念与推导步骤是否覆盖；
- 步骤依赖是否合理，是否存在跳步；
- 正确性说明是否解释了为什么；
- 复杂度是否正确且有依据；
- 易错点是否题目特异、准确且给出纠正；
- 解释是否与代码实际算法一致。

`next_hint` 的 critic 重点判断：

- 是否准确诊断学生当前错误；
- 是否选择了 rubric 中合适的下一层提示；
- 提示是否能推动一步，而不是复述题面；
- 是否过早泄漏完整算法或代码；
- 是否针对学生状态，而不是输出通用模板。

对抗验证集至少 200 对，并覆盖：

- 正确且讲解好；
- 正确但只有代码；
- 错误但讲得漂亮；
- 答非所问；
- 重复步骤；
- 注释堆砌；
- 解释与代码算法不一致；
- 易错点是通用套话或事实错误；
- 学生只问一个提示却直接得到完整答案；
- 同一回答放到不同学生状态下的适切性变化。

准入门槛：

- 人工金标上的 pairwise ranking accuracy ≥80%；
- 比外部基线 `LocalTeachingReward` 至少高 10 个百分点；
- 对“错误但漂亮”样本的误判率 ≤10%；
- 对模板堆砌的高分率 ≤10%；
- `next_hint` 过早答案泄漏的高分率 ≤10%；
- 同一题内排序与跨题排序分别报告，不能只用全局 Spearman；
- critic 推理吞吐满足 GRPO 在线训练预算。

正式 Teaching Reward 使用执行正确性软门控：

```text
R_teach_effective = R_teach_raw × R_exec
```

这样错误但漂亮的回答不能仅凭教学文风获得高总奖励，同时部分正确答案仍保留一定教学梯度。

未达准入门槛时，可信的降级主实验为：

```text
0.60 × (0.05 × R_static + 0.70 × R_pass + 0.25 × R_strict)
+ 0.40 × (R_contract × (0.25 + 0.75 × R_pass))
```

但该结果只能称为“执行正确性 GRPO + 教学能力保持”，不得声称在核心主题上超越外部基线。要通过第 19 节超越基线 Gate，必须使用：

```text
0.80 × R_exec + 0.10 × R_contract + 0.10 × R_teach_effective
```

禁止为了凑“三级奖励”把未经验证的 Teaching Reward 强行接入。

### 9.3 奖励缩放

1. 删除旧的自定义“先合并再 batch Z-score”逻辑。
2. 使用一个 canonical composite reward callable；每条 completion 只调用一次统一 verifier，再同时计算并记录 static、pass-rate、code、contract 与 total。
3. 第一版主实验设置 `scale_rewards=false`，避免额外的题目难度标准差偏置。
4. `scale_rewards="batch"` 仅可作为后续消融，不得在主实验中临时切换。

TRL 官方文档明确提供 `scale_rewards=False` 和 batch scaling，并说明组内标准差缩放可能引入题目难度偏置：[TRL GRPO 文档](https://huggingface.co/docs/trl/en/grpo_trainer)。

### 9.4 初始 GRPO 配置

GRPO 不重新全参数训练，也不丢弃 QLoRA。正确的热启动链路为：

```text
4-bit NF4 冻结基座
+ best SFT LoRA adapter（r/alpha/target_modules 保持不变）
→ 复制为独立 GRPO run
→ 继续只更新 adapter 参数
→ 分别保存 SFT adapter 与 GRPO adapter，禁止覆盖
```

训练前不得把 SFT adapter 合并进 BF16 基座再重新量化，否则会引入额外量化误差，也破坏 Base/SFT/GRPO 的清晰版本关系。

```yaml
start_from: best_sft_adapter
method: QLoRA
load_in_4bit: true
quantization: NF4
lora_r: SAME_AS_SFT
num_generations: 4
max_prompt_length: TO_BE_FROZEN_FROM_PROMPT_AUDIT
max_completion_length: TO_BE_FROZEN_FROM_COMPLETION_AUDIT
temperature: 0.8
top_p: 0.95
learning_rate: 1.0e-5
beta: 0.05
loss_type: grpo
scale_rewards: false
num_iterations: 1
gradient_checkpointing: true
mask_truncated_completions: false
seed: 20260728
```

说明：

- 正式云端协议固定 `beta=0.05`，保留 KL 约束。
- 正式云端协议固定 TRL 0.22.2、`loss_type=grpo`、`scale_rewards=false`；不得因版本迁移改成 DR-GRPO、DAPO 或 GSPO。
- `max_completion_length` 必须以 SFT 输出长度审计为依据，不能让代码普遍被截断。

### 9.5 正式静态难度课程与后续可学习度消融

2026-08-15 正式实验固定使用静态 `easy → medium → hard` 三阶段课程，不得在主运行中替换为随机单 epoch 或动态采样：

1. easy：3,228 条，1 epoch，`max_completion_length=512`；
2. medium：1,735 条，1 epoch，`max_completion_length=768`；
3. hard：1,488 条，1 epoch，`max_completion_length=1024`。

下面的可学习度采样只作为完成正式静态课程后的候选消融，不覆盖主实验协议。题库 difficulty 不完全等于当前模型的可学习度，因此消融可考察同一 prompt 的多次生成是否存在有意义的 reward 差异。

在正式 GRPO 前，对候选 prompt 使用 SFT checkpoint 各采样 4 次，记录：

```text
p_strict_pass
mean_test_pass_rate
group_reward_std
zero_advantage
mean_completion_length
```

候选消融可按可学习度分三阶段：

1. 阶段 A：优先 `0.25 ≤ p_strict_pass ≤ 0.75` 且 reward 有方差的题，建立有效梯度；
2. 阶段 B：混入更难题与少量易题，保持 difficulty/io_mode 覆盖；
3. 阶段 C：回到接近原始分层分布，避免只对中等可解题过拟合。

动态规则：

- 连续评测发现某题组 `zero_advantage_ratio` 过高时降低其采样权重；
- 不因单步 reward 低就永久丢弃 hard 题；
- 所有采样权重和阶段切换写入日志；
- 不能把 reward 相同直接解释为生成文本坍缩；
- 必须与静态 difficulty curriculum 比较“有效优势样本比例、吞吐与 held-out 结果”，验证后才可声称更优。

### 9.6 训练步骤

1. 20-step GRPO pilot；
2. 检查 reward 分布、全零优势、截断、显存、吞吐、容器执行队列；
3. pilot 通过后运行主训练；
4. 主训练目标按 800 prompts、约 1 epoch 规划；
5. 每 50 steps 冻结评测；
6. 最多一次因确定 bug 重启；不做无边界 trial-and-error。

### 9.7 必须记录的监控

- 各 reward 分量 mean/std；
- total reward；
- `frac_reward_zero_std`；
- 严格全通过比例；
- 语法/接口/超时比例；
- completion mean/P95/max length；
- clipped completion ratio；
- 代码 exact-match 多样性；
- AST hash 多样性；
- Distinct-n；
- step time；
- GPU 利用率和显存；
- reward worker 队列长度；
- checkpoint held-out Pass@1；
- 教学质量监控集得分。

“组内奖励方差为零”只能叫 `zero_advantage_ratio`，不得直接叫 generation collapse。

### 9.8 GRPO 停止条件

出现任一情况立即停止并回滚到最近健康 checkpoint：

- 连续 3 次评测严格 Pass@1 下降；
- completion 截断率 >2%；
- 语法/接口错误率比 SFT 高 5 个百分点以上；
- 教学质量下降超过预设容忍线；
- reward hacking 明显出现；
- 超时/容器故障占比 >10%；
- NaN/Inf；
- 同一模板或相同 AST 大量坍缩；
- H60 仍未完成稳定 GRPO 单步。

---

## 10. 评测设计

### 10.1 模型对比

必须至少比较：

1. Base；
2. SFT；
3. SFT + GRPO。

如果 GRPO 失败，应明确只报告前两者。

### 10.2 评测集

#### A. TACO 冻结测试

- 300 题；
- 优先来自 TACO 独立 test split，并重新完成 reference/tests 可执行性验证；
- 与 TACO train 中的 SFT/GRPO 题完全不重叠，并做题面近重复检查；
- 分层覆盖 difficulty 和 io_mode；
- 使用 hidden/held-out tests；
- 主指标：严格 Pass@1。

#### B. 通用代码能力

- HumanEval+；
- MBPP+；
- 固定 EvalPlus 版本和容器。

EvalPlus 为 HumanEval 和 MBPP 提供远多于原始基准的附加测试，适合检查代码是否仅通过少量样例以及训练后是否退化：[EvalPlus 官方仓库](https://github.com/evalplus/evalplus)。

#### C. ExplainBench：一次性题解讲解能力

- 100 道冻结题；
- 每题具备人工抽查后的 private teaching rubric；
- Base、SFT、GRPO 输出随机匿名排序；
- 自动 judge 只能作为辅助；
- 至少 50 道由人工盲评；
- 技术硬门控：
  - 代码严格通过；
  - 核心算法解释正确；
  - 解释与代码不存在关键矛盾。

硬门控失败时总体教学结果记为不合格，不能因语言更漂亮而获胜。门控通过后再报告：

| 指标 | 定义 |
|---|---|
| 关键概念覆盖率 | rubric 中必要 `key_concepts` 被正确解释的比例 |
| 推导步骤完整率 | 必要 `reasoning_steps` 被覆盖且依赖顺序合理的比例 |
| 正确性论证覆盖率 | `proof_invariants` 被准确说明的比例 |
| 解释—代码一致率 | 讲解的算法、数据结构、边界与实际代码一致 |
| 易错点 Precision | 模型指出的易错点中，真实且适用于本题的比例 |
| 易错点 Recall | rubric 中重要易错点被模型覆盖的比例 |
| 纠错可操作率 | 指出错误后是否说明原因和具体修正 |
| 初学者清晰度 | 术语是否解释、步骤粒度是否合适 |
| 冗余度 | 是否存在不推动理解的重复与模板话术 |

不能只报告一个混合总分；主表至少单列“技术合格率、解释—代码一致率、易错点 F1、人工教学胜率”。

#### D. TutorBench：交互式教学能力

TutorBench 单独评估“会教”，不能从完整题解分数推断。设计为：

- 150～200 道与 Tutor-SFT 不重叠的冻结题；
- 每题构造至少 2 个学生状态；
- 学生消息包含当前思路、代码片段或明确困惑；
- 模型只需给出下一轮教学反馈，不默认交付完整题解；
- 至少 100 个场景由人工复核 student state、目标诊断与允许的 hint level。

学生状态至少覆盖：

- 题意/接口误解；
- 暴力解正确但超时；
- 关键观察缺失；
- 状态或递推式错误；
- 边界/下标错误；
- 复杂度误判；
- 实现与思路不一致。

TutorBench 指标：

| 指标 | 定义 |
|---|---|
| Diagnosis Accuracy | 是否准确识别学生当前真正的阻塞或错误 |
| Next-Action Accuracy | 是否选择了 rubric 中合适的下一教学动作 |
| Hint-Level Accuracy | 提示力度是否匹配 H1～H4 目标层级 |
| Hint Usefulness | 提示是否带来可验证的下一步进展 |
| Premature-Reveal Rate | 首轮不必要地泄漏完整算法或代码的比例，越低越好 |
| Hallucinated-Mistake Rate | 错误指责学生不存在的问题的比例，越低越好 |
| State Grounding | 回答是否引用并处理学生的具体思路/代码，而非通用模板 |
| Adaptation Consistency | 多轮后是否根据学生进展调整，而非重复同一提示 |

`Hint Usefulness` 使用三层证据：

1. rubric 规则判断提示是否覆盖预期的下一步；
2. 弱学生模型在获得提示前后的解题进展变化，只作为辅助指标；
3. 人工评审判断提示是否“足以推动一步但未越级泄漏”，作为最终可信证据。

不得只用同一个大模型同时生成 student、生成 gold、担任 judge 并得出最终结论。

### 10.3 自动 Judge 约束

如果使用 DeepSeek-V4-Flash 作为 judge：

- 与 teacher 同源偏差必须在报告中披露；
- A/B 顺序随机；
- 做 position-swap 一致性检查；
- judge 不得只看到被截断的前 3,000 字符；
- API 失败记为缺失，不记 0；
- 自动 judge 与人工结果分别报告；
- ExplainBench 与 TutorBench 使用不同 rubric 和 judge prompt；
- 不使用“严格双盲”措辞，应称“匿名随机顺序成对评测”。

### 10.4 统计

- Pass@1：Wilson 95% CI；
- Base/SFT/GRPO 同题二元结果：McNemar 或配对 bootstrap；
- 教学 pairwise：胜/负/平 + bootstrap CI；
- 同时报告效应量，不只报告 p 值；
- 所有结果保留逐题 JSONL，确保可重算。

### 10.5 主结果表

| 模型 | TACO 严格 Pass@1 | HumanEval+ | MBPP+ | Explain 技术合格率 | 易错点 F1 | Tutor 诊断准确率 | 提示泄漏率 | 人工教学胜率 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Base |  |  |  |  |  |  |  |  |
| SFT |  |  |  |  |  |  |  |  |
| SFT + GRPO |  |  |  |  |  |  |  |  |

---

## 11. 双 4090 五天执行表

### 11.1 云端开启前必须完成

以下任一未完成，不得启动五天计时：

- 仓库 G0 验收通过；
- 300 条 pilot 通过；
- 10,415 条 verified 候选已全部尝试生成，目标 10,000 条 accepted 数据已完成验证；
- 约 1,000 条 Tutor-SFT 辅助对话与 private teaching rubric 已完成；
- TeachingCritic 已在人工金标和对抗集上通过准入门槛；
- ExplainBench、TutorBench 与外部基线 reward 对照脚本已冻结；
- SFT/GRPO/最终测试 split 冻结；
- tokenizer 长度审计完成；
- Qwen2.5-Coder-7B-Instruct 的 revision、tokenizer、chat template 与 QLoRA 配置已冻结；
- 模型权重、数据、wheel/容器、脚本准备完成；
- 代理模型端到端 dry-run 完成；
- final eval 能在本地小样本跑通；
- DeepSeek API 数据生成不再依赖云端 GPU；
- 云端启动命令写入 runbook；
- 训练中断恢复演练完成。

### 11.2 120 小时安排

| 时间 | 任务 | 必须产物 | 失败处理 |
|---|---|---|---|
| H0–H2 | 硬件、磁盘、驱动、网络、容器、数据 hash 自检 | `environment_report.json` | 环境不符立即修复或停止计时 |
| H2–H6 | 正式基座 SFT 单 batch、GRPO 单 step、vLLM server、reward 容器 | dry-run 日志与 checkpoint | H6 未过则回退稳妥模型/单卡 |
| H6–H12 | Base 冻结集推理；200 条 SFT calibration；吞吐估计 | Base 结果、calibration 曲线 | 不满足长度/显存则调 batch，不改数据 |
| H12–H44 | 约 9,500 条 full-explanation + 约 1,000 条 Tutor-SFT 的正式 SFT 主训练（先 1 epoch） | 多个 checkpoint、完整日志 | 只允许一次有原因的恢复 |
| H44–H52 | SFT checkpoint 评测与选择；必要时续训但总计不超过 1.5 epoch | best SFT adapter | SFT 退化则暂停 GRPO |
| H52–H60 | 外部基线式/本项目式 reward 各自完成 20-step GRPO pilot | 两分支 reward 分布、吞吐、错误统计 | 本项目分支未过门槛则停止核心超越实验 |
| H60–H70 | 从同一 SFT adapter 分叉的等预算小规模 GRPO 对照 | 固定 prompt/步数/seed 的 paired run | 不得只保留结果较好的分支 |
| H70–H94 | 本项目正式 GRPO 主训练 | checkpoint、监控日志 | 按停止条件回滚 |
| H94–H101 | GRPO checkpoint 评测与选择 | best GRPO adapter | 无提升则保留 SFT 为主结果 |
| H101–H108 | Base/SFT/GRPO 最终生成、EvalPlus、ExplainBench/TutorBench 推理 | 原始逐题输出 | 优先完成 TACO 与两个教学主指标 |
| H108–H116 | 统计、复算、模型合并/adapter 导出、Demo smoke | 结果表、模型产物 | 禁止开启新训练 |
| H116–H120 | 双重备份、hash 校验、环境与复现说明 | 完整 manifest | 未备份不得关机 |

### 11.3 五天内禁止事项

- 临时生成全量 SFT 数据；
- 临时重做数据划分；
- 下载和比较第三个基座；
- 大规模超参数搜索；
- 为了“结果好看”修改测试集；
- 在 H108 后开启新训练；
- 只保存最后 checkpoint；
- 先关服务器再确认产物 hash。

### 11.4 时间降级路线

#### A. 多卡不稳定

- SFT：单卡 QLoRA；
- 第二张卡用于并行验证和生成；
- 不临时研究 FSDP/ZeRO。

#### B. vLLM server 不稳定

- 回退 TRL colocate 或普通 generation；
- 降低 group/batch；
- 保留 4 generations；
- 优先减少训练 prompt 数，不把生成长度砍到代码普遍截断。

#### C. GRPO 到 H60 仍不稳定

- 终止 GRPO；
- 完成 Base vs SFT 的完整评测；
- 输出 reward offline 分析和失败复盘；
- 项目暂定位为“高可信 SFT + 可验证奖励框架”，不声称 RL 结果。

#### D. 最终评测时间不足

优先级：

1. TACO 300；
2. ExplainBench/TutorBench 模型输出生成；
3. HumanEval+；
4. MBPP+；
5. 人工评测可在云端结束后继续。

---

## 12. 本地阶段时间表

建议在云端开启前安排约 16 个本地工作日。数据生成和 CPU 验证可以并发并允许跨日运行，但不能挤占五天云端：

| 本地日 | 主要任务 | 验收 |
|---|---|---|
| D1 | 代码声明—入口—测试对账，修语法与旧入口 | compileall 通过 |
| D2 | 锁依赖、容器化 verifier、统一 reward API | 单元测试通过 |
| D3 | 代理模型 SFT/GRPO dry-run、断点恢复 | G0 通过 |
| D4 | 300～500 条 DeepSeek-V4-Flash pilot + pilot token 审计 | 数据门槛通过 |
| D5–D9 | 对 10,415 条候选进行首次生成、CPU 验证、两轮定向修复 | 目标 10,000 accepted |
| D10 | 全量 tokenizer 审计、超长样本压缩/隔离；生成 teaching rubric 与 Tutor-SFT | 长度上限冻结、约 1,000 条提示对话 |
| D11 | 构造 TeachingCritic preference pairs 与对抗集；人工复核 200 对 | critic 数据冻结 |
| D12 | 本地训练/验证 TeachingCritic；与外部基线 LocalTeachingReward 对照 | critic 准入报告 |
| D13 | split、ExplainBench/TutorBench、近重复与泄漏检查 | manifest/hash |
| D14 | Qwen2.5-Coder-7B-Instruct 本地 4-bit smoke、配置冻结 | model frozen |
| D15 | 完整云端 runbook、离线包、最终评测 smoke | cloud-ready |
| D16 | 只做缓冲与复核，不新增功能 | Go/No-Go 决策 |

若 API 生成或 CPU 验证耗时较长，可延长本地阶段，不得挤占五天训练窗口。

---

## 13. 消融与误差分析

### 13.1 必做且不额外训练的消融

在同一批至少 100 题上比较 teacher 标签：

- scratch；
- reference-guided；
- reference-guided + seed。

指标：

- 首次严格通过率；
- 修复后可用率；
- 代码—解释一致性；
- token 成本；
- 失败类型。

这回答的是“数据构造策略是否更好”，不冒充“模型训练消融”。

### 13.2 必做的外部基线式训练对照

为了证明本项目独立方案不是“文档上更严谨”，必须从同一个 best SFT adapter 分叉：

```text
B_external_style:
  0.60 × execution + 0.40 × legacy format
  teaching 仅监控
  按外部基线公开设计复现其 reward 行为

B_ours:
  0.80 × execution
  + 0.10 × anti-hacking contract
  + 0.10 × gated TeachingCritic
  + learnability-aware sampling
```

控制变量：

- 相同 SFT 起点；
- 相同 prompts 与 held-out tests；
- 相同 `num_generations`；
- 相同 optimizer steps；
- 相同 seed；
- 相同 completion 上限；
- 相同算力预算；
- 两分支全部保留，不得只展示较好结果。

最低规模为 100～200 prompts 的等预算小训练，并在同一 GRPO dev、ExplainBench 子集和 TutorBench 子集上评测。它是训练消融，不是对固定文本重新算两个 reward。

如果时间允许，可再加一组：

```text
B_exec_only:
  1.00 × execution
```

该组仍不得挤占最终评测和备份。

### 13.3 必做误差分析

至少分析：

- `standard_input` reference 验证率低的 100 个失败样本；
- SFT 生成失败的 50 个样本；
- GRPO reward=0 的 50 个样本；
- Base 正确但 SFT/GRPO 错误的所有退化题；
- 自动 judge 与人工意见相反的样本。

错误类别至少包括：

- 真算法错误；
- I/O 解析；
- 接口不匹配；
- 输出格式比较；
- timeout；
- verifier bug；
- 特殊判题；
- 题目/测试噪声；
- 解释与代码不一致；
- 截断；
- reward hacking。

---

## 14. 产物、目录与复现

建议新增：

```text
configs/
  model_frozen.yaml
  sft_frozen.yaml
  grpo_frozen.yaml
data/
  manifests/
  splits/
  rejected/
artifacts/
  runs/<run_id>/
    config.yaml
    environment.json
    metrics.jsonl
    checkpoints/
    eval_outputs/
    manifest.sha256
reports/
  data_pilot_report.md
  sft_report.md
  grpo_report.md
  final_report.md
docs/
  CLOUD_RUNBOOK.md
  DECISION_LOG.md
  CLAIMS_MATRIX.md
```

每次正式 run 必须记录：

- `run_id`；
- git commit；
- dirty worktree 状态；
- 配置完整副本；
- 数据 manifest/hash；
- 模型与 tokenizer revision；
- 环境版本；
- random seed；
- 开始/结束时间；
- GPU 信息；
- checkpoint；
- stdout/stderr；
- 逐题评测输出。

---

## 15. 后续 Codex 的行为约束

### 15.1 每次改动前

Codex 必须先回答：

1. 当前处于哪个阶段？
2. 该改动解决哪个验收阻塞？
3. 是否改变数据、模型、奖励或评测口径？
4. 是否会消耗云端时间？
5. 最小可验证改动是什么？

### 15.2 每次改动后

必须：

1. 运行相关静态检查和单元测试；
2. 给出实际命令与结果；
3. 更新 `DECISION_LOG.md`；
4. 更新 `CLAIMS_MATRIX.md` 中的状态；
5. 不把未跑结果写入 README；
6. 保留失败日志，不能只展示成功日志。

### 15.3 禁止的叙事

未经证据不得写：

- “显著提升”；
- “稳定达到”；
- “鲁棒”；
- “安全沙箱”；
- “双盲”；
- “最优 checkpoint”；
- “Teaching Reward 有效”；
- “解决 generation collapse”；
- “实现完整 GRPO 闭环”。

应替换为可核验表述，例如：

- “在 300 题冻结集上由 X 提升至 Y，95% CI 为……”；
- “代码已实现，尚未进行训练型验证”；
- “在 call-based smoke 5/5 通过，样本量有限”。

### 15.4 变更控制

以下变更必须先写决策记录，不能直接实施：

- 更换基座；
- 更换 teacher；
- 改变 SFT 数据量超过 ±10%；
- 修改 split；
- 调整 reward 权重；
- 改变 Pass@1 定义；
- 更换 judge；
- 将 4096 降为 2048；
- 引入新的训练框架；
- 新增第三个正式实验模型。

---

## 16. 阶段门槛总表

| Gate | 进入条件 | 通过标准 | 不通过时 |
|---|---|---|---|
| G0 仓库可信 | P0 修复开始 | compileall、tests、代理 SFT/GRPO dry-run 全过 | 继续本地修复 |
| G1 数据 pilot | G0 通过 | 300 条 pilot 达到第 5.9 节门槛 | 修 prompt/verifier |
| G2 正式数据 | G1 通过 | 10,415 条全部尝试、目标 10,000 accepted、100% 全测试通过、0 泄漏/截断 | 不开云端 |
| G3 TeachingCritic | G2 通过且人工金标/对抗集冻结 | 达到第 9.2 节全部准入门槛并能在线推理 | 只能做执行正确性 GRPO，不得启动核心超越实验 |
| G4 模型冻结 | G2–G3 通过 | Qwen2.5-Coder-7B-Instruct revision、tokenizer、chat template、QLoRA 与长度上限全部冻结 | 继续本地修复 |
| G5 Cloud-ready | G0–G4 通过 | 数据/权重/环境/runbook/评测全部就绪 | 延迟开机 |
| G6 SFT | 正式 SFT 完成 | 教学提升且代码不明显退化 | 不做 GRPO |
| G7 GRPO pilot | 两 reward 分支各 20 step 完成 | reward、截断、执行、吞吐稳定 | 停止核心超越实验 |
| G8 最终交付 | 三阶段评测完成 | 可复算结果、模型、日志、hash、报告齐全 | 先补复现，不写结论 |
| G9 超越基线 | G8 完成 | 第 19 节全部核心条件通过 | 只能称独立实现，不能称优于外部基线 |

---

## 17. 最终交付标准

项目完成时应至少具备：

1. 一套目标约 10,000 条、全部代码通过可用测试的 reference-guided 教学 SFT 数据；
2. 约 1,000 条带 student state 和分级提示目标的 Tutor-SFT 数据；
3. 可复现的 DeepSeek-V4-Flash 数据构造、教学 rubric 与执行反馈修复管线；
4. Base、SFT、SFT+GRPO 三阶段 checkpoint，或诚实标明 GRPO 未完成；
5. 真正进入梯度的执行、契约与 TeachingCritic reward；
6. TACO、HumanEval+、MBPP+、ExplainBench 与 TutorBench 结果；
7. 冻结 split、环境 lock、完整 run manifest；
8. 至少一项数据策略消融；
9. 外部基线式 reward 与本项目 reward 的等预算训练对照；
10. 至少一项错误分析；
11. CLI/简易 Demo；
12. README 与 `PROVENANCE.md` 明确区分自主实现和公共依赖，不得暗示代码来自外部同类项目。

### 17.1 理想简历表述模板

完成全部实验后再填入数字：

> 独立设计并实现面向 OI/ACM 初学者的算法教学后训练系统：解析并验证 TACO 24,237 道题的多候选参考解，获得 10,415 道高可信候选；使用 DeepSeek-V4-Flash 构造并经执行反馈修复约 10K 条全测试通过的题解数据及分级提示对话，在 2×RTX 4090 上完成 Qwen2.5-Coder-7B-Instruct 的 NF4 QLoRA SFT 与可验证奖励 GRPO。构建 ExplainBench/TutorBench，从代码正确性、易错点识别、学生错误诊断与提示泄漏等维度评测，使严格 Pass@1 从 X 提升至 Y、Tutor 诊断准确率达到 Z、首轮答案泄漏率降至 W。

任何 X/Y/Z/W 只能来自最终冻结评测。

---

## 18. 当前立即执行的顺序

从本规划书生效起，下一步固定为：

1. 修复仓库 P0，建立 `CLAIMS_MATRIX.md`；
2. 锁定 TRL/Transformers/PEFT/vLLM/Unsloth 版本；
3. 在本地完成代理模型 SFT/GRPO dry-run；
4. 将 verifier 升级为受限容器执行；
5. 修复“失败标签仍写入 SFT”；
6. 用 DeepSeek-V4-Flash 生成 300 条 pilot；
7. 达标后对 10,415 条 verified 候选全部生成并验证，目标接受约 10,000 条；
8. 生成 teaching rubric 和约 1,000 条 Tutor-SFT，冻结 ExplainBench/TutorBench；
9. 构造 TeachingCritic 偏好数据与对抗集，在本地完成训练和准入验证；
10. 冻结所有 split；
11. 冻结 Qwen2.5-Coder-7B-Instruct revision、tokenizer、chat template、QLoRA 与长度配置；
12. 准备外部基线式/本项目式 GRPO 对照配置、云端离线包和 runbook；
13. 最后才启动 120 小时双 4090 窗口。

截至 2026-07-28 的执行状态：

- 第 1、4、5 项已完成本机验收；
- 第 4 项的机器可读证据为
  `artifacts/g0/docker_verifier_report.json`，受限 Docker 合同全部检查通过；
- 当前下一步仍是第 2 项依赖锁定，随后执行第 3 项代理模型 dry-run；
- G0 尚未通过，不得提前进入第 6 项或任何正式训练。

在第 1～12 步未全部完成前，后续 Codex 不得建议“先上云跑起来看看”。

---

## 19. 超越外部同类基线 Gate

本节只用于内部验收和技术报告，不要求在简历开场提及任何博主或同名项目。项目是否“更强”分四层判断。

### 19.1 数据层胜出

必须同时满足：

- 10,415 条 verified 候选全部尝试生成；
- accepted 数据代码严格通过率 100%；
- reference 泄漏与 tokenizer 截断均为 0；
- reference-guided 相比 scratch 的严格通过率有明确正向效应并报告区间；
- 人工抽检解释—代码一致率 ≥95%；
- 教学 rubric 正确率与题目特异性 ≥95%。

达到后可说：

> 本项目的数据可信度与可追溯性强于只做格式/长度过滤的公开同类方案。

### 19.2 Reward 层胜出

必须同时满足：

- 新 contract reward 已在正式入口进入梯度；
- TeachingCritic 读取 problem/rubric/student state，而非只读回答表面；
- 人工金标 pairwise accuracy ≥80%；
- 相比外部基线 LocalTeachingReward 至少提高 10 个百分点；
- 错误但漂亮、模板堆砌、答非所问和过早泄漏四类攻击的误判率均 ≤10%；
- reward unit test、trainer integration test 和在线吞吐门槛全部通过。

达到后可说：

> 本项目实现了更题目相关、更难刷分且真正进入 GRPO 梯度的教学奖励。

### 19.3 训练层胜出

从同一 SFT adapter 分叉的等预算小训练中，本项目分支必须：

- GRPO dev 严格 Pass@1 相对外部基线式分支不下降超过 1 个百分点；
- ExplainBench 技术合格回答中的人工教学胜率 >50%，目标 ≥55%；
- TutorBench 的 Diagnosis Accuracy 或 Hint-Level Accuracy 至少一项提高 ≥5 个百分点；
- Premature-Reveal Rate 不高于外部基线式分支；
- 结果不依赖删除困难题或改变 completion 上限。

若样本不足以显著，只报告效应量和区间，不使用“显著优于”。

### 19.4 最终模型层胜出

最终 SFT+GRPO 相比 Base/SFT 必须满足：

- 相对 SFT，TACO 严格 Pass@1 有正提升，目标至少 +2 个百分点；
- 相对 SFT，ExplainBench 技术合格率不下降超过 1 个百分点；
- 相对 SFT，TutorBench 人工偏好不下降；若声称 GRPO 提升教学，胜率必须 >50%；
- 相对 Base，SFT/GRPO 的易错点 F1、诊断准确率和提示层级准确率均有正提升；
- HumanEval+/MBPP+ 任一基准不下降超过 2 个百分点；
- 最终 adapter、原始输出、统计脚本和 Demo 均可复现。

### 19.5 结论等级

| 等级 | 允许表述 |
|---|---|
| 仅数据层通过 | 独立完成高可信算法教学数据与执行验证管线 |
| 数据 + Reward 通过 | 独立实现题目相关、抗刷分的教学奖励框架 |
| 数据 + Reward + 训练通过 | 在等预算受控实验中优于公开同类 reward 方案 |
| 四层全部通过 | 完成并验证了更可信的算法教学 SFT + GRPO 闭环 |

任何层级都不允许把外部同类项目写成本项目的代码来源。没有通过的层级不进入简历结论。

## 20. 2026-07-29 SFT 生成执行调整

本调整不废弃已经 accepted 的数据，正式标签生成采用 A/B 混合策略：

1. A 类保留现有 reference-guided 教学化改写，每题只进行一次 API 请求，
   且必须通过全部 Docker 测试才可 accepted。
2. A 类任何失败以及历史 rejected 都进入 B 类。
3. B 类把 verified TACO reference 视为不可变代码真值，teacher 只生成
   教学层，并且只允许增加 `#` 注释。
4. 去掉注释后的 Python token 必须与 reference 一致；否则由程序注入
   原始 reference，再执行语法检查和 Docker 全测试。
5. 历史 rejected 的 B 类恢复阶段必须先于剩余新题的 A 类生成。
6. 正式并发默认设为 3 且可配置；同类自由改写重试关闭，
   `distill_retries=1`。

该调整在保留 A 类讲解与代码融合优势的同时，减少错误改写造成的数据
浪费和重复 API 消耗。执行验证仍只能证明通过当前 TACO 测试，不能外推
为对未知隐藏测试的绝对正确性保证。
### 2026-07-30 执行记录：正式 SFT 断点续跑

- rejected 文件已改为仅保存当前未解决题目的紧凑快照。
- Windows 并发写快照采用唯一临时文件和有限原子替换重试，避免 `WinError 5` 中断正式生成。
- 已接纳样本继续通过 ID 断点跳过，不因本修复重新调用教师 API。
### 2026-07-30 SFT 生成口径修正

- accepted 数只统计 accepted JSONL 的唯一 ID；“已完成”不得混同 accepted 与终态 rejected。
- Docker verifier 必须在教师 API 请求前通过可用性预检。
- Docker 连接失败属于可恢复基础设施故障，不属于 wrong answer；历史误分类样本恢复后重新进入 B 类。
## 2026-08-01 阶段状态：正式 SFT 标签数据已达到进入训练条件

- 候选口径：24,237 道去重 TACO 题中，仅使用 reference 离线验证 pass rate 1.0 的 10,415 题。
- 最终状态：10,340 accepted、75 unresolved rejected；覆盖全部候选且 accepted/rejected 无重叠。
- 接纳率：99.28%。accepted 中 standard-input 7,889、call-based 2,451；所有样本记录执行 pass rate 1.0 且 student 输入不含 reference solution。
- 未接纳原因：教师响应失败或输出截断 40、Docker Python 镜像缺少 numpy 28、执行超时 7。
- Gate 判定：SFT 数据构造通过。下一阶段为冻结数据 manifest、划分训练/验证集、统计 tokenizer 长度并启动 SFT；不以回收剩余 75 条作为前置条件。
- 当前 accepted 快照 SHA-256：`CBFA65F00AD635654431004D8C20AD4BCB1D64EF6CFE7DB984331DB5F9E7042D`。
# 2026-08-01 执行状态补充：数据冻结完成

- canonical SFT 已固定为 10,306 条，SHA256：`08ef448f4be6b6b34ee2b6b7af5748827feeba0a0f36cc393350374671c86a1b`。
- 固定 SFT train/dev ID 为 9,791/515，随机种子 20260728，不允许训练脚本重新随机切分。
- 正式基座为 `Qwen/Qwen2.5-Coder-7B-Instruct`，正式 chat template 下最大序列长度 8,173，SFT `max_seq_length` 固定为 8192。
- 34 条超过 8K 的质量通过样本保留在隔离清单，不进入当前 canonical，也不得通过截断重新混入。
- 下一有效阶段调整为“QLoRA SFT 训练管线落地与500条云端校准”；本地只完成数据、collator、配置、preflight 和入口验证，不下载完整7B权重。
# 2026-08-01 执行状态补充：SFT训练管线已落地，等待云端校准

- assistant-only label、固定 split、动态 padding、500条确定性 calibration、QLoRA配置、双4090 preflight、校准启动与Base/Adapter评估入口已实现。
- 全量10,306条按训练格式检查：通过10,306、截断0、最大8,173、监督 token 比例77.5844%。
- 当前本地环境未下载完整7B模型，未执行500条真实训练，因此“正式 full SFT Gate”仍未通过。
- 下一步唯一主线是在双RTX 4090服务器同步 canonical/source bank，执行 `bash scripts/run_sft_calibration_dual_4090.sh`，收集显存、吞吐、loss、checkpoint和adapter重载证据；通过后才启动full SFT。
# 2026-08-01 云端校准预检补充

- 云端实际环境为 PyTorch 2.9.0+cu130；bitsandbytes 0.47.0缺少CUDA 13原生库，首次preflight是假通过，未进入训练。
- CUDA 13下bitsandbytes最低采用0.48系列；preflight新增NF4前后向和PagedAdamW8bit真实算子探针。
- 只有原生算子探针、双rank绑定和数据校验均通过，才允许启动500条校准。
# 2026-08-01 云端分布式入口补充

- 双卡启动不得依赖PATH中的裸 `torchrun`，统一使用当前环境的 `python -m torch.distributed.run`。
- preflight必须确认两个rank均使用项目虚拟环境解释器、可导入训练依赖并分别绑定GPU 0/1。
## 2026-08-01 执行状态补充：8K QLoRA 首步显存修复

- 双 4090 已真实完成 7B 4-bit 模型加载，但首个 8K 长 batch 在 cross-entropy/backward 阶段 OOM，尚未产生 optimizer step。
- 8192 是冻结数据完整性的硬要求，本次不通过降长或删除长样本规避问题。
- SFT 训练栈增加 Liger fused linear cross-entropy 与 CUDA expandable segments；preflight 必须实际编译并反传 fused kernel。
- 恢复顺序固定为：1 step 最长样本探针 -> 500 条完整校准 -> Base/Adapter 对照。只有三者完成后，正式 full SFT Gate 才能通过。
## 2026-08-02 执行状态补充：500 条 SFT 校准训练完成

- 双 RTX 4090 已使用 8K QLoRA + Liger 完成固定 500 条、32 optimizer steps 校准；平均 train loss 0.7215665，100 条 dev loss 0.6630970，无 OOM/NaN。
- SFT 训练可运行性与数值稳定性子项通过。
- 正式 full SFT 之前剩余硬验收为：保存产物存在、adapter 独立重载并真实生成、Base/Adapter 固定 dev 对照。完成前不启动 full SFT。
## 2026-08-02 执行状态补充：校准 adapter 重载通过

- 校准 adapter 已从 4-bit 基座独立重载并完成 64-token 真实生成，保存与推理链路通过。
- 校准阶段剩余唯一门槛为固定 dev Base/Adapter 对照；通过后方可启动 9,791 条 full SFT。
## 2026-08-02 执行状态补充：校准对照采用跨环境两阶段评测

- 双 4090 训练节点不要求 Docker 或 verified source bank；训练环境保持最小化。
- 固定 dev 的 Base/Adapter 回答在云端生成并落盘，随后下载到本地，以冻结 source bank 和 Docker digest 执行统一代码验证。
- 两阶段必须共享固定 selection manifest；本地不得重新生成模型回答，云端不得改用非隔离执行器。
## 2026-08-02 执行状态补充：校准首轮质量对照

- 固定 40 条 dev 中，Adapter 严格 Docker Pass@1 为 17.5%，Base 为 10%，净提升 7.5 个百分点；教学结构完整数 18 vs 0，接口匹配均为 40/40。
- Adapter 的回答显著更长，8 条撞 2048 completion 上限并全部失败，因此首轮结果含非对称截断偏差。
- full SFT 暂缓；只对 8 条撞限 Adapter 回答以 4096 上限复验，其他自然结束结果复用。恢复轮完成后再关闭校准 Gate。

## 2026-08-02 执行状态补充：4096-token 恢复轮完成

- 固定 40 条 dev 的 4096-token 对照已完成；Base/Adapter Docker Pass@1 为 10.0%/17.5%，Adapter 净提升 7.5 个百分点。
- Adapter 教学模板完整数为 21/40，Base 为 0/40；代码块与题目接口均为 40/40。
- 两组都没有撞上 4096 上限，也没有未闭合代码围栏；截断假设已排除，不再继续增加 completion budget。
- 本轮证明 500 条校准的训练方向为正，但它仍是诊断实验，不代替 9,791 条 full SFT 的最终评测。
- 33 条失败的主因是算法/题意误解（19）与边界状态错误（5），其余是 I/O（4）、代码块提取（2）、实现异常（2）和资源超限（1）。不为这些错误继续增加生成长度。

## 2026-08-02 执行状态补充：全量 SFT 启动前最后修复

- 多代码块提取器已修复，离线重放后 Base/Adapter Pass@1 为 4/40 和 8/40。
- full 入口 validate-only 已确认固定 9,791 train / 515 dev，训练配置不变；本地不启动 7B 训练。
- 云容器首次 full 启动在 NCCL `/dev/shm` attach 阶段失败；双卡入口已默认禁用 NCCL SHM 通道，待保持其余配置不变重启。
- NCCL 修复后 full 已完成 25 steps，但全量 dev 评估因完整 logits OOM；评估已改为 loss-only，不缩减 515 条 dev 或 8K 长度。

## 2026-08-03 外部实现复核后的执行口径补充

1. 正式 GRPO 从已完成的 full SFT adapter 暖启动；“warm-start”专指该阶段，不把从官方 Instruct 基座开始的 QLoRA SFT表述为 adapter 暖启动。
2. 正确性奖励继续要求可执行测试和统一 verifier；无测试样本不得以 AST/格式代理分冒充 correctness。
3. TeachingCritic 准入前，LocalTeachingReward 只作表面教学特征监控，不进入梯度；正式主奖励使用第 9.2 节冻结的 code/contract 门控 composite 公式。
4. GRPO 自带组相对优势归一化，额外 batch Z-score 默认关闭，只能作为命名消融。
5. 候选增强按优先级验证：独立 heldout strict Pass@1 best checkpoint、zero-advantage 监控、curriculum 消融、Bootstrap 教学评测。任何 README 声明均须以本项目冻结数据、Docker 口径和落盘结果重新验证。
6. HumanEval 88.4% 是 Qwen2.5-Coder-7B-Instruct 官方基座结果，不是 CodeGuide 训练收益；正式报告需要单独运行 HumanEval+/MBPP+ 等保持性评测。

## 2026-08-14 仓库清理后的执行约束

1. 正式训练实现集中在 `src/training/`，`scripts/` 只保留 CLI 编排与必要兼容入口；新增实验不得复制整套代码到 bundle 子目录。
2. 冻结数据只认 `data/final/sft_accepted.jsonl`、source bank、固定 split 与版本化 manifest；smoke/probe/传输压缩包不得长期留在主工作区。
3. 正式评测统一写入 `outputs/eval/<versioned_run>/`，至少保留 selection、generation、verification 与 summary；旧运行不得覆盖。
4. TACO train 与 reference cache 已在下游 source bank/manifest/hash 验证后本地删除；如重新构造数据可按来源下载，但训练与评测不得依赖项目根目录外的绝对路径。
5. 下一阶段继续以 full SFT adapter 暖启动 GRPO，不因清理重新运行 SFT、API 蒸馏或已完成评测。

## 2026-08-15 训练实现基线更新

1. 训练依赖以根目录唯一 `requirements.txt` 为准，不再维护阶段性 requirements 文件。
2. SFT 与 GRPO 唯一入口分别为 `scripts/train_sft.py` 和 `scripts/train_grpo.py`，底层使用 TRL Trainer；实验变化通过 YAML 与 CLI 参数表达，不新增一次性训练脚本。
3. 双卡进程由 Accelerate 管理；默认 MULTI_GPU，DeepSpeed ZeRO-2 是可选启动配置。项目代码不得再次手工初始化或销毁分布式进程组。
4. SFT 继续使用冻结的 assistant-only 标签；数据已经预分词并审计时，TRL 必须跳过二次数据准备，禁止静默改变 mask 或截断代码。
5. GRPO 从 full SFT adapter 热启动，correctness 必须调用统一 verifier，teaching contract 只衡量明确的教学结构合同；正式教学质量提升仍需独立评测，不能由启发式 reward 直接宣称。
6. 本轮完成的是代码主线重构与本地合同验证。重构后的训练入口在云端至少完成一个最小 GPU smoke 后，才可更新为“运行验证通过”。
