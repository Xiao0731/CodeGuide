# CodeGuide

> 面向 OI / ACM 初学者的算法教学代码模型后训练项目  
> **已验证参考答案 → 参考答案引导蒸馏 → QLoRA SFT → 可验证 GRPO → 冻结执行评测**

CodeGuide 旨在训练一个不仅“会写代码”，而且“会教算法”的代码大模型。与仅追求最终可执行答案的代码生成任务不同，CodeGuide 同时强调两项能力：

1. **代码正确性**：生成能够通过测试用例的可执行代码；
2. **教学完整性**：回答需要覆盖题意理解、关键观察、分步推导、复杂度分析与常见错误。

项目基座为 `Qwen/Qwen2.5-Coder-7B-Instruct`。正式主线围绕 TACO 构建，从多候选参考答案验证、教师侧参考答案引导蒸馏、8K QLoRA SFT、检查点轨迹分析，到从最佳 SFT 适配器热启动的可验证 GRPO，再使用 TACO-515 与 EvalPlus 做冻结评测。

## 项目背景

传统代码大模型往往更擅长直接输出最终代码，但对于 OI / ACM 初学者而言，真正有价值的不只是“答案”，CodeGuide希望在保证基座模型代码能力的基础上，输出遵循一个稳定的教学结构：

题意理解→观察分析→分步推导→复杂度分析→可执行代码→常见错误

整个项目始终坚持**正确性优先于教学模式**，即使回答格式完整、语言流畅，只要代码或核心算法错误就不会被视为高质量教学样本。

## 核心流程

```
原始题目数据（DeepMind/CodeContests + BAAI/TACO）
        │
        ▼
多候选参考答案验证与质量过滤（执行验证 + 接口匹配 + AST静态检查）
        │
        ▼
参考答案引导的数据蒸馏（50道人工高质量seed题，构建约1万条步进式算法教学SFT数据）
        │
        ▼
监督微调 SFT（Qwen2.5-Coder-7B-Instruct + NF4 QLoRA）
  ├── Assistant-only Loss（仅优化模型回答部分）
  └── Checkpoint轨迹分析（验证集Pass@1选择最佳SFT模型）
        │
        ▼
GRPO 强化学习微调
  ├── Reward 1：代码正确性奖励（沙箱执行 / AST检查 / 测试通过率 / Strict Pass） 
  ├── Reward 2：教学契约奖励（解题步骤 / 思路解释 / 代码块 / 复杂度分析）
  ├── 正确性门控（防止错误代码通过教学奖励获得高分）
  ├── 课程教学（Easy → Medium → Hard）
  └── 训练稳定性监测（组奖励方差 / zero-advantage预警）
  └── 最佳 Checkpoint 选择（验证集Pass@1，而非最后一个epoch）
        │
        ▼
模型能力评估
  ├── 竞赛级代码能力：TACO-515  
  ├── 通用代码能力：  HumanEval / HumanEval+ / MBPP / MBPP+   
  └── 教学能力：DeepSeek-V4 Flash + Qwen 3.8 Max 双盲评测（Base vs SFT vs GRPO）  
           
额外诊断实验： 双专家训练（Standard Input × Call Based）
        │
        ▼
接口拆分未带来整体收益，因此主实验采用混合训练方案
```

## 目录结构

```
CodeGuide/
│
├── configs/
│   ├── sft.yaml 						SFT 的模型、8K 长度、NF4、LoRA、优化器、保存/评估等配置
│   ├── grpo.yaml 						GRPO 数据划分、课程阶段、生成参数、奖励公式和训练超参数
│   ├── accelerate/
│   │   ├── dual_gpu.yaml 				普通双卡 Accelerate：2 个进程、bf16、MULTI_GPU
│   │   └── dual_gpu_deepspeed.yaml 	可选 DeepSpeed 启动配置
│   ├── deepspeed/
│   │   └── zero2.json 					ZeRO-2：切分优化器状态和梯度，并启用通信重叠
│   └── eval/
│       ├── taco515_compact_code_first_v1.yaml
│       ├── evalplus_code_capability_v1.yaml
│       └── expert_online_*.yaml 		TACO-515、EvalPlus 与 Standard/Call 专家实验的冻结协议
│ 
├── data/
│   ├── final/
│   │   ├── sft_accepted.jsonl 8K 					审计后的 canonical SFT 数据
│   │   └── taco_verified_source_bank.jsonl.zst 	已验证 TACO 原题、接口、测试等冻结 source bank
│   ├── raw/ 原始数据
│   └── splits/
│       ├── sft_train_ids.json / sft_dev_ids.json
│       └── grpo_minimal_v1/
│           ├── grpo_train.jsonl
│           ├── grpo_dev.jsonl
│           └── freeze_manifest.json 				固定数据划分与交集检查
│ 
├── scripts/
│   ├── verify_taco_references.py TACO 多候选 Python reference 离线执行验证
│   ├── build_sft_dataset.py reference-guided 教学标签生成、断点续跑、质量与执行硬过滤
│   ├── freeze_sft_data.py 冻结 canonical SFT、source bank 与固定 split
│   ├── train_sft.py SFT 命令入口
│   ├── train_grpo.py GRPO 命令入口
│   ├── evaluate_sft_matrix.py TACO-515：prepare → generate → verify → summarize
│   ├── prepare_evalplus_datasets_offline.py 准备固定版本 EvalPlus 数据
│   ├── generate_evalplus_code_capability.py 生成 HumanEval(+)/MBPP(+) 代码样本
│   ├── validate_docker_verifier.py Docker 验证器 smoke / 合同检查
│   └── inference_demo.py adapter / 合并模型命令行推理
│ 
├── src/
│   ├── data/
│   │   ├── loader.py TACO 加载、多候选解析、I/O 类型与 fn_name 等元数据处理
│   │   ├── code_validator.py 从模型回答中提取最终代码、语法检查
│   │   ├── quality.py 蒸馏回答的质量过滤
│   │   └── source_bank.py 冻结 source bank 的读取逻辑
│   ├── reward/
│   │   ├── execution.py standard_input / call_based 共用的统一执行验证器
│   │   ├── format.py 教学契约 contract_score：步骤、代码块、复杂度、长度、教学词
│   │   └── grpo.py  verifier 正式 GRPO 单一 composite reward；每条 completion 只执行一次 
│ 
│   └── training/
│       ├── common.py 配置加载、路径、SHA256、FlashAttention/SDPA 后端选择
│       ├── sft_data.py assistant-only 标签掩码、冻结 split、动态 padding
│       ├── train_sft.py TRL SFTTrainer + PEFT + BitsAndBytes 的正式实现
│       └── grpo_train.py TRL GRPOTrainer、SFT adapter 热启动、三阶段课程与开发集评估
│ 
├── tests/ 将数据、奖励、训练协议写成可回归检查的“合同”
│       
├── EXPERIMENT_LOG.md 实验结果、OOM/NCCL/评测故障与修复记录
├── DECISION_LOG.md 关键技术决策
├── CLAIMS_MATRIX.md
├── requirements.txt
└── README.md
```

## 数据构造

### TACO参考答案验证

本项目选用TACO作为数据源，该数据源收纳了2w多道竞赛级算法题，TACO中同一道题可能包含多个Python候选答案，但是公开答案并不天然等于可信标准答案，因此本项目在使用TACO数据之前对每道题的多个候选答案进行解析、排序和执行验证，而不是默认采用solutions[0]作为标答。主要考虑：

- `standard_input` 是否正确读写 stdin / stdout；
- `call_based` 是否满足 `fn_name` / 起始接口约束；
- 语法是否有效；
- 是否能够通过实际测试；
- 第一候选失败时是否可由其他候选回退救回。

最终，24237道去重的TACO题目经过验证，完全通过测试的有10415道题，其中7,952是standard_input，2,463道是call\_based。

### 参考答案引导蒸馏

正式蒸馏阶段，已验证参考答案仅作为 **教师侧特权上下文** 提供给教师模型，用于降低自由解题时的算法幻觉。

教师模型输入：题目+已验证参考答案+接口信息+测试摘要

学生模型最终训练输入：题目+接口约束

本项目人工整理出了涉及十类共50题的高质量seed题，代码层面通过连接DeepSeek V4 falsh作为教师模型，最终规范SFT数据保存至`data/final/sft_accepted.jsonl`，约10k条。

在进入SFT阶段前，冻结出9797道SFT训练集，留下515道作为全流程评测集，记作TACO515；GRPO训练集6451道，开发集50道用以选择最佳checkpoint。以下是GRPO训练集和开发集题目维度数量：

| **维度数量**    | GRPO训练集 | GRPO开发集 |
| --------------- | ---------- | ---------- |
| standard\_input | 4,864      | 40         |
| call\_based     | 1,587      | 10         |
| Easy            | 3,228      | 28         |
| Medium          | 1,735      | 15         |
| Hard            | 1,488      | 7          |

##  QLoRA SFT

基座模型固定为`Qwen/Qwen2.5-Coder-7B-Instruct`，在SFT阶段使用4-bit NF4、LoRA / PEFT、TRL `SFTTrainer`。仅计算助手回复损失（assistant-only loss）并独立冻结评测进行检查点选择。

| 参数                   | 当前配置                     |
| ---------------------- | ---------------------------- |
| 最大序列长度           | 8192                         |
| 量化                   | 4-bit NF4 + double quant     |
| 计算精度               | bf16                         |
| LoRA                   | r=32, alpha=64, dropout=0.05 |
| 单卡 batch             | 1                            |
| 梯度累积               | 8                            |
| 优化器                 | `paged_adamw_8bit`           |
| 学习率调度             | cosine                       |
| Gradient Checkpointing | 开启                         |
| Liger Kernel           | 开启                         |
| Packing                | 关闭                         |

####  TRL `SFTTrainer` 到底是什么？

`SFTTrainer` 可以理解成 TRL 对 Hugging Face `Trainer` 的 SFT 专用封装：它负责常见的 causal LM 监督微调训练循环、PEFT 集成、保存/评估等通用工作。

固定 split、8K 不静默截断、assistant-only 标签怎么构造，仍然由项目自己的 `src/training/sft_data.py` 控制。

#### “只计算助手回复损失”是什么？

模型看完完整对话，但只因为Assistant的输出对错而被惩罚。本项目的ChatML样本belike：

```
<System> 你是算法教学助手……
<User>   题目……
<Assistant> 题意、思路、代码……
```

训练时模型当然需要读到 System 和 User 才知道自己在回答什么，但我们并不希望它学习去“预测用户问题本身”。

当前项目不是简单设置一个 `assistant_only_loss=True` 开关，而是自己构造 `labels`：

```text
input_ids:
[System tokens][User tokens][Assistant tokens]

labels:
[-100 ... -100][-100 ... -100][Assistant tokens]
```

PyTorch 的交叉熵会忽略 `label=-100` 的位置，所以：

- System / User：**只作为上下文输入，不贡献 loss**；
- Assistant：**真正参与监督学习**；
- 动态 padding 位置也设为 `-100`，同样不参与 loss。

实验审计中，训练数据的 supervised token ratio 约为 **77.58%**。

#### 为什么做学习率 × checkpoint 轨迹，而不是只看训练 loss？

SFT 的训练 loss 下降只能说明模型越来越会拟合教师标签，不等价于“算法正确率一定越来越高”。教学数据本身同时包含风格和代码，训练过深可能出现：

- 教学模板学得更牢；
- Standard Input 适应更好；
- 但 Call-based 能力或外部分布能力发生遗忘。

因此项目对 `1e-4` 与 `2e-4` 两条训练轨迹做冻结执行评测，而不是默认最后一步最好。



项目没有直接将“训练最后一步”视为最佳模型，而是观察不同学习率和优化器步数的能力轨迹。下图是学习率设置为1e-4,2e-4时，两组模型训练至611步的结果，其中，蓝色虚线表示基座模型在`standard_input`、`call_based`以及测试集的得分。

从下图可见，模型在训练过程中，标准输入能力随训练显著增强；函数调用能力逐渐出现性能回退；整体最佳点并非简单出现在训练末端。

![CodeGuide SFT Checkpoint Trajectory](docs/pictures/CodeGuide%20SFT%20Checkpoint%20Trajectory.png)

最终我们选出LR 2e-4 step50，LR 1e-4 step20，2e-4 step200三个checkpoint进行TACO515检查点对比。其中LR 2e-4 step50测试集整体得分率最高，LR 1e-4 step20训练前期仍能保持基座模型解决`Call-based`类型题目的能力，2e-4 step200则在测试集整体和`standard_input`得分率都位居高位。

正式 TACO-515 评测使用固定 515 道留出题、统一生成协议与 Docker 验证器。CodeGuide使用统一验证器，支持`standard_input`、`call_based`两类题目，并对接口、执行结果与测试通过情况进行统一处理，最终严格评测要求：能够提取有效代码、接口匹配、Docker 执行成功、所有测试通过、`pass_rate == 1.0` 才计入 Strict Pass@1。

SFT训练集中，标准输入（`standard_input`）7952题，函数调用（`call_based`）2463题。

| **模型**            | 严格评分   | 标准输入   | 函数调用   | 测试通过率 | 平均 Token 数 |
| ------------------- | ---------- | ---------- | ---------- | ---------- | ------------- |
| Base                | 15.53%     | 4.33%      | 51.64%     | 21.50%     | 506.87        |
| LR 2e-4 step50      | 23.30%     | 15.52%     | 48.36%     | 36.15%     | 627.10        |
| LR 1e-4 step20      | 21.94%     | 12.47%     | **52.46%** | 33.01%     | 672.11        |
| **LR 1e-4 step200** | **24.85%** | **18.07%** | 46.72%     | **37.35%** | 626.90        |
| **GRPO**           | **27.77%** |  **18.31%** |  49.73    | **38.51%**| 654.32        |

这组结果说明：SFT 显著提升标准输入能力；但函数调用能力在较深训练阶段出现一定遗忘；检查点选择不能只依据最终训练步数。最终选择：**LR 1e-4 step200**作为 GRPO 热启动模型。

此处一并给出TACO515，Base、SFT、GRPO三阶段的代码通过率，可以看到GRPO相较于SFT best在竞赛级代码能力上也有了2.92%的提升。

## 可验证GRPO

### 正式奖励

GRPO 从最佳 SFT 适配器热启动，而不是从基座模型冷启动。SFT 先让模型学会基本教学输出分布和目标域行为，GRPO 再利用真实可执行奖励进一步调整“哪些回答更值得产生”。

正式总奖励：

$$
R = 0.60 R\_{\text{code}} + 0.40 R\_{\text{contract,gated}}
$$

分为代码正确性奖励和教学契约奖励。

其中，代码正确性奖励细分如下：

$$
R\_{\text{code}} = 0.05R\_{\text{static}} + 0.70R\_{\text{pass}} + 0.25R\_{\text{strict}}
$$

定义：

static_validity ∈ {0, 1}，静态分析AST：是否通过编译，检查危险模块 / 调用，以及 `call_based` 的函数名或 `standard_input` 的输入输出接口是否存在。
pass_rate       ∈ [0, 1]， 测试通过率
strict                = 1 if pass_rate == 1 else 0，严格测试通过率，只有通过所有测试用例才为1

#### 为什么既要`pass_rate`又要`strict`？

1. 只用 strict只用 strict：0% 通过和 99% 通过都可能得到 0 → 奖励过稀疏

2. 只用 pass_rate：模型可能长期满足于“差一点全对” → 没有足够动力跨过完整正确门槛


教学契约奖励通过执行正确率进行门控：

$$
gate = 0.25 + 0.75 \times pass\_rate
$$

$$
R\_{\text{contract,gated}} = R\_{\text{contract}} \times gate
$$

教学契约内部权重：

| **项目权重**          |      |
| --------------------- | ---- |
| 教学步骤 / 栏目完整性 | 0.40 |
| Python 代码块         | 0.30 |
| 复杂度                | 0.20 |
| 合理长度              | 0.05 |
| 教学词汇              | 0.05 |

其中语义教学启发式指标只用于诊断，不直接进入梯度。奖励门控用于抑制一种典型的奖励投机（Reward Hacking）：

如果直接把格式分加入总奖励，那么模型可能学会通过堆砌“题意 / 步骤 / 复杂度”等结构获得格式分，但代码本身并不正确，因此本项目通过门控：

1. pass_rate = 0   → 教学契约最多只保留 25%
2. pass_rate = 0.5 → 保留 62.5%
3. pass_rate = 1   → 教学契约完整生效

### 课程学习

在 `configs/grpo.yaml` 中启用 `curriculum.enabled: true`，硬检查课程学习参数并按难度分阶段训练：

```
阶段 1（easy）   → 建立基础能力，3228题，max_new_tokens=512
阶段 2（medium） → 中等难度泛化，1735题，max_new_tokens=768
阶段 3（hard）   → 挑战难题，1488题，max_new_tokens=1024
```

设计动机：GRPO 早期直接遇到 hard 题时 Pass@1≈0，reward≈0，梯度完全消失。

GRPO 依赖**同一题多条 completion 之间存在可区分的奖励**。如果一上来全是难题，4 条回答可能全部 0 分，组内几乎没有“谁更好”的信息；先从 Easy 建立可学习信号，再逐步混入更难的分布，更容易获得有效优势。

| **参数**          | 数值      | 意义                                                         |
| ----------------- | --------- | ------------------------------------------------------------ |
| 量化              | 4-bit NF4 |                                                              |
| `num_generations` | 4         | 每个 prompt 采 4 个候选，GRPO 在同题候选之间做相对比较       |
| `temperature`     | 0.8       | 保留探索性；太低 4 条回答可能高度相似，太高又可能代码质量过乱 |
| `top_p`           | 0.95      | 核采样，截掉极低概率长尾 token                               |
| 学习率            | 1e-5      |                                                              |
| 梯度累积          | 8         | 多次微批次后再做一次 optimizer update                        |
| KL β              | 0.05      | 非零 KL 约束，限制策略过快偏离参考策略                       |
| 精度              | bf16      | 混合精度                                                     |
| 奖励缩放          | 关闭      | 关闭额外“按标准差缩放奖励”                                   |

#### `scale_rewards=false` 到底是什么意思？为什么要关？

GRPO本质是组相对优化，同一prompt的多条completion会形成相对优势。`scale_rewards=false` 关闭的是TRL额外的“再除以组内/批内奖励标准差”这一步。因为如果某些简单题的4个奖励非常接近、标准差很小，除以一个很小的标准差会人为放大它们；不同难度题的梯度权重就可能被“标准差”而不是业务价值主导，因此当前实验默认关闭。

### 最优checkpoint选择

GRPO正式训练采用异步评测，每100步，对当前最近完整检查点做冻结Dev-50计算Pass@1，Pass@1超越历史最高时自动保存。因为GRPO后期奖励虚高，但是实际Pass@1在下降，按最后一个epoch保存无意义。

####  “训练 reward 越来越高，但 Pass@1 反而下降”是什么意思？

训练 reward 是一个**代理目标**，最终真正想要的是模型在未见题上的严格执行正确率。

哪怕我们做了 gating，仍然可能出现：

- 模型把部分测试或结构分优化得很好，但没有增加全测试通过数量；
- 训练题分布被反复见到，reward 上升但泛化变差；
- 某些 checkpoint 生成更长、更符合契约，但代码错误更多；
- on-policy 训练时，当前采样分布上的 reward 提升，不保证冻结 greedy Pass@1 同步提升。

所以训练 reward 最高 ≠ 最终模型最好，因此GRPO训练阶段中，我们使用冻结开发集去选择当前最优checkpoint。

## TACO-515 竞赛级代码能力

| **模型**            | 严格评分   | 标准输入   | 函数调用   | 测试通过率 | 平均 Token 数 |
| ------------------- | ---------- | ---------- | ---------- | ---------- | ------------- |
| Base                | 15.53%     | 4.33%      | 51.64%     | 21.50%     | 506.87        |
| **SFT** | **24.85%** | **18.07%** | 46.72%     | **37.35%** | 626.90        |
| **GRPO**           | **27.77%** |  **18.31%** |  49.73    | **38.51%**| 654.32        |

## EvalPlus：检查模型的通用代码能力

只在 TACO 上上涨不能证明模型整体代码能力变强，因为模型可能只是越来越适应 TACO 的题型和提示分布，因此本项目在SFT和GRPO环节都会使用EvalPlus评测集检查 **模型的通用代码能力**，包括：**HumanEval、HumanEval+、MBPP、MBPP+**。

| **模型**    | HumanEval | HumanEval+ | MBPP      | MBPP+     |
| ----------- | --------- | ---------- | --------- | --------- |
| Base        | 84.1%     | 78.0%      | **81.7%** | **69.8%** |
| SFT step20  | **86.0%** | 78.0%      | 79.1%     | 66.9%     |
| SFT step200 | **86.0%** | **80.5%**  | 74.3%     | 63.8%     |
| GRPO        | **88.4%** | **83.7%**  | **83.5%** | **72.2%** |

> HumanEval 是 OpenAI 提出的代码生成测试，共 164 道 Python 函数生成题，要求模型根据函数契约理解需求，并实现正确逻辑；HumanEval+则是在HumanEval的基础上补了大约80倍的测试用例
>
> MBPP（Mostly Basic Python Problems）是Google设计的入门级Python编程题，约1k道Python杂活；MBPP+测试数量约增加35倍。

因为SFT数据本身明显偏向“读题 → 分析算法 → 推导 → 实现算法”，因此经过训练后模型参数会往竞赛算法题分布、长题面、算法推理、复杂度、结构化解题这个方向偏，因此在强调算法/逻辑综合能力的HumanEval（+）上得分提升，而对于MBPP这类Python杂活熟练度下降。

**领域适配+选择性遗忘**：SFT 让模型发生了“能力重分配”——更擅长一类代码问题，同时遗忘了一部分另一类代码模式。

而在GRPO后，在进一步提升TACO的同时，恢复并超越了基座模型的通用代码能力。

#### **为什么GRPO又把MBPP救回来了？**

1. **执行正确性是主奖励。** 相比 SFT 只模仿教师标签，GRPO 会直接奖励测试真正通过的候选；

2. **`pass_rate` 提供稠密语义反馈。** 它不关心模型有没有复刻 TACO 教学措辞，而关心代码到底通过了多少测试；

3. **Strict bonus 把“全对”重新设为终点。** 这可能把优化重心从风格模仿拉回功能正确性；

4. **KL β=0.05 限制策略漂移。** 在优化 TACO reward 的同时，减少对原始 SFT/基座分布的过度偏离；

5. **GRPO 是同题候选相对比较。** 它强化“当前 4 个答案里更能通过测试的那个”，而不是再次强制模型复制某一个教师答案。

   

## 教学格式双盲检测

代码能够通过测试，并不代表回答真正适合初学者。为单独衡量教学能力，项目实现了
`scripts/evaluate_teaching.py`，在不重新训练模型、也不向评审泄漏参考答案的前提下，
对 Base、SFT 和 GRPO 三个阶段进行双盲比较。

### 评测协议

- **输入隔离**：生成阶段只使用原始 ChatML 的 `system + user`；SFT 中的`assistant/reference`标签不会进入模型输入。Judge 只看到题面与匿名回答 A/B，看不到模型阶段、reference、训练标签或 Docker 正确性结果。
- **三组配对**：每题分别比较 Base vs SFT、Base vs GRPO、SFT vs GRPO。使用固定种子对 A/B 位置做近似平衡交换，降低位置偏差。
- **双 Judge**：DeepSeek V4 Flash 与 Qwen3.8 Max 独立评分，均关闭 thinking，`temperature=0`。

Judge 被明确要求不能因为回答更长、标题更多或排版更漂亮而加分，并且只能返回严格JSON，最终汇总保存到 `outputs/teaching_eval/report.md`。

### 胜率


| 对比         | 胜率 |
| ------------ | ---------: |
| SFT vs Base  |  **63.2%** |
| GRPO vs Base |  **58.5%** |
| GRPO vs SFT  |  **52.3%** |

##### 分维度结果

SFT 对 Base 的教学偏好较明确；GRPO 仍优于 Base，但没有在整体教学偏好上完全超过 SFT。双 Judge 的五维平均绝对分如下：
| 维度       |权重|   Base |        SFT |       GRPO | 最优阶段 |
| ---------- |---| -----: | ---------: | ---------: | -------- |
| 题意理解   |20%| 5.8800 |     5.9225 | **5.9950** | GRPO     |
| 算法讲解   |30%| 5.0575 |     5.1225 | **5.3100** | GRPO     |
| 推导流程   |20%| 5.4750 |     5.5175 | **5.5900** | GRPO     |
| 代码一致性 |20%| 5.4700 |     5.5650 | **5.7400** | GRPO     |
| 初学者友好 |10%| 5.2575 | **5.5550** |     5.4200 | SFT      |

##### 与代码正确性的联合结论

当前证据支持**SFT 明显塑造了教学表达并提升了代码正确率；GRPO 进一步小幅提升严格 Pass@1 和代码一致性，总体教学分与SFT持平**。 最终模型相对 Base 的严格Pass@1 提升 `8.54` 个百分点。



## 快速复现

### 环境配置

正式云端GRPO环境记录：

```
GPU: 2 × NVIDIA RTX 4090 24GB
PyTorch: 2.11.0 + CUDA 13.0
Transformers: 4.56.2
TRL: 0.22.2
Unsloth: 2026.7.5
```

所有Python依赖统一维护于`requirements.txt`，推荐安装：

```
python -m venv .venv
source .venv/bin/activate

pip install -U pip packaging ninja
pip install --no-build-isolation -r requirements.txt
```

> CUDA 环境建议先安装与驱动匹配的 PyTorch，再安装其余依赖和 `flash-attn`。如果 FlashAttention 2 不可用，当前 `attention_backend: auto` 会回退 SDPA。

### 核心命令

#### 1. 只验证数据和配置

无需加载 7B 模型：该命令会检查冻结数据、数据划分和正式 SFT / GRPO 配置是否完整。

```
python scripts/quickstart.py check
```

#### 2. SFT

默认运行正式双卡 QLoRA SFT，并使用：learning rate = 1e-4

```
python scripts/quickstart.py sft
```

如需使用 DeepSpeed ZeRO-2：`--deepspeed`；如需修改学习率：`--learning-rate`

#### 3. GRPO

从选定的 SFT adapter 热启动，在这里的`best_sft_adapter`只作为示范：

```
python scripts/quickstart.py grpo \
  --sft-adapter /path/to/sft/best_sft_adapter
```

正式 GRPO 自动执行：Easy → Medium → Hard，三阶段课程学习，并使用代码正确性与教学契约组成的可验证奖励。

训练期间 TRL 会记录组内 `reward_std` 与 `frac_reward_zero_std`。当大量 prompt 的多个 completion 获得相同 reward、导致组内相对优势消失时，CodeGuide 会自动触发组奖励坍缩预警。

#### 4. 代码能力Docker一键评测

同时运行 TACO-515 与 EvalPlus：

```
python scripts/quickstart.py eval \
  --sft-adapter /path/to/sft/best_sft_adapter \
  --grpo-adapter /path/to/grpo/best_grpo_adapter
```

该命令会自动完成：

```
Base / SFT / GRPO
        │
        ├── TACO-515
        │     ├── 模型生成
        │     ├── Docker 严格执行验证
        │     └── Strict Pass@1 汇总
        │
        └── EvalPlus
              ├── HumanEval
              ├── HumanEval+
              ├── MBPP
              └── MBPP+
```

TACO-515 的生成结果与 Docker 执行验证彼此分离，因此已有 generation 可以重复验证和统计，无需重新运行 7B 模型推理。

该命令同样支持单独测试某个评测集，如只运行 TACO-515：

```
python scripts/quickstart.py eval-taco \
  --sft-adapter /path/to/sft/checkpoint-200 \
  --grpo-adapter /path/to/grpo/best_adapter
```

也可以在完整评测中跳过其中一项：

```
# 只跑 EvalPlus
python scripts/quickstart.py eval \
  --sft-adapter /path/to/sft/checkpoint-200 \
  --grpo-adapter /path/to/grpo/best_adapter \
  --skip-taco
```



## ChatML数据格式

```json
{
  "id": "taco_xxxxxxxxxx",
  "messages": [
    {
      "role": "system",
      "content": "你是 CodeGuide，一位专为 OI/ACM 初学者设计的算法教学助手……"
    },
    {
      "role": "user",
      "content": "请按上述格式讲解以下算法题：\n\n【题目描述】\n……\n\n- io_mode: call_based\n- fn_name: largest_power\n\nstarter_code:\n```python\ndef largest_power(n):\n    pass\n```"
    },
    {
      "role": "assistant",
      "content": "### 第一步：理解题意\n……\n\n### 关键观察\n……\n\n### 算法步骤\n……\n\n### 复杂度\n……\n\n### 常见错误\n……\n\n```python\n# 完整可运行代码\n```"
    }
  ],
  "metadata": {
    "source": "codewars",
    "difficulty": "easy",
    "tags": ["Mathematics"],
    "io_mode": "call_based",
    "fn_name": "largest_power",
    "starter_code": "def largest_power(n):\n    ",
    "pass_rate": 1.0,
    "reference_verified": true,
    "reference_pass_rate": 1.0,
    "selected_reference_index": 0,
    "label_strategy": "pedagogical_rewrite",
    "code_source": "teacher_generated",
    "reference_hash": "...",
    "verification_evidence": {},
    "freeze_schema_version": "..."
  }
}
```



## 其余实验：Standard / Call 双专家

在验收到SFT后，模型的通用代码能力发生部分退化，本项目在排查是否是因为训练环节没有区分训练两类代码题而导致的部分衰退，因此层尝试把两种 I/O 模式拆开训练“专家模型”：

当时的 64 题专家验证结果：

| 模型            | Strict Pass@1 | 平均测试通过率 | 平均生成 Token |
| --------------- | ------------: | -------------: | -------------: |
| Standard Expert |         6.25% |          7.35% |          504.0 |
| Call Expert     |        45.31% |         62.18% |          482.1 |

这组实验表明，**把接口模式拆开，并没有带来比 mixed SFT 更好的整体收益。**

可能原因包括：

- Standard 与 Call 的底层算法能力高度共享，拆开后每个模型能看到的数据更少；
- 两类题真正差异主要在 I/O / 函数接口，而不是需要两套完全不同的算法知识；
- 专家模型容易把容量过度适配到接口模式，却损失跨模式共享的表示；
- 最终真实部署仍希望一个模型同时处理两种合同，mixed 方案更自然。

因此专家路线作为本项目**诊断实验**：它证明 Call-based 和 Standard Input 的学习动态确实不同，但不适合作为当前最终架构。



## 训练与显存基建

| 组件                   | 当前项目中的作用                                | 主要解决的问题                          |
| ---------------------- | ----------------------------------------------- | --------------------------------------- |
| `Transformers`         | 加载 Qwen2.5-Coder、Tokenizer、注意力后端       | 模型与推理/训练基础接口                 |
| `TRL`                  | `SFTTrainer`、`GRPOTrainer`                     | 后训练算法与训练循环                    |
| `PEFT` / LoRA          | 只训练低秩适配器                                | 减少可训练参数与优化器状态              |
| `bitsandbytes`         | 4-bit NF4 + double quant + `paged_adamw_8bit`   | 压缩基座权重和优化器状态                |
| FlashAttention 2       | 高效 Attention kernel                           | 降低长序列 Attention 中间显存和访存开销 |
| SDPA                   | FlashAttention 不可用时的 PyTorch 原生后端      | 保证训练仍可运行                        |
| Liger Kernel           | SFT 中启用 fused kernel，尤其是 fused linear CE | 避免大词表 logits / CE 带来的显存峰值   |
| Gradient Checkpointing | 反向时重算部分前向激活                          | 用计算时间换激活显存                    |
| `bf16`                 | 混合精度训练                                    | 降低显存/带宽，且指数范围比 fp16 更宽   |
| Accelerate             | 双卡两进程启动、设备与分布式管理                | 不手写 DDP                              |
| DeepSpeed ZeRO-2       | 可选切分优化器状态与梯度                        | 进一步降低数据并行训练状态显存          |
| Docker verifier        | 最终严格执行评测                                | 可复现、隔离的代码正确性验证            |
| subprocess verifier    | GRPO 在线 reward                                | 避免每个 rollout 都启动容器的巨大开销   |





