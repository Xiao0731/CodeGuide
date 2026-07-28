# CodeGuide G0 仓库可信性验收报告

> 日期：2026-07-27（America/Los_Angeles）  
> 审计对象：`review/codeguide` 源码快照  
> 规划依据：`CodeGuide_后训练项目实施与验收规划书_v1.2.md`  
> 总结论：**G0 尚未通过；静态/P0 与 Docker verifier 子阶段已通过，CUDA
> 依赖锁和代理模型 dry-run 待完成。**

## 1. 修复前基线

| 检查 | 修复前结果 |
|---|---|
| `python -m compileall -q .` | 失败：5 个语法错误 |
| 单元测试 | 36 passed，1 failed |
| GRPO 正式入口 | `scripts/train_grpo.py` 与 `src/training/grpo_train.py` 两套分叉 |
| 无测试 correctness | AST 外观打分，正确样例仅得 0.4；仍可能向 GRPO 提供伪正确性信号 |
| teacher 执行失败标签 | 统计失败但继续写入 SFT |
| checkpoint 数据 | 从训练 JSONL 抽题 |
| checkpoint 指标 | 平均测试用例通过率被称为 Pass@1 |
| 格式奖励 | 正式入口使用旧 `reward_functions.format_reward`，新版 anti-hacking 实现未接线 |
| Batch Z-score | 默认开启，文档声称能校准多路奖励权重，但实现只对合并总分再次标准化 |
| generation collapse | 只凭低 reward 方差下结论 |
| 正式长度 | SFT/GRPO 仍残留 2048/4096 猜测值 |
| 项目完成度叙事 | README 存在无训练日志支撑的“发现、提升、鲁棒”等完成态表述 |

## 2. 本轮已完成修复

### 2.1 入口与语法

- 修复 CLI Demo、Gradio Demo 的字符串语法错误；
- 将 Notebook `!pip/!git` 从 Python setup 文件中移除；
- 重写 Colab/AI Studio shell setup，保证 `bash -n` 通过；
- `scripts/train_grpo.py` 改为薄 wrapper，正式实现唯一指向
  `src/training/grpo_train.py`。

### 2.2 数据可信性

- teacher 输出必须同时满足：
  - 有代码块；
  - 语法合法；
  - verifier 支持；
  - 测试数大于 0；
  - 无执行错误；
  - `pass_rate == 1.0`；
- 任一条件不满足就直接退出，不写入 accepted SFT；
- 正式数据生成默认 `run_code=true`；
- 只有显式 `--no-run-code --allow-unverified-output` 才能产生独立诊断数据；
- 新增 `scripts/verify_sft_labels.py`，可对已有 JSONL 全量重执行并 fail closed。

### 2.3 执行合同

- 新增明确的 `subprocess` 与 `docker` 两种 backend；
- `subprocess` 标为开发单测后端，不称安全沙箱；
- Docker 后端要求：
  - 镜像必须固定 `@sha256:` digest；
  - 非 root 用户；
  - `--network none`；
  - 只读根文件系统；
  - `--cap-drop ALL`；
  - `no-new-privileges`；
  - CPU、内存、PID、wall-clock 限制；
  - 独立 tmpfs；
  - 只挂载只读 runner，不挂项目、密钥或数据目录；
- 初始审计环境无 Docker；2026-07-28 已在 Windows 10 + Docker Desktop
  环境完成真实运行验收，详见第 8 节。

### 2.4 Reward 与 GRPO 数据

- 无测试用例时 `accuracy_reward=0`，AST 不再代理语义正确性；
- 正式 Contract Reward 已统一接入 `src.reward.format._score_format`；
- 中文步骤重复检测改用字符 bigram Jaccard；
- TeachingCritic 准入前：
  - correctness 权重 0.9；
  - contract 权重 0.1；
  - teaching 权重 0；
- LocalTeachingReward 改称 `diagnostic/teaching_surface`，不进入梯度；
- 主配置关闭额外 batch Z-score；
- 低组内奖励方差改记为 `zero_advantage_ratio`，不再直接称生成坍缩；
- GRPO 样本至少需要 4 个可执行测试；
- 使用确定性 SHA-256 顺序拆分约 70% online reward tests 与约 30% held-out tests；
- reward 函数看不到 held-out test 内容。

### 2.5 Checkpoint 与长度

- `save_best=true` 时强制读取独立 `grpo.eval_data`；
- dev metadata 必须含 `heldout_tests`；
- 禁止从 `train_data` 抽题；
- Pass@1 改为“整题所有 held-out tests 全通过”的题目均值；
- 删除固定 20 题、512 completion token 的静默缩减，改由配置明确控制；
- SFT 与 GRPO LoRA 统一为 `r=16, alpha=32`；
- SFT/GRPO `max_seq_length` 设为未冻结；
- 正式入口在长度不是 4096/6144/8192 时 fail fast。

### 2.6 声明与归属

- 新建根目录 `CLAIMS_MATRIX.md`；
- 新建根目录 `DECISION_LOG.md`；
- 新建 `PROVENANCE.md`；
- README 已删除未经证据支持的“12 项改进已生效”、训练提升数字和完成态叙事；
- 当前快照不含 `.git`，公开发布仍被来源审计阻塞。

## 3. 实际验收命令与结果

### 3.1 静态编译

```bash
.venv-g0/bin/python -m compileall -q \
  -x '(^|/)(\.venv-g0|__pycache__)(/|$)' .
```

结果：**PASS，exit 0**。

### 3.2 单元测试

```bash
.venv-g0/bin/python -m pytest -q
```

结果：

```text
42 passed in 3.74s
```

### 3.3 Shell 语法

```bash
bash -n scripts/setup_colab.sh scripts/setup_aistudio.sh \
  scripts/setup_kaggle.sh scripts/setup_modelscope.sh \
  scripts/prepare_data.sh scripts/run_distill.sh \
  scripts/run_train.sh scripts/colab_quickstart.sh
```

结果：**PASS，exit 0**。

### 3.4 静态配置合同

```bash
.venv-g0/bin/python scripts/validate_config.py \
  --config configs/train_config.yaml --allow-unfrozen
```

结果：**PASS with warnings**：

```text
sft.max_seq_length 未冻结
grpo.max_seq_length 未冻结
container image 运行时从环境变量解析
```

严格模式：

```bash
.venv-g0/bin/python scripts/validate_config.py \
  --config configs/train_config.yaml
```

结果：**预期失败，exit 1**。这证明未做长度审计时无法把配置误用为正式训练配置。

### 3.5 数据 smoke

结构审计：

```bash
.venv-g0/bin/python scripts/audit_sft_data.py \
  --data data/sft_train_smoke_ref_label.jsonl --top-k 5
```

结果：5 条均为 `reward_compatible`，I/O metadata 可解析。

重新执行 call-based smoke：

```bash
.venv-g0/bin/python scripts/verify_sft_labels.py \
  --data data/sft_train_smoke_ref_label_call_based.jsonl \
  --backend subprocess
```

结果：

```text
records=5, full_pass=5, failed=0
```

重新执行旧 standard-input smoke：

```bash
.venv-g0/bin/python scripts/verify_sft_labels.py \
  --data data/sft_train_smoke_ref_label.jsonl \
  --backend subprocess
```

结果：**预期失败**：

```text
records=5, full_pass=4, failed=1
id=taco_b74ba2ec45
got '3\n0\n2', expected '4\n0\n2'
```

该负例证明旧管线确实写入过错误 teacher 标签；新 hard gate 会拒绝同类样本。

### 3.6 冻结 manifest 读取

```bash
.venv-g0/bin/python scripts/evaluate_model.py \
  --manifest data/manifests/g0_smoke.json --validate-only
```

结果：**PASS**。脚本验证了记录文件存在、SHA-256 一致、记录数为 5。

### 3.7 密钥扫描

```bash
rg -n --hidden -g '!.venv-g0/**' -g '!artifacts/**' \
  '(sk-[A-Za-z0-9_-]{16,}|API_KEY\s*=\s*["'\''][^"'\'']+["'\''])' .
```

结果：未发现硬编码密钥。

## 4. G0 Gate 状态

| G0 条目 | 状态 | 证据/阻塞 |
|---|---:|---|
| `compileall` | PASS | 全仓静态编译 exit 0 |
| 全部单元测试 | PASS | 44 passed |
| 数据 smoke | PASS（开发后端） | call-based 5/5；旧 standard-input 负例 4/5 被准确检出 |
| 代理模型 SFT 单 batch | BLOCKED | 当前环境无 torch、transformers、trl、peft、unsloth、GPU |
| adapter 保存/读取 | BLOCKED | 同上 |
| 代理模型 GRPO 单 step | BLOCKED | 同上 |
| 中断后断点恢复 | BLOCKED | 同上 |
| checkpoint callback 实际运行 | BLOCKED | 无模型/GPU；合同与单测已修复 |
| 正式评测 CLI 读取 manifest | PASS | SHA-256 + count 验证通过 |
| Docker reward smoke | PASS | digest 固定；隔离、资源、超时、并发及两种 I/O 实测通过 |
| 训练环境 lock | BLOCKED | 仅完成 `requirements.g0.lock.txt`；CUDA 训练栈未验证 |
| 无密钥入库 | PASS | 静态扫描无命中 |
| Git 私有基线 | PASS | `main`/`origin/main` 已同步至 `af2a0fb`，未强推 |
| pre-baseline 来源审计 | PARTIAL | 旧远端提交已保留；导入文件作者/许可证仍待公开发布前审计 |

## 5. 当前环境证据

初始审计环境的 `artifacts/g0/environment.json` 记录：

- Python 3.12.13；
- pytest 8.4.1；
- OmegaConf 2.3.0；
- 无 NVIDIA GPU / `nvidia-smi`；
- 无 Docker；
- 无 torch、Transformers、TRL、PEFT、Unsloth、vLLM；
- 当前目录无 Git 元数据。

因此本报告不得声称：

- QLoRA SFT 已跑通；
- GRPO 已跑通；
- Docker 是经过第三方安全审计的通用安全沙箱；
- checkpoint 恢复已跑通；
- 训练依赖已锁定；
- G0 已全部通过。

## 6. 下一步固定顺序

1. 在用户本地 RTX 4060 或等价 CUDA 环境复制本修复版；
2. 固定 Python 3.11、CUDA、PyTorch、Transformers、TRL、PEFT、
   bitsandbytes、Unsloth、vLLM 的精确兼容版本，生成
   `requirements.lock.txt`；
3. 使用 0.5B～1.5B 代理模型运行：
   - SFT 单 batch；
   - adapter 保存与恢复；
   - GRPO 2 prompts × 4 generations × 1 step；
   - 独立 dev checkpoint；
   - 中断恢复；
4. 运行 tokenizer 长度审计并冻结 4096/6144/8192；
5. 严格模式 `validate_config.py` 通过后，才可将 G0 标记为 PASS；
6. G0 通过后进入 300 条 DeepSeek-V4-Flash 数据 pilot。

## 7. 2026-07-28 Windows / RTX 4060 本机复验

本节是对 2026-07-27 修复快照的第二环境复验，不覆盖上文 Linux 静态审计。

实际环境：

- Python 3.11.9；
- NVIDIA GeForce RTX 4060 Laptop GPU，8188 MiB；
- 驱动 576.40；
- Docker CLI 28.1.1 已安装，但 Docker Desktop Linux daemon 未运行；
- 当前 `.venv` 未安装 torch、Transformers、TRL、PEFT、bitsandbytes、
  Unsloth 或 vLLM。

实际结果：

| 检查 | 结果 |
|---|---|
| `python -m compileall` | PASS，exit 0 |
| `python -m pytest -q` | PASS，修复后 44 passed |
| shell `bash -n` | PASS，exit 0；WSL 输出网络提示但不影响语法检查 |
| `validate_config.py --allow-unfrozen` | PASS，保留长度和镜像三项 warning |
| 严格 `validate_config.py` | 预期失败：SFT/GRPO 长度未冻结 |
| call-based SFT 重验 | PASS，5/5 |
| historical standard-input SFT 重验 | 预期失败，4/5；准确检出 `taco_b74ba2ec45` |
| 冻结 smoke manifest | PASS，SHA-256 与记录数一致 |
| 密钥扫描 | PASS，无硬编码 key/token 命中 |
| 个人绝对路径扫描 | PASS，无工作区/用户名路径命中 |
| Docker verifier 实测 | PASS，受限合同全部通过；证据见第 8 节 |
| 代理 SFT/GRPO dry-run | BLOCKED，训练依赖尚未安装 |

本机复验结论：**G0 静态/P0 与 Docker verifier 子阶段通过；完整 G0
仍未通过。** 下一任务是冻结 RTX 4060/CUDA 依赖并运行代理模型 dry-run。

Git 基线随后已完成：私有远端 `Xiao0731/CodeGuide` 的旧提交通过非强制
merge 保留，本地与远端 `main` 同步至 `af2a0fb`。下一阻塞现为 CUDA lock
和代理模型 dry-run。

## 8. 2026-07-28 Docker verifier 真实运行验收

冻结镜像：

```text
python:3.11.9-slim-bookworm@sha256:8fb099199b9f2d70342674bd9dbccd3ed03a258f26bbd1d556822c6dfc60c317
```

验收命令：

```powershell
.\.venv\Scripts\python.exe scripts\validate_docker_verifier.py `
  --image $image --timeout 2 --concurrency 4 `
  --output artifacts\g0\docker_verifier_report.json
```

首轮真实无限循环测试发现一个执行链路缺陷：宿主侧
`subprocess.run(..., timeout=...)` 超时只终止 `docker run` 客户端，容器仍在
后台运行。修复后每次执行都有唯一名称和标签，并在 `finally` 中执行
`docker rm --force`。CPU 限制导致的退出码 137/152 也被明确归类为
`timeout/resource limit`。

最终结果：

| 检查 | 结果 |
|---|---|
| digest 固定、浮动 tag fail closed | PASS |
| standard-input | PASS，2/2 |
| call-based 顶层函数 | PASS，2/2 |
| call-based `Solution` 方法 | PASS，2/2 |
| wrong answer 检出 | PASS |
| `network=none` | PASS |
| 非 root、只读根目录、受限 `/tmp` | PASS |
| cap drop / no-new-privileges | PASS |
| 256 MiB memory+swap、1 CPU、64 PIDs、CPU ulimit | PASS |
| timeout 分类与强制清理 | PASS |
| 4 路并发与运行后无泄漏 | PASS |
| 现有 call-based SFT smoke 经 Docker 重验 | PASS，5/5 |
| 连续 3 次无限循环后的残留容器 | 0 |
| 全部单元测试 | PASS，44 passed |

机器可读报告为 `artifacts/g0/docker_verifier_report.json`，其中
`all_passed=true`。该结果支持“受限 Docker 执行合同已在本机实测通过”，
不支持“经过第三方安全审计的安全沙箱”这一更强表述。未运行 API 蒸馏、
SFT、GRPO 或模型训练。
