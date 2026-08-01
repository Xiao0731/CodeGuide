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
