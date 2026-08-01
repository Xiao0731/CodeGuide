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
