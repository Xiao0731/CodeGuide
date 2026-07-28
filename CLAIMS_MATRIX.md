# CodeGuide Claims Matrix

> 依据：`CodeGuide_后训练项目实施与验收规划书_v1.2.md`。  
> 用途：防止把“代码存在”写成“已跑通”，或把小样本 smoke 写成总体结论。  
> 状态仅使用：未实现 / 已实现 / 已跑通 / 已验证 / 阻塞。

| Claim | 当前状态 | 证据 | 可对外表述 | 当前禁止表述 |
|---|---|---|---|---|
| TACO train 去重后有 24,237 题 | 已跑通 | full reference cache 与加载日志 | 已完成 TACO train 去重统计 | 所有题均适合训练 |
| 18,733 题有 Python-like reference 候选 | 已跑通 | full reference cache | 已解析 Python-like 候选 | 这些 reference 全部正确 |
| 当前 verifier 下有 10,415 题存在全测试通过 reference | 已跑通 | `data/cache/taco_reference_verification_train_full.jsonl` | 获得 10,415 道高可信候选 | 已获得 10,415 条最终 SFT 数据 |
| 多候选回退额外救回 960 题 | 已跑通 | candidate results cache | 多候选机制提高了 reference 可用题数 | 回退机制保证 reference 绝对正确 |
| `reference_guided_label` 不向 student user 泄漏 reference | 已跑通（smoke） | 两份 reference-guided smoke | smoke 中未发现泄漏 | 全量数据泄漏率为 0 |
| reference-guided 比 scratch 更值得扩大 | 已跑通（小样本） | scratch 0/4；guided 4/5 | 小样本显示明确正向信号 | 显著提升、稳定达到 80% |
| call-based 接口约束有效 | 已跑通（5 条 smoke） | call-based smoke 5/5 | 5 条专项 smoke 全通过且无接口错 | 全量 call-based 已解决 |
| 统一 `verify_code()` 支持 standard-input 与 call-based | 已实现并在 reference/smoke 中跑通 | verifier tests、cache、smoke | 两种 I/O 链路已用于离线验证 | verifier 是安全沙箱 |
| 正式容器安全执行 | 已实现，未在目标环境验证 | `src/reward/execution.py`、G0 合同测试 | 已实现 fail-closed 的受限 Docker 合同 | Docker 安全执行已跑通或是安全沙箱 |
| teacher 错误代码不会进入 accepted SFT | 已实现并通过本机单测 | `scripts/build_sft_dataset.py::_is_accepted_verification`、42 项测试 | 生成管线设置并测试了全测试通过硬门槛 | 已正式生成约 10K 高质量 SFT 标签 |
| 正式 SFT 数据生成与两轮修复闭环 | 未实现 | 无 300-500 pilot 报告 | 尚未开始正式 pilot | 已有约 10K 高质量 SFT 标签 |
| QLoRA SFT 训练闭环 | 未验证 | 旧训练代码存在，无 v1.2 G6 证据 | 训练入口待 G0 dry-run | SFT 已提升教学质量 |
| GRPO 唯一正式入口与 online/held-out 拆分 | 已实现并通过本机单测，待代理 dry-run | `scripts/train_grpo.py`、`src/training/grpo_train.py`、42 项测试 | 入口和确定性测试拆分已实现 | GRPO 已训练完成或有效 |
| Teaching Reward 真正进入梯度 | 未实现/未验证 | 当前仅有旧 reward 代码 | TeachingCritic 尚待构建 | Teaching Reward 有效 |
| anti-hacking contract reward 进入正式训练 | 未验证 | 代码与正式入口尚未完成接线验收 | 待 integration test | 已解决 reward hacking |
| best checkpoint 基于独立冻结 dev 严格 Pass@1 | 已实现，待代理 dry-run | `src/training/grpo_train.py`、独立 `grpo.eval_data` 合同 | checkpoint 选择合同已修复 | callback 已在目标训练环境跑通 |
| ExplainBench / TutorBench | 未实现 | 仅规划设计 | 评测集待冻结 | 已验证模型会交互式教学 |
| Base / SFT / GRPO 最终对比 | 未开始 | 无冻结输出 | 尚无最终模型实验结论 | 显著优于基座或外部方案 |
| 仓库 G0 | 阻塞 | 私有远端已同步至 `af2a0fb`；Docker/CUDA/代理 dry-run 未完成 | 已建立并推送静态可信基线 | 仓库开箱即复现或 G0 已通过 |
| GPT 代码包包含项目治理文档且排除敏感大文件 | 已跑通 | ZIP 84 条目检查，必需入口全存在，排除项命中 0 | 当前代码包已通过内容清单检查 | 所有未来代码包天然正确，无需复验 |

## 更新记录

### 2026-07-28

- 首次建立 claims matrix。
- 按 v1.2 将历史成果、小样本 smoke、现有代码和未闭环能力分开标记。
- GPT 打包清单改为自动纳入根目录 Markdown；该变更不改变任何模型能力 claim。
- 本次更新不产生新的模型或数据实验结论。
- 导入 G0/D1 修复快照后，将 Docker、teacher hard gate、GRPO 测试拆分和
  checkpoint 条目更新为“已实现，待目标环境验证”；G0 仍为阻塞。
- 首个可信 Git 基线尚未发布；GitHub CLI 缺失是当前外部阻塞。
- Windows/Python 3.11.9 本机复验完成：compileall、42 项测试、call-based
  5/5 和 manifest 通过；Docker daemon 与训练栈未就绪，因此 G0 仍为阻塞。
- 选定复用 `Xiao0731/CodeGuide`；发现其当前为 Public 且含一个旧 commit，
  私有化、远端历史审计和非强制合并尚未完成。
- 远端现已 Private；旧提交 `e6c7bae` 经扫描后通过 `af2a0fb` 非强制 merge
  保留，本地/远端 `main` 已一致。该证据只覆盖基线后的 provenance。
