# CodeGuide Claims Matrix

> 用途：防止把“代码存在”写成“已跑通”，把小样本诊断写成总体结论，或把格式学习写成真实教学能力。  
> 状态仅使用：未实现 / 已实现 / 已跑通 / 已验证 / 阻塞。  
> 最近更新：2026-08-05。

| Claim | 当前状态 | 证据 | 可对外表述 | 当前禁止表述 |
|---|---|---|---|---|
| TACO train 去重后有 24,237 题 | 已跑通 | full reference cache 与加载日志 | 已完成 TACO train 去重统计 | 所有题均适合训练 |
| 18,733 题有 Python-like reference 候选 | 已跑通 | full reference cache | 已解析 Python-like 候选 | 这些 reference 全部正确 |
| 当前 verifier 下有 10,415 题存在全测试通过 reference | 已验证 | full verification cache、source bank、Docker 抽样复验 20/20 | 获得 10,415 道当前执行合同下的高可信 reference | reference 在所有可能测试上绝对正确 |
| 多候选回退额外救回 960 题 | 已验证 | candidate results cache | 多候选机制比只取第一候选多获得 960 道可用题 | 回退机制保证 reference 绝对正确 |
| 正式 SFT 标签生成完成 | 已验证 | 10,415 候选中 accepted 10,340、rejected 75；accepted SHA256 | 已完成 10,340 条全测试通过硬过滤的教学标签 | 10,340 条都具有同等教学质量 |
| student user 不泄漏完整 reference solution | 已验证 | accepted 全量字段审计、A/B 构造合同 | 正式 accepted 数据未把 reference solution 字段暴露给 student user | 当前 user prompt 不含任何人工接口元数据 |
| canonical SFT 数据与固定划分完成 | 已验证 | canonical 10,306；train/dev=9,791/515；固定 manifest/hash | 已冻结 10,306 条 canonical 与固定划分 | 所有 accepted 都进入了正式训练 |
| 8K 长度选择有全量审计依据 | 已验证 | P50/P95/P99/max=2,603/5,062/6,728/8,173；4096 截断 13.10%，8192 截断 0 | 正式 SFT 使用经全量 tokenizer 审计选择的 8192 长度 | 8K 对所有未来训练任务都最优 |
| 统一 `verify_code()` 支持 standard-input 与 call-based | 已验证 | reference cache、标签生成、本地 Docker 验收、TACO 评测 | 两种 I/O 链路已用于正式数据与冻结评测 | verifier 是经过第三方安全审计的通用沙箱 |
| 正式受限 Docker 执行合同 | 已在本机实测通过 | `scripts/validate_docker_verifier.py`、`artifacts/g0/docker_verifier_report.json` | digest 固定、隔离、资源限制、超时清理和并发合同已在本机 Docker Desktop 验证 | 在所有宿主环境均安全或经过第三方审计 |
| 双 RTX 4090、8K QLoRA SFT 训练闭环 | 已验证 | 612/612 steps、train/dev loss、adapter 文件、manifest、独立重载生成 | 正式 SFT 的训练、保存、重载和推理闭环已经完成 | loss 下降等同于教学或代码能力提升 |
| TACO-100 checkpoint 轨迹 | 已验证 | Base、step50、step20、step200 冻结生成与 Docker 报告 | 不同 I/O 模式出现不同 checkpoint 偏好 | TACO-100 足以决定最终 checkpoint |
| TACO-515 Base 严格 Pass@1 为 80/515 | 已验证 | 冻结 generation、Docker verification、汇总报告 | 当前提示协议下 Base Overall=15.53% | 该数字等同于官方 TACO 基准成绩 |
| TACO-515 step200 严格 Pass@1 为 128/515 | 已验证 | 冻结 generation、Docker verification、汇总报告 | step200 Overall=24.85%，相对同协议 Base 增加 48 题、约提升 60% | step200 已经是最终 SFT_CHAMPION |
| step200 的 standard-input 明显提升 | 已验证 | 17/393 → 71/393 | Standard 从 4.33% 提升到 18.07%，约为 Base 的 4.18 倍 | 已证明所有标准输入外源任务都提升 |
| step200 的 call-based 存在退化风险 | 已验证 | 63/122 → 57/122；step20=64/122 | 训练后期可能牺牲部分 Call 能力，需外源 EvalPlus 复核 | 已证明发生广泛灾难性遗忘 |
| TACO-515 配对显著性 | 未完成 | 当前仅有比例与粗略 Wilson 区间 | 正式配对统计待 McNemar、paired bootstrap 与 Holm 校正 | 已完成论文级显著性检验 |
| 当前文本 I/O 路由器 100% 准确 | 已验证为无效负结果 | forbidden literal scan 失败、41,224 标签命中、top features 直接含 `io_mode` | 路由实验暴露了人工元数据泄漏，已退出主线 | 文本路由器真实达到 100% 准确率 |
| 当前 TACO-515 为自然题面接口推断评测 | 不成立 | user prompt 明示 `io_mode`、`fn_name` 与 starter metadata | 当前结果是在显式接口元数据下的代码生成成绩 | 模型只看自然题面自主判断接口并取得 24.85% |
| clean-interface 评测 | 已设计，未跑通 | 已冻结删除人工 `io_mode/fn_name` 的原则 | 将先在 TACO-100 做独立提示词消融 | clean track 已完成或与现有 515 等价 |
| Standard/Call 双 LoRA 专家训练 | 已跑通，选优进行中 | 专家 adapter、模式纯净在线探针；Standard 尚余 13 个探针 | 双专家用于模式解耦诊断 | 双专家 Oracle 已经优于 Mixed |
| 双专家 Oracle-515 | 未完成 | 计划 Standard→393、Call→122 | 结果待两个 best checkpoint 冻结后生成 | 已完成可部署专家路由系统 |
| 最终系统采用单一 Mixed Adapter | 已决策 | 部署、泛化、路由泄漏与维护成本分析 | 双专家只做 Oracle 消融，正式 GRPO 使用 Mixed | 最终系统依赖 gold `io_mode` 路由 |
| HumanEval(+)/MBPP(+) 外源评测协议 | 已实现 | EvalPlus 0.3.1 配置、生成脚本、本地 Docker 评分与汇总脚本 | 已冻结 Base/step20/step200 的代码能力评测协议 | 外源结果已经生成 |
| HumanEval(+)/MBPP(+) 结果 | 阻塞 | 官方 release 数据尚待本地上传；三次失败均发生在第一题生成前 | 当前没有任何 EvalPlus 模型结果 | 网络或数据 schema 失败属于模型能力失败 |
| 教学结构学习 | 已有部分证据 | calibration/full 的 `template_complete`、生成样例 | SFT 学会了部分教学栏目与表达风格 | 已证明讲解技术正确、初学者友好或显著优于 Base |
| 教学客观格式全量审计 | 已设计，未完成 | 已冻结代码块、栏目、顺序、非空、冗余与截断指标 | 将复用已有 515 generations，无需重新生成 | 已完成教学格式总评 |
| TACO-100 教学盲评 | 已设计，未完成 | 已冻结评价维度与全样本/双方通过交集两种口径 | 将比较 Base、step20、step200 | 已验证教学质量显著提升 |
| GRPO 唯一正式入口与冻结 train/dev 拆分 | 已实现，待正式 smoke | TRL `GRPOTrainer` 入口、6,451/50 freeze manifest、参数化合同测试 | 入口、SFT adapter 热启动和冻结拆分已静态验证 | 正式 7B GRPO 已训练完成 |
| Teaching Reward 真正进入梯度 | 未实现/未验证 | 当前可靠教学 critic 尚未准入 | 现阶段教学启发式只作诊断 | Teaching Reward 已有效提升教学质量 |
| anti-hacking contract reward 进入正式训练 | 未验证 | 代码存在，待 GRPO smoke 与集成验收 | 正式 smoke 后再更新状态 | 已解决 reward hacking |
| best checkpoint 基于独立冻结 dev 严格 Pass@1 | 已实现，待 GRPO smoke | held-out 合同与 checkpoint 逻辑 | checkpoint 选择合同已修复 | callback 已在正式 GRPO 跑通 |
| Base / SFT / GRPO 最终双模块对比 | 未开始 | GRPO 尚未训练，教学协议未执行 | 尚无最终模型实验结论 | 显著优于基座或外部方案 |
| GitHub 项目记录使用中文提交与中文注释 | 已决策并开始执行 | 2026-08-05 后中文 commit 与新增中文注释 | 后续提交和注释优先中文 | 修改第三方固定接口名称以追求全中文 |
| GPT 代码包包含治理文档且排除敏感大文件 | 已跑通 | ZIP 清单检查、必需入口存在、排除项命中 0 | 当前代码包已通过内容清单检查 | 所有未来代码包天然正确，无需复验 |

## 更新记录

### 2026-07-28

- 首次建立 claims matrix。
- 将历史成果、小样本 smoke、现有代码和未闭环能力分开标记。
- 私有 Git 基线、Docker verifier 和治理文档完成初步验收。

### 2026-08-01 至 2026-08-02

- 正式标签生成定版为 accepted 10,340、canonical 10,306。
- 冻结 8K 长度和 9,791/515 train/dev。
- 完成双 4090 正式 QLoRA SFT、adapter 保存和独立重载。
- 固定 40 题显示教学结构学习，但未证明 full Adapter 代码正确率提升。

### 2026-08-04 至 2026-08-05

- 增加 TACO-100 checkpoint 轨迹和 TACO-515 严格执行结果。
- step200 在 TACO-515 上从 15.53% 提升到 24.85%，但 Call 从 51.64% 降到 46.72%；step20 保持到 52.46%。
- 文本 I/O 路由器 100% 结果经审计确认为标签泄漏，退出主线。
- 双专家重新定位为 gold-mode Oracle 消融，最终系统仍采用单一 Mixed Adapter。
- 评测正式拆成代码能力与教学能力两个模块。
- EvalPlus 外源协议已实现，但官方数据尚未完成云端验收，因此结果保持为空。
- GRPO 仍未启动；需先冻结唯一 SFT_CHAMPION、教学基线和 reward smoke。

### 2026-08-15

- 训练主线收敛到 TRL 0.22.2 + Accelerate + PEFT/bitsandbytes；删除手写训练循环和重复 reward 实现。初次重构曾误用 0.19.1，现已纠正。
- SFT/GRPO 数据数量、配置和 CLI 已完成本地合同验证，但重构后的入口尚未在云端 GPU 重新运行，因此不新增训练效果声明。
- 正式历史 generation、Docker verification 和报告继续作为实验结果；删除的是一次性入口与被替代的中间产物，不是实验结论。
- GRPO formal config 已静态锁定为 6,451 train、dev50 选优、TACO-515 最终评测；正式三阶段 curriculum 和 composite reward 均有测试，但尚未形成新的 GPU 训练结果，因此只能声明“实现与合同验证通过”。
