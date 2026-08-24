# CodeGuide 实验、决策与故障记录

> 作用：持续记录后续开发中的实验对比、失败现象、根因、应对措施和可复算证据，供项目复盘、论文写作与面试讲解使用。  
> 最高优先级依据：`CodeGuide_后训练项目实施与验收规划书_v1.2.md`。  
> 当前阶段：G0 仓库可信 / P0 修复。  
> 状态口径：`已实现`、`已跑通`、`已验证`必须严格区分。

### EXP-009: reference-locked comment-plan recovery

- Date: 2026-07-29.
- Decision: preserve all accepted Class-A labels. Route each new A failure and
  each historical rejected item to one Class-B request.
- API ceiling: A success uses one request; A failure uses one A plus one B;
  historical rejected uses one B. Same-mode retries are disabled.
- Class B generates teaching text and a JSON comment plan. The program injects
  comments into the verified reference at AST-safe statement lines.
- Acceptance requires comment-stripped executable-token equality, valid
  syntax, and full Docker `verify_code()` success.
- Recovery records are versioned so legacy failures receive one improved
  attempt and version-2 failures do not loop.
- Evidence: 15 tests passed. First four live refined recoveries had pass rate
  1.0, token equivalence true, and 7/10/8/11 inserted comments.
- Resume state: concurrency 3, 1,343 accepted preserved, 204 historical
  rejected prioritized, 9,072 total pending.

### DEC-010: raise production teacher concurrency to 1,000

- Date: 2026-07-29.
- Account limit supplied by the operator: 2,500 concurrent requests for
  `deepseek-v4-flash`.
- Production wrapper default changed from 3 to 1,000 API requests.
- Safety boundary: the API semaphore does not wrap Docker verification;
  synchronous verification prevents 1,000 simultaneous local containers.
- Operational trade-off: generation and API spend can advance very quickly.
  JSONL flush and ID-based resume remain mandatory.
- Observation: the SDK still retried transport failures despite
  `distill_retries=1`. It is now constructed with `max_retries=0`, preventing
  hidden same-request API consumption.

### DEC-011: roll concurrency back to 20 after timeout amplification

- Date: 2026-07-29.
- Evidence: 1,999 logged HTTP 200 responses, 454 request timeouts, and only
  about 1,478 newly accepted records in the observed run.
- Root cause: synchronous Docker verification blocks the asyncio event loop,
  delaying receipt of already-running API responses.
- Decision: pause generation, reduce production API concurrency from 1,000 to
  20, and do not resume automatically until the operator confirms.
- The service limit of 2,500 remains documented as an account ceiling only.

### DEC-012: retry external Class-B API failures after recharge

- Date: 2026-07-29.
- Trigger: HTTP 402 `Insufficient Balance` ended production with 3,298
  `recovery_llm_failed` records.
- Decision: teacher/API failures remain retryable on a later resume. Code and
  execution failures remain terminal.
- Cost boundary: affected records enter Class B directly; accepted labels and
  prior Class-A requests are never regenerated.
- Follow-up: the first insufficient-balance response now trips a batch circuit
  breaker instead of exhausting the remaining queue with guaranteed failures.

### DEC-013: rejected JSONL represents unresolved current state

- Date: 2026-07-29.
- The previous append-only file mixed historical failures with recovered IDs
  and grew to about 375 MB.
- Decision: keep one compact record per unresolved ID, remove an ID
  atomically as soon as it is accepted, and compact accepted IDs at startup.
- Historical request evidence remains in process logs; rejected is now a
  current work queue/status artifact.

## 1. 记录规则

每个有效步骤至少记录：

```text
记录 ID：
日期：
阶段 / Gate：
类型：实现 / 实验 / 决策 / 故障 / 文档
目标与验收阻塞：
是否改变数据、模型、reward、split 或评测口径：
是否消耗云端时间：
配置与输入：
实际命令：
实际结果：
错误现象：
根因：
应对措施：
证据文件：
结论边界：
状态：已实现 / 已跑通 / 已验证 / 阻塞
下一步：
```

失败记录不得删除。修复后追加结果，并保留原始错误、根因和修复前后差异。

## 2. 已有实验与故障基线

### EXP-001：TACO train 多候选参考解离线验证

- 日期：2026-05 至 2026-07，按现有缓存与此前运行记录整理
- 阶段：数据准备基线，G0 前已有成果
- 类型：实验
- 目标：避免第一份 Python reference 失败时错误放弃整道题
- 输入：TACO train 去重后 24,237 题
- 实现：
  - loader 保留并排序多个 Python-like 候选；
  - 每题最多依次验证多个候选；
  - 首个全通过候选立即选中；
  - 使用统一 `verify_code()`；
  - 支持并行、断点续传、逐行 flush 和失败分类
- 实际结果：
  - 有 Python 候选：18,733
  - 第一候选直接通过：9,455
  - 多候选回退救回：960
  - 最终全通过 reference：10,415
  - 无 Python reference：5,504
  - 平均尝试候选数：1.348
- 主要失败：
  - `wrong_answer`：4,444
  - `runtime_error`：2,818
  - `unsupported`：607
  - `timeout`：282
  - `interface_mismatch`：153
  - `syntax_error`：14
- 结论：
  - 多候选回退确实增加了 960 道可用题；
  - 10,415 是“通过当前 verifier 全部可用测试”的候选数，不等于最终 accepted SFT 数；
  - 当前执行器仍是受限子进程，不能称为安全沙箱。
- 证据：
  - `data/cache/taco_reference_verification_train_full.jsonl`
  - `scripts/verify_taco_references.py`
  - `src/reward/execution.py`
- 状态：已跑通；统计结果可复算

### EXP-002：scratch 分层 smoke4

- 类型：实验
- 目标：观察 teacher 不看 reference 时的标签代码正确性
- 输入：easy / medium / hard / very_hard 各 1 条
- 实际结果：0/4 生成代码全测试通过
- 主要错误：
  - call-based 样本把 `special_number` 改成 `specialNumber`；
  - standard-input 高难题存在真实算法错误；
  -部分样本虽语法正确、讲解完整，但代码测试为 0。
- 根因：
  - teacher 需要从零解题；
  - call-based 接口约束在旧 prompt 中不充分；
  - 语法与表面质量过滤无法保证算法正确。
- 应对：
  - 引入 teacher 私有 verified reference；
  - 将 `io_mode`、`fn_name`、`starter_code` 显式加入接口约束；
  - 生成代码统一用 `verify_code()` 计算通过率。
- 证据：`data/sft_train_smoke4.jsonl`
- 结论边界：样本只有 4 条，只能用于发现问题，不能估计总体通过率。
- 状态：已跑通

### EXP-003：reference-guided 分层 smoke

- 类型：实验
- 目标：验证 teacher 私有 reference 是否值得扩大
- 输入：easy / medium / medium_hard / hard / very_hard 各 1 条，reference 全部预先验证通过
- 实际结果：
  - 5 条成功生成并写入；
  - 4/5 生成代码全测试通过；
  - reference 未泄漏到 `messages.user`；
  - 无语法错误和明显截断。
- 失败样本：
  - `taco_b74ba2ec45`
  - expected：`4\n0\n2`
  - actual：`3\n0\n2`
  - 类型：`wrong_answer`
- 根因：teacher 在可读性改写时改变了 reference 的算法语义，不是执行器或接口问题。
- 应对：
  - 强化 teacher prompt：必须忠实遵循 reference 核心算法；
  - 允许改注释和可读性，不允许改算法语义；
  - 失败标签不得进入正式 accepted SFT。
- 证据：`data/sft_train_smoke_ref_label.jsonl`
- 结论边界：4/5 仅说明方向有效，不能声称总体 80%。
- 状态：已跑通

### EXP-004：reference-guided call-based 专项 smoke

- 类型：实验
- 目标：检查函数名、starter contract 与 reference 泄漏
- 输入：5 条 `reference_verified=true` 且 pass rate 为 1.0 的 call-based 题
- 实际结果：
  - 5/5 生成代码全测试通过；
  - 5/5 严格声明所需 `fn_name`；
  - 5/5 遵守 starter contract；
  - 0 reference 泄漏；
  - 0 `interface_mismatch`；
  - 0 语法错误、截断和执行错误。
- 证据：`data/sft_train_smoke_ref_label_call_based.jsonl`
- 结论边界：专项样本量为 5，不能直接外推全量 call-based 通过率。
- 状态：已跑通

### INC-001：PowerShell 将 Python stderr 日志包装为 NativeCommandError

- 类型：故障
- 现象：DeepSeek 请求主流程仍在运行，但 PowerShell 报
  `python.exe ... NativeCommandError`。
- 根因：PowerShell 5.1 对 native stderr 经管道传给 `Tee-Object` 的包装行为，不等价于 Python 进程失败。
- 应对：以 `$LASTEXITCODE` 判断真实退出状态，并允许日志继续输出。
- 结论：这是脚本输出通道问题，不是蒸馏 API 或 Python 主流程失败。
- 状态：已修复并在后续 smoke 中跑通

### INC-002：本地 TACO 加载时出现 Hugging Face HEAD 404

- 类型：故障
- 现象：日志出现对
  `datasets/parquet/parquet.py` 的 HTTP HEAD 404。
- 根因：`datasets` 内部模块探测产生的网络日志；随后仍从本地 Parquet 加载了 24,237 条去重题目。
- 应对：以最终本地加载结果和退出码判断；后续环境冻结应进一步避免无必要网络探测。
- 结论：该次运行中不是数据加载失败，但离线复现仍需治理。
- 状态：主流程已跑通，离线依赖治理待 G0 完成

### INC-003：smoke report 临时 Python 文件编码损坏

- 类型：故障
- 现象：临时脚本中的中文字符串变成乱码并触发
  `SyntaxError: unterminated string literal`。
- 根因：PowerShell 写临时 Python 文件时编码不一致。
- 应对：按 UTF-8 无 BOM 写入临时脚本。
- 结果：后续 call-based smoke report 正常输出 5 条完整报告。
- 状态：已修复并跑通

### INC-004：GPT 代码包遗漏中文文件名的 v1.2 规划书

- 日期：2026-07-28
- 阶段：G0
- 类型：故障
- 现象：
  - `scripts/package_gpt_context.ps1` 的源代码已列出规划书；
  - ZIP 检查却显示
    `CodeGuide_后训练项目实施与验收规划书_v1.2.md` 不存在；
  - 其他 ASCII 文件名的治理文档均正常进入 ZIP。
- 根因：Windows PowerShell 5.1 对 `.ps1` 中非 ASCII 字符串字面量的解码不稳定，
  导致 `Test-Path -LiteralPath` 实际检查了错误路径。
- 最小修复：不再硬编码中文文件名，改为自动收集项目根目录下全部 `*.md`。
- 验收：
  - 命令：
    `powershell -ExecutionPolicy Bypass -File scripts\package_gpt_context.ps1 -OutputPath codeguide_gpt_context.zip`
  - 重新生成根目录 `codeguide_gpt_context.zip`；
  - 检查规划书、上下文、实验日志、claims matrix 和打包脚本五个入口；
  - 再检查敏感目录和全量缓存未进入压缩包。
- 实际结果：
  - ZIP 条目数：84；
  - 五个必需入口全部存在；
  - 敏感/排除路径命中数：0；
  - 解压后总大小：1,368,745 bytes。
- 结论：这是打包清单的编码问题，不是规划书文件损坏。
- 状态：已修复并跑通验收

## 3. 当前决策

### DEC-001：v1.2 规划书成为最高优先级项目约束

- 日期：2026-07-28
- 阶段：G0
- 类型：决策 / 文档
- 目标与阻塞：避免后续按陈旧 README、旧配置或未验证功能声明推进
- 决策：
  - `CodeGuide_后训练项目实施与验收规划书_v1.2.md` 高于旧 README、配置与 Notebook；
  - 当前阶段固定为 G0/P0 修复；
  - 未通过上一 Gate 不进入下一阶段；
  - 后续计划调整必须先更新规划书；
  - 每个有效步骤同步更新 `GPT_PROJECT_CONTEXT.md` 与本文件；
  - claim 证据变化时同步更新 `CLAIMS_MATRIX.md`。
- 是否改变核心口径：不改变数据、模型、reward、split 或评测口径
- 云端消耗：无
- 实际检查：
  - 完整阅读规划书全部 19 节；
  - 更新项目上下文；
  - 建立本日志和 claims matrix；
  - 更新 GPT 打包清单。
- 状态：已实现
- 下一步：按规划书第 18 节执行 P0 仓库审计与修复，不启动 pilot 或训练。

### DEC-002：采用任务级 Git 提交与 Gate 审核工作流

- 日期：2026-07-28
- 阶段：G0
- 类型：工程治理
- 决策：
  - 当前修复版作为首个可信基线，后续每个明确任务单独 commit 并 push；
  - commit 使用 Conventional Commits 风格；
  - 每次汇报分支、commit、完成内容、改动文件、测试、未通过项和下一步；
  - 大阶段通过后才打里程碑 tag，并只在里程碑生成一次 ZIP；
  - API key、`.env`、checkpoint、正式生成数据、全量缓存、W&B 缓存、
    大日志和个人路径不得进入 Git。
- 外部阻塞：当前机器已安装 Git，但未安装 GitHub CLI `gh`，因此尚不能创建
  私有远端仓库或认证推送。
- 状态：本地规则已实现，GitHub 发布阻塞

### DEC-003：G0/D1 可信性修复合同

- 日期：2026-07-28
- 阶段：G0/P0
- 类型：数据 / reward / split / 评测合同
- 输入快照：`CodeGuide_G0_D1_fixed_20260727.zip`，SHA-256
  `601DE8B1CF77B1FE97F3BD73E390C7E5A90C88F59877B2DC5543797A8A85AF6A`。
- 决策：
  1. `scripts/train_grpo.py` 是唯一公开 GRPO 命令；
  2. 无可执行测试时 correctness 为 0，AST 不代理语义正确性；
  3. GRPO 题至少 4 个测试，并按 SHA-256 确定性拆分 online/held-out；
  4. teacher 输出 unsupported、报错、无测试或非全通过时不得进入 accepted SFT；
  5. Docker 是正式执行后端，subprocess 只用于受信任的本地检查；
  6. TeachingCritic 准入前 reward 为 correctness 0.9 + contract 0.1；
  7. 主实验关闭额外 batch Z-score，只记录 `zero_advantage_ratio`；
  8. checkpoint 读取独立 `grpo.eval_data`，按整题全通过率计算 Pass@1；
  9. SFT 与 GRPO adapter 统一为 `r=16, alpha=32`；
  10. tokenizer 审计前 `max_seq_length` 保持未冻结且训练 fail fast；
  11. v1.2 的 SFT 起始配置固定为 `1e-4`、1 epoch、completion-only loss、
      length-grouped sampling。
- 隐私修复：环境快照只记录 Python 可执行文件名；早期 APPS 脚本改为相对路径/CLI。
- 云端消耗：无。
- 状态：代码已合并，当前机器复验待执行

### EXP-005：G0/D1 修复快照本机静态复验

- 日期：2026-07-28
- 阶段：G0/P0
- 环境：Windows 10、Python 3.11.9、RTX 4060 Laptop 8 GB。
- 实际命令与结果：
  - `python -m compileall`：PASS；
  - `python -m pytest -q`：42 passed in 4.86s；
  - `bash -n`：PASS；
  - `validate_config.py --allow-unfrozen`：PASS with 3 warnings；
  - 严格配置验证：按设计因两个长度未冻结而失败；
  - call-based 标签重验：5/5；
  - historical standard-input 标签重验：4/5，准确检出已知错误标签；
  - manifest SHA-256/count：PASS；
  - 密钥与个人绝对路径扫描：0 命中。
- 环境阻塞：
  - RTX 4060 可见，驱动 576.40；
  - Docker CLI 已安装，但 Linux daemon 未运行；
  - 当前虚拟环境没有 torch/Transformers/TRL/PEFT/bitsandbytes/Unsloth/vLLM。
- 云端消耗：无。
- 结论：G0 静态/P0 子阶段通过；Docker verifier、CUDA lock 和代理模型
  SFT/GRPO dry-run 未通过，完整 G0 仍为阻塞。

### DEC-004：复用现有 GitHub 仓库并先完成私有化与历史审计

- 日期：2026-07-28
- 阶段：G0
- 类型：版本控制 / 发布治理
- 远端：`Xiao0731/CodeGuide`。
- 发现：
  - 远端当前为 Public；
  - 远端 `main` 已有 1 个旧 commit；
  - 本地可信基线 `b3fb3a8` 是独立 root commit；
  - 本地尚未配置 `origin`，GitHub CLI 尚未安装。
- 决策：
  - 不另建仓库，优先复用现有仓库；
  - 推送前先将仓库改为 Private；
  - `gh auth login` 后先 fetch 和审查远端旧提交；
  - 禁止直接 force-push，需保留可审计历史并显式解决无共同祖先问题；
  - 评审者通过 private repository collaborator 方式访问。
- 状态：仓库已选定；私有化、认证、远端历史合并和 push 待完成。

### EXP-006：可信基线 GitHub 私有仓库交付

- 日期：2026-07-28
- 阶段：G0
- 远端：`https://github.com/Xiao0731/CodeGuide`，Private。
- 远端旧提交：`e6c7bae`，仅含早期 APPS 下载/环境文件。
- 审计：
  - 旧提交密钥扫描 0 命中；
  - 旧提交个人绝对路径扫描 0 命中；
  - 本地与远端无共同祖先。
- 整合：
  - 使用 `ours` merge 保留旧提交历史；
  - merge commit：`af2a0fb`；
  - merge 前后工作树 diff 为 0；
  - 未使用 force-push。
- 复验：42 passed in 5.21s。
- 推送：普通 fast-forward push 成功，本地 `main` 与 `origin/main` 均为
  `af2a0fb368fd0e9e4722eded9b1c055900ff307e`。
- 状态：首个可信私有 Git 基线已交付；未打 G0 tag。

### EXP-007：Docker verifier 真实隔离与超时清理验收

- 日期：2026-07-28
- 阶段：G0
- 环境：Windows 10、Docker Engine 28.1.1、Linux/amd64 容器。
- 冻结镜像：
  `python:3.11.9-slim-bookworm@sha256:8fb099199b9f2d70342674bd9dbccd3ed03a258f26bbd1d556822c6dfc60c317`。
- 首轮发现：
  - standard-input 与 call-based 均通过；
  - 无限循环能触发超时，但本地 `docker run` 客户端被杀后，后台容器仍存活；
  - 根因是 `subprocess.run(..., timeout=...)` 只终止客户端进程，
    `--rm` 不会主动终止仍在运行的容器。
- 修复：
  - 每次运行使用唯一容器名和 `codeguide.verifier=true` 标签；
  - `finally` 中按唯一名称执行 `docker rm --force`；
  - 增加 `--pull never`、`--init`、`--memory-swap 256m`、CPU ulimit、
    `/tmp` 工作目录和 `TMPDIR=/tmp`；
  - CPU 限制导致的 137/152 退出统一记录为
    `container timeout/resource limit`。
- 真实验收覆盖：
  - digest 固定与浮动 tag fail closed；
  - standard-input；
  - call-based 顶层函数与 `Solution` 方法；
  - wrong answer；
  - `network=none`；
  - 非 root、只读根目录和受限 tmpfs；
  - CPU/内存/swap/PID 限制；
  - timeout 后无容器泄漏；
  - 4 路并发后无容器泄漏。
- 结果：
  - `scripts/validate_docker_verifier.py` 全部检查 PASS；
  - 单元测试 44 passed；
  - 现有 call-based SFT 标签通过 Docker 后端 5/5；
  - 连续 3 次无限循环均被终止，残留 verifier 容器 0。
- 证据：`artifacts/g0/docker_verifier_report.json`。
- 边界：可称“受限 Docker 执行合同已在本机实测通过”，不得称为经过
  第三方安全审计的“安全沙箱”。
- 云端/API/训练消耗：无。
- 状态：Docker verifier 子项通过；G0 仍被 CUDA lock 与代理模型
  SFT/GRPO dry-run 阻塞。

### EXP-008：rejected 根因抽检与 A/B 混合恢复

- 日期：2026-07-29。
- 抽样：从当时 203 条正式 rejected 中用 `seed=42` 固定抽取 100 条。
- 失败分布：84 条 wrong answer、15 条 runtime error、1 条 syntax error。
- 通过率分布：46 条为 0；7 条位于 `(0, 0.1)`；18 条位于
  `[0.1, 0.5)`；20 条位于 `[0.5, 0.9)`；9 条位于 `[0.9, 1.0)`。
- reference 差异：80 条已是不同算法或结构，18 条大幅改写，1 条少量
  可执行 token 修改，1 条接口变化，0 条仅修改注释或格式。
- token 相似度均值为 0.430，中位数为 0.417。
- verifier 旁路发现：至少 3 条首失败仅为空白差异；11 条 wrong answer
  带 constructive 标签，可能需要 checker-aware 解释，不能全部直接归因
  为算法错误。
- 决策：通过全部测试的自由教学化改写保留为 A 类；失败样本使用不可变
  verified reference 生成 B 类讲解；不直接把旧讲解与另一份代码拼接。
- 成本约束：A 类只请求一次；失败后只请求一次 B 类，不重复自由改写；
  已 accepted 样本永不重跑。
- 首批在线验证：2 条历史 rejected（runtime error、wrong answer）均恢复
  到 pass rate 1.0；模型修改了可执行 token，系统按预期注入原 reference。
- 状态：已实现并通过 13 项测试；正式任务以并发 3 恢复运行。
### DEC-014：rejected 快照采用唯一临时文件与有限替换重试

- **日期**：2026-07-30
- **背景**：20 并发正式续跑时，Windows 对共享 `.tmp` 到 rejected JSONL 的替换返回 `WinError 5`，生成进程退出。
- **决定**：每次 flush 使用 PID 与纳秒时间组成的唯一临时文件；仅对 `os.replace` 的短暂 `PermissionError` 做 6 次有限退避重试。
- **边界**：不增加 API 请求重试，不改变当前未解决 rejected 快照的业务定义。
### DEC-015：基础设施失败不得归类为 wrong answer

- **日期**：2026-07-30
- **证据**：410 条 B 类 `recovery_wrong_answer` 的错误文本全部为 Docker daemon 连接失败，Docker Desktop 服务处于 stopped。
- **决定**：Docker 生成前必须预检；连接失败使用 `docker_unavailable`，不记作代码错误，并允许断点恢复。
- **影响**：避免在 verifier 不可用时先付费生成讲解，也避免把基础设施故障错误计入模型代码失败率。
### DEC-016：以 10,340 条 accepted 快照进入 SFT 下一阶段

- **日期**：2026-08-01
- **审计范围**：10,415 条 verified TACO reference-guided 候选。
- **结果**：10,340 accepted，75 unresolved rejected，唯一 ID 全覆盖且无交叉；accepted 接纳率 99.28%。
- **失败隔离**：40 条教师响应失败/截断、28 条 verifier 镜像缺少 numpy、7 条超时。它们不混入训练集，也不阻塞已通过样本定版。
- **决定**：结束正式标签生成主线，以 accepted 快照进入数据冻结、划分和 SFT 训练准备。若后续回收 75 条，必须生成新版本 manifest，不静默修改当前快照。
- **复现标识**：accepted SHA-256 `CBFA65F00AD635654431004D8C20AD4BCB1D64EF6CFE7DB984331DB5F9E7042D`。

### DEC-021：4096 恢复轮后关闭“截断导致低通过率”假设

- **日期**：2026-08-02
- **证据**：固定 40 题在 4096 completion 上限下，Base/Adapter 均无撞限和未闭合代码围栏；Adapter Pass@1 仍为 7/40。
- **决定**：不再通过继续增大生成上限来解释或修复剩余失败。剩余 30 条 wrong answer 和 3 条 runtime/timeout 按模型能力问题记录。
- **阶段判定**：500 条校准已证明 SFT 对教学结构和严格代码正确率的方向为正；不将小样本 17.5% 误写为最终模型能力。

### DEC-026：多代码块使用接口感知的保守选择

- **日期**：2026-08-02
- **决定**：单代码块与已完整的最后代码块保持兼容；仅在最后一块为示例、语法错误或接口不完整时，按目标函数或 stdin/stdout 完整性选择其他块。
- **边界**：不使用测试结果反向挑选代码块，不加入题目 ID 特例。

### DEC-027：受限云容器默认禁用 NCCL SHM

- **日期**：2026-08-02
- **证据**：full SFT 在 DDP 包装阶段因 `/dev/shm/nccl-*` attach 失败退出，模型加载和双 rank GPU 绑定已通过。
- **决定**：双 4090 入口默认 `NCCL_SHM_DISABLE=1`，使 NCCL 避开故障共享内存通道；不同时禁用 P2P，避免无证据地扩大性能损失。

### DEC-028：SFT dev 评估只保留 loss

- **日期**：2026-08-02
- **证据**：full 训练前 25 steps 稳定，但 Trainer 评估的 `return_outputs=True` 在长样本上实体化全量 vocabulary logits 并 OOM。
- **决定**：设置 `prediction_loss_only=True`，保留全量 dev loss 而不收集 logits/predictions；不缩减 dev 或序列长度。
# DEC-017：冻结 10,306 条 canonical SFT 并采用 8K 正式训练长度

- **日期**：2026-08-01
- **输入**：10,340 条已接纳且具有执行通过证据的教学样本。
- **决定**：保留完整质量通过快照，同时将 34 条超过 8192 tokens 的极端样本隔离；正式 canonical 固定为 10,306 条，不对最终代码做右截断。
- **依据**：Qwen2.5-Coder-7B-Instruct 正式 chat template 下 canonical 最大长度 8,173；4096 会影响 13.10% 样本并主要伤及代码块，8192 下截断数为 0。
- **划分**：固定种子 20260728，SFT train/dev=9,791/515；未来 GRPO prompt ID=900/100，所有集合通过无泄漏与可复现 hash 检查。
- **溯源**：建立 10,415 条独立 verified source bank，抽样 20 条从压缩母库读取 reference/tests 后用固定 Docker verifier 复验，20/20 通过。
- **发布边界**：代码、manifest、报告与固定 ID 可进入 Git；canonical、source bank、原始 TACO、验证缓存及密钥不得进入 Git。
# DEC-018：SFT 使用显式 assistant-only labels 与标准 PEFT DDP

- **日期**：2026-08-01
- **决定**：废弃旧入口中随机切分和 `train_on_responses_only` 字符串模板猜测，改用冻结 ID、正式 chat template token 前缀边界和显式 labels。
- **训练栈**：Transformers + PEFT + bitsandbytes，普通双进程 DDP；每个 rank 显式加载一份4-bit模型到对应 GPU，不使用 `device_map=auto`，不允许CPU或全参数静默回退。
- **校准**：固定500条 train 与100条 dev sanity subset，先完成1 epoch云端校准，未过门槛不得启动9,791条 full SFT。
- **长度**：8192硬门槛，任何超长样本直接报错，不允许静默截断代码。
# DEC-019：CUDA 13 环境要求 bitsandbytes 0.48+ 并执行原生算子预检

- **日期**：2026-08-01
- **故障**：云端 `torch 2.9.0+cu130` 搭配 bitsandbytes 0.47.0，缺少 `libbitsandbytes_cuda130.so`；旧 preflight 因包导入时内部吞掉异常而错误报告通过。
- **决定**：CUDA 13 环境锁定 `bitsandbytes>=0.48.2,<0.49`，不为此重装 PyTorch；preflight 必须实际执行 NF4 Linear 前后向和 PagedAdamW8bit step。
- **结果边界**：双卡、canonical hash、split 已通过；只有新版原生探针通过后才允许开始校准训练。
# DEC-020：分布式启动器必须继承当前虚拟环境解释器

- **日期**：2026-08-01
- **故障**：激活 `.venv` 后直接调用 `torchrun`，PATH 命中 `/opt/conda/bin/torchrun`，两个 rank 使用 base Python，导致找不到仅安装在 venv 的 transformers。
- **决定**：所有入口统一使用 `python -m torch.distributed.run`；preflight 的双rank探针额外导入 transformers 并输出 `sys.executable`。
- **收益**：解释器不一致会在模型下载前 fail closed，且单卡/双卡均沿用用户当前激活环境。
# DEC-021：8K QLoRA 使用 Liger fused linear cross-entropy

- **日期**：2026-08-01
- **证据**：双 RTX 4090 在首个长 batch 的交叉熵/反向传播阶段 OOM；每卡模型已占约 18.6--18.9 GiB，额外需要约 3.7--4.3 GiB，且尚未完成 optimizer step。
- **决定**：不降低 8192 长度、不删除冻结长样本；启用 Transformers 原生 `use_liger_kernel`，并显式启用 `fused_linear_cross_entropy`，同时设置 CUDA expandable segments。
- **验证门槛**：preflight 必须实际完成 Liger fused loss backward；随后先完成一个最长样本 optimizer step，再运行完整 500 条校准。
- **边界**：当前仅完成故障修复，不能将失败运行表述为校准通过；若 fused loss 后仍 OOM，再依据实测峰值讨论 activation offload 等下一层措施。
# DEC-022：500 条 SFT 校准训练通过，进入 adapter 重载验收

- **日期**：2026-08-02
- **结果**：双 4090 完成 32/32 optimizer steps；平均 train loss 0.7215665，dev loss 0.6630970，无 OOM/NaN，证明 8K QLoRA + Liger 训练配置可运行。
- **判定**：训练执行子项通过；在 adapter 独立重载和真实生成成功前，完整校准 Gate 保持未关闭。
- **下一步**：使用保存目录 `outputs/sft/qwen25_coder_7b_qlora_8k/calibration_seed20260728/adapter` 运行重载 smoke，再决定是否进入 Base/Adapter 对照与 full SFT。
# DEC-023：adapter 重载通过后先做固定 dev 对照

- **日期**：2026-08-02
- **证据**：校准 adapter 已独立重载，并对固定样本生成 64 tokens 非空中文教学回答。
- **决定**：保存/重载 Gate 通过；下一步先比较 Base 与 Adapter 在固定 dev 子集上的教学结构、接口遵循和代码正确性，再决定是否启动 full SFT。
- **边界**：`do_sample=False` 时忽略 temperature/top-p/top-k 的提示不影响确定性重载结论。
# DEC-024：模型生成和 Docker 执行验证跨环境解耦

- **日期**：2026-08-02
- **背景**：双 4090 云端没有 Docker，也没有 source bank；强行安装会污染训练环境并重复传输大文件。
- **决定**：云端只运行 `stage=generate`，输出固定 Base/Adapter generations；本地运行 `stage=verify`，复用既有 source bank 和固定 Docker verifier。
- **一致性**：两个阶段由同一个 `selection.json` 和 problem ID 关联；验证只消费已落盘文本，不重新生成、不调用 API。
# DEC-025：对撞限回答做 4096-token 定向复验

- **日期**：2026-08-02
- **证据**：Adapter Pass@1 相对 Base 提升 7.5 个百分点，但 Adapter 有 8/40 回答恰好生成 2048 tokens、全部失败；Base 无撞限。
- **决定**：不重跑全部 80 份回答。复用所有低于旧上限、已自然结束的 generation，只对撞限回答以 4096 上限重生成，再执行相同 Docker 验证。
- **Gate**：定向复验完成前不启动 full SFT；最终报告同时保留 2048 首轮与 4096 恢复轮，禁止覆盖原始证据。

# DEC-026：SFT dev 评估使用显式 loss-only prediction_step

- **日期**：2026-08-02
- **证据**：`TrainingArguments(prediction_loss_only=True)` 后，云端 traceback 仍显示 Trainer 请求 `return_outputs=True` 并在完整 logits 交叉熵处 OOM。
- **决定**：不缩减 dev、不降低 8K 长度、不改变评估频率；通过 Trainer 子类在评估阶段强制 `return_outputs=False`，只聚合 loss。
- **验收**：同一 full 启动入口必须越过 step 25、产出 eval loss 并继续训练，方可确认该故障关闭。
- **实现修正**：Liger 评估前向显式传入 `skip_logits=True`；不通过递归 `model.train()` 激活 fused loss，以免 LoRA dropout 0.05 污染 dev loss。

# DEC-027：full SFT 训练完成后先做产物重载再进入质量评测

- **日期**：2026-08-02
- **证据**：固定 9,791/515 数据完成 612/612 steps，train loss 0.5693143，dev loss 0.5323175，训练与评估均无 OOM/NaN。
- **决定**：不重复训练；先检查 `full_seed20260728/adapter` 与 `run_manifest.json`，独立重载 adapter 完成确定性生成，再沿用校准阶段的云端生成、本地 Docker 验证流程评估 full Adapter。
- **边界**：在 adapter 重载成功前只宣称“full SFT 训练执行完成”，不宣称正式模型质量 Gate 已通过。
- **验收更新**：full Adapter 已独立重载并完成非空确定性生成，训练保存/重载 Gate 关闭；模型质量仍以固定 40 题本地 Docker Pass@1 为准。

# DEC-028：full SFT 工程通过但算法质量 Gate 暂不提升

- **日期**：2026-08-02
- **证据**：固定 40 题 Base/full Adapter Pass@1 均为 4/40；full 输出更长且具教学结构，但出现 2 条 4096-token 撞限。
- **决定**：保留 full Adapter 作为已完成的 SFT 产物，但不将 loss 下降或教学格式学习等同于算法正确率提升；后续决策需分别报告 pedagogical metrics 与 execution Pass@1。
- **评测完整性**：保留本次原始 generation、verification 与 comparison report，不重新生成或覆盖以追求更高分数。

# DEC-029：不按训练进度推断 A/B 数据阶段

- **日期**：2026-08-03
- **证据**：calibration 500 本身含 163 条 B 类；full 数据经固定 ID 重排、DDP 随机采样和长度分桶，epoch 百分比不对应原始 accepted 行号。
- **决定**：不将 65% loss 平台归因于后段 B 类；后续若验证“早期 A 类质量更高”，必须做 A-only 与 matched mixed 的等样本、等 optimizer-step、等评测集消融。
- **优先控制变量**：固定样本数、更新步数和学习率日程后再比较 label strategy，避免把 32 对 612 steps 的训练剂量差异误认为数据类别效应。

# DEC-030：区分外部生成上限与模型可见的回答预算

- **日期**：2026-08-03
- **决定**：评测继续保留 4096 上限以避免机械截断，不把降低 `max_new_tokens` 当作提升正确率的手段。
- **依据**：模型看不见推理 API 的 `max_new_tokens`；降低上限只会截断同一 greedy 前缀，不能让模型自动提前思考或写代码。
- **后续方向**：若要控制教学篇幅与代码位置，应通过训练标签规范或模型可见 prompt 明确章节/篇幅预算，并用 execution reward 强化代码正确性，而不是依赖隐藏的解码停止参数。

# DEC-031：不以外部原型 README 指标替代可复现实验

- **日期**：2026-08-03
- **决定**：保留外部项目作为设计参考，不据其 README 宣称 GPT-4o SFT 数据或 GRPO 效果优于本项目，也不回退到其简化验证逻辑。
- **依据**：外部包缺少数据、模型和结果产物；TACO 无执行测试，SFT 接纳条件允许失败代码，训练存在 2048 静默截断风险。
- **可借鉴项**：统一且简洁的教学模板、对回答篇幅的显式约束、最佳 checkpoint 与课程学习概念；引入前必须在当前 verified source bank 和 Docker 口径下重新验证。

# DEC-032：选择性吸收外部 GRPO 设计，不回退正确性口径

- **日期**：2026-08-03
- **决定**：正式 GRPO 继续从 full SFT adapter 暖启动；保留并验证 zero-advantage 监控、独立 heldout strict Pass@1 选优和 curriculum 消融，不采用无测试 AST 分替代 correctness。
- **奖励主线**：TeachingCritic 准入前保持 Docker correctness 0.9 + contract 0.1；LocalTeachingReward 只能记录为表面诊断，不能称为真实教学质量，也不进入梯度。
- **归一化**：默认关闭 reward 函数中的 batch Z-score。GRPO 已做 prompt 组内相对优势标准化；如研究额外预标准化，必须作为独立消融并同时报告组内方差和 reward 排序变化。
- **框架选择**：不因外部项目使用 Unsloth 而重写已在双 4090、8K 上通过的 Transformers/PEFT/bitsandbytes/Liger SFT 管线。Unsloth 仅在同配置显存与吞吐基准证明收益后考虑用于后续入口。
- **评测补齐**：后续需要执行 HumanEval+/MBPP+ 等代码保持评测和带技术正确性门控的教学评测；官方 HumanEval 88.4% 只作为基座来源数据，不作为 CodeGuide 训练后结果。

# DEC-033：冻结后删除可恢复中间资产，统一正式入口

- **日期**：2026-08-14
- **决定**：canonical SFT、source bank、TACO test、固定 split、GRPO 正式数据和版本化评测目录为必须保留资产；TACO train、reference cache、传输压缩包、API smoke 与 superseded 输出可在其下游冻结产物验证后删除。
- **代码边界**：SFT 唯一实现为 `src.training.train_sft`，TACO checkpoint 唯一通用评测器为 `scripts/evaluate_sft_matrix.py`；历史入口只在确有兼容价值时保留薄包装器，不再维护平行实现。
- **证据边界**：删除一次性运行脚本不等于删除实验结论；原始正式 generation/verification、汇总报告、固定哈希与 `EXPERIMENT_LOG.md` 继续保留。

# DEC-034：训练基础设施交回成熟框架

- **日期**：2026-08-15
- **决定**：SFT/GRPO 分别以 TRL `SFTTrainer`/`GRPOTrainer` 为唯一训练循环，Accelerate 负责双卡进程，PEFT/bitsandbytes 负责 NF4 QLoRA；不再维护手写分布式、rollout 或优化器循环。
- **项目保留边界**：只保留 CodeGuide 特有的数据冻结合同、assistant-only 标签、接口感知代码提取、Docker correctness 和 teaching contract。
- **配置边界**：实验差异进入 `configs/sft.yaml`、`configs/grpo.yaml` 和 Accelerate 配置；不再为每个实验复制 shell/PowerShell 或 Python 训练脚本。
- **DeepSpeed/FlashAttention**：FlashAttention 2 为可选自动后端，缺失时回退 SDPA；DeepSpeed ZeRO-2 通过独立 Accelerate 配置启用，默认双卡入口仍为普通 MULTI_GPU。
- **证据边界**：本轮框架化只在本地完成静态、测试和数据合同验证；云端正式训练前仍需执行最小 GPU smoke，历史 full SFT 结果不因代码重构而失效。

# DEC-035：新架构必须保持正式 GRPO 实验语义

- **日期**：2026-08-15
- **决定**：保留 TRL `GRPOTrainer`、Accelerate、PEFT 和 bitsandbytes，不恢复任何手写训练循环；同时将配置和奖励恢复为 2026-08-15 正式云端运行协议。
- **数据边界**：train 6,451、dev 50、TACO final 515 两两互斥。dev50 唯一用于 checkpoint selection；TACO-515 永不参与训练、调参或选优。
- **课程边界**：easy 3,228/512、medium 1,735/768、hard 1,488/1024，固定三阶段各 1 epoch。curriculum 不得作为普通开关关闭。
- **算法边界**：TRL 0.22.2、`loss_type=grpo`、`scale_rewards=false`、beta 0.05；不得因 API 迁移改成 DR-GRPO、DAPO 或 GSPO。
- **奖励边界**：梯度只接收冻结公式的单一 composite total reward；执行正确性统一调用 `verify_code()` 且每 completion 一次。static 不冒充 correctness，teaching heuristic 只作 diagnostic，训练 backend 为 subprocess。

# DEC-036：教学能力使用双盲双 Judge，与代码正确性评测分离

- **日期**：2026-08-24
- **决定**：最终教学评测固定比较 Base/SFT/GRPO，使用同题 ChatML prompt、平衡 A/B 位置和 DeepSeek V4 Flash/豆包两个独立 Judge；reference assistant 标签对生成模型和 Judge 都不可见。
- **评分边界**：一次 pairwise 请求同时产出 winner 与五维 absolute score；程序从维度分重算 weighted score。回答长度、标题数量和排版不得直接形成优势。
- **数据边界**：默认 Blind50 来自冻结 TACO-515 池，不进入训练或 GRPO checkpoint selection。现有 TACO/EvalPlus 继续负责可执行代码能力，LLM-as-Judge 分数只代表教学质量测量。
- **证据边界**：流水线通过本地合同测试不等于模型教学能力已经提升；只有正式 checkpoint 的落盘 generation、两个 Judge 原始结果和报告齐全后才能形成对比 claim。
