# CodeGuide Project Context

## 2026-07-29: deterministic Class-B comment-plan recovery

- Class A remains the original pedagogical rewrite and receives one API
  request. Existing accepted labels are immutable.
- Any Class-A failure and each historical rejected item enter Class B. Class B
  uses one API response for the teaching explanation and a JSON line-comment
  plan; it does not ask the teacher to rewrite the full Python program.
- The pipeline inserts comments at AST-safe statement boundaries in the
  verified reference. Removing comments with `tokenize` must reproduce the
  exact executable token stream; syntax and Docker execution remain hard
  gates.
- Recovery records use `recovery_version=2`, allowing old recovery failures
  one refined attempt without creating an unbounded retry loop.
- Validation: 15 tests passed. The first four live version-2 recoveries passed
  every test, preserved executable-token equivalence, and inserted 7, 10, 8,
  and 11 comments.
- Production resumed at concurrency 3 with 1,343 accepted records preserved,
  204 historical rejected records prioritized, and 9,072 total pending.

### 2026-07-29: DeepSeek API concurrency increase

- The production wrapper API concurrency was raised from 3 to 1,000 after the
  account limit for `deepseek-v4-flash` was confirmed as 2,500.
- The semaphore covers only asynchronous teacher API calls. Docker
  `verify_code()` remains synchronous in the event loop, so this change does
  not launch 1,000 verifier containers simultaneously.
- Resume-safe JSONL output preserves completed accepted/rejected records.
  Higher concurrency substantially increases short-term API spend and may be
  capped below 1,000 by local networking or the HTTP connection pool.
- The OpenAI-compatible SDK's implicit transport retry was explicitly disabled
  with `max_retries=0`; `--distill-retries 1` now truly means one Class-A
  request before Class-B fallback.

### 2026-07-29: concurrency rollback after cost amplification

- A production run at API concurrency 1,000 produced 1,999 logged HTTP 200
  responses and 454 client-side request timeouts while accepted records grew
  by only about 1,478.
- Root cause: Docker verification is synchronous inside the same asyncio event
  loop. While code is being verified, hundreds of API responses cannot be
  consumed promptly; server-side work may finish and be billed after the
  client has timed out.
- The 2,500 account concurrency limit is a service ceiling, not a suitable
  local pipeline setting. Production was paused and the wrapper default was
  reduced to 20 pending operator approval to resume.
- Single-attempt log messages now state failure directly instead of claiming a
  retry that cannot occur when `distill_retries=1`.

### 2026-07-29: payment-failure resume semantics

- DeepSeek returned HTTP 402 `Insufficient Balance`. The run ended with 7,100
  accepted lines and 3,534 rejected lines safely flushed.
- 3,298 Class-B attempts were recorded as `recovery_llm_failed`. These are
  external teacher/API failures, not code correctness failures.
- Resume logic now treats `recovery_llm_failed` as retryable after service
  restoration. Syntax, interface, execution, and wrong-answer recovery
  failures remain final.
- Retried records enter Class B directly; accepted labels and Class A are not
  regenerated.
- A second HTTP 402 run recovered 1,317 additional labels before balance was
  exhausted again. HTTP 402/`Insufficient Balance` is now a fatal batch
  condition: remaining tasks are cancelled immediately and the flushed JSONL
  checkpoint is retained.

### 2026-07-29: rejected output becomes current-state snapshot

- `sft_train_ref_label_rejected.jsonl` is no longer an append-only event log.
  It contains one compact latest record for each currently unresolved ID.
- IDs already present in accepted are removed during startup compaction. When
  a recovery is accepted, the ID is atomically removed from rejected.
- Rejected records no longer duplicate full test suites or assistant outputs;
  they retain failure, error, interface, reference, and recovery fields needed
  for diagnosis and resume.
- Writes use a temporary file plus `os.replace`, so interruption leaves either
  the previous or the new complete snapshot.

This file is the entry point for reviewing the packaged CodeGuide source code.
It describes the code that currently exists on disk and the evidence currently
available for each project stage.

## 0. Governing Plan And Status Rules

The highest-priority project specification is:

`CodeGuide_后训练项目实施与验收规划书_v1.2.md`

It overrides stale README text, old configs, notebooks, and unverified feature
claims. If the implementation plan changes, that document must be updated
before or together with the implementation.

The project is currently in **G0: repository credibility and P0 repair**. It
must not move to the 300-500 problem data pilot, full SFT generation, cloud
training, or final evaluation until the corresponding prior Gate is passed.

All capability statements use three distinct statuses:

- **implemented**: code exists and passes relevant static checks;
- **run end-to-end**: the declared workflow completed in the declared
  environment;
- **validated**: a frozen dataset and explicit metric produced a reproducible
  result.

Code existence alone is never evidence that GRPO, Teaching Reward, sandboxing,
best-checkpoint selection, or evaluation is effective.

Every effective future project step must update:

1. this context file with architecture, implementation, configuration, and
   current-stage changes;
2. `DECISION_LOG.md` with experiments, comparisons, failures, root causes,
   responses, commands, and evidence;
3. `CLAIMS_MATRIX.md` when the evidence level of any project claim changes.

## 1. Project Goal

CodeGuide is intended to be an algorithm teaching model. At inference time the
user provides an algorithm problem, and the model should produce:

1. a plain-language restatement;
2. key observations;
3. a progression from a simple solution to the final algorithm;
4. a correctness-oriented explanation;
5. complexity analysis and common mistakes;
6. runnable, commented Python code.

Correct code is a hard requirement. The teaching explanation is the main
product distinction from a generic code model.

## 2. Current End-to-End Design

```text
TACO local Parquet
    |
    v
src/data/loader.py
  - normalize problems and tests
  - parse all Python-like reference candidates
  - rank candidates for standard_input/call_based contracts
    |
    v
scripts/verify_taco_references.py
  - verify up to N candidates with src/reward/execution.py::verify_code
  - stop at the first fully passing candidate
  - write resumable JSONL verification cache
    |
    v
scripts/build_sft_dataset.py
  - merge the offline cache
  - keep verified references before stratified sampling
  - use the reference only as teacher-side privileged context
  - ask the teacher model for a Chinese teaching label
  - quality, syntax, and execution checks
    |
    v
Final ChatML SFT record
  user: problem plus required interface contract, no reference solution
  assistant: teaching explanation plus runnable code
```

The intended main distillation mode is `reference_guided_label`:

```text
teacher input: question + verified reference + interface + test summary + optional seeds
student input: question + interface only
training target: complete teaching explanation + runnable code
```

The reference solution must never leak into the final student-facing user
message.

## 3. Pre-v1.2 Repository Baseline

Before the v1.2 plan was adopted, the repository already contained a broad
training and evaluation framework:

- problem loading and ChatML data construction;
- data quality checks and code block extraction;
- SFT with Qwen2.5-Coder-7B-Instruct and QLoRA;
- GRPO training with accuracy, format, and teaching rewards;
- curriculum stages and best-checkpoint selection;
- blind comparison against the base model;
- reward ablation and teaching-reward alignment tests;
- CLI and Gradio inference demos.

This section describes the pre-v1.2 repository state, not verified completion
and not source provenance. A formal provenance and license audit is still
required before public release.

There is also an earlier APPS-oriented design under
`scripts/data_generate/`. It uses validated seed examples as few-shot teaching
demonstrations and asks a teacher model to produce structured labels.

## 4. Main Improvements Added On Top

### TACO reference parsing

`src/data/loader.py` now:

- reads local TACO Parquet files;
- parses the JSON-encoded `solutions` field;
- keeps multiple Python-like candidates instead of only one string;
- ranks call-based candidates by exact `fn_name`, then `class Solution`;
- ranks standard-input candidates by complete stdin/stdout behavior;
- preserves selected and raw candidate indices.

### Unified execution verifier

`src/reward/execution.py::verify_code()` is the execution authority for both:

- `standard_input`: run a complete program and compare stdout;
- `call_based`: import the candidate through a harness and invoke the required
  function/class contract.

`src/data/code_validator.py` is retained primarily for code extraction and
syntax checks. Dataset construction and reference validation use
`verify_code()` for actual tests and pass-rate calculation.

### Offline multi-candidate reference verification

`scripts/verify_taco_references.py` supports:

- multiple candidates per problem;
- configurable candidate limit;
- multiprocessing;
- resumable JSONL output;
- incremental flush and progress reporting;
- typed failures such as `no_reference`, `unsupported`, `syntax_error`,
  `timeout`, `runtime_error`, `wrong_answer`, and `interface_mismatch`.

The completed full train cache is intentionally not included in this package
because it is large. Its on-disk path in the working project is:

`data/cache/taco_reference_verification_train_full.jsonl`

Full-cache summary:

- deduplicated train problems: 24,237;
- problems with Python candidates: 18,733;
- first-candidate passes: 9,455;
- additional problems rescued by fallback candidates: 960;
- final fully verified references: 10,415;
- no Python reference: 5,504;
- average attempted candidates: 1.348.

### Reference-guided label construction

`scripts/build_sft_dataset.py` supports three explicit modes:

- `scratch`: teacher and student do not see a reference;
- `reference_guided_label`: teacher sees a verified reference, student does
  not; this is the main path;
- `code_explanation`: the reference is visible to the student as an auxiliary
  code-explanation task, not the main training objective.

The script also supports:

- offline reference-cache merge;
- minimum reference pass-rate filtering;
- required verified-reference filtering;
- I/O mode filtering;
- difficulty-stratified sampling after verification filtering;
- optional seed teaching examples;
- output truncation detection;
- quality filtering, syntax checks, and `verify_code()` pass-rate metadata.

## 5. Smoke Results Reproduced By Included Samples

The included scratch smoke file contains four generated records whose code
pass rates were all zero.

The included reference-guided stratified smoke contains five records:

- all five used fully verified references;
- four of five generated code solutions fully passed;
- one standard-input medium-hard label changed algorithm semantics and failed.

The included call-based reference-guided smoke contains five records:

- five of five generated code solutions fully passed;
- all required function names were declared;
- all starter-code contracts were respected;
- no reference leaked into `messages.user`;
- no interface mismatch, syntax error, truncation, or execution error occurred.

These are small diagnostic runs, not final benchmark results.

## 6. Current Code Defaults Versus Frozen v1.2 Targets

The current files still contain legacy or pre-freeze defaults. They are useful
for auditing the implementation but are not automatically the formal
experiment configuration.

Current on-disk defaults include:

- backbone: Qwen2.5-Coder-7B-Instruct;
- SFT: QLoRA, LoRA rank 16, alpha 32, learning rate `2e-4`;
- GRPO: LoRA rank 64, alpha 128, learning rate `1e-5`;
- GRPO generation: temperature 0.8, top-p 0.95, four generations;
- curriculum output limits: easy 512, medium 768, hard 1024 tokens;
- blind evaluation generation: greedy decoding (`do_sample=False`);
- GPT judge temperature: 0.0.

The distillation request in `scripts/build_sft_dataset.py` currently uses
temperature 0.3, up to 8192 output tokens by default, and DeepSeek-compatible
environment variables:

- `DISTILL_API_KEY`;
- `DISTILL_BASE_URL`;
- `DISTILL_MODEL`.

No API keys are included in this package.

The v1.2 target configuration that must be implemented and frozen through the
Gate process is:

- teacher: `deepseek-v4-flash`;
- first teacher generation: non-thinking, temperature 0.2, top-p 0.9;
- up to two execution-feedback repair attempts before rejection;
- SFT: 4-bit NF4 QLoRA, rank 16, alpha 32, learning rate `1e-4`;
- SFT sequence length selected only after tokenizer audits from 4096, 6144,
  and 8192;
- GRPO starts from the best SFT adapter without merging/requantizing;
- GRPO keeps the same adapter structure, learning rate `5e-6`,
  temperature 0.8, top-p 0.95, `scale_rewards=false`;
- formal execution must use a restricted container before being called a safe
  sandbox.

The mismatch between current code defaults and these targets is a G0 repair
item, not an authorized silent config change.

## 7. Recommended Reading Order

Read these files first:

1. `CodeGuide_后训练项目实施与验收规划书_v1.2.md`
2. `DECISION_LOG.md`
3. `CLAIMS_MATRIX.md`
4. `scripts/build_sft_dataset.py`
5. `src/data/loader.py`
6. `src/reward/execution.py`
7. `scripts/verify_taco_references.py`
8. `src/data/quality.py`
9. `src/data/code_validator.py`
10. `scripts/train_sft.py`
11. `src/training/grpo_train.py`
12. `configs/train_config.yaml`
13. `evals/blind_eval.py` and `evals/ablation.py`

For the earlier seed-driven APPS design, then inspect:

1. `scripts/data_generate/generate_sft.py`
2. `scripts/data_generate/prepare_apps.py`
3. `scripts/data_generate/validate_seed_sandbox.py`
4. `data/seeds/`

## 8. Current Project Boundary

Completed:

- local TACO loading;
- multi-candidate Python reference extraction;
- full offline reference verification and cache;
- privileged-reference label mode;
- verified-reference filtering;
- standard-input and call-based smoke tests.

Not yet completed as a final experiment:

- large-scale production SFT label generation;
- final SFT/GRPO model training on the improved labels;
- post-training blind comparison and paper-ready final metrics.

The next authorized sequence is the one in section 18 of the v1.2 plan. The
immediate work is P0 repository repair, claims auditing, dependency locking,
and local proxy-model dry-runs. The 300-500 problem pilot begins only after G0
passes.

## 9. Package Scope

This review package contains source code, configs, docs, tests, seed examples,
small smoke outputs, and small reference-cache examples. It excludes:

- `.env` files and credentials;
- `.venv` and Python caches;
- raw TACO Parquet files;
- the full 24,237-row verification cache;
- model weights and checkpoints;
- logs and generated training corpora.

## 10. Effective-Step Log

### 2026-07-28: v1.2 governance adopted

- Read the complete v1.2 implementation and acceptance plan.
- Established it as the highest-priority specification.
- Set the current stage to G0/P0 repair.
- Added `DECISION_LOG.md` for experiments, failures, causes, and responses.
- Added `CLAIMS_MATRIX.md` to separate implemented, run, and validated claims.
- Updated the GPT review package to include all three governance documents.
- This documentation step changed no dataset, model, reward, split, metric, or
  cloud resource allocation.

### 2026-07-28: GPT package governance-file fix

- Package verification found that the Chinese-named v1.2 plan was absent from
  the ZIP even though it was listed in the PowerShell source.
- Root cause: Windows PowerShell 5.1 can decode non-ASCII string literals in a
  `.ps1` file inconsistently, causing the literal path check to miss the file.
- Changed the packager to include every root-level Markdown file instead of
  hard-coding a Chinese filename.
- Rebuilt and inspected the ZIP: 84 entries, all five required governance/code
  entry points present, and zero excluded/sensitive-path matches.
- This packaging fix changed no dataset, model, reward, split, metric, or cloud
  resource allocation.

### 2026-07-28: G0/D1 repaired snapshot import

- Stage: G0/P0 repository trust repair.
- Input snapshot: `CodeGuide_G0_D1_fixed_20260727.zip`, SHA-256
  `601DE8B1CF77B1FE97F3BD73E390C7E5A90C88F59877B2DC5543797A8A85AF6A`.
- Imported contracts:
  - teacher code enters accepted SFT only when verification is supported,
    contains executable tests, and passes every test;
  - formal verification uses a digest-pinned Docker backend, while subprocess
    remains a trusted-development backend;
  - correctness is zero when no executable tests exist; AST is not a semantic
    correctness proxy;
  - GRPO requires at least four tests and deterministically separates online
    reward tests from held-out tests;
  - checkpoint selection uses strict whole-problem Pass@1 on independent
    `grpo.eval_data`;
  - before TeachingCritic admission, formal reward is correctness 0.9 plus
    contract 0.1;
  - SFT and GRPO share `r=16, alpha=32`, and training fails fast while sequence
    length is unfrozen.
- Additional v1.2 alignment:
  - SFT starts with `learning_rate=1e-4` and `num_train_epochs=1.0`;
  - `completion_only_loss=true` and `length_grouped_sampling=true` are explicit
    config contracts;
  - config validation and G0 contract tests enforce these values.
- Commit hygiene now excludes credentials, checkpoints, production generated
  data, full caches, W&B state, large logs, and ZIP files. Small smoke fixtures,
  manifests, tests, and acceptance reports remain eligible.
- Removed personal absolute paths from the legacy APPS utility and environment
  snapshot.
- Boundary: code is merged but still requires local static, Docker, and GPU
  revalidation before G0 can be marked passed.

### 2026-07-28: local G0 static revalidation

- Python 3.11.9 compileall passed.
- Unit tests passed: 42/42 in 4.86 seconds.
- Shell syntax checks passed.
- The allow-unfrozen configuration contract passed; strict validation failed
  as intended because SFT/GRPO sequence lengths are not frozen.
- Call-based accepted-label re-execution passed 5/5.
- Historical standard-input smoke remained 4/5 and correctly exposed
  `taco_b74ba2ec45`; the new hard gate would reject that record.
- Frozen smoke manifest hash/count validation passed.
- Secret and personal-path scans had no matches in commit-eligible files.
- Local GPU is an RTX 4060 Laptop GPU with 8188 MiB. Docker CLI exists but its
  Linux daemon is stopped. The current environment lacks the training stack,
  so Docker verifier and proxy SFT/GRPO remain blocked.

### 2026-07-28: GitHub delivery repository selected

- Existing repository: `Xiao0731/CodeGuide`.
- The repository was publicly visible when checked and contained one legacy
  commit, while the repaired local baseline has an independent root history.
- Decision: reuse the existing repository after changing it to private; do not
  create a second repository unless the legacy history later proves unsuitable.
- Before any push, authenticate GitHub CLI, fetch the remote commit, inspect the
  history, and merge without force-pushing.
- A private repository can be shared with the intended reviewer by explicitly
  adding that GitHub account as a collaborator.

### 2026-07-28: trustworthy baseline pushed

- GitHub authentication succeeded for `Xiao0731`.
- `Xiao0731/CodeGuide` was confirmed private with `main` as default branch.
- Legacy remote commit `e6c7bae` contained only early APPS/environment files;
  secret and personal-path scans had no matches.
- Local and remote histories had no common ancestor. Merge `af2a0fb` used the
  `ours` strategy to retain `e6c7bae` for audit while keeping the repaired local
  tree byte-for-byte unchanged.
- Tests after the merge passed: 42/42.
- Normal fast-forward push succeeded; local `main` and `origin/main` both point
  to `af2a0fb368fd0e9e4722eded9b1c055900ff307e`.
- No milestone tag was created because Docker, CUDA lock, and proxy-model
  dry-runs remain incomplete.

### 2026-07-28: Docker verifier real-runtime acceptance

- Docker Desktop Linux daemon was started and Docker Engine 28.1.1 was used.
- The formal verifier image was frozen as
  `python:3.11.9-slim-bookworm@sha256:8fb099199b9f2d70342674bd9dbccd3ed03a258f26bbd1d556822c6dfc60c317`.
- `src/reward/execution.py` now assigns every run a unique name and verifier
  label, disables image pulls, runs under `tini`, and always force-removes the
  named container in a `finally` block.
- The runtime contract now explicitly includes:
  - no network;
  - read-only root filesystem;
  - all capabilities dropped and no-new-privileges;
  - UID/GID 65534;
  - `/tmp` as the only writable working area with `noexec,nosuid`;
  - 256 MiB memory and swap ceiling, one CPU, 64 PIDs, CPU ulimit, and host
    wall timeout;
  - exactly one read-only mount containing the generated runner.
- A real infinite-loop probe exposed an earlier bug: timing out the local
  `docker run` client did not stop its container. The first probe left one live
  verifier container. The new unique-name cleanup fixed the leak; three
  consecutive timeout probes left zero containers.
- CPU-limit exits 137/152 are now reported as
  `container timeout/resource limit` instead of the ambiguous
  `no harness output`.
- `scripts/validate_docker_verifier.py` verifies the live Docker configuration,
  standard-input, top-level call-based functions, `Solution` methods, wrong
  answers, blocked networking, filesystem restrictions, timeout cleanup,
  four-way concurrency, no residual containers, and fail-closed rejection of
  floating image tags.
- Real acceptance result: every check passed. The existing five-record
  call-based SFT smoke also passed 5/5 through the Docker backend.
- Machine-readable evidence:
  `artifacts/g0/docker_verifier_report.json`.
- Unit tests after the repair: 44/44.
- Evidence boundary: this validates the declared restricted-container contract
  on the local Docker Desktop environment. It is not a third-party security
  audit and must not be described as a universally secure sandbox.
- G0 remains blocked by the CUDA dependency lock and proxy-model SFT/GRPO
  dry-runs. No training, API distillation, or milestone tag was performed.

### 2026-07-29: hybrid A/B SFT label generation

- The production `reference_guided_label` path now keeps successful teacher
  rewrites as class A (`label_strategy=pedagogical_rewrite`).
- Class A makes one API attempt. Any generation, structure, syntax, interface,
  execution, timeout, or wrong-answer failure immediately switches the same
  problem to class B instead of repeating the same free rewrite.
- Class B (`label_strategy=reference_locked`) asks the teacher to explain the
  verified reference and permits only `#` comments in the final code.
- Python `tokenize` removes comment-only tokens and compares the remaining
  token stream with the verified reference. Any executable-token difference
  causes deterministic injection of the original reference code.
- The assembled class B label is still syntax checked and executed through the
  same Docker `verify_code()` hard gate.
- Existing accepted records remain immutable and are skipped on resume.
  Existing rejected records are processed directly by class B before any
  unprocessed problem enters class A.
- The production wrapper now uses concurrency 3 (configurable) and explicitly
  sets `--distill-retries 1`.
- Unit/contract checks: 13 passed.
- First live recovery evidence: two historical rejected records
  (`runtime_error`, `wrong_answer`) were recovered to pass rate 1.0. In both
  cases the teacher changed executable tokens, so the pipeline correctly
  injected the verified reference.
### 2026-07-30：Windows 并发 rejected 快照写入修复

- 正式续跑在 20 并发下触发 `WinError 5`，定位为 rejected 快照替换时的临时文件争用/短暂占用。
- 快照写入改为每次使用唯一临时文件；`os.replace` 对 Windows 短暂 `PermissionError` 做有限退避重试，并在结束时清理临时文件。
- 该修复不改变蒸馏策略、API 重试次数、accepted 内容或断点语义。
### 2026-07-30：B 类集中失败根因纠正

- 核对发现 accepted 为 9,691 条；日志“跳过已完成 10,133 条”还包含 442 条终态 rejected，不能解释为 accepted 数。
- 410 条所谓 `recovery_wrong_answer` 的诊断文本全部是 Docker daemon 未运行，并非 reference 或 DeepSeek 代码错误；`pass_rate=0.0` 是验证器未执行任何测试时的默认失败值。
- 正式生成增加 Docker API 前置预检。Docker 不可用时在调用教师 API 前退出。
- Docker 连接失败新增 `docker_unavailable` 分类；历史误分类记录按错误文本重新视为可恢复。
## 2026-08-01：TACO reference-guided SFT 数据阶段定版审计

- 全量 TACO 去重后共 24,237 题，其中 10,415 题具有离线执行完全通过（pass rate 1.0）的 Python reference，构成本轮正式生成候选集；其余 13,822 题未进入该高置信主线。
- 正式生成结果按唯一 ID 统计为：accepted 10,340，当前 unresolved rejected 75；二者无交集、无重复，合计精确覆盖 10,415 个候选，无缺失 ID。
- accepted 接纳率为 99.28%；10,340 条均记录 `reference_verified=true`、`pass_rate=1.0`，无坏 JSON、无缺失 messages、无空 assistant，student user 中未发现 `reference_solution` 字段泄漏。
- accepted I/O 分布：standard-input 7,889，call-based 2,451。难度分布：easy 5,041、medium 1,061、medium_hard 1,464、hard 961、very_hard 446、unknown 1,367。
- rejected 75 条：教师响应失败/截断 40，缺少 numpy 的运行环境错误 28，超时 7。前者可恢复，后两类暂按环境/执行隔离，不阻塞主训练集。
- accepted 文件 SHA-256：`CBFA65F00AD635654431004D8C20AD4BCB1D64EF6CFE7DB984331DB5F9E7042D`；rejected 文件 SHA-256：`AE9149D9FED92EB235F118DB3DFEC949E069271FD37ED3CC7BDAE4BFB350B594`。
- 阶段判定：SFT 数据构造主线通过，可以进入数据冻结、训练/验证划分与 SFT 训练准备；75 条失败样本作为隔离集合处理，不再阻塞下一阶段。
# 2026-08-01 数据冻结与训练前审计

- 冻结后的 canonical SFT 为 `data/final/sft_accepted.jsonl`，共 10,306 条，SHA256 为 `08ef448f4be6b6b34ee2b6b7af5748827feeba0a0f36cc393350374671c86a1b`。
- 完整质量通过快照有 10,340 条；其中 34 条因正式 Qwen chat template 长度超过 8192 被隔离，未截断后混入 canonical。
- A 类 `pedagogical_rewrite` 6,892 条，B 类 `reference_locked` 3,414 条；standard-input 7,856 条，call-based 2,450 条。
- 独立 verified source bank 共 10,415 条，保存题面、完整测试、选中 reference、hash 与原始行溯源；固定 Docker 抽样复验 20/20 通过。
- 使用 `Qwen/Qwen2.5-Coder-7B-Instruct` 正式 chat template 审计：完整序列 P50=2,603、P95=5,062、P99=6,728、max=8,173，正式 SFT 推荐 `max_seq_length=8192`。
- 固定种子 20260728：SFT train/dev 为 9,791/515；预留 GRPO train/validation 为 900/100。划分无交叉、覆盖全部 canonical，且与 TACO test 重叠为 0。
- canonical 和 source bank 属于大数据产物，不进入 Git；通过 `data/manifests/sft_manifest.json` 的路径与 SHA256 管理，云端需单独同步并校验。
# 2026-08-01 QLoRA SFT 管线与500条校准准备

- 新统一入口：`python -m src.training.train_sft --config configs/sft/qwen25_coder_7b_qlora_8k.yaml`，支持 calibration/full、CLI路径覆盖、resume、单卡与 torchrun DDP。
- 训练采用原生 Transformers + PEFT + bitsandbytes：4-bit NF4、LoRA r=32/alpha=64、8K、bf16；不使用 `device_map=auto`，每个 rank 显式绑定本地 GPU。
- completion-only loss 通过 chat template 的 token 前缀关系确定边界，不使用字符串猜测；全量10,306条复审通过，截断0，最大8,173，监督 token 比例77.5844%。
- 500条 calibration 仅从冻结 SFT train 确定性分层抽取，ID内容 hash 为 `6d6975dd2938257150ab7b297d7d39d5ef5c55481163ee7e0011ee07eccd4a11`。
- 本地未下载7B、未执行训练；真实校准、显存、吞吐、checkpoint和adapter重载必须在双RTX 4090云端完成。
# 2026-08-01 云端预检兼容修复

- 双RTX 4090、CUDA、canonical hash和固定split已在云端确认。
- 云镜像为 PyTorch 2.9.0+cu130；bitsandbytes 0.47.0不含CUDA 13原生库，因此QLoRA不可运行。
- 依赖升级至bitsandbytes 0.48.2系列，并将 preflight 改为真实执行NF4前后向和PagedAdamW8bit更新，杜绝import假通过。
# 2026-08-01 云端分布式解释器修复

- 云端 bitsandbytes CUDA 13 原生探针已通过。
- 直接调用 `torchrun` 会命中conda base而不是项目 `.venv`；训练与preflight现统一使用 `python -m torch.distributed.run`。
- 双rank绑定探针会输出每个rank的 `sys.executable`、GPU和transformers版本，解释器漂移将提前失败。
## 2026-08-01：双 4090 首个 8K batch 显存峰值修复

- 云端已完成 4-bit 模型加载并进入第一个真实 forward/backward，但在 `0/32`、尚未完成 optimizer step 时 OOM；rank 0 在交叉熵处额外申请 4.32 GiB，rank 1 在反向传播处申请 3.73 GiB。
- 数据、DDP rank 绑定、NF4 与 PagedAdamW8bit 预检均已通过。本次失败定位为 SDPA 下 8K 序列与约 15 万词表形成完整 logits 的瞬时显存峰值，不是 canonical 数据或分布式入口故障。
- 保持冻结的 `max_seq_length=8192` 与最长样本压力测试不变；训练改用 Liger fused linear cross-entropy，避免显式保留完整 logits，并开启 CUDA expandable segments。
- `requirements-sft.txt` 锁定 `liger-kernel==0.8.0`；训练 manifest 将记录 Liger 开关与配置；云端 preflight 新增真实 fused-loss forward/backward 探针。
- 下一次云端先执行 `--max-steps 1` 的最长样本探针，成功后再执行 500 条、32 optimizer steps 的完整校准。本地仍未宣称 SFT 校准通过。
## 2026-08-02：500 条双 4090 SFT 校准训练完成

- 云端使用双 RTX 4090、Qwen2.5-Coder-7B-Instruct、4-bit NF4 QLoRA、8K 序列与 Liger fused linear cross-entropy，完成固定 500 条训练和 100 条 dev 的完整校准。
- 共完成 32/32 optimizer steps，无 OOM、NaN 或 rank 失败；训练 runtime 580.62 秒，平均 train loss 0.7215665，吞吐 0.861 samples/s、0.055 steps/s。
- 第 25 step 的 100 条 dev 评估 loss 为 0.6630970；逐步 loss 与 grad norm 全部有限，末步 loss 0.6023、grad norm 0.27746。
- 训练与评估执行已通过；完整 Gate 仍需执行刚保存 adapter 的独立重载生成，并核对产物 manifest。不能在该证据完成前宣称正式 full SFT 已启动。
## 2026-08-02：SFT calibration adapter 独立重载通过

- 云端从 `outputs/sft/qwen25_coder_7b_qlora_8k/calibration_seed20260728/adapter` 独立加载 4-bit 基座与 PEFT adapter 成功。
- 固定 smoke 题 `taco_616bc08bca` 完成 64-token 确定性生成，输出非空并保持中文分步教学格式；`adapter_reloaded=true`。
- adapter 保存、重载与真实推理子项通过。校准 Gate 仅剩固定 dev 的 Base/Adapter 对照；对照完成前不启动 9,791 条 full SFT。
## 2026-08-02：SFT 校准评测拆分为云端生成与本地验证

- 云端训练节点没有 Docker，且未同步 219 MiB verified source bank；它们不再作为训练环境依赖。
- Base/Adapter 对照改为两阶段：云端 GPU 仅对固定 dev ID 做确定性生成并逐条落盘；本地读取 generation、独立 source bank 和固定 Docker digest 执行 `verify_code()`。
- `selection.json` 固定 seed、ID、模型和最大生成长度；已有 generation 可按 problem ID 断点跳过，配置变化会 fail closed。
- 修复旧评测脚本乱码章节关键词，generation 与 verification 分文件保存，避免本地执行结果覆盖云端原始输出。
## 2026-08-02：40 条 Base/Adapter 首轮对照与截断归因

- 固定 40 条 dev 的 Docker 严格 Pass@1：Base 4/40（10%），Adapter 7/40（17.5%）；Adapter 净提升 3 题，教学结构完整数从 0 提升到 18。
- 两者代码块与接口匹配均为 40/40，说明 SFT 没有造成集中接口退化。
- Adapter 平均生成长度 1727.3，约为 Base 841.6 的两倍；8 条 Adapter 回答撞到 2048 上限且全部失败，4 条代码围栏未闭合。
- 评测增加自然结束回答复用机制：4096 恢复轮复用旧轮 72 份未撞限回答，只重生成 8 条 Adapter 截断项，不调用 API。

## 2026-08-02：4096-token 校准恢复轮完成

- 固定 40 条 dev 在 4096 生成上限下完成本地 Docker 复验。
- Base/Adapter 严格 Pass@1 分别为 4/40 和 7/40，Adapter 净提升 7.5 个百分点。
- 教学模板完整数为 0/40 和 21/40；两者代码块、接口匹配均为 40/40。
- 4096 轮无撞限、无未闭合代码围栏，因此剩余失败不再归因于截断，而是小规模校准后的算法泛化与运行时问题。
- 当前证据支持“SFT 方向有效”，不支持将 17.5% 当作正式 full SFT 结果。
- 失败根因复查显示：33 条失败中 19 条为核心语义/算法错误，5 条为局部边界，4 条为 I/O，2 条为多代码块提取，2 条为实现异常，1 条为资源超限。本轮重生成的 8 条全部自然结束，但仍因上述非截断原因失败。

## 2026-08-02：多代码块提取修复与 full SFT 入口确认

- 多代码块改为按语法、I/O 完整性和 call-based 目标接口选择；最后一块已完整时保持旧行为。
- 复用已保存 40 题 4096-token 输出重放：Base 4/40 不变，Adapter 7/40 提升为 8/40；`michael_pays` 由接口错误恢复为全测试通过。
- full validate-only 已确认固定 train/dev=9,791/515；新增双 4090 正式启动脚本，但本地未启动训练。

## 2026-08-02：云实例 NCCL 共享内存兼容修复

- full SFT 在模型加载后的 DDP 参数校验阶段失败，NCCL 无法 attach 当前容器 `/dev/shm` 中的共享内存段。
- 双 4090 calibration/full 入口默认 `NCCL_SHM_DISABLE=1`，保留环境变量覆盖；该修复不改变数据、模型、LoRA 或优化参数。
- 后续命令日志统一写入 `logs/`；云端增量 ZIP 排除 Markdown 文档。

## 2026-08-02：full dev 评估 logits OOM 修复

- NCCL SHM 修复后 full SFT 已真实完成 25 optimizer steps，loss 由约 0.90 下降到 0.59，证明双卡训练主路正常。
- 第 25 step 首次 515 条 full-dev 评估在长样本上 OOM；调用栈显示 Trainer `return_outputs=True` 使 Liger 评估路径回退到完整 logits cross entropy。
- SFT 评估只需 eval loss，因此固定 `prediction_loss_only=True`，仍覆盖全部 515 条 dev，不保留无用 logits。

## 2026-08-02：full dev 强制 loss-only 评估修复

- 云端复跑再次在 step 25 评估 OOM，调用栈仍进入 `Trainer.prediction_step(... return_outputs=True)`，证明当前 Transformers 4.53.3 与 Liger 组合未落实仅靠配置声明的 `prediction_loss_only`。
- 训练入口改用最小 `LossOnlyTrainer` 覆盖：评估时显式调用 `compute_loss(..., return_outputs=False)`，只返回标量 loss，不返回 logits 或 labels。
- 数据、515 条 dev、评估间隔、LoRA、学习率和其余 full SFT 超参数均未改变。
- 二次云端复验表明 `return_outputs=False` 本身仍不足：Liger 0.8.0 在 eval mode 默认 `skip_logits=False`。最终修复在 Liger 评估输入中显式设置 `skip_logits=True`，保持 dropout 关闭并启用 fused loss。

## 2026-08-02：9,791 条 full SFT 训练完成

- 双 RTX 4090 完成固定 9,791 train / 515 dev、1 epoch、612/612 optimizer steps 的 Qwen2.5-Coder-7B-Instruct 4-bit NF4 QLoRA SFT。
- 总 runtime 15,032.64 秒（约 4 小时 10 分），平均 train loss 0.5693143；末段 loss 与 grad norm 均有限，无 OOM、NaN 或 rank failure。
- step 600 附近完整 dev 评估成功，`eval_loss=0.5323175`，runtime 182.406 秒；证明显式 `skip_logits=True` 修复有效。
- 当前训练执行已完成；正式产物仍需独立检查 adapter、run manifest 并完成 adapter 重载生成，之后再进入 full SFT 质量评测。
