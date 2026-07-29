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
