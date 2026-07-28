# CodeGuide Project Context

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
