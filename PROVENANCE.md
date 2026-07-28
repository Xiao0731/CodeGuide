# Provenance

CodeGuide is intended to be an independently implemented algorithm-teaching
post-training project.

## Public assets and infrastructure

- TACO and any separately documented evaluation datasets;
- Qwen2.5-Coder-7B-Instruct model and tokenizer;
- DeepSeek API used as a teacher;
- PyTorch, Transformers, TRL, PEFT, Unsloth, bitsandbytes, datasets, vLLM,
  EvalPlus, Docker and related open-source dependencies.

These assets remain subject to their own licenses and citation requirements.

## Repository source status

The review snapshot available on 2026-07-27 did not contain `.git` metadata.
On 2026-07-28, the repaired tree was committed as baseline `b3fb3a8` and pushed
to the private `Xiao0731/CodeGuide` repository. The repository's single legacy
commit, `e6c7bae`, was inspected and preserved as a parent of merge commit
`af2a0fb`; the audited baseline tree remained unchanged by that merge.

This establishes provenance for work after the baseline. It does not
retroactively prove authorship of every file imported into `b3fb3a8`, so public
release still requires source and license auditing.

Required release checks:

1. audit the pre-baseline source of files imported into `b3fb3a8`;
2. identify copied or adapted snippets, if any, and preserve their licenses;
3. remove external promotional wording and directory descriptions;
4. record AI-assisted code generation honestly;
5. publish a dependency/data/model license and citation inventory;
6. rename the current `CodeGuide-LLM` working title before public release if it
   can be confused with an unrelated project.

No external community project is treated as an upstream repository by the
current implementation plan. Similar projects may be used only as internal
competition baselines in controlled experiments.
