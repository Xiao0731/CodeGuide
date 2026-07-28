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

The review snapshot available on 2026-07-27 does not contain `.git` metadata.
Consequently this snapshot alone cannot prove file authorship, commit dates, or
whether every historical file was independently authored. Public release is
blocked until the original Git repository is audited.

Required release checks:

1. inspect the complete Git history and remotes;
2. identify copied or adapted snippets, if any, and preserve their licenses;
3. remove external promotional wording and directory descriptions;
4. record AI-assisted code generation honestly;
5. publish a dependency/data/model license and citation inventory;
6. rename the current `CodeGuide-LLM` working title before public release if it
   can be confused with an unrelated project.

No external community project is treated as an upstream repository by the
current implementation plan. Similar projects may be used only as internal
competition baselines in controlled experiments.
