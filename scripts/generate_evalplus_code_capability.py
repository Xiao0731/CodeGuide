#!/usr/bin/env python3
"""Generate EvalPlus code-only samples for Base, SFT step20 and SFT step200.

This is the external code-capability module. It deliberately does not score
teaching structure or teaching quality. Cloud execution only generates code;
the saved samples must be executed later inside EvalPlus Docker locally.
"""

from __future__ import annotations

import argparse
import ast
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import yaml

ROOT = Path(__file__).resolve().parents[1]


def resolve_path(value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        handle.flush()


def read_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            task_id = row.get("task_id")
            if not isinstance(task_id, str) or not task_id:
                raise RuntimeError(f"missing task_id at {path}:{line_no}")
            if task_id in rows:
                raise RuntimeError(f"duplicate task_id at {path}:{line_no}: {task_id}")
            rows[task_id] = row
    return rows


def batches(items: list[Any], size: int) -> Iterable[list[Any]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def load_evalplus_dataset(name: str) -> dict[str, dict[str, Any]]:
    try:
        from evalplus.data import get_human_eval_plus, get_mbpp_plus
    except ImportError as exc:
        raise RuntimeError(
            "EvalPlus is missing. Install the project requirements.txt."
        ) from exc

    if name == "humaneval":
        return get_human_eval_plus()
    if name == "mbpp":
        return get_mbpp_plus()
    raise ValueError(f"unsupported EvalPlus dataset: {name}")


def build_prompt(
    *,
    tokenizer: Any,
    problem_prompt: str,
    system_prompt: str,
    user_template: str,
    assistant_prefix: str,
) -> str:
    """Render the frozen Qwen instruct EvalPlus prompt."""
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": user_template.format(prompt=problem_prompt.rstrip()),
        },
    ]
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return rendered + assistant_prefix


def extract_python_solution(raw_completion: str) -> str:
    """Extract the first Python code block body without semantic rewriting."""
    value = raw_completion.replace("\r\n", "\n").replace("\t", "    ").strip()

    if value.startswith("```python"):
        value = value[len("```python") :].lstrip("\n")
    elif value.startswith("```"):
        value = value[len("```") :].lstrip("\n")

    stop_positions = []
    for marker in ("\n```", "<|im_end|>", "<|endoftext|>", "</s>"):
        position = value.find(marker)
        if position >= 0:
            stop_positions.append(position)
    if stop_positions:
        value = value[: min(stop_positions)]

    return value.rstrip() + "\n" if value.strip() else ""


def syntax_error(solution: str) -> str | None:
    if not solution.strip():
        return "empty_solution"
    try:
        ast.parse(solution)
    except SyntaxError as exc:
        return f"{exc.msg} (line {exc.lineno}, column {exc.offset})"
    return None


def adapter_fingerprint(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None

    config_path = path / "adapter_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)

    weights = next(
        (
            candidate
            for candidate in (
                path / "adapter_model.safetensors",
                path / "adapter_model.bin",
            )
            if candidate.is_file()
        ),
        None,
    )
    if weights is None:
        raise FileNotFoundError(f"adapter weights missing under {path}")

    return {
        "path": str(path),
        "adapter_config_sha256": sha256_file(config_path),
        "adapter_weights_file": weights.name,
        "adapter_weights_sha256": sha256_file(weights),
    }


def load_model(
    *,
    config: dict[str, Any],
    adapter_path: Path | None,
) -> tuple[Any, Any]:
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "The SFT environment must provide torch, transformers and peft."
        ) from exc

    model_cfg = config["base_model"]
    model_name = str(model_cfg["name_or_path"])
    revision = model_cfg.get("revision")
    trust_remote_code = bool(model_cfg.get("trust_remote_code", False))

    dtype_name = str(model_cfg.get("dtype", "bfloat16"))
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if dtype_name not in dtype_map:
        raise ValueError(f"unsupported dtype: {dtype_name}")
    dtype = dtype_map[dtype_name]

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        revision=revision,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model_kwargs: dict[str, Any] = {
        "revision": revision,
        "trust_remote_code": trust_remote_code,
        "torch_dtype": dtype,
        "device_map": {"": 0},
        "low_cpu_mem_usage": True,
        "attn_implementation": str(model_cfg.get("attention_backend", "sdpa")),
    }

    if bool(model_cfg.get("load_in_4bit", False)):
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    if adapter_path is not None:
        model = PeftModel.from_pretrained(
            model,
            str(adapter_path),
            is_trainable=False,
        )
    model.eval()
    return model, tokenizer


def completion_tokens(
    token_ids: Any,
    *,
    eos_token_id: int | None,
    pad_token_id: int | None,
) -> tuple[list[int], bool]:
    """Return IDs before EOS/padding and whether a natural stop was observed."""
    values = [int(item) for item in token_ids.tolist()]
    natural_stop = False
    stop_ids = {
        value for value in (eos_token_id, pad_token_id) if value is not None
    }
    for index, value in enumerate(values):
        if value in stop_ids:
            values = values[:index]
            natural_stop = True
            break
    return values, natural_stop


def write_manifest(
    *,
    config_path: Path,
    config: dict[str, Any],
    variant: str,
    adapter: dict[str, Any] | None,
    model: Any,
    tokenizer: Any,
    output_root: Path,
) -> None:
    import torch
    import transformers
    import peft

    model_config = getattr(model, "config", None)
    payload = {
        "schema_version": "codeguide-evalplus-code-manifest-v1",
        "evaluation_module": "code_capability",
        "teaching_metrics_in_scope": False,
        "protocol_name": config["protocol_name"],
        "variant": variant,
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "base_model": config["base_model"],
        "resolved_model_commit": (
            getattr(model_config, "_commit_hash", None)
            if model_config is not None
            else None
        ),
        "adapter": adapter,
        "generation": config["generation"],
        "datasets": config["datasets"],
        "tokenizer_class": type(tokenizer).__name__,
        "chat_template_sha256": sha256_text(
            str(getattr(tokenizer, "chat_template", ""))
        ),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
            "evalplus": importlib.metadata.version("evalplus"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "created_unix": time.time(),
    }
    write_json(output_root / "manifests" / f"{variant}.json", payload)


def generate_dataset(
    *,
    dataset_name: str,
    config: dict[str, Any],
    variant: str,
    model: Any,
    tokenizer: Any,
    output_root: Path,
) -> None:
    import torch

    dataset = load_evalplus_dataset(dataset_name)
    expected_count = int(config["datasets"][dataset_name]["expected_tasks"])
    if len(dataset) != expected_count:
        raise RuntimeError(
            f"{dataset_name} task count mismatch: expected={expected_count}, "
            f"actual={len(dataset)}. Do not compare a different EvalPlus release."
        )

    sample_path = output_root / "samples" / dataset_name / f"{variant}.jsonl"
    raw_path = output_root / "raw" / dataset_name / f"{variant}.jsonl"
    stats_path = output_root / "stats" / dataset_name / f"{variant}.json"

    existing_samples = read_jsonl(sample_path)
    existing_raw = read_jsonl(raw_path)
    if set(existing_samples) != set(existing_raw):
        raise RuntimeError(
            f"resume artifact mismatch for {dataset_name}/{variant}: "
            f"samples={len(existing_samples)}, raw={len(existing_raw)}"
        )

    task_ids = list(dataset)
    unknown = set(existing_samples) - set(task_ids)
    if unknown:
        raise RuntimeError(f"unknown task IDs: {sorted(unknown)[:5]}")
    pending = [task_id for task_id in task_ids if task_id not in existing_samples]

    generation_cfg = config["generation"]
    batch_size = int(generation_cfg["batch_size"])
    max_new_tokens = int(generation_cfg["max_new_tokens"])

    print(
        f"[{variant}:{dataset_name}] completed={len(existing_samples)} "
        f"pending={len(pending)} batch_size={batch_size}",
        flush=True,
    )

    for batch_index, current_ids in enumerate(batches(pending, batch_size), 1):
        prompts = [
            build_prompt(
                tokenizer=tokenizer,
                problem_prompt=str(dataset[task_id]["prompt"]),
                system_prompt=str(generation_cfg["system_prompt"]),
                user_template=str(generation_cfg["user_template"]),
                assistant_prefix=str(generation_cfg["assistant_prefix"]),
            )
            for task_id in current_ids
        ]

        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=False,
        ).to(model.device)

        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=int(generation_cfg.get("num_beams", 1)),
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        prompt_width = int(encoded["input_ids"].shape[1])
        generated_suffix = generated[:, prompt_width:]

        for task_id, rendered_prompt, sequence in zip(
            current_ids,
            prompts,
            generated_suffix,
        ):
            valid_ids, natural_stop = completion_tokens(
                sequence,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
            raw_completion = tokenizer.decode(
                valid_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            solution = extract_python_solution(raw_completion)
            syntax_message = syntax_error(solution)
            hit_limit = len(valid_ids) >= max_new_tokens and not natural_stop

            append_jsonl(sample_path, {"task_id": task_id, "solution": solution})
            append_jsonl(
                raw_path,
                {
                    "task_id": task_id,
                    "variant": variant,
                    "dataset": dataset_name,
                    "solution": solution,
                    "raw_completion": raw_completion,
                    "generated_tokens": len(valid_ids),
                    "solution_characters": len(solution),
                    "hit_generation_limit": hit_limit,
                    "syntax_error": syntax_message,
                    "problem_prompt_sha256": sha256_text(
                        str(dataset[task_id]["prompt"])
                    ),
                    "rendered_prompt_sha256": sha256_text(rendered_prompt),
                },
            )

        completed = len(existing_samples) + min(
            batch_index * batch_size,
            len(pending),
        )
        print(f"[{variant}:{dataset_name}] {completed}/{len(task_ids)}", flush=True)

    final_samples = read_jsonl(sample_path)
    final_raw = read_jsonl(raw_path)
    if set(final_samples) != set(task_ids) or set(final_raw) != set(task_ids):
        raise RuntimeError(
            f"incomplete output for {dataset_name}/{variant}: "
            f"samples={len(final_samples)}, raw={len(final_raw)}, "
            f"expected={len(task_ids)}"
        )

    raw_rows = [final_raw[task_id] for task_id in task_ids]
    stats = {
        "schema_version": "codeguide-evalplus-generation-stats-v1",
        "evaluation_module": "code_capability",
        "dataset": dataset_name,
        "variant": variant,
        "samples": len(raw_rows),
        "average_generated_tokens": (
            sum(int(row["generated_tokens"]) for row in raw_rows) / len(raw_rows)
        ),
        "average_solution_characters": (
            sum(int(row["solution_characters"]) for row in raw_rows) / len(raw_rows)
        ),
        "hit_generation_limit": sum(
            bool(row["hit_generation_limit"]) for row in raw_rows
        ),
        "syntax_failures": sum(row["syntax_error"] is not None for row in raw_rows),
        "syntax_failure_task_ids": [
            row["task_id"] for row in raw_rows if row["syntax_error"] is not None
        ],
        "sample_file": str(sample_path),
        "raw_file": str(raw_path),
        "sample_sha256": sha256_file(sample_path),
        "raw_sha256": sha256_file(raw_path),
    }
    write_json(stats_path, stats)
    print(json.dumps(stats, ensure_ascii=False, indent=2), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/eval/evalplus_code_capability_v1.yaml",
    )
    parser.add_argument(
        "--variant",
        required=True,
        choices=("base", "mixed_lr1e4_step020", "mixed_lr1e4_step200"),
    )
    parser.add_argument(
        "--dataset",
        action="append",
        choices=("humaneval", "mbpp"),
        help="May be repeated. Defaults to both datasets.",
    )
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-new-tokens", type=int)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config_path = resolve_path(args.config)
    assert config_path is not None
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise RuntimeError(f"invalid config: {config_path}")
    if config.get("evaluation_module") != "code_capability":
        raise RuntimeError("this script only accepts the code-capability module")
    if config.get("teaching_metrics_in_scope") is not False:
        raise RuntimeError("teaching metrics must remain out of scope here")

    if args.batch_size is not None:
        if args.batch_size <= 0:
            raise ValueError("--batch-size must be positive")
        config["generation"]["batch_size"] = args.batch_size
    if args.max_new_tokens is not None:
        if args.max_new_tokens <= 0:
            raise ValueError("--max-new-tokens must be positive")
        config["generation"]["max_new_tokens"] = args.max_new_tokens

    seed = int(config["generation"]["seed"])
    random.seed(seed)

    import torch

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for model generation")

    variant_cfg = config["variants"][args.variant]
    adapter_path = resolve_path(variant_cfg.get("adapter_path"))
    adapter = adapter_fingerprint(adapter_path)

    output_root = resolve_path(config["output"]["root"])
    assert output_root is not None
    output_root.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model(config=config, adapter_path=adapter_path)
    write_manifest(
        config_path=config_path,
        config=config,
        variant=args.variant,
        adapter=adapter,
        model=model,
        tokenizer=tokenizer,
        output_root=output_root,
    )

    datasets = args.dataset or list(config["datasets"])
    for dataset_name in datasets:
        generate_dataset(
            dataset_name=dataset_name,
            config=config,
            variant=args.variant,
            model=model,
            tokenizer=tokenizer,
            output_root=output_root,
        )

    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"[success] generation complete: {args.variant}", flush=True)


if __name__ == "__main__":
    main()
