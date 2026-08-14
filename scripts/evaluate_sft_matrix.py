#!/usr/bin/env python3
"""TACO-515 checkpoint matrix evaluation.

This is the canonical checkpoint-matrix evaluator. Historical 40-problem
calibration outputs remain as experiment artifacts, but their one-off evaluator
has been retired.

Stages
------
prepare
    Freeze the 515 dev IDs and protocol manifest. No GPU required.
generate
    Generate one variant only (base OR one adapter), with batched greedy decoding.
verify
    Reuse saved generations and run strict Docker verification offline.
summarize
    Combine every available variant into one checkpoint trajectory report.

Key changes from the old evaluator
----------------------------------
1. Base and adapters are unbound: one process writes one variant file.
2. The legacy training system prompt is replaced at evaluation time.
3. A compact, code-first protocol is appended consistently to every model.
4. Decoder-only batching uses LEFT padding and length buckets.
5. Generation is persisted once; Docker/static rescoring can be replayed forever.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import statistics
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.source_bank_io import iter_source_bank
from src.data.code_validator import extract_code
from src.reward.execution import verify_code
from src.training.sft_data import load_canonical, load_id_list


# ---------------------------------------------------------------------------
# Generic I/O
# ---------------------------------------------------------------------------

def read_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    """Read a JSONL file keyed by problem_id; missing files yield an empty dict."""
    if not path.exists():
        return {}
    records: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            problem_id = item.get("problem_id")
            if not problem_id:
                raise ValueError(f"missing problem_id at {path}:{line_no}")
            records[str(problem_id)] = item
    return records


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        return result.stdout.strip() or None
    except Exception:
        return None


def percentile(values: list[int | float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


# ---------------------------------------------------------------------------
# Protocol loading and prompt construction
# ---------------------------------------------------------------------------

def load_protocol(config_path: Path) -> dict[str, Any]:
    """Load and validate the frozen compact-code-first protocol."""
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    payload = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    if not isinstance(payload, dict):
        raise ValueError(f"invalid protocol config: {config_path}")

    required = ("protocol_name", "model", "dataset", "generation", "system_prompt")
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"protocol config missing fields: {missing}")

    generation = payload["generation"]
    if generation.get("do_sample") is not False:
        raise ValueError("official matrix evaluation requires do_sample=false")
    if generation.get("temperature") is not None:
        raise ValueError(
            "temperature must be null; with do_sample=false it must not be passed"
        )
    if int(generation.get("max_new_tokens", 0)) != 2048:
        raise ValueError("compact-code-first-v1 freezes max_new_tokens=2048")
    return payload


def build_eval_messages(record: dict[str, Any], protocol: dict[str, Any]) -> list[dict[str, str]]:
    """Replace the legacy system prompt and append the compact user reminder.

    【改动核心】
    The canonical SFT record is never modified in place. We deep-copy the prompt
    messages, replace/prepend the system message, then append a short reminder to
    the final user message. Every model receives the exact same result.
    """
    raw_messages = record.get("messages")
    if not isinstance(raw_messages, list) or len(raw_messages) < 2:
        raise ValueError("canonical record has invalid messages")
    if raw_messages[-1].get("role") != "assistant":
        raise ValueError("canonical record must end with an assistant label")

    messages = copy.deepcopy(raw_messages[:-1])
    new_system = {"role": "system", "content": str(protocol["system_prompt"])}

    if messages and messages[0].get("role") == "system":
        messages[0] = new_system
    else:
        messages.insert(0, new_system)

    user_suffix = str(protocol.get("user_suffix") or "")
    if user_suffix:
        user_indexes = [
            index for index, message in enumerate(messages)
            if message.get("role") == "user"
        ]
        if not user_indexes:
            raise ValueError("evaluation prompt has no user message")
        last_user_index = user_indexes[-1]
        original = str(messages[last_user_index].get("content") or "").rstrip()
        messages[last_user_index]["content"] = original + user_suffix

    return messages


def build_prompt(
    record: dict[str, Any],
    protocol: dict[str, Any],
    tokenizer: Any,
) -> str:
    messages = build_eval_messages(record, protocol)
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


# ---------------------------------------------------------------------------
# Run layout and frozen selection
# ---------------------------------------------------------------------------

def ensure_run_layout(run_dir: Path) -> None:
    for name in ("logs", "generations", "verification", "reports", "static_proxy"):
        (run_dir / name).mkdir(parents=True, exist_ok=True)


def resolve_repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def choose_dev_ids(
    dev_ids: list[str],
    *,
    use_all_dev: bool,
    samples: int | None,
    seed: int,
) -> list[str]:
    if use_all_dev:
        return list(dev_ids)
    if samples is None or samples <= 0:
        raise ValueError("samples must be positive when use_all_dev=false")
    if samples > len(dev_ids):
        raise ValueError(f"samples={samples} exceeds dev size={len(dev_ids)}")
    rng = random.Random(seed)
    chosen = set(rng.sample(dev_ids, samples))
    # Preserve frozen dev order rather than sorting by ID.
    return [problem_id for problem_id in dev_ids if problem_id in chosen]


def expected_manifest(
    protocol: dict[str, Any],
    config_path: Path,
    selected: list[str],
    batch_size: int,
) -> dict[str, Any]:
    generation = protocol["generation"]
    return {
        "schema_version": "codeguide-taco-matrix-manifest-v1",
        "protocol_name": protocol["protocol_name"],
        "protocol_config": str(config_path.relative_to(ROOT))
        if config_path.is_relative_to(ROOT)
        else str(config_path),
        "protocol_config_sha256": sha256_file(config_path),
        "system_prompt_sha256": sha256_text(str(protocol["system_prompt"])),
        "problem_ids_sha256": sha256_text(
            json.dumps(selected, ensure_ascii=False, separators=(",", ":"))
        ),
        "samples": len(selected),
        "model": protocol["model"]["name"],
        "generation": {
            "max_new_tokens": int(generation["max_new_tokens"]),
            "do_sample": False,
            "temperature": None,
            "top_p": None,
            "num_beams": int(generation.get("num_beams", 1)),
            "batch_size": int(batch_size),
            "length_bucket": bool(generation.get("length_bucket", True)),
        },
        "git_commit": get_git_commit(),
    }


def persist_exact_json(path: Path, payload: dict[str, Any]) -> None:
    """Create a frozen JSON file, or reject any attempt to change it in-place."""
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise RuntimeError(
                f"frozen run metadata mismatch: {path}\n"
                "Use a new run directory instead of mutating an existing protocol."
            )
        return
    write_json(path, payload)


def prepare_run(
    protocol: dict[str, Any],
    config_path: Path,
    run_dir: Path,
    *,
    batch_size: int,
    samples_override: int | None,
) -> list[str]:
    ensure_run_layout(run_dir)
    dataset = protocol["dataset"]
    dev_ids = load_id_list(resolve_repo_path(str(dataset["dev_ids"])))
    selected = choose_dev_ids(
        dev_ids,
        use_all_dev=bool(dataset.get("use_all_dev", True)),
        samples=samples_override,
        seed=int(dataset.get("seed", 20260728)),
    )

    canonical = load_canonical(resolve_repo_path(str(dataset["canonical"])))
    missing = [problem_id for problem_id in selected if problem_id not in canonical]
    if missing:
        raise RuntimeError(f"canonical data misses selected IDs: {missing[:5]}")

    selection = {
        "schema_version": "codeguide-taco-selection-v1",
        "protocol_name": protocol["protocol_name"],
        "problem_ids": selected,
        "samples": len(selected),
        "seed": int(dataset.get("seed", 20260728)),
    }
    manifest = expected_manifest(protocol, config_path, selected, batch_size)
    persist_exact_json(run_dir / "selection.json", selection)
    persist_exact_json(run_dir / "manifest.json", manifest)
    print(
        json.dumps(
            {
                "stage": "prepare",
                "run_dir": str(run_dir),
                "samples": len(selected),
                "protocol": protocol["protocol_name"],
                "batch_size": batch_size,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return selected


def load_frozen_run(
    protocol: dict[str, Any],
    config_path: Path,
    run_dir: Path,
    batch_size: int,
) -> tuple[list[str], dict[str, Any]]:
    selection_path = run_dir / "selection.json"
    manifest_path = run_dir / "manifest.json"
    if not selection_path.exists() or not manifest_path.exists():
        raise RuntimeError("run is not prepared; execute --stage prepare first")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selected = selection.get("problem_ids")
    if not isinstance(selected, list) or not all(isinstance(item, str) for item in selected):
        raise ValueError(f"invalid selection: {selection_path}")
    expected = expected_manifest(protocol, config_path, selected, batch_size)
    existing = json.loads(manifest_path.read_text(encoding="utf-8"))
    if existing != expected:
        raise RuntimeError(
            "current config/code metadata does not match the frozen manifest. "
            "Use the original settings or create a new run directory."
        )
    return selected, existing


# ---------------------------------------------------------------------------
# Batched generation: one process == one variant == one visible GPU
# ---------------------------------------------------------------------------

def chunked(items: list[Any], size: int) -> Iterable[list[Any]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def normalize_eos_ids(eos_token_id: Any) -> set[int]:
    if eos_token_id is None:
        return set()
    if isinstance(eos_token_id, int):
        return {eos_token_id}
    return {int(item) for item in eos_token_id}


def trim_completion_ids(
    token_ids: list[int],
    *,
    eos_token_ids: set[int],
    pad_token_id: int | None,
) -> tuple[list[int], bool]:
    """Trim one padded completion and report whether a natural EOS was seen."""
    for index, token_id in enumerate(token_ids):
        if token_id in eos_token_ids:
            return token_ids[:index], True
    if pad_token_id is not None:
        while token_ids and token_ids[-1] == pad_token_id:
            token_ids.pop()
    return token_ids, False


def prepare_generation_items(
    selected: list[str],
    records: dict[str, dict[str, Any]],
    protocol: dict[str, Any],
    tokenizer: Any,
) -> list[dict[str, Any]]:
    """Build prompts, count tokens, then length-bucket by sorting."""
    prepared: list[dict[str, Any]] = []
    for position, problem_id in enumerate(selected, 1):
        prompt = build_prompt(records[problem_id], protocol, tokenizer)
        prompt_ids = tokenizer(
            prompt,
            add_special_tokens=False,
            truncation=False,
        )["input_ids"]
        prepared.append(
            {
                "position": position,
                "problem_id": problem_id,
                "prompt": prompt,
                "prompt_tokens": len(prompt_ids),
            }
        )
    if bool(protocol["generation"].get("length_bucket", True)):
        prepared.sort(key=lambda item: (item["prompt_tokens"], item["problem_id"]))
    return prepared


def load_model_for_variant(
    protocol: dict[str, Any],
    *,
    variant: str,
    adapter_path: str | None,
    cache_dir: str | None,
) -> tuple[Any, Any]:
    """Load the base model or exactly one adapter on the process-visible GPU."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU fallback is forbidden")
    if torch.cuda.device_count() != 1:
        print(
            f"[warning] process sees {torch.cuda.device_count()} GPUs; "
            "for one-GPU-per-model launch with CUDA_VISIBLE_DEVICES=<id>",
            flush=True,
        )

    model_cfg = protocol["model"]
    model_name = str(model_cfg["name"])
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)

    # 【改动核心】decoder-only batched generation must use LEFT padding.
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype_name = str(model_cfg.get("dtype", "bfloat16"))
    dtype = getattr(torch, dtype_name)
    quantization_config = None
    if bool(model_cfg.get("load_in_4bit", True)):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=str(model_cfg.get("bnb_4bit_quant_type", "nf4")),
            bnb_4bit_use_double_quant=bool(
                model_cfg.get("bnb_4bit_use_double_quant", True)
            ),
            bnb_4bit_compute_dtype=dtype,
        )

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        quantization_config=quantization_config,
        torch_dtype=dtype,
        device_map={"": 0},
    ).eval()

    if variant == "base":
        if adapter_path:
            raise ValueError("base variant must not receive --adapter-path")
        model = base_model
    else:
        if not adapter_path:
            raise ValueError(f"variant {variant!r} requires --adapter-path")
        adapter = Path(adapter_path)
        if not adapter.is_absolute():
            adapter = ROOT / adapter
        if not adapter.exists():
            raise FileNotFoundError(adapter)
        model = PeftModel.from_pretrained(
            base_model,
            str(adapter),
            is_trainable=False,
        ).eval()

    model.config.use_cache = True
    return model, tokenizer


def preflight_variant(
    args: argparse.Namespace,
    protocol: dict[str, Any],
    config_path: Path,
    run_dir: Path,
) -> None:
    """Generate only the longest prompts to test batch-size memory safely.

    Nothing is written to the official generation JSONL. Run this after prepare
    and before launching the four-model matrix.
    """
    import torch

    if not args.variant:
        raise ValueError("--variant is required for preflight")
    batch_size = int(args.batch_size or protocol["generation"]["batch_size"])
    selected, _ = load_frozen_run(protocol, config_path, run_dir, batch_size)
    records = load_canonical(resolve_repo_path(str(protocol["dataset"]["canonical"])))
    model, tokenizer = load_model_for_variant(
        protocol,
        variant=args.variant,
        adapter_path=args.adapter_path,
        cache_dir=args.cache_dir,
    )
    prepared = prepare_generation_items(selected, records, protocol, tokenizer)
    longest = sorted(
        prepared,
        key=lambda item: (item["prompt_tokens"], item["problem_id"]),
        reverse=True,
    )[: max(1, int(args.preflight_items))]

    generation_cfg = protocol["generation"]
    max_new_tokens = int(generation_cfg["max_new_tokens"])
    num_beams = int(generation_cfg.get("num_beams", 1))
    reports: list[dict[str, Any]] = []

    for batch_index, batch in enumerate(chunked(longest, batch_size), 1):
        encoded = tokenizer(
            [item["prompt"] for item in batch],
            return_tensors="pt",
            padding=True,
            truncation=False,
            add_special_tokens=False,
        ).to(model.device)
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        try:
            with torch.inference_mode():
                sequences = model.generate(
                    **encoded,
                    do_sample=False,
                    max_new_tokens=max_new_tokens,
                    num_beams=num_beams,
                    use_cache=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
        except torch.cuda.OutOfMemoryError as exc:
            raise RuntimeError(
                f"preflight OOM with batch_size={batch_size}; use a NEW run "
                "directory and a smaller shared batch size for every variant"
            ) from exc
        elapsed = time.perf_counter() - started
        peak_gib = torch.cuda.max_memory_allocated() / (1024**3)
        generated_width = int(sequences.shape[1] - encoded.input_ids.shape[1])
        reports.append(
            {
                "batch": batch_index,
                "items": [item["problem_id"] for item in batch],
                "prompt_tokens": [item["prompt_tokens"] for item in batch],
                "generated_tensor_width": generated_width,
                "elapsed_seconds": round(elapsed, 3),
                "peak_memory_gib": round(peak_gib, 3),
            }
        )
        del sequences, encoded
        torch.cuda.empty_cache()

    report = {
        "schema_version": "codeguide-batch-preflight-v1",
        "variant": args.variant,
        "batch_size": batch_size,
        "max_new_tokens": max_new_tokens,
        "longest_items": len(longest),
        "batches": reports,
        "max_peak_memory_gib": max(row["peak_memory_gib"] for row in reports),
    }
    output = run_dir / "reports" / f"preflight_{args.variant}.json"
    write_json(output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def generate_variant(
    args: argparse.Namespace,
    protocol: dict[str, Any],
    config_path: Path,
    run_dir: Path,
) -> None:
    import torch

    if not args.variant:
        raise ValueError("--variant is required for generate")
    batch_size = int(args.batch_size or protocol["generation"]["batch_size"])
    selected, manifest = load_frozen_run(
        protocol, config_path, run_dir, batch_size
    )
    records = load_canonical(resolve_repo_path(str(protocol["dataset"]["canonical"])))
    model, tokenizer = load_model_for_variant(
        protocol,
        variant=args.variant,
        adapter_path=args.adapter_path,
        cache_dir=args.cache_dir,
    )
    prepared = prepare_generation_items(selected, records, protocol, tokenizer)

    output_path = run_dir / "generations" / f"{args.variant}.jsonl"
    completed = read_jsonl(output_path)
    pending = [item for item in prepared if item["problem_id"] not in completed]
    print(
        f"[{args.variant}] selected={len(selected)} completed={len(completed)} "
        f"pending={len(pending)} batch_size={batch_size}",
        flush=True,
    )
    if not pending:
        return

    generation_cfg = protocol["generation"]
    max_new_tokens = int(generation_cfg["max_new_tokens"])
    num_beams = int(generation_cfg.get("num_beams", 1))
    eos_ids = normalize_eos_ids(tokenizer.eos_token_id)

    # IMPORTANT: temperature/top_p are intentionally NOT passed.
    # do_sample=False means deterministic greedy decoding.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        for batch_index, batch in enumerate(chunked(pending, batch_size), 1):
            prompts = [item["prompt"] for item in batch]
            encoded = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=False,
                add_special_tokens=False,
            ).to(model.device)
            input_width = int(encoded.input_ids.shape[1])

            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            try:
                with torch.inference_mode():
                    sequences = model.generate(
                        **encoded,
                        do_sample=False,
                        max_new_tokens=max_new_tokens,
                        num_beams=num_beams,
                        use_cache=True,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
            except torch.cuda.OutOfMemoryError as exc:
                raise RuntimeError(
                    f"OOM with batch_size={batch_size}. Stop all variants and rerun "
                    "the matrix in a NEW run directory with one shared smaller batch size."
                ) from exc

            elapsed = time.perf_counter() - started
            peak_gib = torch.cuda.max_memory_allocated() / (1024**3)
            completion_rows = sequences[:, input_width:]

            for item, row in zip(batch, completion_rows):
                raw_ids = [int(token_id) for token_id in row.tolist()]
                effective_ids, saw_eos = trim_completion_ids(
                    raw_ids,
                    eos_token_ids=eos_ids,
                    pad_token_id=tokenizer.pad_token_id,
                )
                text = tokenizer.decode(effective_ids, skip_special_tokens=True)
                generated_tokens = len(effective_ids)
                record = {
                    "schema_version": "codeguide-generation-v1",
                    "problem_id": item["problem_id"],
                    "variant": args.variant,
                    "position": item["position"],
                    "protocol_name": protocol["protocol_name"],
                    "prompt_tokens": item["prompt_tokens"],
                    "generated_tokens": generated_tokens,
                    "hit_generation_limit": (
                        not saw_eos and generated_tokens >= max_new_tokens
                    ),
                    "natural_eos": saw_eos,
                    "batch_size": batch_size,
                    "model": manifest["model"],
                    "adapter_path": str(args.adapter_path) if args.adapter_path else None,
                    "text": text,
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                completed[item["problem_id"]] = record

            print(
                f"[{args.variant}] batch={batch_index} wrote={len(batch)} "
                f"total={len(completed)}/{len(selected)} elapsed={elapsed:.1f}s "
                f"peak={peak_gib:.2f}GiB input_width={input_width}",
                flush=True,
            )


# ---------------------------------------------------------------------------
# Offline strict Docker verification
# ---------------------------------------------------------------------------

def load_source_subset(source_path: Path, selected: set[str]) -> dict[str, dict[str, Any]]:
    source = {
        item["problem_id"]: item
        for item in iter_source_bank(source_path)
        if item.get("problem_id") in selected
    }
    missing = selected - set(source)
    if missing:
        raise RuntimeError(f"source bank misses selected IDs: {sorted(missing)[:5]}")
    return source


def interface_matches(code: str | None, source: dict[str, Any]) -> bool:
    if not code:
        return False
    if source.get("io_mode") == "call_based":
        fn_name = source.get("fn_name")
        return bool(fn_name and str(fn_name) in code)
    return True


def verify_one(
    problem_id: str,
    generation: dict[str, Any],
    source: dict[str, Any],
    *,
    required_sections: list[str],
    container_image: str,
    timeout: float,
    variant: str,
) -> dict[str, Any]:
    text = str(generation.get("text") or "")
    fn_name = source.get("fn_name")
    code = extract_code(
        text,
        io_mode=source.get("io_mode"),
        fn_name=fn_name,
        starter_code=source.get("starter_code"),
    )
    interface_ok = interface_matches(code, source)
    template_complete = all(
        section.lower() in text.lower() for section in required_sections
    )
    result = verify_code(
        code or "",
        {
            "test_cases": source.get("test_cases") or [],
            "io_mode": source.get("io_mode"),
            "fn_name": fn_name,
            "starter_code": source.get("starter_code"),
        },
        timeout=timeout,
        backend="docker",
        container_image=container_image,
    )
    strict_pass = (
        result.pass_rate == 1.0
        and not result.unsupported
        and result.error is None
    )
    if not code:
        failure_type = "missing_code"
    elif result.unsupported:
        failure_type = "unsupported"
    elif result.error:
        failure_type = "runtime_or_timeout"
    elif result.pass_rate < 1.0:
        failure_type = "wrong_answer"
    else:
        failure_type = None

    return {
        "schema_version": "codeguide-verification-v1",
        "problem_id": problem_id,
        "variant": variant,
        "strict_pass": strict_pass,
        "passed_cases": result.passed_cases,
        "total_cases": result.total_cases,
        "pass_rate": result.pass_rate,
        "io_mode": source.get("io_mode"),
        "interface_match": interface_ok,
        "template_complete": template_complete,
        "has_code": bool(code),
        "failure_type": failure_type,
        "error": result.error,
        "first_failure": result.first_failure,
        "execution_backend": result.execution_backend,
    }


def summarize_variant(
    selected: list[str],
    generations: dict[str, dict[str, Any]],
    verifications: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    generation_rows = [generations[problem_id] for problem_id in selected]
    verification_rows = [verifications[problem_id] for problem_id in selected]
    generated_tokens = [int(row["generated_tokens"]) for row in generation_rows]
    prompt_tokens = [int(row["prompt_tokens"]) for row in generation_rows]
    pass_rates = [float(row["pass_rate"]) for row in verification_rows]
    failures = Counter(
        row.get("failure_type")
        for row in verification_rows
        if row.get("failure_type")
    )
    return {
        "samples": len(selected),
        "passed": sum(bool(row["strict_pass"]) for row in verification_rows),
        "pass_at_1": sum(bool(row["strict_pass"]) for row in verification_rows)
        / len(selected),
        "mean_test_pass_rate": statistics.fmean(pass_rates),
        "code_block": sum(bool(row["has_code"]) for row in verification_rows),
        "interface_match": sum(
            bool(row["interface_match"]) for row in verification_rows
        ),
        "template_complete": sum(
            bool(row["template_complete"]) for row in verification_rows
        ),
        "average_generated_tokens": statistics.fmean(generated_tokens),
        "p50_generated_tokens": percentile(generated_tokens, 0.50),
        "p95_generated_tokens": percentile(generated_tokens, 0.95),
        "max_generated_tokens": max(generated_tokens),
        "average_prompt_tokens": statistics.fmean(prompt_tokens),
        "hit_generation_limit": sum(
            bool(row.get("hit_generation_limit")) for row in generation_rows
        ),
        "unclosed_code_fence": sum(
            str(row.get("text") or "").count("```") % 2 != 0
            for row in generation_rows
        ),
        "failure_types": dict(failures),
    }


def verify_variant(
    args: argparse.Namespace,
    protocol: dict[str, Any],
    config_path: Path,
    run_dir: Path,
) -> None:
    if not args.variant:
        raise ValueError("--variant is required for verify")
    if not args.container_image:
        raise ValueError("--container-image is required for verify")
    if "@sha256:" not in args.container_image:
        raise ValueError("container image must be pinned by digest")

    batch_size = int(args.batch_size or protocol["generation"]["batch_size"])
    selected, manifest = load_frozen_run(protocol, config_path, run_dir, batch_size)
    generations_path = run_dir / "generations" / f"{args.variant}.jsonl"
    generations = read_jsonl(generations_path)
    missing_generation = [pid for pid in selected if pid not in generations]
    if missing_generation:
        raise RuntimeError(
            f"generation is incomplete for {args.variant}: {missing_generation[:5]}"
        )

    source = load_source_subset(
        resolve_repo_path(str(protocol["dataset"]["source_bank"])),
        set(selected),
    )
    required_sections = [str(item) for item in protocol.get("required_sections", [])]
    timeout = float(protocol.get("verification", {}).get("timeout_seconds", 5.0))
    workers = int(
        args.verify_workers
        or protocol.get("verification", {}).get("workers", 4)
    )

    output_path = run_dir / "verification" / f"{args.variant}.jsonl"
    completed = read_jsonl(output_path)
    pending = [problem_id for problem_id in selected if problem_id not in completed]
    print(
        f"[verify:{args.variant}] completed={len(completed)} pending={len(pending)} "
        f"workers={workers}",
        flush=True,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_id = {
                executor.submit(
                    verify_one,
                    problem_id,
                    generations[problem_id],
                    source[problem_id],
                    required_sections=required_sections,
                    container_image=args.container_image,
                    timeout=timeout,
                    variant=args.variant,
                ): problem_id
                for problem_id in pending
            }
            for index, future in enumerate(as_completed(future_to_id), 1):
                problem_id = future_to_id[future]
                try:
                    row = future.result()
                except Exception as exc:
                    row = {
                        "schema_version": "codeguide-verification-v1",
                        "problem_id": problem_id,
                        "variant": args.variant,
                        "strict_pass": False,
                        "passed_cases": 0,
                        "total_cases": 0,
                        "pass_rate": 0.0,
                        "io_mode": source[problem_id].get("io_mode"),
                        "interface_match": False,
                        "template_complete": False,
                        "has_code": False,
                        "failure_type": "verifier_exception",
                        "error": f"{type(exc).__name__}: {exc}",
                        "first_failure": None,
                        "execution_backend": "docker",
                    }
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                completed[problem_id] = row
                if index % 10 == 0 or index == len(pending):
                    print(
                        f"[verify:{args.variant}] {len(completed)}/{len(selected)}",
                        flush=True,
                    )

    if set(completed) != set(selected):
        missing = set(selected) - set(completed)
        raise RuntimeError(f"verification incomplete: {sorted(missing)[:5]}")
    report = {
        "schema_version": "codeguide-strict-variant-report-v1",
        "protocol_name": protocol["protocol_name"],
        "variant": args.variant,
        "model": manifest["model"],
        "container_image": args.container_image,
        "metrics": summarize_variant(selected, generations, completed),
    }
    report_path = run_dir / "reports" / f"strict_{args.variant}.json"
    write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


# ---------------------------------------------------------------------------
# Matrix summary: Base / calibration / checkpoints / Full
# ---------------------------------------------------------------------------

def discover_variants(run_dir: Path) -> list[str]:
    return sorted(path.stem for path in (run_dir / "generations").glob("*.jsonl"))


def summarize_matrix(
    protocol: dict[str, Any],
    config_path: Path,
    run_dir: Path,
    batch_size: int,
) -> None:
    selected, manifest = load_frozen_run(protocol, config_path, run_dir, batch_size)
    variants = discover_variants(run_dir)
    if not variants:
        raise RuntimeError("no generation variants found")

    matrix: dict[str, Any] = {
        "schema_version": "codeguide-checkpoint-matrix-v1",
        "protocol_name": protocol["protocol_name"],
        "samples": len(selected),
        "model": manifest["model"],
        "variants": {},
    }
    pass_sets: dict[str, set[str]] = {}

    for variant in variants:
        generations = read_jsonl(run_dir / "generations" / f"{variant}.jsonl")
        verifications = read_jsonl(run_dir / "verification" / f"{variant}.jsonl")
        if any(problem_id not in generations for problem_id in selected):
            matrix["variants"][variant] = {"status": "generation_incomplete"}
            continue
        if any(problem_id not in verifications for problem_id in selected):
            matrix["variants"][variant] = {"status": "verification_incomplete"}
            continue
        metrics = summarize_variant(selected, generations, verifications)
        matrix["variants"][variant] = {"status": "complete", **metrics}
        pass_sets[variant] = {
            problem_id
            for problem_id in selected
            if bool(verifications[problem_id].get("strict_pass"))
        }

    if "base" in pass_sets:
        base_pass = pass_sets["base"]
        for variant, passed in pass_sets.items():
            if variant == "base":
                continue
            matrix["variants"][variant]["rescued_vs_base"] = len(passed - base_pass)
            matrix["variants"][variant]["lost_vs_base"] = len(base_pass - passed)
            matrix["variants"][variant]["common_pass_vs_base"] = len(
                passed & base_pass
            )
            matrix["variants"][variant]["rescued_problem_ids"] = sorted(
                passed - base_pass
            )
            matrix["variants"][variant]["lost_problem_ids"] = sorted(
                base_pass - passed
            )

    output = run_dir / "reports" / "checkpoint_matrix.json"
    write_json(output, matrix)
    print(json.dumps(matrix, ensure_ascii=False, indent=2))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("prepare", "preflight", "generate", "verify", "summarize"),
        required=True,
    )
    parser.add_argument(
        "--protocol-config",
        default="configs/eval/taco515_compact_code_first_v1.yaml",
    )
    parser.add_argument(
        "--run-dir",
        default="outputs/sft/taco515_compact_code_first_v1",
    )
    parser.add_argument(
        "--variant",
        help="base / calibration500 / checkpoint_step_xxx / full",
    )
    parser.add_argument("--adapter-path")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--samples", type=int)
    parser.add_argument("--cache-dir")
    parser.add_argument("--container-image")
    parser.add_argument("--verify-workers", type=int)
    parser.add_argument(
        "--preflight-items",
        type=int,
        default=8,
        help="number of longest prompts to generate during the memory preflight",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config_path = resolve_repo_path(args.protocol_config)
    run_dir = resolve_repo_path(args.run_dir)
    protocol = load_protocol(config_path)
    batch_size = int(args.batch_size or protocol["generation"]["batch_size"])

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if args.stage in {"preflight", "generate", "verify"} and not args.variant:
        raise ValueError(f"--variant is required for stage={args.stage}")
    if args.variant == "base" and args.adapter_path:
        raise ValueError("base variant must not receive --adapter-path")
    if args.stage in {"preflight", "generate"} and args.variant != "base" and not args.adapter_path:
        raise ValueError("non-base generate requires --adapter-path")

    if args.stage == "prepare":
        prepare_run(
            protocol,
            config_path,
            run_dir,
            batch_size=batch_size,
            samples_override=args.samples,
        )
    elif args.stage == "preflight":
        preflight_variant(args, protocol, config_path, run_dir)
    elif args.stage == "generate":
        generate_variant(args, protocol, config_path, run_dir)
    elif args.stage == "verify":
        verify_variant(args, protocol, config_path, run_dir)
    elif args.stage == "summarize":
        summarize_matrix(protocol, config_path, run_dir, batch_size)


if __name__ == "__main__":
    main()
