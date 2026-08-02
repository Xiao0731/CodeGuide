#!/usr/bin/env python3
"""Generate Base/Adapter answers on GPU, then verify them offline with Docker."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.source_bank_io import iter_source_bank
from src.data.code_validator import extract_code
from src.reward.execution import verify_code
from src.training.sft_data import load_canonical, load_id_list

SECTIONS = ("题意", "关键", "步骤", "复杂度", "错误", "```python")
VARIANTS = ("base", "adapter")


def read_jsonl(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    records = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                item = json.loads(line)
                records[item["problem_id"]] = item
    return records


def select_dev_ids(dev_ids_path: Path, samples: int, seed: int) -> list[str]:
    dev_ids = load_id_list(dev_ids_path)
    rng = random.Random(seed)
    return sorted(rng.sample(dev_ids, min(samples, len(dev_ids))))


def persist_selection(output_dir: Path, selected: list[str], args: argparse.Namespace) -> None:
    path = output_dir / "selection.json"
    selection = {
        "schema_version": "codeguide-sft-comparison-v1",
        "problem_ids": selected,
        "samples": len(selected),
        "seed": args.seed,
        "model": args.model,
        "max_new_tokens": args.max_new_tokens,
    }
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != selection:
            raise RuntimeError(f"existing generation selection does not match this run: {path}")
        return
    path.write_text(json.dumps(selection, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def reuse_natural_completions(
    source_dir: Path,
    output_dir: Path,
    selected: list[str],
    args: argparse.Namespace,
) -> None:
    source_selection_path = source_dir / "selection.json"
    if not source_selection_path.exists():
        raise RuntimeError(f"reuse source has no selection.json: {source_dir}")
    source_selection = json.loads(source_selection_path.read_text(encoding="utf-8"))
    for key, expected in (("problem_ids", selected), ("seed", args.seed), ("model", args.model)):
        if source_selection.get(key) != expected:
            raise RuntimeError(f"reuse source selection mismatch for {key}")
    source_limit = int(source_selection["max_new_tokens"])
    if source_limit > args.max_new_tokens:
        raise RuntimeError("cannot reuse generations produced with a larger token limit")

    for variant in VARIANTS:
        target_path = output_dir / f"{variant}_generations.jsonl"
        target = read_jsonl(target_path)
        reusable = read_jsonl(source_dir / f"{variant}_generations.jsonl")
        count = 0
        with target_path.open("a", encoding="utf-8") as handle:
            for problem_id in selected:
                item = reusable.get(problem_id)
                if problem_id in target or not item:
                    continue
                if int(item["generated_tokens"]) >= source_limit:
                    continue
                copied = dict(item)
                copied["reused_from"] = str(source_dir)
                handle.write(json.dumps(copied, ensure_ascii=False) + "\n")
                handle.flush()
                target[problem_id] = copied
                count += 1
        print(f"[{variant}] reused {count} natural completions from {source_dir}", flush=True)


def generate_answers(args: argparse.Namespace, output_dir: Path) -> list[str]:
    if not args.adapter_path:
        raise RuntimeError("adapter_path is required for generate/all stage")

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU fallback is forbidden")
    records = load_canonical(ROOT / args.canonical)
    selected = select_dev_ids(ROOT / args.dev_ids, args.samples, args.seed)
    persist_selection(output_dir, selected, args)
    if args.reuse_generations_from:
        reuse_natural_completions(
            ROOT / args.reuse_generations_from, output_dir, selected, args
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=args.cache_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    base = AutoModelForCausalLM.from_pretrained(
        args.model,
        cache_dir=args.cache_dir,
        quantization_config=quantization,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
    )
    base.eval()

    def generate(model, record: dict) -> tuple[str, int]:
        prompt = tokenizer.apply_chat_template(
            record["messages"][:-1], tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
            )
        token_count = int(generated.shape[1] - inputs.input_ids.shape[1])
        text = tokenizer.decode(
            generated[0, inputs.input_ids.shape[1] :], skip_special_tokens=True
        )
        return text, token_count

    models = {"base": base}
    for variant in VARIANTS:
        if variant == "adapter":
            models[variant] = PeftModel.from_pretrained(base, args.adapter_path).eval()
        path = output_dir / f"{variant}_generations.jsonl"
        completed = read_jsonl(path)
        with path.open("a", encoding="utf-8") as handle:
            for position, problem_id in enumerate(selected, 1):
                if problem_id in completed:
                    continue
                text, generated_tokens = generate(models[variant], records[problem_id])
                handle.write(json.dumps({
                    "problem_id": problem_id,
                    "variant": variant,
                    "position": position,
                    "generated_tokens": generated_tokens,
                    "text": text,
                }, ensure_ascii=False) + "\n")
                handle.flush()
                print(f"[{variant}] {position}/{len(selected)} {problem_id} tokens={generated_tokens}", flush=True)
    return selected


def verify_answers(args: argparse.Namespace, output_dir: Path) -> dict:
    if not args.container_image:
        raise RuntimeError("--container-image is required for verify/all stage")
    selection_path = output_dir / "selection.json"
    if not selection_path.exists():
        raise RuntimeError(f"missing cloud generation selection: {selection_path}")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selected = selection["problem_ids"]
    source = {
        item["problem_id"]: item
        for item in iter_source_bank(ROOT / args.source_bank)
        if item["problem_id"] in selected
    }
    if len(source) != len(selected):
        missing = sorted(set(selected) - set(source))
        raise RuntimeError(f"source bank does not cover selected dev prompts: {missing[:5]}")

    report = {
        "schema_version": "codeguide-sft-comparison-report-v1",
        "samples": len(selected),
        "selection": selection,
        "container_image": args.container_image,
        "models": {},
    }
    for variant in VARIANTS:
        generations = read_jsonl(output_dir / f"{variant}_generations.jsonl")
        missing = sorted(set(selected) - set(generations))
        if missing:
            raise RuntimeError(f"{variant} generation is incomplete: {missing[:5]}")
        failures = Counter()
        passed = code_blocks = interface_matches = complete = 0
        verification_path = output_dir / f"{variant}_verification.jsonl"
        with verification_path.open("w", encoding="utf-8") as handle:
            for problem_id in selected:
                text = generations[problem_id]["text"]
                code = extract_code(text)
                code_blocks += bool(code)
                complete += all(section.lower() in text.lower() for section in SECTIONS)
                item = source[problem_id]
                fn_name = item.get("fn_name")
                interface_ok = not fn_name or bool(code and fn_name in code)
                interface_matches += interface_ok
                result = verify_code(
                    code or "",
                    {
                        "test_cases": item["test_cases"],
                        "io_mode": item["io_mode"],
                        "fn_name": fn_name,
                        "starter_code": item.get("starter_code"),
                    },
                    backend="docker",
                    container_image=args.container_image,
                )
                passed += result.pass_rate == 1.0
                if result.pass_rate < 1.0:
                    failure_type = (
                        "unsupported" if result.unsupported
                        else "runtime_or_timeout" if result.error
                        else "wrong_answer"
                    )
                    failures[failure_type] += 1
                handle.write(json.dumps({
                    "problem_id": problem_id,
                    "pass_rate": result.pass_rate,
                    "interface_match": interface_ok,
                    "error": result.error,
                    "first_failure": result.first_failure,
                }, ensure_ascii=False) + "\n")
                handle.flush()
        report["models"][variant] = {
            "template_complete": complete,
            "code_block": code_blocks,
            "interface_match": interface_matches,
            "pass_at_1": passed / len(selected),
            "passed": passed,
            "average_generated_tokens": sum(
                int(generations[problem_id]["generated_tokens"]) for problem_id in selected
            ) / len(selected),
            "max_generated_tokens": max(
                int(generations[problem_id]["generated_tokens"]) for problem_id in selected
            ),
            "hit_generation_limit": sum(
                int(generations[problem_id]["generated_tokens"])
                >= int(selection["max_new_tokens"])
                for problem_id in selected
            ),
            "unclosed_code_fence": sum(
                generations[problem_id]["text"].count("```") % 2 != 0
                for problem_id in selected
            ),
            "failure_types": dict(failures),
        }
    report_path = output_dir / "comparison_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("adapter_path", nargs="?")
    parser.add_argument("--stage", choices=("generate", "verify", "all"), default="all")
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--canonical", default="data/final/sft_accepted.jsonl")
    parser.add_argument("--dev-ids", default="data/splits/sft_dev_ids.json")
    parser.add_argument("--source-bank", default="data/final/taco_verified_source_bank.jsonl.zst")
    parser.add_argument("--output-dir", default="outputs/sft/calibration_eval")
    parser.add_argument("--samples", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--cache-dir")
    parser.add_argument("--reuse-generations-from")
    parser.add_argument("--container-image")
    args = parser.parse_args()

    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.stage in ("generate", "all"):
        generate_answers(args, output_dir)
    if args.stage in ("verify", "all"):
        verify_answers(args, output_dir)


if __name__ == "__main__":
    main()
