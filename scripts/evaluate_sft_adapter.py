#!/usr/bin/env python3
"""Deterministic Base/Adapter generation comparison after SFT calibration."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.source_bank_io import iter_jsonl
from src.data.code_validator import extract_code
from src.reward.execution import verify_code
from src.training.sft_data import load_canonical, load_id_list

SECTIONS = ("题意", "关键", "步骤", "复杂度", "错误", "```python")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("adapter_path")
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--canonical", default="data/final/sft_accepted.jsonl")
    parser.add_argument("--dev-ids", default="data/splits/sft_dev_ids.json")
    parser.add_argument("--source-bank", default="data/final/taco_verified_source_bank.jsonl.zst")
    parser.add_argument("--output-dir", default="outputs/sft/calibration_eval")
    parser.add_argument("--samples", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--container-image", required=True)
    args = parser.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    records = load_canonical(ROOT / args.canonical)
    dev_ids = load_id_list(ROOT / args.dev_ids)
    rng = random.Random(args.seed)
    selected = sorted(rng.sample(dev_ids, min(args.samples, len(dev_ids))))
    source = {item["problem_id"]: item for item in iter_jsonl(ROOT / args.source_bank) if item["problem_id"] in selected}
    if len(source) != len(selected):
        raise RuntimeError("source bank does not cover selected dev prompts")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    base = AutoModelForCausalLM.from_pretrained(args.model, quantization_config=quant, torch_dtype=torch.bfloat16, device_map={"": 0})
    base.eval()

    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    def generate(model, record):
        prompt = tokenizer.apply_chat_template(record["messages"][:-1], tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            generated = model.generate(**inputs, do_sample=False, max_new_tokens=args.max_new_tokens, pad_token_id=tokenizer.pad_token_id)
        return tokenizer.decode(generated[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)

    outputs = {"base": {}, "adapter": {}}
    for problem_id in selected:
        outputs["base"][problem_id] = generate(base, records[problem_id])
    adapted = PeftModel.from_pretrained(base, args.adapter_path)
    adapted.eval()
    for problem_id in selected:
        outputs["adapter"][problem_id] = generate(adapted, records[problem_id])

    report = {"samples": len(selected), "generation": {"do_sample": False, "max_new_tokens": args.max_new_tokens}, "models": {}}
    for variant in ("base", "adapter"):
        failures = Counter()
        passed = code_blocks = interface_matches = complete = 0
        raw_path = output_dir / f"{variant}_generations.jsonl"
        with raw_path.open("w", encoding="utf-8") as handle:
            for problem_id in selected:
                text = outputs[variant][problem_id]
                code = extract_code(text)
                code_blocks += bool(code)
                complete += all(section.lower() in text.lower() for section in SECTIONS)
                item = source[problem_id]
                fn_name = item.get("fn_name")
                interface_ok = not fn_name or (code and fn_name in code)
                interface_matches += interface_ok
                verification_metadata = {
                    "test_cases": item["test_cases"], "io_mode": item["io_mode"],
                    "fn_name": fn_name, "starter_code": item.get("starter_code"),
                }
                result = verify_code(
                    code or "", verification_metadata, backend="docker",
                    container_image=args.container_image,
                )
                passed += result.pass_rate == 1.0
                if result.pass_rate < 1.0:
                    failure_type = "unsupported" if result.unsupported else ("runtime_or_timeout" if result.error else "wrong_answer")
                    failures[failure_type] += 1
                handle.write(json.dumps({"problem_id": problem_id, "text": text, "pass_rate": result.pass_rate, "error": result.error, "first_failure": result.first_failure}, ensure_ascii=False) + "\n")
        report["models"][variant] = {
            "template_complete": complete, "code_block": code_blocks, "interface_match": interface_matches,
            "pass_at_1": passed / len(selected), "failure_types": dict(failures),
        }
    (output_dir / "comparison_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
