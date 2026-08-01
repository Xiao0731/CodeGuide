#!/usr/bin/env python3
"""Reload a saved QLoRA adapter and perform one deterministic generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("adapter_path")
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--canonical", default="data/final/sft_accepted.jsonl")
    parser.add_argument("--cache-dir")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    adapter_path = Path(args.adapter_path)
    if not (adapter_path / "adapter_config.json").exists():
        raise RuntimeError(f"adapter_config.json is missing: {adapter_path}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU fallback is forbidden")

    with Path(args.canonical).open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    record = min(records, key=lambda item: sum(len(message["content"]) for message in item["messages"][:-1]))

    tokenizer = AutoTokenizer.from_pretrained(adapter_path, cache_dir=args.cache_dir)
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
    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()
    prompt = tokenizer.apply_chat_template(
        record["messages"][:-1], tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=args.max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
        )
    completion = tokenizer.decode(
        output[0, inputs.input_ids.shape[1] :], skip_special_tokens=True
    )
    if not completion.strip():
        raise RuntimeError("reloaded adapter produced an empty completion")
    print(json.dumps({
        "adapter_reloaded": True,
        "problem_id": record["id"],
        "generated_tokens": int(output.shape[1] - inputs.input_ids.shape[1]),
        "completion_preview": completion[:300],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
