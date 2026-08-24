#!/usr/bin/env python3
"""Blind Base/SFT/GRPO teaching evaluation with two independent API judges."""

from __future__ import annotations

import argparse
import asyncio
import copy
import gc
import json
import os
import random
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Reuse the canonical model loader and generation helpers. This evaluator must
# not grow a second Transformers/PEFT loading implementation.
from scripts.evaluate_sft_matrix import (  # noqa: E402
    chunked,
    load_model_for_variant,
    normalize_eos_ids,
    trim_completion_ids,
)
from src.training.sft_data import (  # noqa: E402
    load_canonical,
    load_id_list,
    stratified_sample_ids,
)

VARIANTS = ("base", "sft", "grpo")
PAIR_SPECS = (
    ("base_vs_sft", "base", "sft"),
    ("base_vs_grpo", "base", "grpo"),
    ("sft_vs_grpo", "sft", "grpo"),
)

JUDGE_SYSTEM_PROMPT = """You are an expert programming instructor evaluator.

You are comparing two answers to the same algorithm problem. Judge ONLY their
teaching quality. Do not infer which model produced either answer. Do not reward
length, verbosity, decorative formatting, or more section headings by themselves.
Focus on whether the response builds correct intuition, explains the algorithm
and why it works, keeps the reasoning coherent, aligns code with the explanation,
and is genuinely useful to a beginner.

Return one JSON object only. Do not use Markdown fences or add prose outside it."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("validate", "import", "generate", "judge", "report", "all"),
        default="all",
    )
    parser.add_argument(
        "--config", default="configs/eval/teaching_eval.yaml"
    )
    parser.add_argument("--base-model")
    parser.add_argument("--sft-adapter")
    parser.add_argument("--grpo-adapter")
    parser.add_argument("--base-results")
    parser.add_argument("--sft-results")
    parser.add_argument("--grpo-results")
    parser.add_argument("--num-samples", type=int)
    parser.add_argument("--cache-dir")
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_config(path: Path) -> dict[str, Any]:
    import yaml

    if not path.exists():
        raise FileNotFoundError(path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"invalid teaching evaluation config: {path}")

    required = {
        "dataset_path",
        "selection_ids_path",
        "output_dir",
        "evaluation_size",
        "seed",
        "model",
        "adapters",
        "generation_config",
        "judge_api",
        "criteria",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"teaching evaluation config missing: {missing}")

    criteria = payload["criteria"]
    if not isinstance(criteria, dict) or len(criteria) != 5:
        raise ValueError("teaching evaluation requires exactly five criteria")
    weight_sum = sum(float(item["weight"]) for item in criteria.values())
    if abs(weight_sum - 1.0) > 1e-9:
        raise ValueError(f"teaching criterion weights must sum to 1, got {weight_sum}")

    generation = payload["generation_config"]
    if bool(generation.get("do_sample")):
        if generation.get("temperature") is None or generation.get("top_p") is None:
            raise ValueError("sampled generation requires temperature and top_p")
    if int(generation.get("max_new_tokens", 0)) <= 0:
        raise ValueError("max_new_tokens must be positive")
    if int(generation.get("batch_size", 0)) <= 0:
        raise ValueError("batch_size must be positive")
    return payload


def prompt_messages(record: Mapping[str, Any]) -> list[dict[str, str]]:
    """Return system+user ChatML only; the reference label is never exposed."""
    raw = record.get("messages")
    if not isinstance(raw, list) or len(raw) < 2:
        raise ValueError("invalid canonical ChatML record")
    if raw[-1].get("role") != "assistant":
        raise ValueError("canonical ChatML record must end with assistant label")
    messages = copy.deepcopy(raw[:-1])
    if not messages or any(item.get("role") not in {"system", "user"} for item in messages):
        raise ValueError("blind teaching prompt may contain only system and user messages")
    if not any(item.get("role") == "user" for item in messages):
        raise ValueError("blind teaching prompt has no user question")
    return [
        {"role": str(item["role"]), "content": str(item.get("content") or "")}
        for item in messages
    ]


def question_text(record: Mapping[str, Any]) -> str:
    users = [
        message["content"]
        for message in prompt_messages(record)
        if message["role"] == "user"
    ]
    return "\n\n".join(users).strip()


def select_records(
    config: Mapping[str, Any], num_samples: int | None
) -> tuple[list[str], dict[str, dict[str, Any]]]:
    canonical = load_canonical(resolve_path(str(config["dataset_path"])))
    frozen_ids = load_id_list(resolve_path(str(config["selection_ids_path"])))
    missing = sorted(set(frozen_ids) - set(canonical))
    if missing:
        raise RuntimeError(f"selection IDs missing from canonical data: {missing[:5]}")

    size = int(num_samples or config["evaluation_size"])
    if size <= 0 or size > len(frozen_ids):
        raise ValueError(f"num_samples must be in [1, {len(frozen_ids)}]")
    pool = [canonical[problem_id] for problem_id in frozen_ids]
    selected = stratified_sample_ids(pool, size, int(config["seed"]))
    return selected, canonical


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(
        f"{path.suffix}.{os.getpid()}.{time.time_ns()}.tmp"
    )
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        for attempt in range(8):
            try:
                os.replace(temporary, path)
                return
            except PermissionError:
                if attempt == 7:
                    raise
                time.sleep(min(0.05 * (2**attempt), 1.0))
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except PermissionError:
            # A scanner may briefly retain the temporary file after a failed
            # replace. It is uniquely named and safe to clean up later.
            pass


def initialize_results(
    path: Path,
    selected: list[str],
    canonical: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    expected = [
        {
            "id": problem_id,
            "question": question_text(canonical[problem_id]),
            "base": "",
            "sft": "",
            "grpo": "",
        }
        for problem_id in selected
    ]
    if not path.exists():
        atomic_write_json(path, expected)
        return expected

    existing = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(existing, list):
        raise ValueError(f"results must be a JSON list: {path}")
    if [item.get("id") for item in existing] != selected:
        raise RuntimeError(
            "existing teaching results use a different frozen selection; "
            "use a new output directory instead of overwriting them"
        )
    for current, baseline in zip(existing, expected):
        if current.get("question") != baseline["question"]:
            raise RuntimeError(f"question changed for {baseline['id']}")
        for key in ("base", "sft", "grpo"):
            current.setdefault(key, "")
    return existing


def generation_source_plan(
    config: Mapping[str, Any], args: argparse.Namespace
) -> dict[str, dict[str, str]]:
    configured = config.get("generation_sources") or {}
    if configured and not isinstance(configured, dict):
        raise ValueError("generation_sources must be a mapping")
    overrides = {
        "base": args.base_results,
        "sft": args.sft_results,
        "grpo": args.grpo_results,
    }
    plan: dict[str, dict[str, str]] = {}
    for variant in VARIANTS:
        source = configured.get(variant) or {}
        if isinstance(source, str):
            source = {"path": source}
        if not isinstance(source, dict):
            raise ValueError(f"invalid generation source for {variant}")
        path = overrides[variant] or source.get("path")
        if not path:
            continue
        plan[variant] = {
            "path": str(path),
            "expected_variant": str(source.get("expected_variant") or variant),
        }
    return plan


def load_generation_source(
    path: Path, *, expected_variant: str
) -> tuple[dict[str, str], set[str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    answers: dict[str, str] = {}
    protocols: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            problem_id = str(row.get("problem_id") or row.get("id") or "").strip()
            answer = str(row.get("text") or row.get("answer") or "").strip()
            actual_variant = str(row.get("variant") or expected_variant)
            if not problem_id:
                raise ValueError(f"missing problem_id at {path}:{line_number}")
            if not answer:
                raise ValueError(f"empty answer for {problem_id} at {path}:{line_number}")
            if actual_variant != expected_variant:
                raise ValueError(
                    f"variant mismatch at {path}:{line_number}: "
                    f"expected {expected_variant}, got {actual_variant}"
                )
            if problem_id in answers:
                raise ValueError(f"duplicate problem_id {problem_id} in {path}")
            answers[problem_id] = answer
            protocol = str(row.get("protocol_name") or "").strip()
            if protocol:
                protocols.add(protocol)
    return answers, protocols


def import_generated_answers(
    *,
    results: list[dict[str, Any]],
    results_path: Path,
    source_plan: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    if set(source_plan) != set(VARIANTS):
        missing = sorted(set(VARIANTS) - set(source_plan))
        raise ValueError(f"generation sources missing variants: {missing}")

    selected = {str(item["id"]) for item in results}
    imported: dict[str, Any] = {}
    protocols: set[str] = set()
    for variant in VARIANTS:
        source = source_plan[variant]
        path = resolve_path(source["path"])
        answers, source_protocols = load_generation_source(
            path, expected_variant=source["expected_variant"]
        )
        missing = sorted(selected - set(answers))
        if missing:
            raise RuntimeError(
                f"generation source {path} is missing {len(missing)} selected IDs: "
                f"{missing[:5]}"
            )
        for item in results:
            problem_id = str(item["id"])
            current = str(item.get(variant) or "").strip()
            incoming = answers[problem_id]
            if current and current != incoming:
                raise RuntimeError(
                    f"existing {variant} answer differs for {problem_id}; "
                    "use a new output directory instead of overwriting it"
                )
            item[variant] = incoming
        protocols.update(source_protocols)
        imported[variant] = {
            "path": str(path),
            "available_records": len(answers),
            "imported_records": len(results),
            "expected_variant": source["expected_variant"],
            "protocols": sorted(source_protocols),
        }
    if len(protocols) > 1:
        raise RuntimeError(f"generation sources use different protocols: {sorted(protocols)}")
    atomic_write_json(results_path, results)
    return {
        "mode": "existing_generations",
        "protocols": sorted(protocols),
        "sources": imported,
    }


def model_protocol(config: Mapping[str, Any], base_model: str | None) -> dict[str, Any]:
    model = dict(config["model"])
    if base_model:
        model["name"] = base_model
    return {"model": model}


def generate_variant(
    *,
    variant: str,
    adapter_path: str | None,
    config: Mapping[str, Any],
    base_model: str | None,
    cache_dir: str | None,
    canonical: Mapping[str, Mapping[str, Any]],
    results: list[dict[str, Any]],
    results_path: Path,
) -> None:
    pending = [item for item in results if not str(item.get(variant) or "").strip()]
    if not pending:
        print(f"[{variant}] all {len(results)} answers already exist", flush=True)
        return
    if variant != "base" and not adapter_path:
        raise ValueError(f"--{variant}-adapter is required for generation")

    import torch

    protocol = model_protocol(config, base_model)
    model, tokenizer = load_model_for_variant(
        protocol,
        variant=variant,
        adapter_path=adapter_path,
        cache_dir=cache_dir,
    )
    generation = config["generation_config"]
    prepared: list[dict[str, Any]] = []
    for item in pending:
        messages = prompt_messages(canonical[item["id"]])
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_tokens = len(
            tokenizer(prompt, add_special_tokens=False, truncation=False)["input_ids"]
        )
        prepared.append(
            {"id": item["id"], "prompt": prompt, "prompt_tokens": prompt_tokens}
        )
    if bool(generation.get("length_bucket", True)):
        prepared.sort(key=lambda item: (item["prompt_tokens"], item["id"]))

    result_by_id = {item["id"]: item for item in results}
    batch_size = int(generation["batch_size"])
    max_new_tokens = int(generation["max_new_tokens"])
    eos_ids = normalize_eos_ids(tokenizer.eos_token_id)
    completed = len(results) - len(pending)

    try:
        for batch in chunked(prepared, batch_size):
            encoded = tokenizer(
                [item["prompt"] for item in batch],
                return_tensors="pt",
                padding=True,
                truncation=False,
                add_special_tokens=False,
            ).to(model.device)
            input_width = int(encoded.input_ids.shape[1])
            kwargs: dict[str, Any] = {
                "max_new_tokens": max_new_tokens,
                "do_sample": bool(generation.get("do_sample", False)),
                "num_beams": int(generation.get("num_beams", 1)),
                "use_cache": True,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
            }
            if kwargs["do_sample"]:
                kwargs["temperature"] = float(generation["temperature"])
                kwargs["top_p"] = float(generation["top_p"])
            with torch.inference_mode():
                sequences = model.generate(**encoded, **kwargs)

            for source, row in zip(batch, sequences[:, input_width:]):
                effective, _ = trim_completion_ids(
                    [int(token) for token in row.tolist()],
                    eos_token_ids=eos_ids,
                    pad_token_id=tokenizer.pad_token_id,
                )
                result_by_id[source["id"]][variant] = tokenizer.decode(
                    effective, skip_special_tokens=True
                )
                completed += 1
            atomic_write_json(results_path, results)
            print(f"[{variant}] {completed}/{len(results)}", flush=True)
            del sequences, encoded
    finally:
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()


def balanced_blind_orders(
    problem_ids: Iterable[str], seed: int
) -> dict[tuple[str, str], bool]:
    """Return deterministic swaps, with each pair split as evenly as possible."""
    ids = list(problem_ids)
    rng = random.Random(seed)
    assignments: dict[tuple[str, str], bool] = {}
    for pair_name, _, _ in PAIR_SPECS:
        swaps = [True] * (len(ids) // 2) + [False] * (len(ids) - len(ids) // 2)
        rng.shuffle(swaps)
        assignments.update(
            {(problem_id, pair_name): swap for problem_id, swap in zip(ids, swaps)}
        )
    return assignments


def build_judge_prompt(
    *,
    question: str,
    answer_a: str,
    answer_b: str,
    criteria: Mapping[str, Mapping[str, Any]],
) -> str:
    criteria_lines = "\n".join(
        f"- {key}: {item['label']} (weight {float(item['weight']):.2f})"
        for key, item in criteria.items()
    )
    dimension_schema = ",\n".join(
        f'    "{key}": {{"A": 0, "B": 0}}' for key in criteria
    )
    return f"""Algorithm problem:
<QUESTION>
{question}
</QUESTION>

Candidate answer A:
<ANSWER_A>
{answer_a}
</ANSWER_A>

Candidate answer B:
<ANSWER_B>
{answer_b}
</ANSWER_B>

Score each criterion from 0 to 10:
{criteria_lines}

The top-level score_A and score_B must be the weighted sums of the five
dimensions. Choose A, B, or TIE based only on teaching quality. Do not prefer an
answer merely because it is longer, more verbose, or more attractively formatted.

Return exactly this JSON structure:
{{
  "winner": "A/B/TIE",
  "score_A": 0,
  "score_B": 0,
  "dimensions": {{
{dimension_schema}
  }},
  "reason": "brief evidence-based explanation"
}}"""


def _extract_json(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    start = stripped.find("{")
    if start < 0:
        raise ValueError("judge response contains no JSON object")
    payload, _ = json.JSONDecoder().raw_decode(stripped[start:])
    if not isinstance(payload, dict):
        raise ValueError("judge response JSON must be an object")
    return payload


def parse_judgment(
    text: str, criteria: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    payload = _extract_json(text)
    winner = str(payload.get("winner") or "").upper()
    if winner not in {"A", "B", "TIE"}:
        raise ValueError(f"invalid judge winner: {winner!r}")
    dimensions = payload.get("dimensions")
    if not isinstance(dimensions, dict) or set(dimensions) != set(criteria):
        raise ValueError("judge dimensions do not match the frozen criteria")

    normalized: dict[str, dict[str, float]] = {}
    for key in criteria:
        value = dimensions[key]
        if not isinstance(value, dict):
            raise ValueError(f"invalid dimension {key}")
        scores = {label: float(value[label]) for label in ("A", "B")}
        if any(score < 0 or score > 10 for score in scores.values()):
            raise ValueError(f"dimension {key} score outside [0, 10]")
        normalized[key] = scores

    weighted = {
        label: sum(
            normalized[key][label] * float(criteria[key]["weight"])
            for key in criteria
        )
        for label in ("A", "B")
    }
    reported = {label: float(payload[f"score_{label}"]) for label in ("A", "B")}
    if any(score < 0 or score > 10 for score in reported.values()):
        raise ValueError("top-level judge score outside [0, 10]")
    return {
        "winner": winner,
        "score_A": round(weighted["A"], 4),
        "score_B": round(weighted["B"], 4),
        "reported_score_A": reported["A"],
        "reported_score_B": reported["B"],
        "dimensions": normalized,
        "reason": str(payload.get("reason") or "").strip(),
    }


def _first_env(names: Any) -> str | None:
    if isinstance(names, str):
        names = [names]
    for name in names or []:
        value = os.environ.get(str(name))
        if value:
            return value
    return None


def judge_connection(name: str, config: Mapping[str, Any]) -> dict[str, str]:
    api_key = _first_env(config.get("api_key_env"))
    base_url = _first_env(config.get("base_url_env")) or config.get("base_url_default")
    model = _first_env(config.get("model_env")) or config.get("model_default")
    missing = [
        field
        for field, value in (("api key", api_key), ("base URL", base_url), ("model", model))
        if not value
    ]
    if missing:
        raise RuntimeError(f"judge {name} missing {', '.join(missing)} environment/config")
    return {
        "api_key": str(api_key),
        "base_url": str(base_url),
        "model": str(model),
        "display_name": str(config.get("display_name") or name),
    }


async def run_judges(
    *,
    results: list[dict[str, Any]],
    results_path: Path,
    config: Mapping[str, Any],
) -> None:
    from openai import AsyncOpenAI

    api_cfg = config["judge_api"]
    judge_cfgs = api_cfg.get("judges")
    if not isinstance(judge_cfgs, dict) or set(judge_cfgs) != {"deepseek", "qwen"}:
        raise ValueError("exactly the deepseek and qwen judges must be configured")
    connections = {
        name: judge_connection(name, settings)
        for name, settings in judge_cfgs.items()
    }
    clients = {
        name: AsyncOpenAI(
            api_key=connection["api_key"],
            base_url=connection["base_url"],
            timeout=float(api_cfg["timeout_seconds"]),
        )
        for name, connection in connections.items()
    }
    semaphore = asyncio.Semaphore(int(api_cfg["concurrency"]))
    orders = balanced_blind_orders(
        [str(item["id"]) for item in results], int(config["seed"])
    )
    criteria = config["criteria"]

    jobs: list[tuple[int, str, str, str, str, str, str]] = []
    for index, item in enumerate(results):
        item.setdefault("judgments", {})
        for judge_name in judge_cfgs:
            item["judgments"].setdefault(judge_name, {})
            for pair_name, left, right in PAIR_SPECS:
                if pair_name in item["judgments"][judge_name]:
                    continue
                swap = orders[(str(item["id"]), pair_name)]
                model_a, model_b = (right, left) if swap else (left, right)
                jobs.append(
                    (
                        index,
                        judge_name,
                        pair_name,
                        model_a,
                        model_b,
                        str(item[model_a]),
                        str(item[model_b]),
                    )
                )

    async def execute(job: tuple[int, str, str, str, str, str, str]):
        index, judge_name, pair_name, model_a, model_b, answer_a, answer_b = job
        if not answer_a.strip() or not answer_b.strip():
            raise RuntimeError(f"missing generated answer for {results[index]['id']} {pair_name}")
        prompt = build_judge_prompt(
            question=str(results[index]["question"]),
            answer_a=answer_a,
            answer_b=answer_b,
            criteria=criteria,
        )
        connection = connections[judge_name]
        last_error: Exception | None = None
        for attempt in range(int(api_cfg["max_retries"]) + 1):
            try:
                async with semaphore:
                    request: dict[str, Any] = {
                        "model": connection["model"],
                        "messages": [
                            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                            {"role": "user", "content": prompt},
                        ],
                        "temperature": float(api_cfg["temperature"]),
                        "max_tokens": int(api_cfg["max_tokens"]),
                    }
                    settings = judge_cfgs[judge_name]
                    if settings.get("response_format") == "json_object":
                        request["response_format"] = {"type": "json_object"}
                    if settings.get("extra_body"):
                        request["extra_body"] = dict(settings["extra_body"])
                    response = await clients[judge_name].chat.completions.create(
                        **request
                    )
                content = response.choices[0].message.content or ""
                parsed = parse_judgment(content, criteria)
                parsed.update(
                    {
                        "order": {"A": model_a, "B": model_b},
                        "winner_model": (
                            parsed["winner"]
                            if parsed["winner"] == "TIE"
                            else {"A": model_a, "B": model_b}[parsed["winner"]]
                        ),
                        "judge_model": connection["model"],
                    }
                )
                return index, judge_name, pair_name, parsed
            except Exception as exc:  # API and schema failures share bounded retry.
                last_error = exc
                if attempt < int(api_cfg["max_retries"]):
                    await asyncio.sleep(2**attempt)
        raise RuntimeError(
            f"judge {judge_name} failed for {results[index]['id']} {pair_name}: {last_error}"
        ) from last_error

    tasks = [asyncio.create_task(execute(job)) for job in jobs]
    completed = 0
    try:
        for task in asyncio.as_completed(tasks):
            index, judge_name, pair_name, parsed = await task
            results[index]["judgments"][judge_name][pair_name] = parsed
            completed += 1
            atomic_write_json(results_path, results)
            print(f"[judge] {completed}/{len(tasks)}", flush=True)
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        for client in clients.values():
            await client.close()


def _mean(values: Iterable[float]) -> float:
    materialized = list(values)
    return statistics.fmean(materialized) if materialized else 0.0


def aggregate_report(
    results: list[dict[str, Any]], config: Mapping[str, Any]
) -> dict[str, Any]:
    judge_names = tuple(config["judge_api"]["judges"])
    criteria = config["criteria"]
    absolute: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    pairwise: dict[str, dict[str, dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: {"left": 0, "right": 0, "tie": 0, "total": 0})
    )

    for item in results:
        judgments = item.get("judgments") or {}
        for judge_name in judge_names:
            if set((judgments.get(judge_name) or {})) != {pair[0] for pair in PAIR_SPECS}:
                raise RuntimeError(f"incomplete {judge_name} judgments for {item['id']}")
            for pair_name, left, right in PAIR_SPECS:
                judgment = judgments[judge_name][pair_name]
                order = judgment["order"]
                for label in ("A", "B"):
                    model = order[label]
                    for dimension in criteria:
                        absolute[judge_name][model][dimension].append(
                            float(judgment["dimensions"][dimension][label])
                        )
                winner = judgment["winner_model"]
                bucket = pairwise[judge_name][pair_name]
                bucket["total"] += 1
                if winner == "TIE":
                    bucket["tie"] += 1
                elif winner == left:
                    bucket["left"] += 1
                elif winner == right:
                    bucket["right"] += 1
                else:
                    raise ValueError(f"invalid mapped winner {winner}")

    teaching_scores: dict[str, dict[str, float]] = {}
    dimension_scores: dict[str, dict[str, dict[str, float]]] = {}
    for judge_name in (*judge_names, "combined"):
        dimension_scores[judge_name] = {}
        teaching_scores[judge_name] = {}
        for model in VARIANTS:
            dimensions: dict[str, float] = {}
            for dimension in criteria:
                if judge_name == "combined":
                    values = [
                        value
                        for judge in judge_names
                        for value in absolute[judge][model][dimension]
                    ]
                else:
                    values = absolute[judge_name][model][dimension]
                dimensions[dimension] = round(_mean(values), 4)
            dimension_scores[judge_name][model] = dimensions
            teaching_scores[judge_name][model] = round(
                sum(
                    dimensions[key] * float(criteria[key]["weight"])
                    for key in criteria
                ),
                4,
            )

    pooled_pairwise: dict[str, dict[str, float | int]] = {}
    for pair_name, left, right in PAIR_SPECS:
        buckets = [pairwise[judge][pair_name] for judge in judge_names]
        total = sum(bucket["total"] for bucket in buckets)
        left_wins = sum(bucket["left"] for bucket in buckets)
        right_wins = sum(bucket["right"] for bucket in buckets)
        ties = sum(bucket["tie"] for bucket in buckets)
        pooled_pairwise[pair_name] = {
            "left_model": left,
            "right_model": right,
            "left_wins": left_wins,
            "right_wins": right_wins,
            "ties": ties,
            "total": total,
            "left_win_rate": round(left_wins / total, 4),
            "right_win_rate": round(right_wins / total, 4),
            "tie_rate": round(ties / total, 4),
        }

    disagreements = 0
    comparisons = 0
    for item in results:
        for pair_name, _, _ in PAIR_SPECS:
            winners = [
                item["judgments"][judge][pair_name]["winner_model"]
                for judge in judge_names
            ]
            comparisons += 1
            disagreements += int(winners[0] != winners[1])

    return {
        "samples": len(results),
        "teaching_scores": teaching_scores,
        "dimension_scores": dimension_scores,
        "pairwise": pooled_pairwise,
        "judge_disagreement": {
            "disagreements": disagreements,
            "comparisons": comparisons,
            "rate": round(disagreements / comparisons, 4),
        },
    }


def write_report(
    path: Path,
    summary: Mapping[str, Any],
    config: Mapping[str, Any],
) -> None:
    combined = summary["teaching_scores"]["combined"]
    lines = [
        "# Blind Teaching Evaluation",
        "",
        f"- Samples: {summary['samples']}",
        "- Models: Base / SFT / GRPO",
        "- Judges: DeepSeek V4 Flash and Qwen3.8 Max",
        f"- Generation protocol: {config.get('generation_protocol', 'configured ChatML')}",
        "- Protocol: the same frozen prompts, hidden model identities, balanced A/B order",
        "- Reference teaching answers were not provided to either judge.",
        "",
        "## Average Teaching Score",
        "",
        "| Model | Score (0-10) |",
        "|---|---:|",
    ]
    for model in VARIANTS:
        lines.append(f"| {model.upper()} | {combined[model]:.4f} |")

    lines.extend(
        [
            "",
            "## Pairwise Win Rate",
            "",
            "Win rates use all comparisons from both judges; ties remain in the denominator.",
            "",
            "| Comparison | First wins | Second wins | Ties |",
            "|---|---:|---:|---:|",
        ]
    )
    display_pairs = (
        ("base_vs_sft", "SFT", "Base", "right_win_rate", "left_win_rate"),
        ("base_vs_grpo", "GRPO", "Base", "right_win_rate", "left_win_rate"),
        ("sft_vs_grpo", "GRPO", "SFT", "right_win_rate", "left_win_rate"),
    )
    for pair_name, first, second, first_key, second_key in display_pairs:
        row = summary["pairwise"][pair_name]
        lines.append(
            f"| {first} vs {second} | {row[first_key]:.2%} | "
            f"{row[second_key]:.2%} | {row['tie_rate']:.2%} |"
        )

    lines.extend(
        [
            "",
            "## Dimension Scores",
            "",
            "| Dimension | Base | SFT | GRPO |",
            "|---|---:|---:|---:|",
        ]
    )
    dimensions = summary["dimension_scores"]["combined"]
    for key, criterion in config["criteria"].items():
        lines.append(
            f"| {criterion['label']} | {dimensions['base'][key]:.4f} | "
            f"{dimensions['sft'][key]:.4f} | {dimensions['grpo'][key]:.4f} |"
        )

    disagreement = summary["judge_disagreement"]
    lines.extend(
        [
            "",
            "## Judge Disagreement",
            "",
            f"DeepSeek and Qwen selected different winners in "
            f"{disagreement['disagreements']}/{disagreement['comparisons']} comparisons "
            f"({disagreement['rate']:.2%}).",
            "",
            "Scores are LLM-as-Judge measurements of teaching quality, not execution-based "
            "code correctness results. Code capability remains reported by the frozen TACO/EvalPlus protocols.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_summary(
    config: Mapping[str, Any], selected: list[str], args: argparse.Namespace
) -> dict[str, Any]:
    adapters = config["adapters"]
    sft = args.sft_adapter or adapters.get("sft")
    grpo = args.grpo_adapter or adapters.get("grpo")
    return {
        "stage": "teaching_eval",
        "dataset_path": str(resolve_path(str(config["dataset_path"]))),
        "selection_ids_path": str(resolve_path(str(config["selection_ids_path"]))),
        "samples": len(selected),
        "base_model": args.base_model or config["model"]["name"],
        "generation_protocol": config.get("generation_protocol"),
        "sft_adapter": sft,
        "grpo_adapter": grpo,
        "generation_sources": generation_source_plan(config, args),
        "judges": list(config["judge_api"]["judges"]),
        "planned_successful_judgments": len(selected) * len(PAIR_SPECS) * 2,
        "results": str(resolve_path(str(config["output_dir"])) / "results.json"),
        "report": str(resolve_path(str(config["output_dir"])) / "report.md"),
    }


def main() -> None:
    args = parse_args()
    config_path = resolve_path(args.config)
    config = load_config(config_path)
    selected, canonical = select_records(config, args.num_samples)
    summary = validate_summary(config, selected, args)
    if args.stage == "validate":
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    output_dir = resolve_path(str(config["output_dir"]))
    results_path = output_dir / "results.json"
    report_path = output_dir / "report.md"
    results = initialize_results(results_path, selected, canonical)

    source_plan = generation_source_plan(config, args)
    if args.stage == "import" or (args.stage == "all" and source_plan):
        import_summary = import_generated_answers(
            results=results,
            results_path=results_path,
            source_plan=source_plan,
        )
        print(json.dumps(import_summary, ensure_ascii=False, indent=2))

    if args.stage == "generate" or (args.stage == "all" and not source_plan):
        adapters = config["adapters"]
        generation_plan = (
            ("base", None),
            ("sft", args.sft_adapter or adapters.get("sft")),
            ("grpo", args.grpo_adapter or adapters.get("grpo")),
        )
        for variant, adapter in generation_plan:
            generate_variant(
                variant=variant,
                adapter_path=str(adapter) if adapter else None,
                config=config,
                base_model=args.base_model,
                cache_dir=args.cache_dir,
                canonical=canonical,
                results=results,
                results_path=results_path,
            )

    if args.stage in {"judge", "all"}:
        asyncio.run(
            run_judges(results=results, results_path=results_path, config=config)
        )

    if args.stage in {"report", "all"}:
        report = aggregate_report(results, config)
        write_report(report_path, report, config)
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
