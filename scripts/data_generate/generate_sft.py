#!/usr/bin/env python3
"""
Generate an SFT dataset from APPS train problems using a DeepSeek/OpenAI-compatible chat model.

Default pilot setting:
- sample size: 500
- difficulty ratio: introductory : interview : competition = 2 : 2 : 1
  => 200 / 200 / 100 when sample_size = 500

What this script does:
1. Load a local APPS train file (JSON or JSONL).
2. Stratified-sample problems by difficulty.
3. Retrieve a few validated seed examples from data/seeds/seed.json.
4. Prompt ds-chat in ENGLISH to produce a fully annotated training record.
5. Validate, normalize, and write JSONL output.
6. Keep failure logs so the run is resumable and auditable.

Notes:
- The prompt is intentionally strict and field-by-field.
- The model is asked to output ENGLISH only.
- Original source fields are preserved by the script even if the model drifts.
- Only generated fields are trusted from the model: category, teaching_answer, quality_tags.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from openai import OpenAI
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "The 'openai' package is required. Install it with: pip install openai"
    ) from exc


ALLOWED_CATEGORIES = [
    "array",
    "string",
    "hash",
    "two_pointers_sliding_window",
    "binary_search",
    "backtracking",
    "dfs_bfs",
    "greedy",
    "dp",
    "graph",
]

ALLOWED_DIFFICULTIES = ["introductory", "interview", "competition"]

SYSTEM_PROMPT = (
    "You are a careful dataset annotator for algorithmic tutoring data. "
    "You must respond in English only. Output one JSON object and nothing else. "
    "Never use markdown fences. Never omit required keys."
)


def build_user_prompt(raw_problem: Dict[str, Any], few_shots: List[Dict[str, Any]], train_id: str) -> str:
    """Build a strict English prompt with full field-by-field rules."""
    raw_json = json.dumps(raw_problem, ensure_ascii=False, indent=2)
    few_shot_block = "\n\n".join(
        f"VALIDATED_SEED_EXAMPLE_{i+1}:\n{json.dumps(ex, ensure_ascii=False, indent=2)}"
        for i, ex in enumerate(few_shots)
    )

    return f"""
You will convert ONE APPS training problem into ONE fully annotated SFT record.
Respond in ENGLISH only.
Return EXACTLY ONE JSON object.
Do not wrap the JSON in markdown.
Do not add explanations before or after the JSON.

TARGET JSON SCHEMA
{{
  "train_ID": "{train_id}",
  "problem_id": <int>,
  "url": "<string>",
  "category": "<one of {ALLOWED_CATEGORIES}>",
  "difficulty": "<introductory|interview|competition>",
  "problem_statement": "<string>",
  "starter_code": "<string>",
  "io_format": {{
    "mode": "<standard_input|call_based>",
    "inputs": <array>,
    "outputs": <array>
  }},
  "teaching_answer": {{
    "problem_restatement": "<string>",
    "key_observation": "<string>",
    "step_by_step_plan": ["<string>", "..."],
    "complexity": {{
      "time": "<string>",
      "space": "<string>"
    }},
    "common_mistakes": ["<string>", "..."],
    "annotated_code": "<string>"
  }},
  "quality_tags": {{
    "clear_explanation": <true|false>,
    "contains_trap": <true|false>,
    "python_friendly": <true|false>,
    "verifiable_by_tests": <true|false>
  }}
}}

HARD RULES FOR EVERY FIELD
1. train_ID
- Use exactly: "{train_id}".

2. problem_id
- Copy the original problem_id exactly.

3. url
- Copy the original url exactly if present, otherwise use "".

4. category
- Choose exactly ONE primary solving pattern from this fixed enum:
  - array
  - string
  - hash
  - two_pointers_sliding_window
  - binary_search
  - backtracking
  - dfs_bfs
  - greedy
  - dp
  - graph
- Choose the category that best matches the main intended solution, not a secondary trick.

5. difficulty
- Copy the original difficulty exactly.
- Allowed values: introductory, interview, competition.

6. problem_statement
- Preserve the original problem statement exactly as the source content.
- Do not summarize it.
- Do not rewrite it.
- Do not delete examples or notes.

7. starter_code
- Preserve the original starter_code exactly.
- If the source starter_code is missing, output "".

8. io_format
- mode must be either "standard_input" or "call_based".
- Keep the source examples faithful.
- inputs and outputs must stay aligned with the source examples.
- Do not invent extra examples.

9. teaching_answer.problem_restatement
- Write in plain, natural English.
- Restate the task for a learner.
- Do NOT copy the original wording line by line.
- 1 to 3 sentences.
- Must be accurate and simpler than the original problem statement.

10. teaching_answer.key_observation
- State the single most decisive insight.
- This should be the “unlock” idea, not a generic comment like “use DP”.
- 1 to 3 sentences.
- It should explain why the intended solution works.

11. teaching_answer.step_by_step_plan
- Write a concrete execution plan.
- Use 4 to 8 short imperative steps.
- Steps must be operational and ordered.
- Avoid vague filler like “solve the problem carefully”.

12. teaching_answer.complexity.time and space
- Must be in Big-O form.
- Also include a brief justification in plain English.
- Example style: "O(N), because we scan the array once." 

13. teaching_answer.common_mistakes
- Write 2 to 4 concrete mistakes.
- Focus on actual traps: complexity mistakes, boundary mistakes, or conceptual blind spots.
- Do not write generic advice like “be careful with bugs”.

14. teaching_answer.annotated_code
- Must be valid, executable Python 3.
- Comments must be in English only.
- The code must match io_format.mode:
  - If mode == "call_based": implement the callable solution in the expected style.
  - If mode == "standard_input": provide a full stdin/stdout program.
- The code must be self-contained.
- Do not include markdown fences.
- Do not include alternative solutions.
- Do not include pseudo-code.

15. quality_tags.clear_explanation
- true if the teaching_answer is genuinely clear enough for an intermediate learner to follow and reproduce.
- false only if the explanation is substantially unclear or incomplete.

16. quality_tags.contains_trap
- Use this strict definition:
  - true if the problem contains at least one notable complexity trap, boundary pitfall, or non-obvious conceptual blind spot.
  - false if the problem is mostly straightforward and errors would mainly come from basic implementation rather than a real trap.
- Do NOT overuse true.

17. quality_tags.python_friendly
- true if a natural Python solution exists without unusual low-level optimization or language-specific hacks.
- false if the problem is awkward in Python because of performance, recursion depth, heavy constant factors, or unusually tricky implementation demands.

18. quality_tags.verifiable_by_tests
- true if the answer can be checked by standard deterministic testing (including normal floating-point tolerance if appropriate).
- false for multi-solution outputs, special-judge style constructions, interactive tasks, or outputs where many different valid answers may exist.

GLOBAL QUALITY RULES
- English only.
- Be faithful to the source problem.
- Do not hallucinate constraints or examples.
- Do not omit keys.
- Do not use null.
- Use empty strings only where the source genuinely has no content.
- The final JSON must be parseable.

SOURCE APPS PROBLEM
{raw_json}

VALIDATED FEW-SHOT SEED EXAMPLES
{few_shot_block}

Now return the final JSON object only.
""".strip()


def load_records(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    if path.suffix.lower() == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        raise ValueError("JSON input must be a list of objects.")

    raise ValueError("Unsupported file format. Use .json or .jsonl")


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_difficulty(value: Any) -> str:
    s = str(value or "").strip().lower()
    if s not in ALLOWED_DIFFICULTIES:
        raise ValueError(f"Unsupported difficulty: {value!r}")
    return s


def infer_mode(raw: Dict[str, Any]) -> str:
    starter = raw.get("starter_code") or ""
    io = raw.get("input_output") or {}

    if isinstance(io, dict) and io.get("fn_name"):
        return "call_based"

    if starter:
        if "class Solution" in starter:
            return "call_based"
        if re.search(r"\bdef\s+\w+\s*\(", starter) and "input(" not in starter and "stdin" not in starter:
            return "call_based"

    return "standard_input"


def build_io_format(raw: Dict[str, Any]) -> Dict[str, Any]:
    io = raw.get("input_output") or {}
    if not isinstance(io, dict):
        io = {}
    return {
        "mode": infer_mode(raw),
        "inputs": io.get("inputs", []),
        "outputs": io.get("outputs", []),
    }


def stratified_sample(
    rows: List[Dict[str, Any]],
    total: int,
    ratio: Tuple[int, int, int],
    seed: int,
) -> List[Dict[str, Any]]:
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        diff = normalize_difficulty(row.get("difficulty"))
        buckets[diff].append(row)

    # Deterministic shuffle per bucket
    rnd = random.Random(seed)
    for diff in ALLOWED_DIFFICULTIES:
        rnd.shuffle(buckets[diff])

    ratio_sum = sum(ratio)
    desired = {
        "introductory": total * ratio[0] // ratio_sum,
        "interview": total * ratio[1] // ratio_sum,
        "competition": total * ratio[2] // ratio_sum,
    }
    # put any rounding remainder on the largest bucket first order: intro, interview, competition
    remainder = total - sum(desired.values())
    for diff in ALLOWED_DIFFICULTIES:
        if remainder <= 0:
            break
        desired[diff] += 1
        remainder -= 1

    for diff, need in desired.items():
        have = len(buckets[diff])
        if have < need:
            raise ValueError(f"Not enough {diff} problems for requested sample. need={need}, have={have}")

    sampled: List[Dict[str, Any]] = []
    for diff in ALLOWED_DIFFICULTIES:
        sampled.extend(buckets[diff][: desired[diff]])

    rnd.shuffle(sampled)
    return sampled


def compact_seed(seed: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "train_ID": seed.get("train_ID", ""),
        "problem_id": seed.get("problem_id", ""),
        "category": seed.get("category", ""),
        "difficulty": seed.get("difficulty", ""),
        "problem_statement": seed.get("problem_statement", ""),
        "starter_code": seed.get("starter_code", ""),
        "io_format": seed.get("io_format", {}),
        "teaching_answer": seed.get("teaching_answer", {}),
        "quality_tags": seed.get("quality_tags", {}),
    }


def select_few_shots(
    seeds: List[Dict[str, Any]],
    difficulty: str,
    count: int,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    same_diff = [s for s in seeds if s.get("difficulty") == difficulty]
    by_cat: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for s in same_diff:
        by_cat[s.get("category", "")].append(s)

    selected: List[Dict[str, Any]] = []
    categories = list(by_cat.keys())
    rng.shuffle(categories)

    # First pass: one per category for diversity
    for cat in categories:
        if len(selected) >= count:
            break
        group = by_cat[cat]
        rng.shuffle(group)
        selected.append(compact_seed(group[0]))

    # Second pass: fill remaining with same-difficulty seeds
    if len(selected) < count:
        pool = same_diff[:]
        rng.shuffle(pool)
        seen = {s["train_ID"] for s in selected}
        for s in pool:
            if len(selected) >= count:
                break
            if s.get("train_ID") not in seen:
                selected.append(compact_seed(s))
                seen.add(s.get("train_ID"))

    # Last fallback: any seed if same-difficulty is insufficient
    if len(selected) < count:
        pool = seeds[:]
        rng.shuffle(pool)
        seen = {s["train_ID"] for s in selected}
        for s in pool:
            if len(selected) >= count:
                break
            if s.get("train_ID") not in seen:
                selected.append(compact_seed(s))
                seen.add(s.get("train_ID"))

    return selected


def extract_first_json_object(text: str) -> Dict[str, Any]:
    text = text.strip()
    if not text:
        raise ValueError("Empty model response")

    # Direct parse first
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    # Remove code fences if present
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    # Find first balanced JSON object
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found in model response")

    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
        else:
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : i + 1]
                    obj = json.loads(candidate)
                    if isinstance(obj, dict):
                        return obj
                    raise ValueError("Parsed JSON is not an object")
    raise ValueError("Could not extract a balanced JSON object")


def validate_boolean(value: Any, field: str) -> bool:
    if isinstance(value, bool):
        return value
    raise ValueError(f"Field {field} must be boolean, got {type(value).__name__}")


def sanitize_generated(
    generated: Dict[str, Any],
    raw: Dict[str, Any],
    train_id: str,
) -> Dict[str, Any]:
    category = generated.get("category", "")
    if category not in ALLOWED_CATEGORIES:
        raise ValueError(f"Invalid category: {category!r}")

    difficulty = normalize_difficulty(raw.get("difficulty"))

    teaching = generated.get("teaching_answer")
    if not isinstance(teaching, dict):
        raise ValueError("teaching_answer must be an object")

    quality = generated.get("quality_tags")
    if not isinstance(quality, dict):
        raise ValueError("quality_tags must be an object")

    # Preserve original source-aligned fields from raw
    final = {
        "train_ID": train_id,
        "problem_id": raw.get("problem_id", ""),
        "url": raw.get("url", "") or "",
        "category": category,
        "difficulty": difficulty,
        "problem_statement": raw.get("question", raw.get("problem_statement", "")) or "",
        "starter_code": raw.get("starter_code", "") or "",
        "io_format": build_io_format(raw),
        "teaching_answer": {
            "problem_restatement": str(teaching.get("problem_restatement", "")),
            "key_observation": str(teaching.get("key_observation", "")),
            "step_by_step_plan": list(teaching.get("step_by_step_plan", [])),
            "complexity": {
                "time": str((teaching.get("complexity") or {}).get("time", "")),
                "space": str((teaching.get("complexity") or {}).get("space", "")),
            },
            "common_mistakes": list(teaching.get("common_mistakes", [])),
            "annotated_code": str(teaching.get("annotated_code", "")),
        },
        "quality_tags": {
            "clear_explanation": validate_boolean(quality.get("clear_explanation"), "quality_tags.clear_explanation"),
            "contains_trap": validate_boolean(quality.get("contains_trap"), "quality_tags.contains_trap"),
            "python_friendly": validate_boolean(quality.get("python_friendly"), "quality_tags.python_friendly"),
            "verifiable_by_tests": validate_boolean(quality.get("verifiable_by_tests"), "quality_tags.verifiable_by_tests"),
        },
    }

    # Minimal structural checks
    if final["io_format"]["mode"] not in {"standard_input", "call_based"}:
        raise ValueError("Invalid io_format.mode after normalization")
    if not isinstance(final["teaching_answer"]["step_by_step_plan"], list):
        raise ValueError("step_by_step_plan must be a list")
    if not isinstance(final["teaching_answer"]["common_mistakes"], list):
        raise ValueError("common_mistakes must be a list")
    if not final["teaching_answer"]["annotated_code"].strip():
        raise ValueError("annotated_code is empty")

    return final


def call_model(client: OpenAI, model: str, system_prompt: str, user_prompt: str, temperature: float, max_retries: int = 5) -> str:
    delay = 2.0
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            return resp.choices[0].message.content or ""
        except Exception as exc:  # pragma: no cover
            if attempt == max_retries:
                raise
            time.sleep(delay)
            delay *= 1.8
    raise RuntimeError("unreachable")


def resolve_default_seed_path(repo_root: Path) -> Optional[Path]:
    candidates = [
        repo_root / "data" / "seeds" / "seed.json",
        Path("/mnt/data/seed.json"),
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def resolve_default_apps_path(repo_root: Path) -> Optional[Path]:
    candidates = [
        repo_root / "data" / "raw" / "apps" / "train.jsonl",
        repo_root / "data" / "raw" / "apps_train.jsonl",
        repo_root / "data" / "apps" / "train.jsonl",
        repo_root / "data" / "raw" / "APPS_train.jsonl",
        repo_root / "data" / "raw" / "apps" / "train.json",
        repo_root / "data" / "raw" / "apps_train.json",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    default_seed_path = resolve_default_seed_path(repo_root)
    default_apps_path = resolve_default_apps_path(repo_root)

    parser = argparse.ArgumentParser(description="Generate an APPS -> SFT dataset using ds-chat with seed-based prompting.")
    parser.add_argument("--apps-train", type=Path, default=default_apps_path, help="Path to the APPS train JSON/JSONL file")
    parser.add_argument("--seed-path", type=Path, default=default_seed_path, help="Path to the validated seed.json file")
    parser.add_argument("--output", type=Path, default=repo_root / "data" / "sft" / "apps_train_sample_500.jsonl", help="Output JSONL path")
    parser.add_argument("--fail-log", type=Path, default=repo_root / "data" / "sft" / "apps_train_sample_500_failures.jsonl", help="Failure log JSONL path")
    parser.add_argument("--sample-size", type=int, default=500, help="Total number of sampled APPS train problems")
    parser.add_argument("--ratio", type=str, default="2:2:1", help="Difficulty ratio as introductory:interview:competition")
    parser.add_argument("--few-shot-count", type=int, default=4, help="How many validated seeds to include per prompt")
    parser.add_argument("--random-seed", type=int, default=42, help="Sampling seed")
    parser.add_argument("--train-id-start", type=int, default=100000, help="Starting integer for generated train_ID values")
    parser.add_argument("--model", type=str, default=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"), help="Model name for the OpenAI-compatible endpoint")
    parser.add_argument("--base-url", type=str, default=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"), help="OpenAI-compatible base URL")
    parser.add_argument("--api-key-env", type=str, default="DEEPSEEK_API_KEY", help="Environment variable name holding the API key")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap after sampling, useful for dry runs")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output file if present")
    parser.add_argument("--dry-run-prompt", action="store_true", help="Build and print the first prompt without calling the API")
    args = parser.parse_args()

    if args.apps_train is None:
        raise SystemExit("Could not find a default APPS train file. Pass --apps-train explicitly.")
    if args.seed_path is None:
        raise SystemExit("Could not find a default seed.json file. Pass --seed-path explicitly.")

    ratio_parts = tuple(int(x) for x in args.ratio.split(":"))
    if len(ratio_parts) != 3 or any(x <= 0 for x in ratio_parts):
        raise SystemExit("--ratio must look like 2:2:1")

    api_key = os.getenv(args.api_key_env)
    if not api_key and not args.dry_run_prompt:
        raise SystemExit(f"Missing API key in environment variable: {args.api_key_env}")

    rows = load_records(args.apps_train)
    seeds = load_records(args.seed_path)
    sampled = stratified_sample(rows, total=args.sample_size, ratio=ratio_parts, seed=args.random_seed)
    if args.limit is not None:
        sampled = sampled[: args.limit]

    processed_ids = set()
    if args.resume and args.output.exists():
        for row in load_records(args.output):
            processed_ids.add(row.get("problem_id"))

    if args.dry_run_prompt:
        first = sampled[0]
        rng = random.Random(args.random_seed + int(first.get("problem_id", 0) or 0))
        few = select_few_shots(seeds, normalize_difficulty(first.get("difficulty")), args.few_shot_count, rng)
        train_id = f"train_{args.train_id_start:06d}"
        raw_problem = {
            "source": first.get("source", "APPS"),
            "problem_id": first.get("problem_id", ""),
            "title": first.get("title", ""),
            "question": first.get("question", first.get("problem_statement", "")),
            "difficulty": normalize_difficulty(first.get("difficulty")),
            "url": first.get("url", "") or "",
            "starter_code": first.get("starter_code", "") or "",
            "input_output": first.get("input_output", {}),
            "reference_solutions": first.get("reference_solutions", first.get("solutions", [])),
        }
        print(build_user_prompt(raw_problem, few, train_id))
        return

    client = OpenAI(api_key=api_key, base_url=args.base_url)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.fail_log.parent.mkdir(parents=True, exist_ok=True)

    completed = 0
    for idx, raw in enumerate(sampled, start=0):
        problem_id = raw.get("problem_id")
        if args.resume and problem_id in processed_ids:
            continue

        difficulty = normalize_difficulty(raw.get("difficulty"))
        train_id = f"train_{args.train_id_start + idx:06d}"

        raw_problem = {
            "source": raw.get("source", "APPS"),
            "problem_id": raw.get("problem_id", ""),
            "title": raw.get("title", ""),
            "question": raw.get("question", raw.get("problem_statement", "")),
            "difficulty": difficulty,
            "url": raw.get("url", "") or "",
            "starter_code": raw.get("starter_code", "") or "",
            "input_output": raw.get("input_output", {}),
            "reference_solutions": raw.get("reference_solutions", raw.get("solutions", [])),
        }

        rng = random.Random(args.random_seed + int(problem_id or 0))
        few_shots = select_few_shots(seeds, difficulty, args.few_shot_count, rng)
        user_prompt = build_user_prompt(raw_problem, few_shots, train_id)

        try:
            text = call_model(
                client=client,
                model=args.model,
                system_prompt=SYSTEM_PROMPT,
                user_prompt=user_prompt,
                temperature=args.temperature,
            )
            generated = extract_first_json_object(text)
            final = sanitize_generated(generated, raw_problem, train_id)
            append_jsonl(args.output, final)
            completed += 1
            print(f"[OK] {train_id} problem_id={problem_id} difficulty={difficulty}")
        except Exception as exc:
            fail_row = {
                "train_ID": train_id,
                "problem_id": problem_id,
                "difficulty": difficulty,
                "title": raw.get("title", ""),
                "error": repr(exc),
                "raw_problem": raw_problem,
            }
            append_jsonl(args.fail_log, fail_row)
            print(f"[FAIL] {train_id} problem_id={problem_id}: {exc}", file=sys.stderr)

    print(
        f"Done. requested={len(sampled)} completed={completed} output={args.output} failures={args.fail_log}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
