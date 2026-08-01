"""Frozen SFT data loading, assistant-only tokenization, and dynamic padding."""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

IGNORE_INDEX = -100


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_id_list(path: Path) -> list[str]:
    payload = load_json(path)
    if isinstance(payload, dict):
        payload = payload.get("ids")
    if not isinstance(payload, list) or not all(isinstance(item, str) for item in payload):
        raise ValueError(f"invalid ID file: {path}")
    return payload


def load_canonical(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            problem_id = record.get("id") or record.get("problem_id")
            if not problem_id:
                raise ValueError(f"missing problem_id at {path}:{line_no}")
            if problem_id in records:
                raise ValueError(f"duplicate problem_id: {problem_id}")
            records[problem_id] = record
    return records


def validate_frozen_splits(
    records: dict[str, dict[str, Any]], train_ids: list[str], dev_ids: list[str]
) -> None:
    train, dev, canonical = set(train_ids), set(dev_ids), set(records)
    if len(train) != len(train_ids) or len(dev) != len(dev_ids):
        raise ValueError("duplicate IDs in frozen split")
    if train & dev:
        raise ValueError("frozen train/dev overlap")
    if train | dev != canonical:
        missing = canonical - (train | dev)
        extra = (train | dev) - canonical
        raise ValueError(f"split coverage mismatch: missing={len(missing)}, extra={len(extra)}")


def split_records(
    canonical_path: Path, train_ids_path: Path, dev_ids_path: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records = load_canonical(canonical_path)
    train_ids = load_id_list(train_ids_path)
    dev_ids = load_id_list(dev_ids_path)
    validate_frozen_splits(records, train_ids, dev_ids)
    return [records[item] for item in train_ids], [records[item] for item in dev_ids]


def _stratum(record: dict[str, Any]) -> tuple[str, str, str, str]:
    metadata = record.get("metadata", {})
    return tuple(
        str(metadata.get(key) or "unknown")
        for key in ("label_strategy", "io_mode", "difficulty", "source")
    )


def stratified_sample_ids(
    records: Iterable[dict[str, Any]], sample_size: int, seed: int
) -> list[str]:
    """Deterministically sample across the four frozen metadata dimensions."""
    groups: dict[tuple[str, str, str, str], list[str]] = defaultdict(list)
    for record in records:
        groups[_stratum(record)].append(record.get("id") or record["problem_id"])

    rng = random.Random(seed)
    for ids in groups.values():
        ids.sort()
        rng.shuffle(ids)

    total = sum(map(len, groups.values()))
    if sample_size > total:
        raise ValueError(f"sample_size {sample_size} exceeds population {total}")

    quotas: dict[tuple[str, str, str, str], int] = {}
    fractions: list[tuple[float, tuple[str, str, str, str]]] = []
    assigned = 0
    for key, ids in sorted(groups.items()):
        exact = sample_size * len(ids) / total
        quota = min(len(ids), int(exact))
        quotas[key] = quota
        assigned += quota
        fractions.append((exact - quota, key))

    for _, key in sorted(fractions, key=lambda item: (-item[0], item[1])):
        if assigned == sample_size:
            break
        if quotas[key] < len(groups[key]):
            quotas[key] += 1
            assigned += 1

    selected = [item for key in sorted(groups) for item in groups[key][: quotas[key]]]
    return sorted(selected)


def ids_sha256(ids: list[str]) -> str:
    payload = json.dumps(ids, ensure_ascii=False, indent=2) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def tokenize_assistant_only(
    record: dict[str, Any], tokenizer: Any, max_seq_length: int
) -> dict[str, Any]:
    messages = record["messages"]
    if not messages or messages[-1].get("role") != "assistant":
        raise ValueError("the final message must be assistant")

    prompt_messages = messages[:-1]
    prompt_ids = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=True,
        add_generation_prompt=True,
    )
    full_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
    )
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("chat template prompt is not a prefix of the full conversation")
    if len(full_ids) > max_seq_length:
        problem_id = record.get("id") or record.get("problem_id")
        raise ValueError(
            f"sample {problem_id} has {len(full_ids)} tokens, exceeding {max_seq_length}; "
            "silent truncation is forbidden"
        )
    labels = [IGNORE_INDEX] * len(prompt_ids) + full_ids[len(prompt_ids) :]
    if not any(label != IGNORE_INDEX for label in labels):
        raise ValueError("sample has no supervised assistant tokens")
    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
        "length": len(full_ids),
        "problem_id": record.get("id") or record.get("problem_id"),
    }


@dataclass
class AssistantOnlyDataCollator:
    tokenizer: Any
    pad_to_multiple_of: int | None = 8

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        import torch
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            raise ValueError("tokenizer.pad_token_id must be configured")
        batch = pad_features(features, pad_id, self.pad_to_multiple_of)
        return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}


def pad_features(
    features: list[dict[str, Any]], pad_id: int, pad_to_multiple_of: int | None = 8
) -> dict[str, list[list[int]]]:
    """Pure-Python padding core, allowing local validation without installing torch."""
    max_length = max(len(item["input_ids"]) for item in features)
    if pad_to_multiple_of:
        max_length = ((max_length + pad_to_multiple_of - 1) // pad_to_multiple_of) * pad_to_multiple_of
    batch: dict[str, list[list[int]]] = {"input_ids": [], "attention_mask": [], "labels": []}
    for item in features:
        padding = max_length - len(item["input_ids"])
        batch["input_ids"].append(item["input_ids"] + [pad_id] * padding)
        batch["attention_mask"].append(item["attention_mask"] + [0] * padding)
        batch["labels"].append(item["labels"] + [IGNORE_INDEX] * padding)
    return batch
