from __future__ import annotations

import json
from argparse import Namespace

import pytest

from scripts.evaluate_sft_adapter import (
    persist_selection,
    read_jsonl,
    reuse_natural_completions,
    select_dev_ids,
)


def test_generation_jsonl_can_resume(tmp_path):
    path = tmp_path / "base_generations.jsonl"
    path.write_text(
        json.dumps({"problem_id": "a", "text": "first"}) + "\n"
        + json.dumps({"problem_id": "a", "text": "latest"}) + "\n",
        encoding="utf-8",
    )
    assert read_jsonl(path)["a"]["text"] == "latest"


def test_dev_selection_is_deterministic(tmp_path):
    path = tmp_path / "dev.json"
    path.write_text(json.dumps({"ids": [str(i) for i in range(20)]}), encoding="utf-8")
    assert select_dev_ids(path, 7, 42) == select_dev_ids(path, 7, 42)


def test_selection_manifest_rejects_changed_generation_contract(tmp_path):
    args = Namespace(seed=42, model="model", max_new_tokens=128)
    persist_selection(tmp_path, ["a", "b"], args)
    persist_selection(tmp_path, ["a", "b"], args)
    changed = Namespace(seed=42, model="model", max_new_tokens=256)
    with pytest.raises(RuntimeError, match="does not match"):
        persist_selection(tmp_path, ["a", "b"], changed)


def test_reuse_skips_only_generations_that_hit_old_limit(tmp_path):
    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir()
    target.mkdir()
    selection = {
        "problem_ids": ["a", "b"],
        "seed": 42,
        "model": "model",
        "max_new_tokens": 128,
    }
    (source / "selection.json").write_text(json.dumps(selection), encoding="utf-8")
    for variant in ("base", "adapter"):
        (source / f"{variant}_generations.jsonl").write_text(
            json.dumps({"problem_id": "a", "generated_tokens": 64, "text": "done"}) + "\n"
            + json.dumps({"problem_id": "b", "generated_tokens": 128, "text": "cut"}) + "\n",
            encoding="utf-8",
        )
    args = Namespace(seed=42, model="model", max_new_tokens=256)
    reuse_natural_completions(source, target, ["a", "b"], args)
    for variant in ("base", "adapter"):
        reused = read_jsonl(target / f"{variant}_generations.jsonl")
        assert set(reused) == {"a"}
        assert reused["a"]["reused_from"] == str(source)
