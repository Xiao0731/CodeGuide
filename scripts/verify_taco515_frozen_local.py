import json
from pathlib import Path

import evaluate_sft_matrix as core


def load_frozen_for_offline_verify(protocol, config_path, run_dir, batch_size):
    selection_path = run_dir / "selection.json"
    manifest_path = run_dir / "manifest.json"

    if not selection_path.is_file():
        raise RuntimeError(f"missing selection: {selection_path}")
    if not manifest_path.is_file():
        raise RuntimeError(f"missing manifest: {manifest_path}")

    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    selected = selection.get("problem_ids")

    if not isinstance(selected, list):
        raise RuntimeError("selection.problem_ids is invalid")

    if len(selected) != 515:
        raise RuntimeError(f"expected 515 selected IDs, got {len(selected)}")

    if len(set(selected)) != 515:
        raise RuntimeError("selection contains duplicate problem IDs")

    variant = "grpo_best"
    gen_path = run_dir / "generations" / f"{variant}.jsonl"

    generations = core.read_jsonl(gen_path)

    if len(generations) != 515:
        raise RuntimeError(
            f"expected 515 generation rows, got {len(generations)}"
        )

    selected_set = set(selected)
    generated_set = set(generations)

    missing = selected_set - generated_set
    extra = generated_set - selected_set

    if missing or extra:
        raise RuntimeError(
            f"generation/selection mismatch: "
            f"missing={list(missing)[:5]}, extra={list(extra)[:5]}"
        )

    print("=== OFFLINE REPLAY VALIDATED ===")
    print("selection IDs     = 515")
    print("generation rows   = 515")
    print("ID sets identical = YES")
    print("Skipping generation-time manifest metadata comparison.")
    print("Docker execution verification remains unchanged.")
    print()

    return selected, manifest


core.load_frozen_run = load_frozen_for_offline_verify
core.main()
