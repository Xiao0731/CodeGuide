from __future__ import annotations

from src.training.sft_data import IGNORE_INDEX, load_id_list, pad_features, stratified_sample_ids, tokenize_assistant_only


class FakeTokenizer:
    pad_token_id = 0

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        ids = []
        for message in messages:
            role = {"system": 10, "user": 20, "assistant": 30}[message["role"]]
            ids.extend([role] + [100 + ord(char) % 50 for char in message["content"]] + [40])
        if add_generation_prompt:
            ids.append(30)
        return ids


def record(problem_id="x", code="```python\nprint(1)\n```"):
    return {"id": problem_id, "messages": [
        {"role": "system", "content": "teach"}, {"role": "user", "content": "problem"},
        {"role": "assistant", "content": "explain\n" + code}],
        "metadata": {"label_strategy": "pedagogical_rewrite", "io_mode": "standard_input", "difficulty": "easy", "source": "taco"}}


def test_assistant_only_mask_and_code_supervision():
    tokenizer = FakeTokenizer()
    item = tokenize_assistant_only(record(), tokenizer, 8192)
    first = next(i for i, label in enumerate(item["labels"]) if label != IGNORE_INDEX)
    assert all(label == IGNORE_INDEX for label in item["labels"][:first])
    assert item["labels"][first:] == item["input_ids"][first:]
    assert first == len(tokenizer.apply_chat_template(record()["messages"][:-1], True, True))


def test_dynamic_padding_masks_padding_labels():
    tokenizer = FakeTokenizer()
    one = tokenize_assistant_only(record("a"), tokenizer, 8192)
    two = tokenize_assistant_only(record("b", "```python\nprint(123456)\n```"), tokenizer, 8192)
    batch = pad_features([one, two], tokenizer.pad_token_id, pad_to_multiple_of=None)
    assert len(batch["input_ids"]) == 2
    for labels, mask in zip(batch["labels"], batch["attention_mask"]):
        assert all(label == IGNORE_INDEX for label, active in zip(labels, mask) if not active)
    assert batch["labels"][0][: len(one["labels"])] == one["labels"]


def test_stratified_sample_is_deterministic():
    records = [record(str(i)) for i in range(20)]
    assert stratified_sample_ids(records, 7, 42) == stratified_sample_ids(records, 7, 42)


def test_load_frozen_id_manifest(tmp_path):
    path = tmp_path / "ids.json"
    path.write_text('{"count": 2, "ids": ["a", "b"]}', encoding="utf-8")
    assert load_id_list(path) == ["a", "b"]
