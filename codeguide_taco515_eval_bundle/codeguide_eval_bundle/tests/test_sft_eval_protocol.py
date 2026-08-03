from scripts.evaluate_sft_matrix import build_eval_messages, trim_completion_ids


def test_eval_messages_replace_system_and_append_user_suffix():
    record = {
        "messages": [
            {"role": "system", "content": "legacy long teaching prompt"},
            {"role": "user", "content": "problem statement"},
            {"role": "assistant", "content": "training label"},
        ]
    }
    protocol = {
        "system_prompt": "compact system",
        "user_suffix": "\ncompact reminder",
    }
    messages = build_eval_messages(record, protocol)
    assert messages == [
        {"role": "system", "content": "compact system"},
        {"role": "user", "content": "problem statement\ncompact reminder"},
    ]
    # Canonical record must stay unchanged.
    assert record["messages"][0]["content"] == "legacy long teaching prompt"


def test_eval_messages_prepend_system_if_missing():
    record = {
        "messages": [
            {"role": "user", "content": "problem"},
            {"role": "assistant", "content": "label"},
        ]
    }
    messages = build_eval_messages(
        record,
        {"system_prompt": "compact", "user_suffix": ""},
    )
    assert messages[0] == {"role": "system", "content": "compact"}
    assert messages[1] == {"role": "user", "content": "problem"}


def test_trim_completion_stops_at_eos():
    trimmed, saw_eos = trim_completion_ids(
        [11, 12, 2, 2, 2],
        eos_token_ids={2},
        pad_token_id=2,
    )
    assert trimmed == [11, 12]
    assert saw_eos


def test_trim_completion_removes_trailing_pad_without_eos():
    trimmed, saw_eos = trim_completion_ids(
        [11, 12, 0, 0],
        eos_token_ids={2},
        pad_token_id=0,
    )
    assert trimmed == [11, 12]
    assert not saw_eos
