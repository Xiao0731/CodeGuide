import asyncio
import json
import os
from pathlib import Path

import pytest

from scripts.evaluate_sft_matrix import build_eval_messages, trim_completion_ids
from scripts.evaluate_teaching import (
    aggregate_report,
    atomic_write_json,
    balanced_blind_orders,
    build_judge_prompt,
    import_generated_answers,
    load_config,
    parse_judgment,
    prompt_messages,
    run_judges,
)


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


CRITERIA = {
    "problem_understanding": {"label": "Problem Understanding", "weight": 0.20},
    "algorithm_explanation": {"label": "Algorithm Explanation", "weight": 0.30},
    "reasoning_flow": {"label": "Reasoning Flow", "weight": 0.20},
    "code_alignment": {"label": "Code Alignment", "weight": 0.20},
    "beginner_friendly": {"label": "Beginner Friendly", "weight": 0.10},
}


def test_teaching_prompt_removes_reference_label():
    record = {
        "messages": [
            {"role": "system", "content": "teacher system"},
            {"role": "user", "content": "problem statement"},
            {"role": "assistant", "content": "SECRET REFERENCE TEACHING ANSWER"},
        ]
    }
    messages = prompt_messages(record)
    assert [item["role"] for item in messages] == ["system", "user"]
    assert "SECRET REFERENCE" not in str(messages)
    judge_prompt = build_judge_prompt(
        question="problem statement",
        answer_a="candidate one",
        answer_b="candidate two",
        criteria=CRITERIA,
    )
    assert "SECRET REFERENCE" not in judge_prompt
    assert "Base" not in judge_prompt
    assert "GRPO" not in judge_prompt


def test_balanced_blind_orders_are_deterministic_and_even():
    ids = [f"problem-{index}" for index in range(10)]
    first = balanced_blind_orders(ids, 20260728)
    second = balanced_blind_orders(ids, 20260728)
    assert first == second
    for pair_name in ("base_vs_sft", "base_vs_grpo", "sft_vs_grpo"):
        assert sum(first[(problem_id, pair_name)] for problem_id in ids) == 5


def test_parse_judgment_recomputes_weighted_scores():
    dimensions = {
        key: {"A": 10, "B": 5}
        for key in CRITERIA
    }
    parsed = parse_judgment(
        json.dumps(
            {
                "winner": "A",
                "score_A": 9.9,
                "score_B": 5.1,
                "dimensions": dimensions,
                "reason": "A explains the algorithm more clearly.",
            }
        ),
        CRITERIA,
    )
    assert parsed["score_A"] == 10.0
    assert parsed["score_B"] == 5.0
    assert parsed["reported_score_A"] == 9.9


def _judgment(left, right, winner, left_score, right_score):
    dimensions = {
        key: {"A": left_score, "B": right_score}
        for key in CRITERIA
    }
    return {
        "winner": "A" if winner == left else "B" if winner == right else "TIE",
        "winner_model": winner,
        "order": {"A": left, "B": right},
        "dimensions": dimensions,
    }


def test_report_uses_model_winners_and_counts_judge_disagreement():
    deepseek = {
        "base_vs_sft": _judgment("base", "sft", "sft", 5, 8),
        "base_vs_grpo": _judgment("base", "grpo", "grpo", 5, 9),
        "sft_vs_grpo": _judgment("sft", "grpo", "grpo", 8, 9),
    }
    qwen = {
        "base_vs_sft": _judgment("base", "sft", "sft", 6, 8),
        "base_vs_grpo": _judgment("base", "grpo", "base", 7, 6),
        "sft_vs_grpo": _judgment("sft", "grpo", "grpo", 8, 9),
    }
    config = {
        "judge_api": {"judges": {"deepseek": {}, "qwen": {}}},
        "criteria": CRITERIA,
    }
    summary = aggregate_report(
        [{"id": "p", "judgments": {"deepseek": deepseek, "qwen": qwen}}],
        config,
    )
    assert summary["judge_disagreement"] == {
        "disagreements": 1,
        "comparisons": 3,
        "rate": 0.3333,
    }
    assert summary["pairwise"]["base_vs_sft"]["right_win_rate"] == 1.0
    assert summary["teaching_scores"]["combined"]["grpo"] > 0


def test_import_generated_answers_reuses_frozen_jsonl(tmp_path):
    results = [
        {"id": "p1", "question": "q1", "base": "", "sft": "", "grpo": ""},
        {"id": "p2", "question": "q2", "base": "", "sft": "", "grpo": ""},
    ]
    source_plan = {}
    for variant, expected in (
        ("base", "base"),
        ("sft", "selected_sft"),
        ("grpo", "grpo_best"),
    ):
        path = tmp_path / f"{variant}.jsonl"
        rows = [
            {
                "problem_id": problem_id,
                "variant": expected,
                "protocol_name": "frozen-protocol",
                "text": f"{variant}-{problem_id}",
            }
            for problem_id in ("p1", "p2")
        ]
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        source_plan[variant] = {
            "path": str(path),
            "expected_variant": expected,
        }

    output = tmp_path / "results.json"
    summary = import_generated_answers(
        results=results,
        results_path=output,
        source_plan=source_plan,
    )

    assert summary["protocols"] == ["frozen-protocol"]
    assert results[0]["base"] == "base-p1"
    assert results[1]["sft"] == "sft-p2"
    assert results[1]["grpo"] == "grpo-p2"
    assert json.loads(output.read_text(encoding="utf-8"))[0]["base"] == "base-p1"


def test_atomic_write_json_retries_transient_windows_lock(tmp_path, monkeypatch):
    output = tmp_path / "results.json"
    real_replace = os.replace
    attempts = 0

    def flaky_replace(source, target):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise PermissionError(5, "temporary Windows file lock")
        real_replace(source, target)

    monkeypatch.setattr("scripts.evaluate_teaching.os.replace", flaky_replace)
    monkeypatch.setattr("scripts.evaluate_teaching.time.sleep", lambda _: None)
    atomic_write_json(output, {"saved": True})

    assert attempts == 2
    assert json.loads(output.read_text(encoding="utf-8")) == {"saved": True}
    assert list(tmp_path.glob("results.json.*.tmp")) == []


def test_teaching_judges_disable_hidden_reasoning():
    config = load_config(Path("configs/eval/teaching_eval.yaml"))
    judges = config["judge_api"]["judges"]
    assert judges["deepseek"]["extra_body"] == {
        "thinking": {"type": "disabled"}
    }
    assert judges["qwen"]["extra_body"] == {"enable_thinking": False}


def test_judge_failure_isolated_until_batch_finishes(tmp_path, monkeypatch):
    valid = json.dumps(
        {
            "winner": "A",
            "score_A": 8,
            "score_B": 6,
            "dimensions": {
                key: {"A": 8, "B": 6}
                for key in CRITERIA
            },
            "reason": "A is clearer.",
        }
    )

    class FakeCompletions:
        def __init__(self, api_key):
            self.api_key = api_key

        async def create(self, **request):
            content = "not JSON" if self.api_key == "deepseek-key" else valid
            message = type("Message", (), {"content": content})()
            choice = type(
                "Choice", (), {"message": message, "finish_reason": "stop"}
            )()
            return type("Response", (), {"choices": [choice]})()

    class FakeClient:
        def __init__(self, api_key, **_):
            self.chat = type(
                "Chat", (), {"completions": FakeCompletions(api_key)}
            )()

        async def close(self):
            return None

    monkeypatch.setattr("openai.AsyncOpenAI", FakeClient)
    results = [
        {
            "id": "p1",
            "question": "problem",
            "base": "base answer",
            "sft": "sft answer",
            "grpo": "grpo answer",
        }
    ]
    config = {
        "seed": 20260728,
        "criteria": CRITERIA,
        "judge_api": {
            "concurrency": 6,
            "timeout_seconds": 1,
            "max_retries": 0,
            "temperature": 0,
            "max_tokens": 100,
            "judges": {
                "deepseek": {
                    "api_key_env": [],
                    "api_key_default": "unused",
                    "base_url_default": "https://deepseek.invalid",
                    "model_default": "deepseek",
                },
                "qwen": {
                    "api_key_env": [],
                    "base_url_default": "https://qwen.invalid",
                    "model_default": "qwen",
                },
            },
        },
    }
    monkeypatch.setenv("DEEPSEEK_TEST_KEY", "deepseek-key")
    monkeypatch.setenv("QWEN_TEST_KEY", "qwen-key")
    config["judge_api"]["judges"]["deepseek"]["api_key_env"] = [
        "DEEPSEEK_TEST_KEY"
    ]
    config["judge_api"]["judges"]["qwen"]["api_key_env"] = ["QWEN_TEST_KEY"]
    output = tmp_path / "results.json"

    with pytest.raises(RuntimeError, match="3 judge comparisons failed"):
        asyncio.run(run_judges(results=results, results_path=output, config=config))

    assert len(results[0]["judgments"]["qwen"]) == 3
    assert len(results[0]["judgments"]["deepseek"]) == 0
    assert len(results[0]["judge_errors"]["deepseek"]) == 3
    persisted = json.loads(output.read_text(encoding="utf-8"))
    assert len(persisted[0]["judgments"]["qwen"]) == 3
