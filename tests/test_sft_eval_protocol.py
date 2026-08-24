import json

from scripts.evaluate_sft_matrix import build_eval_messages, trim_completion_ids
from scripts.evaluate_teaching import (
    aggregate_report,
    balanced_blind_orders,
    build_judge_prompt,
    parse_judgment,
    prompt_messages,
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
    doubao = {
        "base_vs_sft": _judgment("base", "sft", "sft", 6, 8),
        "base_vs_grpo": _judgment("base", "grpo", "base", 7, 6),
        "sft_vs_grpo": _judgment("sft", "grpo", "grpo", 8, 9),
    }
    config = {
        "judge_api": {"judges": {"deepseek": {}, "doubao": {}}},
        "criteria": CRITERIA,
    }
    summary = aggregate_report(
        [{"id": "p", "judgments": {"deepseek": deepseek, "doubao": doubao}}],
        config,
    )
    assert summary["judge_disagreement"] == {
        "disagreements": 1,
        "comparisons": 3,
        "rate": 0.3333,
    }
    assert summary["pairwise"]["base_vs_sft"]["right_win_rate"] == 1.0
    assert summary["teaching_scores"]["combined"]["grpo"] > 0
