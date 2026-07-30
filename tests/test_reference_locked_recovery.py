import asyncio
import io
import json
from types import SimpleNamespace

import scripts.build_sft_dataset as builder
from scripts.build_sft_dataset import (
    Counter,
    CurrentRejectedStore,
    classify_verification_failure,
    comments_only_equivalent,
    inject_reference_comments,
    is_fatal_distill_error,
    is_recoverable_rejected_record,
    load_latest_records,
    process_one,
    replace_final_code_block,
)
from src.data.loader import Problem
from src.data.quality import DataQualityChecker


class MemoryRejectedStore:
    def __init__(self):
        self.records = {}

    def reject(self, record):
        self.records[record["id"]] = record

    def resolve(self, problem_id):
        self.records.pop(problem_id, None)


def test_comments_only_equivalent_accepts_comments_without_code_changes():
    reference = "def add(a, b):\n    return a + b\n"
    annotated = (
        "# Add two numbers.\n"
        "def add(a, b):\n"
        "    # Preserve the verified expression.\n"
        "    return a + b\n"
    )

    assert comments_only_equivalent(annotated, reference)


def test_comments_only_equivalent_rejects_executable_changes():
    reference = "def add(a, b):\n    return a + b\n"
    changed = "def add(a, b):\n    return a - b\n"

    assert not comments_only_equivalent(changed, reference)


def test_comments_only_equivalent_rejects_docstrings():
    reference = "def add(a, b):\n    return a + b\n"
    with_docstring = 'def add(a, b):\n    """Add two numbers."""\n    return a + b\n'

    assert not comments_only_equivalent(with_docstring, reference)


def test_replace_final_code_block_keeps_explanation_and_injects_reference():
    response = "Teaching explanation.\n\n```python\nreturn_wrong_answer()\n```"
    reference = "def solve():\n    return 42\n"

    result = replace_final_code_block(response, reference)

    assert result.startswith("Teaching explanation.")
    assert "return_wrong_answer" not in result
    assert reference.strip() in result


def test_comment_plan_is_inserted_without_changing_executable_tokens():
    reference = "def add(a, b):\n    return a + b\n"
    annotated, count = inject_reference_comments(
        reference,
        [
            {"line": 1, "comment": "定义题目要求的函数接口。"},
            {"line": 2, "comment": "返回两个参数之和。"},
            {"line": 99, "comment": "这个行号不安全，应被忽略。"},
        ],
    )

    assert count == 2
    assert "# 定义题目要求的函数接口。" in annotated
    assert "# 返回两个参数之和。" in annotated
    assert comments_only_equivalent(annotated, reference)


def test_load_latest_records_uses_last_record_and_ignores_bad_tail(tmp_path):
    path = tmp_path / "rejected.jsonl"
    path.write_text(
        json.dumps({"id": "a", "failure_type": "wrong_answer"})
        + "\n"
        + json.dumps(
            {
                "id": "a",
                "failure_type": "recovery_llm_failed",
                "metadata": {"recovery_attempted": True},
            }
        )
        + "\n"
        + "{interrupted",
        encoding="utf-8",
    )

    records = load_latest_records(path)

    assert records["a"]["failure_type"] == "recovery_llm_failed"
    assert records["a"]["metadata"]["recovery_attempted"] is True


def test_current_rejected_store_removes_accepted_and_resolved_ids(tmp_path):
    path = tmp_path / "rejected.jsonl"
    records = {
        "accepted": {"id": "accepted", "failure_type": "wrong_answer"},
        "pending": {
            "id": "pending",
            "failure_type": "recovery_llm_failed",
            "metadata": {"test_cases": [{"large": "payload"}]},
        },
    }

    store = CurrentRejectedStore(path, records, {"accepted"})
    assert set(load_latest_records(path)) == {"pending"}
    assert "test_cases" not in load_latest_records(path)["pending"]["metadata"]

    store.resolve("pending")
    assert load_latest_records(path) == {}


def test_rejected_recovery_records_are_versioned():
    record = builder.rejected_record(
        Problem(
            id="p-version",
            source="taco",
            description="Return 42.",
            difficulty="easy",
        ),
        "recovery_llm_failed",
        distill_mode="reference_guided_label",
        recovery_attempted=True,
    )

    assert record["metadata"]["recovery_attempted"] is True
    assert record["metadata"]["recovery_version"] == 2


def test_teacher_api_failure_remains_recoverable_after_current_version():
    assert is_recoverable_rejected_record(
        {
            "failure_type": "recovery_llm_failed",
            "metadata": {"recovery_attempted": True, "recovery_version": 2},
        }
    )
    assert not is_recoverable_rejected_record(
        {
            "failure_type": "recovery_runtime_error",
            "metadata": {"recovery_attempted": True, "recovery_version": 2},
        }
    )


def test_historical_docker_connection_failure_is_resume_retryable():
    assert is_recoverable_rejected_record(
        {
            "failure_type": "recovery_wrong_answer",
            "error": "docker: error during connect: open //./pipe/docker_engine",
            "metadata": {"recovery_attempted": True, "recovery_version": 2},
        }
    )


def test_docker_connection_failure_has_infrastructure_failure_type():
    result = SimpleNamespace(
        unsupported=False,
        error="docker: error during connect: open //./pipe/docker_engine",
        first_failure=None,
    )
    assert classify_verification_failure(result) == "docker_unavailable"


def test_insufficient_balance_is_a_fatal_batch_error():
    error = RuntimeError(
        "Error code: 402 - {'error': {'message': 'Insufficient Balance'}}"
    )
    assert is_fatal_distill_error(error)
    assert not is_fatal_distill_error(RuntimeError("Request timed out"))


def test_process_one_falls_back_to_reference_locked_after_wrong_answer(monkeypatch):
    problem = Problem(
        id="p1",
        source="taco",
        description="Return 42.",
        difficulty="easy",
        reference_solution="def answer():\n    return 42\n",
        public_tests=[
            {
                "input_args": [],
                "expected_output": 42,
                "fn_name": "answer",
            }
        ],
        input_output={"fn_name": "answer"},
        metadata={"reference_verified": True, "reference_pass_rate": 1.0},
    )
    responses = iter(
        [
            "Explanation.\n```python\ndef answer():\n    return 41\n```",
            (
                "Locked explanation.\n"
                "```json\n"
                '[{"line": 1, "comment": "Keep the verified interface."}, '
                '{"line": 2, "comment": "Return the required value."}]\n'
                "```"
            ),
        ]
    )

    async def fake_call(*args, **kwargs):
        return next(responses)

    def fake_verify(code, metadata, **kwargs):
        passed = "return 42" in code
        return SimpleNamespace(
            unsupported=False,
            error=None,
            total_cases=1,
            pass_rate=1.0 if passed else 0.0,
            first_failure=None if passed else "case #1: got 41, expected 42",
        )

    monkeypatch.setattr(builder, "call_distill_model_async", fake_call)
    monkeypatch.setattr(builder, "verify_code", fake_verify)
    accepted = io.StringIO()
    rejected = MemoryRejectedStore()
    counter = Counter()

    asyncio.run(
        process_one(
            client=object(),
            distill_model="teacher",
            problem=problem,
            semaphore=asyncio.Semaphore(1),
            quality_checker=DataQualityChecker(),
            max_output_tokens=1024,
            thinking_mode="off",
            distill_mode="reference_guided_label",
            max_reference_chars=12000,
            seed_examples={},
            seed_examples_per_prompt=0,
            distill_retries=1,
            verification_timeout=5,
            run_code=True,
            execution_backend="subprocess",
            container_image="",
            counter=counter,
            out_file=accepted,
            rejected_store=rejected,
            lock=asyncio.Lock(),
        )
    )

    record = json.loads(accepted.getvalue())
    assert rejected.records == {}
    assert record["metadata"]["label_strategy"] == "reference_locked"
    assert record["metadata"]["initial_failure"]["failure_type"] == "wrong_answer"
    assert record["metadata"]["comment_only_equivalent"] is True
    assert record["metadata"]["comment_plan_count"] == 2
    assert "# Return the required value." in record["messages"][-1]["content"]
    assert counter.recovery_saved == 1
