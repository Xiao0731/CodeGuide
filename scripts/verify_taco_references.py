#!/usr/bin/env python3
"""Offline verification cache for TACO reference solutions.

This script validates the selected TACO `reference_solution` with the same
execution verifier used by reward/evaluation code. It writes one JSONL record
per problem and supports resume-by-id so long TACO scans can be stopped and
continued safely.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.code_validator import validate_syntax
from src.data.loader import Problem, load_problems
from src.reward.execution import VerificationResult, supports_verification, verify_code

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _format_seconds(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def _io_metadata(problem: Problem) -> dict[str, Any]:
    input_output = problem.input_output if isinstance(problem.input_output, dict) else {}
    io_mode = "call_based" if input_output.get("fn_name") else "standard_input"
    return {
        "io_mode": io_mode,
        "fn_name": input_output.get("fn_name"),
        "starter_code": problem.starter_code,
        "test_cases": problem.public_tests or [],
        "tags": list(problem.tags or []),
        "skill_types": list(problem.skill_types or []),
        "raw_tags": list(problem.raw_tags or []),
    }


def _difficulty_key(problem: Problem) -> str:
    return str(problem.difficulty or "unknown").lower()


def _stratified_sample(
    problems: list[Problem],
    difficulties: list[str],
    per_difficulty: int,
    seed: int,
) -> list[Problem]:
    if not difficulties:
        return problems
    if per_difficulty <= 0:
        raise ValueError("--per-difficulty must be positive")

    by_diff: dict[str, list[Problem]] = {}
    for problem in problems:
        by_diff.setdefault(_difficulty_key(problem), []).append(problem)

    rng = random.Random(seed)
    selected: list[Problem] = []
    seen: set[str] = set()
    for raw_diff in difficulties:
        diff = raw_diff.lower()
        pool = by_diff.get(diff, [])
        if len(pool) < per_difficulty:
            logger.warning(
                "difficulty=%s 可用样本不足：需要 %d，实际 %d",
                diff,
                per_difficulty,
                len(pool),
            )
        take_n = min(per_difficulty, len(pool))
        sampled = rng.sample(pool, take_n) if len(pool) > take_n else list(pool)
        for problem in sampled:
            if problem.id not in seen:
                selected.append(problem)
                seen.add(problem.id)
    return selected


def _load_done_records(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    done: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record.get("id"), str):
                done[record["id"]] = record
    return done


def _is_timeout(result: VerificationResult) -> bool:
    text = " ".join(
        str(part or "") for part in (result.error, result.first_failure)
    ).lower()
    return "timeout" in text or "timed out" in text


def _is_interface_mismatch(result: VerificationResult) -> bool:
    text = str(result.error or result.first_failure or "").lower()
    markers = (
        "no top-level function",
        "solution has no callable method",
        "unsupported call_based metadata",
        "missing",
        "fn_name",
        "callable",
    )
    return any(marker in text for marker in markers)


def _looks_runtime_error(result: VerificationResult) -> bool:
    if result.error:
        return True
    text = str(result.first_failure or "")
    if " got " in text and " expected " in text:
        return False
    return ":" in text


def _classify_result(result: VerificationResult, min_pass_rate: float) -> str:
    if result.unsupported:
        if result.io_mode == "call_based" and _is_interface_mismatch(result):
            return "interface_mismatch"
        return "unsupported"
    if _is_timeout(result):
        return "timeout"
    if result.pass_rate >= min_pass_rate:
        return "passed"
    if result.io_mode == "call_based" and _is_interface_mismatch(result):
        return "interface_mismatch"
    if _looks_runtime_error(result):
        return "runtime_error"
    return "wrong_answer"


def _reference_candidates(problem: Problem, max_candidates: int) -> list[dict[str, Any]]:
    candidates = list(problem.reference_candidates or [])
    if not candidates and problem.reference_solution:
        candidates = [{
            "rank": 0,
            "raw_index": problem.metadata.get("selected_raw_solution_index"),
            "priority": 0,
            "reason": "selected_reference_solution",
            "code": problem.reference_solution,
        }]
    if max_candidates <= 0:
        return candidates
    return candidates[:max_candidates]


def _candidate_result(
    candidate: dict[str, Any],
    metadata: dict[str, Any],
    *,
    timeout: float,
    min_pass_rate: float,
) -> dict[str, Any]:
    code = str(candidate.get("code") or "")
    rank = candidate.get("rank")
    if rank is None:
        rank = 0

    summary: dict[str, Any] = {
        "index": int(rank),
        "raw_index": candidate.get("raw_index"),
        "priority": candidate.get("priority"),
        "reason": candidate.get("reason"),
        "pass_rate": 0.0,
        "error_type": None,
        "error": None,
        "passed_cases": 0,
        "total_cases": len(metadata.get("test_cases") or []),
        "first_failure": None,
    }

    syntax_ok, syntax_error = validate_syntax(code)
    if not syntax_ok:
        summary["error_type"] = "syntax_error"
        summary["error"] = syntax_error or "syntax_error"
        return summary

    result = verify_code(code, metadata, timeout=timeout)
    error_type = _classify_result(result, min_pass_rate)
    verified = error_type == "passed"
    summary.update({
        "pass_rate": round(result.pass_rate, 6),
        "error_type": None if verified else error_type,
        "error": None if verified else (result.error or result.first_failure or error_type),
        "passed_cases": result.passed_cases,
        "total_cases": result.total_cases,
        "first_failure": result.first_failure,
    })
    return summary


def _best_failed_candidate(candidate_results: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not candidate_results:
        return None
    return max(
        candidate_results,
        key=lambda item: (
            float(item.get("pass_rate") or 0.0),
            -int(item.get("index") or 0),
        ),
    )


def verify_reference(
    problem: Problem,
    *,
    timeout: float,
    min_pass_rate: float,
    max_reference_candidates: int,
) -> dict[str, Any]:
    metadata = _io_metadata(problem)
    io_mode = metadata["io_mode"]
    candidates = _reference_candidates(problem, max_reference_candidates)

    base = {
        "id": problem.id,
        "selected_reference_index": None,
        "selected_raw_solution_index": None,
        "candidate_count": len(problem.reference_candidates or ([] if not problem.reference_solution else [problem.reference_solution])),
        "attempted_candidates": 0,
        "candidate_results": [],
        "reference_verified": False,
        "reference_pass_rate": 0.0,
        "reference_error": None,
        "reference_error_type": None,
        "io_mode": io_mode,
        "fn_name": metadata.get("fn_name"),
        "passed_cases": 0,
        "total_cases": len(metadata.get("test_cases") or []),
        "first_failure": None,
        "difficulty": problem.difficulty,
        "source": problem.source,
    }

    if not candidates:
        base["reference_error"] = "no_reference"
        base["reference_error_type"] = "no_reference"
        return base

    if not supports_verification(metadata):
        base["reference_error"] = "unsupported"
        base["reference_error_type"] = "unsupported"
        return base

    for candidate in candidates:
        result = _candidate_result(
            candidate,
            metadata,
            timeout=timeout,
            min_pass_rate=min_pass_rate,
        )
        base["candidate_results"].append(result)
        base["attempted_candidates"] = len(base["candidate_results"])

        if result.get("error_type") is None and float(result.get("pass_rate") or 0.0) >= min_pass_rate:
            base.update({
                "selected_reference_index": result.get("index"),
                "selected_raw_solution_index": result.get("raw_index"),
                "reference_verified": True,
                "reference_pass_rate": result.get("pass_rate"),
                "reference_error": None,
                "reference_error_type": None,
                "passed_cases": result.get("passed_cases", 0),
                "total_cases": result.get("total_cases", base["total_cases"]),
                "first_failure": None,
            })
            return base

    best = _best_failed_candidate(base["candidate_results"])
    if best:
        base.update({
            "selected_reference_index": None,
            "selected_raw_solution_index": None,
            "reference_verified": False,
            "reference_pass_rate": best.get("pass_rate", 0.0),
            "reference_error": best.get("error") or best.get("first_failure") or best.get("error_type"),
            "reference_error_type": best.get("error_type") or "wrong_answer",
            "passed_cases": best.get("passed_cases", 0),
            "total_cases": best.get("total_cases", base["total_cases"]),
            "first_failure": best.get("first_failure"),
        })
    return base


def _verify_reference_worker(payload: tuple[Problem, float, float, int]) -> dict[str, Any]:
    problem, timeout, min_pass_rate, max_reference_candidates = payload
    return verify_reference(
        problem,
        timeout=timeout,
        min_pass_rate=min_pass_rate,
        max_reference_candidates=max_reference_candidates,
    )


@dataclass
class Stats:
    total: int = 0
    with_reference: int = 0
    verified_records: int = 0
    full_pass: int = 0
    partial_pass: int = 0
    unsupported: int = 0
    timeout: int = 0
    runtime_error: int = 0
    syntax_error: int = 0
    no_reference: int = 0
    wrong_answer: int = 0
    interface_mismatch: int = 0
    first_candidate_pass: int = 0
    fallback_pass: int = 0
    total_attempted_candidates: int = 0
    with_candidate_records: int = 0
    by_io_attempted: dict[str, int] = field(default_factory=dict)
    by_io_full_pass: dict[str, int] = field(default_factory=dict)
    by_difficulty_attempted: dict[str, int] = field(default_factory=dict)
    by_difficulty_full_pass: dict[str, int] = field(default_factory=dict)

    def add_problem_seen(self, problem: Problem) -> None:
        self.total += 1
        if problem.reference_solution:
            self.with_reference += 1

    def add_record(self, record: dict[str, Any]) -> None:
        self.verified_records += 1
        io_mode = str(record.get("io_mode") or "unknown")
        err_type = record.get("reference_error_type")
        pass_rate = float(record.get("reference_pass_rate") or 0.0)
        difficulty = str(record.get("difficulty") or "unknown")
        attempted_candidates = int(record.get("attempted_candidates") or 0)
        self.total_attempted_candidates += attempted_candidates
        if int(record.get("candidate_count") or 0) > 0:
            self.with_candidate_records += 1

        if err_type != "no_reference":
            self.by_io_attempted[io_mode] = self.by_io_attempted.get(io_mode, 0) + 1
            self.by_difficulty_attempted[difficulty] = self.by_difficulty_attempted.get(difficulty, 0) + 1
        if record.get("reference_verified"):
            self.full_pass += 1
            self.by_io_full_pass[io_mode] = self.by_io_full_pass.get(io_mode, 0) + 1
            self.by_difficulty_full_pass[difficulty] = self.by_difficulty_full_pass.get(difficulty, 0) + 1
            selected_index = int(record.get("selected_reference_index") or 0)
            if selected_index == 0:
                self.first_candidate_pass += 1
            else:
                self.fallback_pass += 1
        elif 0.0 < pass_rate < 1.0:
            self.partial_pass += 1

        if err_type == "no_reference":
            self.no_reference += 1
        elif err_type == "unsupported":
            self.unsupported += 1
        elif err_type == "timeout":
            self.timeout += 1
        elif err_type == "runtime_error":
            self.runtime_error += 1
        elif err_type == "syntax_error":
            self.syntax_error += 1
        elif err_type == "wrong_answer":
            self.wrong_answer += 1
        elif err_type == "interface_mismatch":
            self.interface_mismatch += 1

    def summary(self) -> str:
        io_lines = []
        for io_mode in sorted(self.by_io_attempted):
            attempted = self.by_io_attempted[io_mode]
            passed = self.by_io_full_pass.get(io_mode, 0)
            rate = passed / attempted if attempted else 0.0
            io_lines.append(f"  - {io_mode}: {passed}/{attempted} = {rate:.2%}")
        if not io_lines:
            io_lines.append("  - 暂无可统计 io_mode")

        difficulty_lines = []
        for difficulty in sorted(self.by_difficulty_attempted):
            attempted = self.by_difficulty_attempted[difficulty]
            passed = self.by_difficulty_full_pass.get(difficulty, 0)
            rate = passed / attempted if attempted else 0.0
            difficulty_lines.append(f"  - {difficulty}: {passed}/{attempted} = {rate:.2%}")
        if not difficulty_lines:
            difficulty_lines.append("  - 暂无可统计 difficulty")

        avg_attempts = (
            self.total_attempted_candidates / self.with_candidate_records
            if self.with_candidate_records
            else 0.0
        )
        first_pass_rate = (
            self.first_candidate_pass / self.with_candidate_records
            if self.with_candidate_records
            else 0.0
        )
        final_pass_rate = (
            self.full_pass / self.with_candidate_records
            if self.with_candidate_records
            else 0.0
        )
        no_reference_ratio = (
            self.no_reference / self.total
            if self.total
            else 0.0
        )

        return "\n".join([
            f"总题数: {self.total}",
            f"有参考解数量: {self.with_reference}",
            f"有 Python 候选参考解记录: {self.with_candidate_records}",
            f"已验证/已写缓存数量: {self.verified_records}",
            f"完全通过数量: {self.full_pass}",
            f"第一候选直接通过数量: {self.first_candidate_pass} ({first_pass_rate:.2%})",
            f"多候选回退后通过数量: {self.fallback_pass}",
            f"多候选最终通过率: {final_pass_rate:.2%}",
            f"平均尝试候选数: {avg_attempts:.2f}",
            f"部分通过数量: {self.partial_pass}",
            f"不支持数量: {self.unsupported}",
            f"超时数量: {self.timeout}",
            f"运行错误数量: {self.runtime_error}",
            f"语法错误数量: {self.syntax_error}",
            f"接口不匹配数量: {self.interface_mismatch}",
            f"答案错误数量: {self.wrong_answer}",
            f"无参考解数量: {self.no_reference} ({no_reference_ratio:.2%})",
            "standard_input / call_based 分别通过率:",
            *io_lines,
            "不同难度通过率:",
            *difficulty_lines,
        ])


def run(args: argparse.Namespace) -> None:
    max_items = args.limit if args.limit is not None else 10_000_000
    problems = load_problems(
        source="taco",
        split=args.split,
        max_items=max_items,
        taco_data_root=args.taco_data_root,
    )
    if args.stratified_difficulties:
        problems = _stratified_sample(
            problems,
            args.stratified_difficulties,
            args.per_difficulty,
            args.stratified_seed,
        )
    if args.io_mode != "any":
        before = len(problems)
        problems = [
            problem for problem in problems
            if _io_metadata(problem)["io_mode"] == args.io_mode
        ]
        logger.info("按 io_mode=%s 过滤：%d → %d 条", args.io_mode, before, len(problems))
    if args.verify_limit is not None:
        problems = problems[:args.verify_limit]
        logger.info("按 --verify-limit 截断待验证集合：%d 条", len(problems))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    done_records = _load_done_records(output_path) if args.resume else {}
    done_ids = set(done_records)
    if done_ids:
        logger.info("resume enabled: 跳过已缓存 %d 条", len(done_ids))

    stats = Stats()
    for problem in problems:
        stats.add_problem_seen(problem)
        if problem.id in done_records:
            stats.add_record(done_records[problem.id])

    pending = [problem for problem in problems if problem.id not in done_ids]
    logger.info("待验证：%d/%d 条", len(pending), len(problems))
    logger.info(
        "执行配置：workers=%d, timeout=%.1fs, max_reference_candidates=%d",
        args.workers,
        args.timeout,
        args.max_reference_candidates,
    )

    def write_record(out, record: dict[str, Any], completed_new: int, total_new: int, started_at: float) -> None:
        stats.add_record(record)
        out.write(json.dumps(record, ensure_ascii=False) + "\n")
        out.flush()
        if completed_new % args.log_every == 0 or completed_new == total_new:
            elapsed = time.time() - started_at
            speed = completed_new / elapsed if elapsed > 0 else 0.0
            remaining = total_new - completed_new
            eta = remaining / speed if speed > 0 else 0.0
            logger.info(
                "progress %d/%d: full_pass=%d fallback=%d partial=%d timeout=%d unsupported=%d speed=%.2f/s eta=%s",
                completed_new,
                total_new,
                stats.full_pass,
                stats.fallback_pass,
                stats.partial_pass,
                stats.timeout,
                stats.unsupported,
                speed,
                _format_seconds(eta),
            )

    try:
        with output_path.open("a", encoding="utf-8") as out:
            started_at = time.time()
            if args.workers <= 1:
                for idx, problem in enumerate(pending, 1):
                    record = verify_reference(
                        problem,
                        timeout=args.timeout,
                        min_pass_rate=args.min_pass_rate,
                        max_reference_candidates=args.max_reference_candidates,
                    )
                    write_record(out, record, idx, len(pending), started_at)
            else:
                max_in_flight = max(args.workers, args.workers * args.queue_factor)
                payload_iter = iter(
                    (problem, args.timeout, args.min_pass_rate, args.max_reference_candidates)
                    for problem in pending
                )
                completed_new = 0
                futures = set()
                with ProcessPoolExecutor(max_workers=args.workers) as executor:
                    def submit_until_full() -> None:
                        while len(futures) < max_in_flight:
                            try:
                                payload = next(payload_iter)
                            except StopIteration:
                                return
                            futures.add(executor.submit(_verify_reference_worker, payload))

                    submit_until_full()
                    while futures:
                        done, futures = wait(futures, return_when=FIRST_COMPLETED)
                        for future in done:
                            completed_new += 1
                            record = future.result()
                            write_record(out, record, completed_new, len(pending), started_at)
                        submit_until_full()
    except KeyboardInterrupt:
        logger.warning("收到中断信号，已写入的缓存可用 --resume 继续")
    finally:
        logger.info("\n%s", stats.summary())
        logger.info("输出缓存：%s", output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="离线验证 TACO reference_solution 并写入 JSONL 缓存")
    parser.add_argument("--taco-data-root", default="data/raw/TACO/ALL")
    parser.add_argument("--split", choices=["train", "test", "valid", "validation", "eval"], default="train")
    parser.add_argument("--limit", type=int, default=None, help="最多加载多少条 TACO 样本")
    parser.add_argument("--output", default="data/cache/taco_reference_verification.jsonl")
    parser.add_argument("--timeout", type=float, default=8.0, help="单题 verify_code 超时时间（秒）")
    parser.add_argument("--min-pass-rate", type=float, default=1.0, help="判定 reference_verified 的最低通过率")
    parser.add_argument("--max-reference-candidates", type=int, default=3,
                        help="每题最多验证前 N 个排序后的 Python 参考候选（默认 3）")
    parser.add_argument("--resume", action="store_true", help="跳过 output 中已有 id，继续追加验证")
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--workers", type=int, default=1,
                        help="并行验证进程数；1 表示串行（默认 1）")
    parser.add_argument("--queue-factor", type=int, default=2,
                        help="并行时每个 worker 预提交多少个任务，用于限制内存占用（默认 2）")
    parser.add_argument("--io-mode", choices=["any", "standard_input", "call_based"], default="any",
                        help="小规模验收用：只验证某一种接口模式（默认 any）")
    parser.add_argument("--verify-limit", type=int, default=None,
                        help="加载/过滤后最多实际验证多少条；用于 call_based 小样本验收")
    parser.add_argument("--stratified-difficulties", nargs="+", default=[],
                        help="小规模验收用：按 difficulty 分层抽样，例如 easy medium hard very_hard")
    parser.add_argument("--per-difficulty", type=int, default=1)
    parser.add_argument("--stratified-seed", type=int, default=42)
    args = parser.parse_args()
    if args.workers < 1:
        args.workers = 1
    if args.queue_factor < 1:
        args.queue_factor = 1
    if args.workers > (os.cpu_count() or args.workers):
        logger.warning("workers=%d 大于 CPU 逻辑核数 %d", args.workers, os.cpu_count() or -1)
    run(args)


if __name__ == "__main__":
    main()
