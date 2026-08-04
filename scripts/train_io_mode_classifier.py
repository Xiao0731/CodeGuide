#!/usr/bin/env python3
"""Train a text-only classifier for TACO standard_input vs call_based routing.

Features come exclusively from user-visible problem text in ``messages``. Labels
come from frozen metadata, but ``io_mode``, ``fn_name`` and ``starter_code`` are
never included in classifier input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.training.sft_data import load_canonical, load_id_list

LABELS = ["standard_input", "call_based"]
CUE_PATTERNS = {
    "explicit_input_heading": re.compile(r"(?im)^\s*(input|输入)\s*:?") ,
    "explicit_output_heading": re.compile(r"(?im)^\s*(output|输出)\s*:?") ,
    "stdin_stdout": re.compile(r"(?i)stdin|stdout|standard input|standard output|标准输入|标准输出"),
    "read_print": re.compile(r"(?i)read\s+from|print\s+the|读取|输出"),
    "function_word": re.compile(r"(?i)\bfunction\b|\bmethod\b|函数|方法"),
    "class_solution": re.compile(r"(?i)class\s+solution|solution\s+class"),
    "return_word": re.compile(r"(?i)\breturn(?:s|ed)?\b|返回"),
    "parameter_word": re.compile(r"(?i)\bparameter(?:s)?\b|参数"),
    "signature_like": re.compile(r"(?m)\b[a-zA-Z_]\w*\s*\([^\n()]{0,160}\)"),
}


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def io_mode(record: dict[str, Any]) -> str:
    metadata = record.get("metadata") or {}
    value = str(metadata.get("io_mode") or record.get("io_mode") or "unknown")
    if value not in LABELS:
        raise ValueError(f"unsupported io_mode: {value}")
    return value


def problem_text(record: dict[str, Any]) -> str:
    messages = record.get("messages")
    if not isinstance(messages, list):
        raise ValueError("record has no messages list")
    user_parts: list[str] = []
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            user_parts.append(content.strip())
    if not user_parts:
        raise ValueError("record has no user-visible problem text")
    return "\n\n".join(user_parts)


def cue_audit(texts: list[str], labels: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {label: {"samples": labels.count(label)} for label in LABELS}
    for label in LABELS:
        label_texts = [text for text, item_label in zip(texts, labels) if item_label == label]
        for name, pattern in CUE_PATTERNS.items():
            hits = sum(bool(pattern.search(text)) for text in label_texts)
            result[label][name] = {
                "hits": hits,
                "rate": hits / len(label_texts) if label_texts else 0.0,
            }
    return result


def top_features(pipeline: Any, top_k: int) -> dict[str, list[dict[str, Any]]]:
    features = pipeline.named_steps["features"].get_feature_names_out()
    classifier = pipeline.named_steps["classifier"]
    if classifier.coef_.shape[0] != 1:
        return {}
    positive_label = str(classifier.classes_[1])
    negative_label = str(classifier.classes_[0])
    weights = classifier.coef_[0]
    positive = sorted(range(len(weights)), key=lambda index: weights[index], reverse=True)[:top_k]
    negative = sorted(range(len(weights)), key=lambda index: weights[index])[:top_k]
    return {
        positive_label: [
            {"feature": str(features[index]), "weight": float(weights[index])}
            for index in positive
        ],
        negative_label: [
            {"feature": str(features[index]), "weight": float(weights[index])}
            for index in negative
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical", default="data/final/sft_accepted.jsonl")
    parser.add_argument("--train-ids", default="data/splits/sft_train_ids.json")
    parser.add_argument("--test-ids", default="data/splits/sft_dev_ids.json")
    parser.add_argument("--output-dir", default="outputs/router/io_mode_text_v1")
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--top-features", type=int, default=30)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        import joblib
        import numpy as np
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import (
            accuracy_score,
            classification_report,
            confusion_matrix,
            f1_score,
        )
        from sklearn.pipeline import FeatureUnion, Pipeline
    except ImportError as exc:
        raise RuntimeError(
            "router dependencies are missing; install requirements-router.txt"
        ) from exc

    canonical = load_canonical(resolve(args.canonical))
    train_ids = load_id_list(resolve(args.train_ids))
    test_ids = load_id_list(resolve(args.test_ids))
    if set(train_ids) & set(test_ids):
        raise RuntimeError("classifier train/test ID overlap")

    def build_rows(ids: list[str]) -> tuple[list[str], list[str]]:
        missing = [pid for pid in ids if pid not in canonical]
        if missing:
            raise RuntimeError(f"canonical misses IDs: {missing[:5]}")
        return (
            [problem_text(canonical[pid]) for pid in ids],
            [io_mode(canonical[pid]) for pid in ids],
        )

    train_texts, train_labels = build_rows(train_ids)
    test_texts, test_labels = build_rows(test_ids)

    features = FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=50000,
                    sublinear_tf=True,
                    strip_accents="unicode",
                    token_pattern=r"(?u)\b\w+\b",
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=2,
                    max_features=100000,
                    sublinear_tf=True,
                ),
            ),
        ]
    )
    classifier = LogisticRegression(
        C=2.0,
        class_weight="balanced",
        max_iter=2000,
        random_state=args.seed,
        solver="liblinear",
    )
    pipeline = Pipeline([("features", features), ("classifier", classifier)])
    pipeline.fit(train_texts, train_labels)

    predictions = pipeline.predict(test_texts)
    probabilities = pipeline.predict_proba(test_texts)
    classes = [str(item) for item in pipeline.named_steps["classifier"].classes_]
    class_index = {label: index for index, label in enumerate(classes)}

    accuracy = float(accuracy_score(test_labels, predictions))
    macro_f1 = float(f1_score(test_labels, predictions, average="macro"))
    majority = Counter(train_labels).most_common(1)[0][0]
    majority_accuracy = sum(label == majority for label in test_labels) / len(test_labels)
    matrix_values = confusion_matrix(test_labels, predictions, labels=LABELS).tolist()
    report = classification_report(
        test_labels,
        predictions,
        labels=LABELS,
        output_dict=True,
        zero_division=0,
    )

    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipeline": pipeline,
            "labels": LABELS,
            "input_contract": "user-role message text only",
            "seed": args.seed,
        },
        output_dir / "io_mode_classifier.joblib",
    )

    prediction_path = output_dir / "taco515_predictions.jsonl"
    with prediction_path.open("w", encoding="utf-8") as handle:
        for pid, text, true_label, predicted, proba in zip(
            test_ids, test_texts, test_labels, predictions, probabilities
        ):
            row = {
                "problem_id": pid,
                "true_io_mode": true_label,
                "predicted_io_mode": str(predicted),
                "correct": bool(predicted == true_label),
                "confidence": float(np.max(proba)),
                "probability_standard_input": float(proba[class_index["standard_input"]]),
                "probability_call_based": float(proba[class_index["call_based"]]),
                "problem_text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    payload = {
        "schema_version": "codeguide-text-io-router-report-v1",
        "input_contract": {
            "included": "only content of messages with role=user",
            "excluded": ["io_mode", "fn_name", "starter_code", "assistant answer"],
        },
        "train_samples": len(train_ids),
        "test_samples": len(test_ids),
        "train_distribution": dict(Counter(train_labels)),
        "test_distribution": dict(Counter(test_labels)),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "majority_class": majority,
        "majority_accuracy": majority_accuracy,
        "confusion_matrix": {"labels": LABELS, "values": matrix_values},
        "classification_report": report,
        "cue_audit_train": cue_audit(train_texts, train_labels),
        "cue_audit_test": cue_audit(test_texts, test_labels),
        "top_features": top_features(pipeline, args.top_features),
        "predictions": str(prediction_path),
    }
    (output_dir / "report.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
