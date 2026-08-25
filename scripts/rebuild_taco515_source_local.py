import hashlib
import json
import re
import sys
from pathlib import Path

import pyarrow.parquet as pq
import zstandard as zstd

if hasattr(sys, "set_int_max_str_digits"):
    sys.set_int_max_str_digits(0)

ROOT = Path(".")
selection_path = ROOT / "outputs/eval/taco515_selected_bs16/selection.json"
data_root = ROOT / "data/raw/TACO/ALL"

output_path = ROOT / "data/final/taco_verified_source_bank.jsonl.zst"
backup_path = ROOT / "data/final/taco_verified_source_bank.BROKEN.jsonl.zst"


def strip_html(text):
    text = re.sub(r"<[^>]+>", "", str(text or ""))
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"&lt;", "<", text)
    text = re.sub(r"&gt;", ">", text)
    text = re.sub(r"&nbsp;", " ", text)
    return text.strip()


def make_id(desc):
    h = hashlib.md5(
        desc.encode(),
        usedforsecurity=False
    ).hexdigest()[:10]
    return f"taco_{h}"


def parse_json_field(value):
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return {}
        try:
            return json.loads(text)
        except Exception:
            return {}
    return {}


def build_tests(io, source):
    inputs = io.get("inputs") or []
    outputs = io.get("outputs") or []
    fn_name = io.get("fn_name")

    if not isinstance(inputs, list) or not isinstance(outputs, list):
        return []

    tests = []
    source_key = str(source or "").lower()

    for inp, out in zip(inputs, outputs):
        expected = out

        if (
            fn_name
            and source_key == "codewars"
            and isinstance(out, list)
            and len(out) == 1
        ):
            expected = out[0]

        tc = {
            "input": (
                inp
                if isinstance(inp, str)
                else json.dumps(inp, ensure_ascii=False)
            ),
            "output": (
                out
                if isinstance(out, str)
                else json.dumps(out, ensure_ascii=False)
            ),
        }

        if fn_name:
            tc["fn_name"] = fn_name
            tc["input_args"] = (
                inp if isinstance(inp, list) else [inp]
            )
            tc["expected_output"] = expected

        tests.append(tc)

    return tests


selection = json.loads(
    selection_path.read_text(encoding="utf-8")
)

ids = selection["problem_ids"]
wanted = set(ids)

assert len(ids) == 515
assert len(wanted) == 515

train_files = sorted(data_root.glob("train-*.parquet"))

if len(train_files) != 9:
    raise RuntimeError(
        f"expected 9 train shards, found {len(train_files)}"
    )

rows = {}

for shard_index, parquet_path in enumerate(train_files, 1):
    print(
        f"[{shard_index}/9] scanning {parquet_path.name} "
        f"(found={len(rows)}/515)",
        flush=True,
    )

    pf = pq.ParquetFile(parquet_path)

    schema_names = set(pf.schema_arrow.names)

    needed_columns = [
        name
        for name in (
            "question",
            "input_output",
            "source",
            "starter_code",
            "difficulty",
        )
        if name in schema_names
    ]

    if "question" not in needed_columns or "input_output" not in needed_columns:
        raise RuntimeError(
            f"missing required columns in {parquet_path.name}: "
            f"{sorted(schema_names)}"
        )

    for batch in pf.iter_batches(
        batch_size=256,
        columns=needed_columns,
    ):
        for row in batch.to_pylist():
            desc = strip_html(row.get("question", ""))

            if len(desc) < 80:
                continue

            pid = make_id(desc)

            if pid not in wanted or pid in rows:
                continue

            io = parse_json_field(
                row.get("input_output", "")
            )

            fn_name = io.get("fn_name")
            source = str(
                row.get("source") or "taco"
            )

            tests = build_tests(io, source)

            rows[pid] = {
                "problem_id": pid,
                "io_mode": (
                    "call_based"
                    if fn_name
                    else "standard_input"
                ),
                "fn_name": fn_name,
                "starter_code": (
                    row.get("starter_code") or ""
                ),
                "test_cases": tests,
                "difficulty": str(
                    row.get("difficulty") or "unknown"
                ).lower(),
                "source": source,
            }

        if len(rows) == 515:
            break

    print(
        f"    after shard: {len(rows)}/515",
        flush=True,
    )

    if len(rows) == 515:
        break


missing = wanted - set(rows)

print()
print("Matched:", len(rows), "/ 515")

if missing:
    print("MISSING:", len(missing))
    print(sorted(missing)[:20])
    raise SystemExit(2)


if output_path.exists():
    output_path.replace(backup_path)
    print(
        "Backed up broken source bank:",
        backup_path,
    )


with output_path.open("wb") as raw:
    with zstd.ZstdCompressor(
        level=3
    ).stream_writer(raw) as writer:

        for pid in ids:
            payload = (
                json.dumps(
                    rows[pid],
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )

            writer.write(
                payload.encode("utf-8")
            )


# 重新读取一遍，确认 zstd + JSON 本身没有损坏
validated = {}

with output_path.open("rb") as raw:
    with zstd.ZstdDecompressor().stream_reader(raw) as reader:
        import io

        with io.TextIOWrapper(
            reader,
            encoding="utf-8",
        ) as text:
            for line in text:
                if not line.strip():
                    continue

                row = json.loads(line)
                validated[row["problem_id"]] = row


assert len(validated) == 515
assert set(validated) == wanted


print()
print("=== TACO515 SOURCE BANK REBUILT ===")
print("records        =", len(validated))
print("output         =", output_path)
print("size           =", output_path.stat().st_size)
print(
    "standard_input =",
    sum(
        x["io_mode"] == "standard_input"
        for x in validated.values()
    ),
)
print(
    "call_based     =",
    sum(
        x["io_mode"] == "call_based"
        for x in validated.values()
    ),
)
print("JSON/ZSTD check = PASS")
print("=== READY FOR DOCKER ===")
