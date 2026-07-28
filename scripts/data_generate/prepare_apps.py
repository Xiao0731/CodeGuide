from datasets import load_dataset
from pathlib import Path
import json

RAW_DIR = Path("data/raw/apps")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def normalize_record(x):
    # 统一字段名
    record = {
        "problem_id": x.get("problem_id", x.get("id")),
        "question": x["question"],
        "difficulty": x["difficulty"],
        "url": x["url"],
        "starter_code": x.get("starter_code", "")
    }

    # solutions / input_output 可能是字符串，也可能已经是解析后的对象
    sols = x.get("solutions", [])
    io_obj = x.get("input_output", {})

    if isinstance(sols, str):
        try:
            sols = json.loads(sols)
        except Exception:
            sols = []

    if isinstance(io_obj, str):
        try:
            io_obj = json.loads(io_obj)
        except Exception:
            io_obj = {}

    record["solutions"] = sols
    record["input_output"] = io_obj
    return record

def main():
    ds = load_dataset(
        "json",
        data_files={
            "train": str(RAW_DIR / "train.jsonl"),
            "test": str(RAW_DIR / "test.jsonl"),
        },
    )

    train_records = [normalize_record(x) for x in ds["train"]]
    test_records = [normalize_record(x) for x in ds["test"]]

    # 保存成规范化 jsonl
    with open(OUT_DIR / "apps_train_normalized.jsonl", "w", encoding="utf-8") as f:
        for item in train_records:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(OUT_DIR / "apps_test_normalized.jsonl", "w", encoding="utf-8") as f:
        for item in test_records:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("train:", len(train_records))
    print("test:", len(test_records))
    print("sample keys:", train_records[0].keys())
    print("sample difficulty:", train_records[0]["difficulty"])
    print("sample problem_id:", train_records[0]["problem_id"])

if __name__ == "__main__":
    main()