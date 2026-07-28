from datasets import load_dataset
import json

ds = load_dataset(
    "json",
    data_files={
        "train": "data/raw/apps/train.jsonl",
        "test": "data/raw/apps/test.jsonl",
    },
)

sample = ds["train"][0]
sample["solutions"] = json.loads(sample["solutions"])
sample["input_output"] = json.loads(sample["input_output"])
print(sample)