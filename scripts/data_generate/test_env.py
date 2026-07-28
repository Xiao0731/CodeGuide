import json
import argparse
from collections import Counter
from pathlib import Path

def count_difficulties(file_path):
    """
    统计 JSONL 文件中三种难度的题目数量。
    :param file_path: JSONL 文件路径
    :return: 字典，键为难度名称，值为对应数量
    """
    counter = Counter()
    invalid_lines = 0
    missing_diff = 0

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    diff = data.get("difficulty")
                    if diff is None:
                        missing_diff += 1
                    else:
                        # 只统计三种主要难度，其他难度可单独记录
                        if diff in ("introductory", "interview", "competition"):
                            counter[diff] += 1
                        else:
                            # 如有其他难度值，可以记录，但题目未要求，这里仅加一个 "other"
                            counter["other"] += 1
                except json.JSONDecodeError:
                    invalid_lines += 1
                    print(f"警告：第 {line_num} 行 JSON 解析失败")

    except FileNotFoundError:
        print(f"错误：文件 {file_path} 不存在。")
        return None
    except Exception as e:
        print(f"发生错误：{e}")
        return None

    # 输出统计结果
    print(f"文件：{file_path}")
    print(f"有效数据行数：{sum(counter.values()) + missing_diff}")
    print(f"无效 JSON 行数：{invalid_lines}")
    print(f"缺少 difficulty 字段的行数：{missing_diff}")
    print("\n难度分布：")
    for diff in ["introductory", "interview", "competition"]:
        count = counter.get(diff, 0)
        print(f"  {diff}: {count}")
    if counter.get("other", 0) > 0:
        print(f"  其他难度: {counter['other']}")

    return {diff: counter.get(diff, 0) for diff in ["introductory", "interview", "competition"]}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "file",
        nargs="?",
        type=Path,
        default=Path("data/processed/apps_train_normalized.jsonl"),
    )
    args = parser.parse_args()
    count_difficulties(str(args.file))
