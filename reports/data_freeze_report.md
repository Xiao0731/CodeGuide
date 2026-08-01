# CodeGuide 数据冻结与训练前审计报告

## 冻结结论

- 正式 accepted 输入：10,340 条；长度隔离 34 条。
- Canonical SFT：`data/final/sft_accepted.jsonl`，10,306 条，SHA-256 `08ef448f4be6b6b34ee2b6b7af5748827feeba0a0f36cc393350374671c86a1b`。
- Verified source bank：`data/final/taco_verified_source_bank.jsonl.zst`，10,415 条，SHA-256 `b74eae83bb538f1a1fb3af24425da8f5f14305dc89a291cf37498e7f590ebda5`。
- Rejected：75 条，{"recovery_llm_failed": 40, "recovery_runtime_error": 28, "recovery_timeout": 7}。
- Source bank 独立 Docker 抽样复验：20/20 通过。

## Canonical 分布

- A/B：`{"pedagogical_rewrite": 6892, "reference_locked": 3414}`。
- I/O：`{"call_based": 2450, "standard_input": 7856}`。
- 难度：`{"easy": 5039, "hard": 954, "medium": 1055, "medium_hard": 1457, "unknown_difficulty": 1360, "very_hard": 441}`。
- Metadata 回填：`{"label_strategy": 1323, "code_source": 10340}`。旧样本缺失 `label_strategy` 按约定回填为 A 类。
- problem ID 冲突：0；accepted/rejected 无交集。
- TACO test 重复：0。

## Token 长度

使用 `Qwen/Qwen2.5-Coder-7B-Instruct` 正式 chat template；推荐 `max_seq_length=8192`。

- 完整序列 P50/P75/P90/P95/P99/max：2603/3427/4398/5062/6728/8173。
- Canonical 超过 8192：0；代码受损：0。
- 全部 validated 快照中 34 条超过 8192，详见 `data/manifests/sft_length_excluded.json`。

## 固定划分

- SFT train/dev：9791/515。
- GRPO train/validation 预留：900/100。
- 固定种子：20260728；重复运行 hash 一致；跨 split 泄漏为 0。

## 验证证据

冻结复用 10,340 条正式生成 Docker pass 证据；本轮从 source bank 实际重新执行 20 条。原 accepted 未逐条重跑，原因是已有统一 verifier 的正式 pass_rate=1.0 证据，而全量 Docker 重跑成本高且不会改变教学标签。
