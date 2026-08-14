# 仓库清理报告（2026-08-14）

## 结果

- 删除 34 个已跟踪的旧文件，包括复制 bundle、早期 APPS 生成链、平台专用安装脚本、API smoke、500 条校准包装器和旧单模型评测器。
- `scripts/train_sft.py` 收敛为 14 行兼容入口，唯一训练实现为 `src.training.train_sft`。
- 删除本地可重建/可重新下载资产约 2.874 GiB：TACO train parquet、reference verification cache、GRPO 传输包、Python cache 和旧 2048-token 校准输出。
- 将规划书从误放的 `outputs/` 恢复到项目根目录。
- 将保留的 4096-token 校准/full 评测分别整理到 `outputs/eval/sft_calibration_4096/` 和 `outputs/eval/sft_full_4096/`。

## 保留的正式数据

| 资产 | SHA256 |
|---|---|
| `data/final/sft_accepted.jsonl` | `08ef448f4be6b6b34ee2b6b7af5748827feeba0a0f36cc393350374671c86a1b` |
| `data/final/taco_verified_source_bank.jsonl.zst` | `c3ca0d7c467629b26761ed2dbd86c849a1f9750ef0332332909644dec51cf1df` |
| `data/raw/TACO/ALL/test-00000-of-00001.parquet` | `5d99adc603500c05751aff9f61bed7bbd54a9ad5ea569d108b645346655aae44` |

同时保留：固定 SFT/GRPO split、GRPO 正式训练数据、教学评测集、EvalPlus 数据、专家消融数据、全部正式 `outputs/eval/` 结果和 G0 verifier 证据。

## 删除边界

- TACO train 已由 verified source bank 覆盖，可从 BAAI/TACO 重新下载；TACO test 未删除。
- 全量 reference cache 已完成 source bank 导出后删除；最终 SFT 不依赖该缓存训练。
- `sft_accepted_all_validated.jsonl`、长度审计、canonical manifest 和 source bank 均保留，避免丢失 8K 过滤与数据溯源证据。
- 只删除 superseded 的 2048-token 校准目录；最终 4096-token 校准与 full 结果保留。

## 验证

- `python -m compileall -q scripts src tests evals`：通过。
- 完整测试：`85 passed`。
- `scripts/train_sft.py --help`：成功转发到正式 SFT CLI。
- `scripts/train_grpo.py --help`：兼容入口仍可用。
- 未调用 DeepSeek API、未运行训练、未执行 Docker、未删除模型 checkpoint。
