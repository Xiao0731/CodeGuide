# 数据资产保留与清理建议

本轮未删除、移动或覆盖任何原始 TACO、生成日志或旧版输出。

## 必须保留

| 路径 | 大小 MiB | 代码引用 | 恢复方式 |
|---|---:|---|---|
| `data/final/sft_accepted.jsonl` | 106.01 | `scripts/audit_token_lengths.py`, `scripts/finalize_data_freeze.py`, `scripts/freeze_sft_data.py` | restore frozen/archive copy |
| `data/final/sft_accepted_all_validated.jsonl` | 107.33 | `scripts/freeze_sft_data.py` | restore frozen/archive copy |
| `data/final/taco_verified_source_bank.jsonl.zst` | 218.54 | `scripts/freeze_sft_data.py`, `scripts/verify_source_bank_sample.py` | restore frozen/archive copy |
| `data/manifests/sft_length_excluded.json` | 0.00 | `scripts/finalize_data_freeze.py`, `scripts/freeze_sft_data.py` | restore frozen/archive copy |
| `data/manifests/sft_manifest.json` | 0.01 | `scripts/finalize_data_freeze.py`, `scripts/freeze_sft_data.py` | restore frozen/archive copy |
| `data/manifests/source_bank_verification.json` | 0.01 | `scripts/finalize_data_freeze.py`, `scripts/freeze_sft_data.py`, `scripts/verify_source_bank_sample.py` | restore frozen/archive copy |
| `data/manifests/taco_train_test_overlap.json` | 0.00 | `scripts/finalize_data_freeze.py` | restore frozen/archive copy |
| `data/manifests/token_length_stats.json` | 1.09 | `scripts/audit_token_lengths.py`, `scripts/finalize_data_freeze.py` | restore frozen/archive copy |
| `data/manifests/token_length_stats_all_validated.json` | 1.11 | `scripts/finalize_data_freeze.py`, `scripts/freeze_sft_data.py` | restore frozen/archive copy |
| `data/raw/TACO/ALL/test-00000-of-00001.parquet` | 234.40 | `scripts/finalize_data_freeze.py` | download BAAI/TACO parquet again |
| `data/seeds/array.json` | 0.02 | 无静态引用 | restore frozen/archive copy |
| `data/seeds/backtracking.json` | 0.03 | 无静态引用 | restore frozen/archive copy |
| `data/seeds/binary_search.json` | 0.02 | 无静态引用 | restore frozen/archive copy |
| `data/seeds/dfs_bfs.json` | 0.03 | 无静态引用 | restore frozen/archive copy |
| `data/seeds/dp.json` | 0.02 | 无静态引用 | restore frozen/archive copy |
| `data/seeds/graph.json` | 0.03 | 无静态引用 | restore frozen/archive copy |
| `data/seeds/greedy.json` | 0.02 | 无静态引用 | restore frozen/archive copy |
| `data/seeds/hash.json` | 0.02 | 无静态引用 | restore frozen/archive copy |
| `data/seeds/seed.json` | 0.24 | `scripts/data_generate/generate_sft.py`, `scripts/data_generate/validate_seed_sandbox.py` | restore frozen/archive copy |
| `data/seeds/string.json` | 0.02 | 无静态引用 | restore frozen/archive copy |
| `data/seeds/two_pointers.json` | 0.02 | 无静态引用 | restore frozen/archive copy |
| `data/splits/grpo_train_ids.json` | 0.02 | `scripts/freeze_sft_data.py` | restore frozen/archive copy |
| `data/splits/grpo_validation_ids.json` | 0.00 | `scripts/freeze_sft_data.py` | restore frozen/archive copy |
| `data/splits/sft_dev_ids.json` | 0.01 | `scripts/freeze_sft_data.py` | restore frozen/archive copy |
| `data/splits/sft_train_ids.json` | 0.22 | `scripts/freeze_sft_data.py` | restore frozen/archive copy |

## 建议归档压缩

| 路径 | 大小 MiB | 代码引用 | 恢复方式 |
|---|---:|---|---|
| `data/cache/ref_label_full_generation.log` | 0.00 | `scripts/run_ref_label_sft_generation.ps1` | restore frozen/archive copy |
| `data/cache/ref_label_full_generation.log.err` | 0.08 | 无静态引用 | restore frozen/archive copy |
| `data/cache/ref_label_pilot50.log` | 0.02 | `scripts/run_ref_label_sft_generation.ps1` | restore frozen/archive copy |
| `data/cache/taco_reference_verification_train_full.jsonl` | 424.72 | `scripts/finalize_data_freeze.py`, `scripts/freeze_sft_data.py`, `scripts/run_ref_label_call_smoke.ps1`, `scripts/run_ref_label_sft_generation.ps1`, `scripts/run_ref_label_smoke.ps1` | restore frozen/archive copy |
| `data/raw/TACO/.cache/huggingface/.gitignore` | 0.00 | `scripts/package_gpt_context.ps1` | restore frozen/archive copy |
| `data/raw/TACO/.cache/huggingface/CACHEDIR.TAG` | 0.00 | 无静态引用 | restore frozen/archive copy |
| `data/raw/TACO/.cache/huggingface/download/.gitattributes.metadata` | 0.00 | 无静态引用 | restore frozen/archive copy |
| `data/raw/TACO/.cache/huggingface/download/ALL/73QwTJMUP35DxkZiaQR035cMdCo=.1934cd4c4e8784bfc231b0294326a308dc2b7abf5e3c7f17622ece10ae8d8b56.incomplete` | 0.00 | 无静态引用 | restore frozen/archive copy |
| `data/raw/TACO/.cache/huggingface/download/ALL/EFPgduGi1zhTP0S7qeV7I03u09g=.934953c5693650a58ede19966db7ec1322f721e9c91b4d62e90a67dced86121d.incomplete` | 0.00 | 无静态引用 | restore frozen/archive copy |
| `data/raw/TACO/.cache/huggingface/download/ALL/ihk3HleDkg-OJTWw4BKH7ppLW1Y=.1ad70829b190935c6cbded92d013b9b1fc10cd154dc0814c929716a1ea9ad1ae.incomplete` | 0.00 | 无静态引用 | restore frozen/archive copy |
| `data/raw/TACO/.cache/huggingface/download/ALL/iln7FAK8ScOKT0OL8tHV8680tg8=.5d99adc603500c05751aff9f61bed7bbd54a9ad5ea569d108b645346655aae44.incomplete` | 0.00 | 无静态引用 | restore frozen/archive copy |
| `data/raw/TACO/.cache/huggingface/download/ALL/IOOW-P0pJherlFfg5KlAFw9S1N4=.e6ebb62153e93828e631f985d29466d7819262a6811a49b00a23a3cdb1904a11.incomplete` | 0.00 | 无静态引用 | restore frozen/archive copy |
| `data/raw/TACO/.cache/huggingface/download/ALL/KoFlAO9ObgOYYXtzex_9BVX4EyM=.c604f98dc0d1568939372f78fc6ed654c31b34ed3e0302636efcc8547802439c.incomplete` | 0.00 | 无静态引用 | restore frozen/archive copy |
| `data/raw/TACO/.cache/huggingface/download/ALL/tUJdi1NpWVS_74HBqU4e8p47ERY=.e3dfc9a787b2234fbb25f31a7d15fad734c518c5fb6fc62a71616dae69944aef.incomplete` | 0.00 | 无静态引用 | restore frozen/archive copy |
| `data/raw/TACO/.cache/huggingface/download/ALL/uiv50W-c9b726BZt2mihpwbzID0=.bee336c14dda183b1f700d54a149173418c7b3def295666159dd72c32aa8b326.incomplete` | 0.00 | 无静态引用 | restore frozen/archive copy |
| `data/raw/TACO/.cache/huggingface/download/ALL/YKwYpXdmZldXm2AsvFIYUDjw_G8=.d634726c27b85171494f7d38abfcd31d773a205002f9218bb54bc2cba49c99d4.incomplete` | 0.00 | 无静态引用 | restore frozen/archive copy |
| `data/raw/TACO/.cache/huggingface/download/README.md.metadata` | 0.00 | 无静态引用 | restore frozen/archive copy |
| `data/raw/TACO/.cache/huggingface/download/TACO.py.metadata` | 0.00 | 无静态引用 | restore frozen/archive copy |
| `data/raw/TACO/.cache/huggingface/download/test/ph-PwKzEizU51qoBvN-kh8yvA-U=.39e312a5096f88929cbdd7324f06da41b1315d1b32f0d803dbe66dc1957c58fe.incomplete` | 0.00 | 无静态引用 | restore frozen/archive copy |
| `data/raw/TACO/.cache/huggingface/download/train/-wlV6Os1lEPl9nc0v2WawbtMOqk=.613f9a94291a262cffe3a5fe0c51096d5a5617f84b19c5cda832e7bc23942be2.incomplete` | 0.00 | 无静态引用 | restore frozen/archive copy |
| `data/raw/TACO/.cache/huggingface/download/train/R3uHPYO1iam6tZmf3xIbBABJgnU=.3af79af93fbad8a1e9d0a5ac37abf5c756719ab266a4b18c2db760377ff6e9fe.incomplete` | 0.00 | 无静态引用 | restore frozen/archive copy |
| `data/raw/TACO/.cache/huggingface/download/train/RJJK1pJaT1zOz6GRp7biCXkxpW0=.b6d3c6c29f97e0ee4bbbe4ab577b851e7aa055905d6a935146128a0190cffa72.incomplete` | 0.00 | 无静态引用 | restore frozen/archive copy |
| `data/raw/TACO/.gitattributes` | 0.00 | 无静态引用 | restore frozen/archive copy |
| `data/raw/TACO/README.md` | 0.01 | `scripts/colab_quickstart.sh`, `scripts/package_gpt_context.ps1` | restore frozen/archive copy |
| `data/raw/TACO/TACO.py` | 0.01 | `scripts/package_gpt_context.ps1` | restore frozen/archive copy |
| `data/sft_train_pilot50_accepted.jsonl` | 0.53 | `scripts/run_ref_label_sft_generation.ps1` | restore frozen/archive copy |
| `data/sft_train_pilot50_rejected.jsonl` | 0.15 | `scripts/run_ref_label_sft_generation.ps1` | restore frozen/archive copy |
| `data/sft_train_ref_label_accepted.jsonl` | 906.09 | `scripts/finalize_data_freeze.py`, `scripts/freeze_sft_data.py`, `scripts/run_ref_label_sft_generation.ps1` | restore frozen/archive copy |
| `data/sft_train_ref_label_rejected.jsonl` | 0.05 | `scripts/finalize_data_freeze.py`, `scripts/freeze_sft_data.py`, `scripts/run_ref_label_sft_generation.ps1` | restore frozen/archive copy |

## 可归档；确认外部备份后可重新下载

| 路径 | 大小 MiB | 代码引用 | 恢复方式 |
|---|---:|---|---|
| `data/raw/TACO/ALL/train-00000-of-00009.parquet` | 273.63 | 无静态引用 | download BAAI/TACO parquet again |
| `data/raw/TACO/ALL/train-00001-of-00009.parquet` | 312.06 | 无静态引用 | download BAAI/TACO parquet again |
| `data/raw/TACO/ALL/train-00002-of-00009.parquet` | 169.31 | 无静态引用 | download BAAI/TACO parquet again |
| `data/raw/TACO/ALL/train-00003-of-00009.parquet` | 171.15 | 无静态引用 | download BAAI/TACO parquet again |
| `data/raw/TACO/ALL/train-00004-of-00009.parquet` | 196.18 | 无静态引用 | download BAAI/TACO parquet again |
| `data/raw/TACO/ALL/train-00005-of-00009.parquet` | 259.13 | 无静态引用 | download BAAI/TACO parquet again |
| `data/raw/TACO/ALL/train-00006-of-00009.parquet` | 203.90 | 无静态引用 | download BAAI/TACO parquet again |
| `data/raw/TACO/ALL/train-00007-of-00009.parquet` | 248.07 | 无静态引用 | download BAAI/TACO parquet again |
| `data/raw/TACO/ALL/train-00008-of-00009.parquet` | 239.92 | 无静态引用 | download BAAI/TACO parquet again |

## 可安全删除并重建

| 路径 | 大小 MiB | 代码引用 | 恢复方式 |
|---|---:|---|---|
| `data/__pycache__/tmp_bad_encoding.cpython-311.pyc` | 0.01 | 无静态引用 | rerun the producing script or restore archive |
| `data/__pycache__/tmp_bad_encoding.cpython-38.pyc` | 0.01 | 无静态引用 | rerun the producing script or restore archive |
| `data/cache/ref_label_api_probe.log` | 0.00 | 无静态引用 | rerun the producing script or restore archive |
| `data/cache/ref_label_api_probe_accepted.log` | 0.00 | 无静态引用 | rerun the producing script or restore archive |
| `data/cache/ref_label_api_probe_accepted5.log` | 0.00 | 无静态引用 | rerun the producing script or restore archive |
| `data/cache/ref_label_call_smoke.log` | 0.00 | `scripts/run_ref_label_call_smoke.ps1` | rerun the producing script or restore archive |
| `data/cache/ref_label_smoke.log` | 0.00 | `scripts/run_ref_label_smoke.ps1` | rerun the producing script or restore archive |
| `data/cache/taco_reference_verification_call_smoke.jsonl` | 0.00 | `scripts/package_gpt_context.ps1` | rerun the producing script or restore archive |
| `data/cache/taco_reference_verification_multi_100.jsonl` | 9.77 | 无静态引用 | rerun the producing script or restore archive |
| `data/cache/taco_reference_verification_multi_call20.jsonl` | 0.02 | 无静态引用 | rerun the producing script or restore archive |
| `data/cache/taco_reference_verification_multi_smoke4.jsonl` | 0.00 | `scripts/package_gpt_context.ps1` | rerun the producing script or restore archive |
| `data/cache/taco_reference_verification_parallel_probe.jsonl` | 0.01 | 无静态引用 | rerun the producing script or restore archive |
| `data/cache/taco_reference_verification_smoke.jsonl` | 0.00 | 无静态引用 | rerun the producing script or restore archive |
| `data/raw/TACO/__pycache__/TACO.cpython-311.pyc` | 0.01 | 无静态引用 | rerun the producing script or restore archive |
| `data/sft_train_ref_label_rejected.jsonl.tmp` | 0.39 | 无静态引用 | rerun the producing script or restore archive |
| `data/tmp_bad_encoding.py` | 0.01 | `scripts/finalize_data_freeze.py` | rerun the producing script or restore archive |

## 空间估算

- 可安全删除并重建：10.23 MiB。
- TACO train 可重新下载部分：2073.35 MiB。
- 建议归档部分：1331.66 MiB。
- TACO test 默认保留。TACO train 仅因 source bank 读取、hash 和 20 条复验均通过而被标记为可重新下载；本轮没有删除。
