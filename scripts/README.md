# Script Entrypoints

`scripts/` 只保留可复现实验流程的入口。训练实现位于 `src/`，脚本不再复制训练框架逻辑。

| 阶段 | 入口 | 作用 |
|---|---|---|
| 数据生成 | `build_sft_dataset.py` | reference-guided 教学标签生成与断点续跑 |
| Reference | `verify_taco_references.py` | 多候选 Python reference 离线执行验证 |
| 数据冻结 | `freeze_sft_data.py` | canonical SFT、source bank 与固定 split |
| SFT | `train_sft.py` | TRL SFTTrainer + PEFT QLoRA |
| GRPO | `train_grpo.py` | TRL GRPOTrainer + 两路本地 reward |
| TACO 评测 | `evaluate_sft_matrix.py` | 配置驱动的 Base/SFT/GRPO 生成与复验 |
| EvalPlus | `prepare_evalplus_datasets_offline.py`、`generate_evalplus_code_capability.py` | HumanEval(+)/MBPP(+) |
| Verifier | `validate_docker_verifier.py` | standard-input/call-based Docker smoke |
| 推理 | `inference_demo.py` | adapter/merged model CLI 推理 |

不同训练规模通过同一入口的参数控制，例如 `--mode calibration|full`、`--max-steps` 和配置文件，不再新增一次性脚本。
