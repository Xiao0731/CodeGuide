# 脚本目录

`scripts/` 只保留当前数据、训练、评测和部署主线。训练实现放在 `src/`，脚本层尽量只做
CLI 编排，避免出现两套逐渐分叉的实现。

## 数据

- `build_sft_dataset.py`：reference-guided 教学标签生成。
- `verify_taco_references.py`：TACO 多候选参考解离线验证。
- `freeze_sft_data.py`、`finalize_data_freeze.py`：冻结 canonical SFT、source bank 和 manifest。
- `audit_sft_training_format.py`、`audit_token_lengths.py`：训练格式与 tokenizer 长度审计。
- `source_bank_io.py`、`verify_source_bank_sample.py`：source bank 读取与抽样复验。

## 训练

- `train_sft.py`：兼容入口，实际调用 `src.training.train_sft`。
- `run_sft_full_dual_4090.sh`：双 RTX 4090 全量 QLoRA SFT。
- `preflight_sft_dual_4090.sh`：CUDA、bitsandbytes、Liger 和数据合同预检。
- `train_grpo.py`：兼容入口，实际调用 `src.training.grpo_train`。
- `train_grpo_minimal.py`：当前冻结的最小可运行 GRPO 管线。
- `prepare_grpo_minimal_config.py`、`validate_grpo_ready.py`：GRPO 数据与配置准备/门禁。

## 评测

- `evaluate_sft_matrix.py`：统一的 TACO checkpoint 矩阵生成、Docker 验证和汇总。
- `verify_saved_matrix_simple.py`：对已保存 generation 离线复验，不重新调用模型。
- `score_sft_static_proxy.py`：静态教学指标，仅作诊断。
- `generate_evalplus_code_capability.py`：HumanEval+/MBPP+ 外源代码能力生成；已完成评分保存在正式输出目录。
- `evaluate_text_routed_experts.py`、`select_best_expert_checkpoint.py`：I/O 专家消融。
- `inference_demo.py`、`gradio_demo.py`：最终模型命令行与 Web 演示。

历史 API smoke、500 条校准包装器、Colab/Kaggle/AIStudio 安装脚本和旧单模型评测入口已删除；
实验结论保留在 `EXPERIMENT_LOG.md` 与正式 `outputs/eval/` 产物中。
