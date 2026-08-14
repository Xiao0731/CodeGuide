# TACO-515 Compact Code First 评测指南

## 需要放入仓库的文件

将压缩包内目录按原路径复制到 CodeGuide 仓库根目录：

- `configs/eval/taco515_compact_code_first_v1.yaml`
- `scripts/evaluate_sft_matrix.py`
- `scripts/run_taco515_compact_dual_4090.sh`
- `scripts/score_sft_static_proxy.py`
- `src/evaluation/__init__.py`
- `src/evaluation/static_proxy.py`
- `tests/test_sft_eval_protocol.py`
- `tests/test_static_proxy.py`

历史 40 题校准结论保留在实验日志中；当前统一使用矩阵评测器，避免维护两套提取与执行逻辑。

## 先做本地检查

```bash
python -m compileall \
  scripts/evaluate_sft_matrix.py \
  scripts/score_sft_static_proxy.py \
  src/evaluation

pytest -q tests/test_sft_eval_protocol.py tests/test_static_proxy.py
```

## 冻结 515 题和协议

```bash
python scripts/evaluate_sft_matrix.py \
  --stage prepare \
  --run-dir outputs/sft/taco515_compact_code_first_v1 \
  --protocol-config configs/eval/taco515_compact_code_first_v1.yaml \
  --batch-size 4
```

`manifest.json` 会冻结 batch size。正式运行过程中不要临时改成 2/6/8；需要改时使用新的 run directory。

## 最长题显存预检

先只让 Base 在一张卡上生成最长的 8 道题；结果不会写入正式 generation：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_sft_matrix.py \
  --stage preflight \
  --variant base \
  --batch-size 4 \
  --preflight-items 8 \
  --run-dir outputs/sft/taco515_compact_code_first_v1 \
  --protocol-config configs/eval/taco515_compact_code_first_v1.yaml
```

查看 `reports/preflight_base.json`。峰值显存明显低于 24GB 时再启动四组正式生成。

## 双 4090 生成

```bash
export CALIBRATION_ADAPTER=/path/to/calibration500
export CHECKPOINT_ADAPTER=/path/to/checkpoint
export FULL_ADAPTER=/path/to/full
export BATCH_SIZE=4

bash scripts/run_taco515_compact_dual_4090.sh
```

实时观察：

```bash
watch -n 1 nvidia-smi

tail -f outputs/sft/taco515_compact_code_first_v1/logs/base.log
tail -f outputs/sft/taco515_compact_code_first_v1/logs/calibration500.log
```

每个 batch 的日志会直接打印峰值显存 `peak=...GiB`。若 batch 4 OOM，应停止四组实验，换一个新目录并让四组统一使用 batch 2。

## 严格 Docker 验证

先设置固定 digest 镜像：

```bash
export CODEGUIDE_IMAGE='your-image@sha256:...'
```

分别验证：

```bash
for variant in base calibration500 checkpoint_step_xxx full; do
  python scripts/evaluate_sft_matrix.py \
    --stage verify \
    --variant "${variant}" \
    --batch-size 4 \
    --run-dir outputs/sft/taco515_compact_code_first_v1 \
    --protocol-config configs/eval/taco515_compact_code_first_v1.yaml \
    --container-image "${CODEGUIDE_IMAGE}" \
    --verify-workers 4
done
```

汇总 checkpoint trajectory：

```bash
python scripts/evaluate_sft_matrix.py \
  --stage summarize \
  --batch-size 4 \
  --run-dir outputs/sft/taco515_compact_code_first_v1 \
  --protocol-config configs/eval/taco515_compact_code_first_v1.yaml
```

## 博主静态代理二次评分

```bash
python scripts/score_sft_static_proxy.py \
  --run-dir outputs/sft/taco515_compact_code_first_v1 \
  --source-bank data/final/taco_verified_source_bank.jsonl.zst \
  --variants base calibration500 checkpoint_step_xxx full
```

输出：

- `static_proxy/<variant>.jsonl`
- `reports/static_proxy_summary.json`

静态分只能命名为 `External Static Code Proxy`，不能叫 Pass@1 或 correctness。
