# EvalPlus 外源代码能力评测

## 评测边界

本模块只评估代码能力，不评估教学格式和教学质量。

CodeGuide 后续总评拆成两个独立模块：

1. **代码能力模块**：执行正确率、接口正确性、外源泛化；
2. **教学能力模块**：格式遵循、讲解准确性、初学者友好度、讲解与代码一致性。

本次 EvalPlus 只属于第一个模块。

## 冻结实验矩阵

模型：Base、Mixed SFT 1e-4 step20、Mixed SFT 1e-4 step200。

数据：HumanEval 164题、MBPP+规范子集378题。

输出：HumanEval/HumanEval+/MBPP/MBPP+ Pass@1、平均生成token、生成上限命中数、Python语法失败数。

所有模型使用相同提示词、BF16精度、贪心解码、代码提取和EvalPlus v0.3.1执行器。实验增量只相对于本次同协议复现的Base计算。

## 云端安装

```bash
cd /home/dataset-assist-0/CodeGuide-main
unzip -o codeguide_evalplus_code_capability_20260805.zip -d .
chmod +x scripts/run_evalplus_code_capability_cloud.sh
python -m compileall -q scripts
python -m unittest tests.test_evalplus_code_capability

python -m pip install -r requirements-external-eval.txt \
  -i https://pypi.tuna.tsinghua.edu.cn/simple \
  --trusted-host pypi.tuna.tsinghua.edu.cn \
  --prefer-binary
```

## 当前只空闲GPU1

```bash
GPU_IDS=1 BATCH_SIZE=4 \
nohup bash scripts/run_evalplus_code_capability_cloud.sh \
  > outputs/evalplus_code_cloud_runner.log 2>&1 &

echo $! > outputs/evalplus_code_cloud_runner.pid
tail -f outputs/evalplus_code_cloud_runner.log
```

运行顺序：Base → step200 → step20。显存不足时把 `BATCH_SIZE=4` 改为 `2`。

## 两张GPU都空闲

```bash
GPU_IDS=0,1 BATCH_SIZE=4 bash scripts/run_evalplus_code_capability_cloud.sh
```

完成后生成 `codeguide_evalplus_code_generations_YYYYMMDD_HHMMSS.tar.gz`。

云端完整性检查：

```bash
cat outputs/eval/evalplus_code_capability_v1/cloud_generation_acceptance.json
```

必须为 `"complete": true`。

## 已冻结结果

本轮 EvalPlus 已完成，原始 sample、执行日志和汇总结果统一保存在下列正式目录。
一次性的本地评分包装器与汇总脚本不再维护；需要重做新实验时，使用当前 EvalPlus CLI
执行并建立新的版本化 run 目录，不覆盖本轮结果。

最终输出位于：

```text
outputs/eval/evalplus_code_capability_v1/reports/evalplus_code_summary_wide.csv
outputs/eval/evalplus_code_capability_v1/reports/evalplus_code_summary_long.csv
outputs/eval/evalplus_code_capability_v1/reports/evalplus_code_summary.json
outputs/eval/evalplus_code_capability_v1/reports/evalplus_code_echarts_option.js
```
