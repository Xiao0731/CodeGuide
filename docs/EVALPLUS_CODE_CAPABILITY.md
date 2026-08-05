# EvalPlus 外源代码能力评测

## 评测边界

本模块只评估代码能力，不评估教学格式和教学质量。

CodeGuide 后续总评拆成两个独立模块：

1. **代码能力模块**：执行正确率、接口正确性、外源泛化；
2. **教学能力模块**：格式遵循、讲解准确性、初学者友好度、讲解与代码一致性。

本次 EvalPlus 只属于第一个模块。

## 冻结实验矩阵

模型：

- `Qwen/Qwen2.5-Coder-7B-Instruct` Base；
- Mixed SFT `1e-4 step20`；
- Mixed SFT `1e-4 step200`。

数据：

- HumanEval：164题；
- MBPP+规范子集：378题。

输出：

- HumanEval Pass@1；
- HumanEval+ Pass@1；
- MBPP Pass@1；
- MBPP+ Pass@1；
- 平均生成 token；
- 生成上限命中数；
- Python语法失败数。

所有模型使用相同提示词、BF16精度、贪心解码、代码提取和
EvalPlus v0.3.1执行器。实验增量只相对于本次同协议复现的Base计算。

## 云端安装

将补丁 ZIP 上传至：

```text
/home/dataset-assist-0/CodeGuide-main
```

在项目根目录覆盖解压：

```bash
cd /home/dataset-assist-0/CodeGuide-main
unzip -o codeguide_evalplus_code_capability_20260805.zip -d .
chmod +x scripts/run_evalplus_code_capability_cloud.sh
python -m compileall -q scripts
python -m unittest tests.test_evalplus_code_capability
```

使用清华镜像安装固定版 EvalPlus：

```bash
python -m pip install -r requirements-external-eval.txt \
  -i https://pypi.tuna.tsinghua.edu.cn/simple \
  --trusted-host pypi.tuna.tsinghua.edu.cn \
  --prefer-binary
```

检查：

```bash
python - <<'PY'
import importlib.metadata
print(importlib.metadata.version("evalplus"))
PY
```

预期为 `0.3.1`。

## 当前只空闲一张GPU

Standard专家仍在GPU0测试探针时，使用GPU1顺序运行三个版本：

```bash
GPU_IDS=1 BATCH_SIZE=4 \
nohup bash scripts/run_evalplus_code_capability_cloud.sh \
  > outputs/evalplus_code_cloud_runner.log 2>&1 &

echo $! > outputs/evalplus_code_cloud_runner.pid
tail -f outputs/evalplus_code_cloud_runner.log
```

运行顺序为：

```text
Base → step200 → step20
```

这只是为了优先得到主比较，不改变最终表格顺序。

显存不足时改成：

```bash
GPU_IDS=1 BATCH_SIZE=2 \
nohup bash scripts/run_evalplus_code_capability_cloud.sh \
  > outputs/evalplus_code_cloud_runner.log 2>&1 &
```

## 两张GPU都空闲

```bash
GPU_IDS=0,1 BATCH_SIZE=4 \
bash scripts/run_evalplus_code_capability_cloud.sh
```

第一轮并行：

```text
GPU0：Base
GPU1：step200
```

随后GPU0运行step20。

完成后生成：

```text
codeguide_evalplus_code_generations_YYYYMMDD_HHMMSS.tar.gz
```

## 云端完整性检查

```bash
cat outputs/eval/evalplus_code_capability_v1/cloud_generation_acceptance.json
```

必须满足：

```json
"complete": true
```

每个模型必须分别具有：

```text
HumanEval：164条
MBPP：378条
```

## 本地Docker评分

将云端压缩包解压到本地 CodeGuide 项目根目录。

PowerShell：

```powershell
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

powershell -ExecutionPolicy Bypass -File `
  .\scripts\evaluate_evalplus_code_capability_local.ps1 `
  -RunRoot outputs/eval/evalplus_code_capability_v1 `
  -Parallel 4 `
  -PullImage
```

需要强制重跑缓存时追加 `-Force`。

最终输出：

```text
outputs/eval/evalplus_code_capability_v1/reports/evalplus_code_summary_wide.csv
outputs/eval/evalplus_code_capability_v1/reports/evalplus_code_summary_long.csv
outputs/eval/evalplus_code_capability_v1/reports/evalplus_code_summary.json
outputs/eval/evalplus_code_capability_v1/reports/evalplus_code_echarts_option.js
```

## 解释规则

官方历史Base分数只作为背景，不直接用于计算微调增益。正式结论使用：

```text
checkpoint分数 − 本次同环境复现Base分数
```

若本次Base与官方历史值有差距，优先核查模型revision、BF16/量化精度、
Transformers版本、EvalPlus版本、提示词、代码提取和数据集任务数。

只要三种模型在同一冻结协议下比较，实验增量仍然有效。
