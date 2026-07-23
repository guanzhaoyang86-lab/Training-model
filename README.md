# TS Grounder VLM

`Training-model-yilong` 当前主线是 **residual_V2 time-series anomaly grounding**：先用 synthetic anomaly data 学基础 JSON grounding，再在真实 TSB-AD-U subset 上做 domain adaptation，最后用 residual mining 驱动 point-F1 / boundary-aware GRPO。

当前系统框架见：

- [docs/residual_v2_framework.md](docs/residual_v2_framework.md)

## Current Framework

```text
Base VLM
  -> Stage 1: synthetic SFT on anomaly_db_v1
  -> Stage 2: real-subset SFT on TSB-AD-U
  -> two-stage SFT model

two-stage SFT model
  -> sft_only baseline
  -> residual_v2:
       residual mining on real train split
       -> residual_pool.jsonl
       -> point-F1 / boundary-aware GRPO
       -> final test
```

当前保留的主结构是：

```text
synthetic SFT -> real-subset SFT -> residual mining -> residual-aware GRPO
```

## Task Definition

- 输入：plain 或 normalized PNG 时序图像 + indexed series values
- 输出：结构化 anomaly evidence + 一句 summary
- 范式：`image + indexed values -> structured evidence + summary`

模型目标输出固定为一个 JSON 对象：

```json
{
  "evidence": [
    {
      "start": 127,
      "end": 130,
      "type": "point",
      "strength": "strong",
      "direction": "becomes irregular"
    }
  ],
  "summary": "A strong anomaly occurs from index 127 to 130, and this segment becomes irregular."
}
```

无异常窗口输出：

```json
{
  "evidence": [],
  "summary": "No anomaly is detected."
}
```

约束：

- `evidence` 必须是列表。
- `type` 只能是 `point`, `freq`, `trend`, `range`。
- `evidence` 是主输出，`summary` 是辅助解释。

## Data

Stage 1 默认使用 synthetic anomaly 数据：

```text
dataset/anomaly_db_v1
```

Stage 2 和 residual_v2 默认使用 TSB-AD-U 八个真实 subset：

```text
dataset/tsb_adu_subset_splits_raw_file_padded_256_128_7_1_2/<SUBSET>
```

当前默认 subsets：

```text
Daphnet, MSL, NEK, Power, SED, TAO, TODS, YAHOO
```

每个 split 样本会被转换为 VLM record，核心字段包括：

- `image_path`
- `system_prompt`
- `user_prompt`
- `assistant_text`
- `target`
- `metadata`

位置提示主要来自文本里的 indexed series values；图片侧使用 plain plot 或自动生成的 normalized plain plot。

## Main Experiment

完整 residual_v2 confirmation run：

```bash
bash submit_sft_vs_residual_v2_8subsets_3seed.sh
```

默认矩阵：

```text
seeds:    3
models:   Qwen2.5-VL-7B, Qwen3-VL-2B, Qwen3-VL-4B, Qwen3-VL-8B
variants: sft_only, residual_v2
subsets:  Daphnet, MSL, NEK, Power, SED, TAO, TODS, YAHOO
```

预期最终 metrics 数量：

```text
3 seeds x 4 models x 2 variants x 8 subsets = 192 test_metrics.json files
```

底层 submitter：

```bash
bash submit_qwen_vl_all_models_sft_residual_ablation_8subsets.sh
```

单模型/单 subset dry run 示例：

```bash
DRY_RUN=1 \
MODELS="2b" \
SUBSETS="YAHOO" \
VARIANTS="residual_v2" \
bash submit_qwen_vl_all_models_sft_residual_ablation_8subsets.sh
```

## Residual V2 Settings

默认 SFT 设置：

```text
STAGE1_SFT_NUM_TRAIN_EPOCHS=1
STAGE2_SFT_NUM_TRAIN_EPOCHS=3
```

默认 residual pool 采样：

```text
false_negative=0.60,
boundary_error=0.20,
false_positive=0.10,
correct_abnormal=0.10,
correct_normal=0.00
```

默认 residual_v2 GRPO 设置：

```text
RESIDUAL_V2_RL_NUM_GENERATIONS=6
RESIDUAL_V2_RL_NUM_TRAIN_EPOCHS=2
RESIDUAL_V2_RL_LEARNING_RATE=2e-6
RESIDUAL_V2_RL_KL_COEF=0.03
```

默认 boundary-aware reward：

```text
point    = 0.60
event    = 0.00
iou      = 0.25
boundary = 0.15
type     = 0.00
```

## Residual Mining

Residual mining 用 two-stage SFT model 在真实 train split 上推理，然后按 ground truth 打分并形成 `residual_pool.jsonl`。

入口：

```bash
python scripts/build_residual_pool.py \
  --model-path outputs/your_two_stage_sft/model \
  --real-train-data dataset/tsb_adu_subset_splits_raw_file_padded_256_128_7_1_2/YAHOO/train.json \
  --source-root dataset/tsb_adu_subset_splits_raw_file_padded_256_128_7_1_2/YAHOO \
  --output outputs/your_run/residual_pool/residual_pool.jsonl
```

每条 residual record 会记录：

- ground truth evidence
- SFT generated text
- parsed prediction evidence
- JSON validity / parse errors
- event F1, mean IoU, boundary MAE
- `error_type`

## Prediction And Evaluation

推理入口：

```bash
python predict.py \
  --config outputs/your_run/resolved_config.yaml \
  --dataset-jsonl outputs/your_run/dataset_cache/test.jsonl \
  --model-path outputs/your_run/model \
  --output outputs/your_run/test_predictions.jsonl
```

评估入口：

```bash
python scripts/eval_vlm_grounder.py \
  --dataset-jsonl outputs/your_run/dataset_cache/test.jsonl \
  --predictions outputs/your_run/test_predictions.jsonl \
  --output outputs/your_run/test_metrics.json
```

批量实验里通常通过 `run_predict_eval_gpu.sh` 自动完成预测和评估。

## Summaries

三 seed ablation 汇总：

```bash
python scripts/summarize_three_seed_ablation.py \
  --root outputs \
  --prefix qwen_vl_all_models_sft_residual_ablation_8subsets_sft_vs_residual_v2_8subsets_3seed
```

reward sweep 汇总：

```bash
python scripts/summarize_reward_weight_sweep.py
```

## Repo Layout

```text
configs/
  vlm_7b_*.yaml
  vlm_qwen3_vl_*.yaml
docs/
  residual_v2_framework.md
scripts/
  build_residual_pool.py
  eval_vlm_grounder.py
  summarize_three_seed_ablation.py
  summarize_reward_weight_sweep.py
src/ts_grounder/
  event_metrics.py
  rl_reward.py
  rl_train_grpo.py
  schema.py
  taxonomy.py
  utils.py
  vlm_data.py
  vlm_eval.py
  vlm_prompting.py
  vlm_training.py
tools/
  build_vlm_sft_dataset.py
  build_tsb_adu_*.py
train.py
predict.py
```

## Key Files

- `submit_sft_vs_residual_v2_8subsets_3seed.sh`: full residual_v2 confirmation launcher.
- `submit_qwen_vl_all_models_sft_residual_ablation_8subsets.sh`: all-model, all-subset matrix submitter.
- `scripts/build_residual_pool.py`: residual mining.
- `src/ts_grounder/rl_train_grpo.py`: residual-aware GRPO training.
- `src/ts_grounder/event_metrics.py`: event scoring, error typing, residual sampling, boundary-aware reward.
- `src/ts_grounder/vlm_training.py`: Hugging Face VLM SFT and generation.
- `src/ts_grounder/vlm_data.py`: raw samples to VLM records.
- `src/ts_grounder/vlm_eval.py`: prediction evaluation.

## Environment

推荐 Python 3.10+：

```bash
python3.10 -m venv .venv
.venv/bin/pip install -r requirements-lock.txt
```

服务器上常用解释器：

```bash
/gpfs/projects/p33222/ybq9740/envs/anomamind/bin/python
```

本地或服务器首次换模型时，先跑小规模 dry run 或 smoke run，确认数据构建、schema、推理和评估链路都能走通。

## Tests

Reward tests：

```bash
python -m pytest tests/test_rl_reward.py
```

Event metric tests：

```bash
python -m pytest tests/test_event_metrics.py
```

Prepared dataset smoke test：

```bash
python scripts/smoke_test.py --dataset-jsonl outputs/your_run/dataset_cache/val.jsonl
```
