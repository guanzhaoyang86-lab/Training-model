# Residual V2 Framework

This is the current system framework for `Training-model-yilong`.
The project trains a vision-language model for time-series anomaly grounding, then improves the two-stage SFT model with residual-focused GRPO.

## Current Pipeline

```text
Base VLM
  -> Stage 1: synthetic SFT on anomaly_db_v1
  -> Stage 2: real-subset SFT on TSB-AD-U
  -> two-stage SFT model

two-stage SFT model
  -> Branch A: SFT-only test baseline
  -> Branch B: residual_v2
       -> residual mining on real train split
       -> residual_pool.jsonl
       -> point-F1 and boundary-aware GRPO
       -> final test
```

The main confirmation launcher is:

```text
submit_sft_vs_residual_v2_8subsets_3seed.sh
```

It runs three seeds across four model families, two variants, and eight TSB-AD-U subsets:

```text
models:   Qwen2.5-VL-7B, Qwen3-VL-2B, Qwen3-VL-4B, Qwen3-VL-8B
variants: sft_only, residual_v2
subsets:  Daphnet, MSL, NEK, Power, SED, TAO, TODS, YAHOO
```

The underlying matrix submitter is:

```text
submit_qwen_vl_all_models_sft_residual_ablation_8subsets.sh
```

## Task Definition

Each model input contains:

- a plain or normalized PNG time-series plot;
- indexed numeric values rendered as `0: value`, `1: value`, ..., `255: value`.

The model output is exactly one JSON object:

```json
{
  "evidence": [
    {
      "start": 253,
      "end": 253,
      "type": "range",
      "strength": "mild",
      "direction": "becomes irregular"
    }
  ],
  "summary": "An anomaly occurs from index 253 to 253, and this segment becomes irregular."
}
```

For a normal window:

```json
{"evidence": [], "summary": "No anomaly is detected."}
```

The structured `evidence` field is the primary output. `summary` is kept as an auxiliary explanation.

## Stage 1: Synthetic SFT

Stage 1 learns the basic image-and-series-to-JSON grounding behavior from synthetic anomaly data.

Default data source:

```text
dataset/anomaly_db_v1
```

Default setting:

```text
STAGE1_SFT_NUM_TRAIN_EPOCHS=1
```

The submitter generates a per-model Stage 1 config under:

```text
outputs/qwen_vl_all_models_sft_residual_ablation_8subsets_<tag>/<model>/stage1/synthetic_sft/
```

Training is executed through:

```text
run_full_train_gpu.sh
train.py
src/ts_grounder/vlm_training.py
```

During SFT, prompt, image, and indexed-value tokens are masked. Only assistant JSON response tokens contribute to the token-level cross-entropy loss.

## Stage 2: Real-Subset SFT

Stage 2 adapts the Stage 1 model to each real TSB-AD-U subset.

Default real subset root:

```text
dataset/tsb_adu_subset_splits_raw_file_padded_256_128_7_1_2/<SUBSET>
```

Default setting:

```text
STAGE2_SFT_NUM_TRAIN_EPOCHS=3
```

Each subset job uses the Stage 1 model as `BASE_MODEL_PATH` and writes a two-stage SFT checkpoint under that subset/variant workspace. This checkpoint is evaluated directly for the `sft_only` baseline and becomes the starting model for `residual_v2`.

## Stage 3: Residual Mining

Residual mining runs the two-stage SFT model on the real train split, scores the generated JSON against ground truth, and writes residual records.

Implementation:

```text
scripts/build_residual_pool.py
src/ts_grounder/event_metrics.py
```

Main output:

```text
<subset_workspace>/residual_pool/residual_pool.jsonl
```

Each residual record stores:

- ground-truth evidence;
- SFT generated text;
- parsed SFT evidence;
- JSON validity and parse errors;
- event precision, recall, F1;
- mean IoU and boundary MAE;
- error type.

The default residual_v2 sampling mix is:

```text
false_negative=0.60,
boundary_error=0.20,
false_positive=0.10,
correct_abnormal=0.10,
correct_normal=0.00
```

This makes GRPO focus on the failures that matter most for point-level detection and localization.

## Stage 4: Residual V2 GRPO

Residual_v2 GRPO starts from the two-stage SFT model and trains on the residual pool.

Implementation:

```text
src/ts_grounder/rl_train_grpo.py
src/ts_grounder/rl_reward.py
src/ts_grounder/event_metrics.py
```

The launcher enables:

```text
--use_residual_pool
--use_boundary_aware_reward
```

Default residual_v2 GRPO settings:

```text
RESIDUAL_V2_RL_NUM_GENERATIONS=6
RESIDUAL_V2_RL_NUM_TRAIN_EPOCHS=2
RESIDUAL_V2_RL_LEARNING_RATE=2e-6
RESIDUAL_V2_RL_KL_COEF=0.03
```

Default boundary-aware reward weights:

```text
point    = 0.60
event    = 0.00
iou      = 0.25
boundary = 0.15
type     = 0.00
```

The training loop samples multiple JSON responses for each residual record, scores each response, normalizes rewards within the group, and applies policy-gradient updates with KL regularization against the two-stage SFT reference.

## Stage 5: Final Test

Final evaluation is run with:

```text
run_predict_eval_gpu.sh
predict.py
scripts/eval_vlm_grounder.py
src/ts_grounder/vlm_eval.py
```

The kept artifact for each run is:

```text
<run>/rl/eval/test_metrics.json
```

The large confirmation run expects:

```text
3 seeds x 4 models x 2 variants x 8 subsets = 192 test_metrics.json files
```

Summaries are generated with:

```text
scripts/summarize_three_seed_ablation.py
```

## Active Framework

This document is the active system structure for the repository: residual_v2 with synthetic SFT, real-subset SFT, residual mining, and residual-aware GRPO.
