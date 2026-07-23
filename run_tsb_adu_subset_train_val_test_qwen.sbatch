#!/bin/bash
#SBATCH --account=p33222
#SBATCH --job-name=tsg-q25vl7b-subset
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=sxm
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

ROOT_DIR="${PROJECT_ROOT:-}"
if [[ -z "$ROOT_DIR" ]]; then
  ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

cd "$ROOT_DIR"
mkdir -p "$ROOT_DIR/outputs" "$ROOT_DIR/.cache" "$ROOT_DIR/.tmp"

SUBSET="${SOURCE_DATASET:-}"
if [[ -z "$SUBSET" ]]; then
  echo "SOURCE_DATASET is required." >&2
  exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-/gpfs/projects/p33222/ybq9740/envs/anomamind/bin/python}"
CONFIG_PATH="${CONFIG_PATH:-$ROOT_DIR/configs/vlm_7b_tsb_adu_windowed_indexed_text.yaml}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-/gpfs/projects/p33222/ybq9740/models/Qwen2.5-VL-7B-Instruct}"
SOURCE_ROOT_BASE="${SOURCE_ROOT_BASE:-$ROOT_DIR/dataset/tsb_adu_subset_splits_raw_file_padded_256_128_7_1_2}"
SOURCE_ROOT="${SOURCE_ROOT:-$SOURCE_ROOT_BASE/$SUBSET}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/outputs/qwen25vl_7b_${SUBSET,,}_${SLURM_JOB_ID:-manual}}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
SAVE_PREDICTIONS="${SAVE_PREDICTIONS:-0}"
EVAL_SPLITS="${EVAL_SPLITS:-test}"
SFT_NUM_TRAIN_EPOCHS="${SFT_NUM_TRAIN_EPOCHS:-}"
ENABLE_RL="${ENABLE_RL:-0}"
SKIP_SFT_BEFORE_RL="${SKIP_SFT_BEFORE_RL:-0}"
RL_OUTPUT_DIR="${RL_OUTPUT_DIR:-$OUTPUT_DIR/rl}"
RL_NUM_GENERATIONS="${RL_NUM_GENERATIONS:-4}"
RL_MAX_NEW_TOKENS="${RL_MAX_NEW_TOKENS:-256}"
RL_TEMPERATURE="${RL_TEMPERATURE:-0.7}"
RL_TOP_P="${RL_TOP_P:-0.9}"
RL_LEARNING_RATE="${RL_LEARNING_RATE:-1e-6}"
RL_NUM_TRAIN_EPOCHS="${RL_NUM_TRAIN_EPOCHS:-1}"
RL_KL_COEF="${RL_KL_COEF:-0.02}"
RL_REWARD_EVENT_F1_WEIGHT="${RL_REWARD_EVENT_F1_WEIGHT:-0.45}"
RL_REWARD_BOUNDARY_IOU_WEIGHT="${RL_REWARD_BOUNDARY_IOU_WEIGHT:-0.35}"
RL_REWARD_HALLUCINATION_PENALTY_WEIGHT="${RL_REWARD_HALLUCINATION_PENALTY_WEIGHT:-0.0}"
RL_SAVE_STEPS="${RL_SAVE_STEPS:-100}"
RL_OPTIMIZER="${RL_OPTIMIZER:-adafactor}"
CLEANUP_SFT_AFTER_RL="${CLEANUP_SFT_AFTER_RL:-1}"
CLEANUP_FINAL_MODEL_AFTER_EVAL="${CLEANUP_FINAL_MODEL_AFTER_EVAL:-1}"
CLEANUP_RL_ARTIFACTS_AFTER_EVAL="${CLEANUP_RL_ARTIFACTS_AFTER_EVAL:-0}"

if [[ "$SKIP_SFT_BEFORE_RL" == "1" && "$ENABLE_RL" != "1" ]]; then
  echo "SKIP_SFT_BEFORE_RL=1 requires ENABLE_RL=1." >&2
  exit 1
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "PYTHON_BIN is not executable: $PYTHON_BIN" >&2
  exit 1
fi
if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "CONFIG_PATH does not exist: $CONFIG_PATH" >&2
  exit 1
fi
if [[ ! -d "$BASE_MODEL_PATH" ]]; then
  echo "BASE_MODEL_PATH does not exist: $BASE_MODEL_PATH" >&2
  exit 1
fi
if [[ ! -d "$SOURCE_ROOT" ]]; then
  echo "SOURCE_ROOT does not exist: $SOURCE_ROOT" >&2
  exit 1
fi
for split in train val test; do
  if [[ ! -f "$SOURCE_ROOT/$split.json" ]]; then
    echo "Missing $split.json under $SOURCE_ROOT" >&2
    exit 1
  fi
done

WORKFLOW_LABEL="SFT+RL"
FINAL_RUN_DIR="$RL_OUTPUT_DIR"
FINAL_MODEL_PATH="$RL_OUTPUT_DIR/model"
FINAL_EVAL_DIR="$RL_OUTPUT_DIR/eval"
FINAL_MODEL_LABEL="RL model"
if [[ "$SKIP_SFT_BEFORE_RL" == "1" ]]; then
  WORKFLOW_LABEL="RL-only"
  RL_BASE_MODEL_PATH="${RL_BASE_MODEL_PATH:-$BASE_MODEL_PATH}"
else
  RL_BASE_MODEL_PATH="${RL_BASE_MODEL_PATH:-$OUTPUT_DIR/model}"
fi
if [[ "$ENABLE_RL" == "0" ]]; then
  WORKFLOW_LABEL="SFT-only"
  FINAL_RUN_DIR="$OUTPUT_DIR"
  FINAL_MODEL_PATH="$OUTPUT_DIR/model"
  FINAL_EVAL_DIR="$OUTPUT_DIR/eval"
  FINAL_MODEL_LABEL="SFT model"
fi

echo "=== TSB-AD-U subset $WORKFLOW_LABEL train+test workflow ==="
echo "HOSTNAME=$(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-manual}"
echo "SOURCE_DATASET=$SUBSET"
echo "SOURCE_ROOT=$SOURCE_ROOT"
echo "BASE_MODEL_PATH=$BASE_MODEL_PATH"
echo "CONFIG_PATH=$CONFIG_PATH"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "ENABLE_RL=$ENABLE_RL"
echo "SKIP_SFT_BEFORE_RL=$SKIP_SFT_BEFORE_RL"
echo "RL_OUTPUT_DIR=$RL_OUTPUT_DIR"
echo "RL_BASE_MODEL_PATH=$RL_BASE_MODEL_PATH"
echo "EVAL_SPLITS=$EVAL_SPLITS"
echo "MAX_NEW_TOKENS=$MAX_NEW_TOKENS"
echo "SAVE_PREDICTIONS=$SAVE_PREDICTIONS"
echo "SFT_NUM_TRAIN_EPOCHS=${SFT_NUM_TRAIN_EPOCHS:-config default}"
echo "RL_NUM_GENERATIONS=$RL_NUM_GENERATIONS"
echo "RL_MAX_NEW_TOKENS=$RL_MAX_NEW_TOKENS"
echo "RL_LEARNING_RATE=$RL_LEARNING_RATE"
echo "RL_NUM_TRAIN_EPOCHS=$RL_NUM_TRAIN_EPOCHS"
echo "RL_KL_COEF=$RL_KL_COEF"
echo "RL_REWARD_EVENT_F1_WEIGHT=$RL_REWARD_EVENT_F1_WEIGHT"
echo "RL_REWARD_BOUNDARY_IOU_WEIGHT=$RL_REWARD_BOUNDARY_IOU_WEIGHT"
echo "RL_REWARD_HALLUCINATION_PENALTY_WEIGHT=$RL_REWARD_HALLUCINATION_PENALTY_WEIGHT"
echo "RL_OPTIMIZER=$RL_OPTIMIZER"
echo "CLEANUP_SFT_AFTER_RL=$CLEANUP_SFT_AFTER_RL"
echo "CLEANUP_FINAL_MODEL_AFTER_EVAL=$CLEANUP_FINAL_MODEL_AFTER_EVAL"
echo "CLEANUP_RL_ARTIFACTS_AFTER_EVAL=$CLEANUP_RL_ARTIFACTS_AFTER_EVAL"
nvidia-smi || true

if [[ "$SKIP_SFT_BEFORE_RL" == "1" ]]; then
  echo "=== Stage 1: prepare real-subset JSONL only (SFT skipped) ==="
  PROJECT_ROOT="$ROOT_DIR" \
  CONFIG_PATH="$CONFIG_PATH" \
  OUTPUT_DIR="$OUTPUT_DIR" \
  MODE=prepare_only \
  PYTHON_BIN="$PYTHON_BIN" \
  MODEL_PATH="$BASE_MODEL_PATH" \
  SOURCE_ROOT="$SOURCE_ROOT" \
  BF16="${BF16:-1}" \
  FP16="${FP16:-0}" \
  LAUNCHER=python \
  NUM_PROCS_PER_NODE=1 \
  bash "$ROOT_DIR/run_full_train_gpu.sh"
else
  echo "=== Stage 1: SFT ==="
  PROJECT_ROOT="$ROOT_DIR" \
  CONFIG_PATH="$CONFIG_PATH" \
  OUTPUT_DIR="$OUTPUT_DIR" \
  MODE=train \
  PYTHON_BIN="$PYTHON_BIN" \
  MODEL_PATH="$BASE_MODEL_PATH" \
  SOURCE_ROOT="$SOURCE_ROOT" \
  BF16="${BF16:-1}" \
  FP16="${FP16:-0}" \
  LAUNCHER=python \
  NUM_PROCS_PER_NODE=1 \
  bash "$ROOT_DIR/run_full_train_gpu.sh"
fi

if [[ "$ENABLE_RL" == "1" ]]; then
  echo "=== Stage 2: RL ==="
  export HF_HOME="${HF_HOME:-$ROOT_DIR/.cache/huggingface}"
  export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
  export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
  export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
  export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
  export TMPDIR="${TMPDIR:-$ROOT_DIR/.tmp}"
  export PYTHONUNBUFFERED=1
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

  PYTHONPATH="$ROOT_DIR/src:${PYTHONPATH:-}" \
  "$PYTHON_BIN" -m ts_grounder.rl_train_grpo \
    --model_name_or_path "$RL_BASE_MODEL_PATH" \
    --reference_model_path "$RL_BASE_MODEL_PATH" \
    --train_file "$OUTPUT_DIR/dataset_cache/train.jsonl" \
    --image_root "$ROOT_DIR" \
    --output_dir "$RL_OUTPUT_DIR" \
    --num_generations "$RL_NUM_GENERATIONS" \
    --max_new_tokens "$RL_MAX_NEW_TOKENS" \
    --temperature "$RL_TEMPERATURE" \
    --top_p "$RL_TOP_P" \
    --learning_rate "$RL_LEARNING_RATE" \
    --num_train_epochs "$RL_NUM_TRAIN_EPOCHS" \
    --kl_coef "$RL_KL_COEF" \
    --reward_event_f1_weight "$RL_REWARD_EVENT_F1_WEIGHT" \
    --reward_boundary_iou_weight "$RL_REWARD_BOUNDARY_IOU_WEIGHT" \
    --reward_hallucination_penalty_weight "$RL_REWARD_HALLUCINATION_PENALTY_WEIGHT" \
    --save_steps "$RL_SAVE_STEPS" \
    --optimizer "$RL_OPTIMIZER"

  if [[ "$CLEANUP_SFT_AFTER_RL" == "1" && "$SKIP_SFT_BEFORE_RL" != "1" ]]; then
    echo "=== Cleanup: removing SFT model/checkpoints after successful RL save ==="
    rm -rf "$OUTPUT_DIR/model" "$OUTPUT_DIR/hf_checkpoints"
  fi
else
  echo "=== Stage 2: RL skipped (ENABLE_RL=0) ==="
fi

for split in $EVAL_SPLITS; do
  case "$split" in
    train|val|test) ;;
    *)
      echo "Unsupported eval split: $split" >&2
      exit 1
      ;;
  esac

  echo "=== Predict+eval $FINAL_MODEL_LABEL split=$split ==="
  PROJECT_ROOT="$ROOT_DIR" \
  PYTHON_BIN="$PYTHON_BIN" \
  RUN_DIR="$FINAL_RUN_DIR" \
  SPLIT="$split" \
  CONFIG_PATH="$OUTPUT_DIR/resolved_config.yaml" \
  MODEL_PATH="$FINAL_MODEL_PATH" \
  DATASET_JSONL="$OUTPUT_DIR/dataset_cache/$split.jsonl" \
  EVAL_DIR="$FINAL_EVAL_DIR" \
  PREDICTIONS_PATH="$FINAL_EVAL_DIR/${split}_predictions.jsonl" \
  METRICS_PATH="$FINAL_EVAL_DIR/${split}_metrics.json" \
  MAX_NEW_TOKENS="$MAX_NEW_TOKENS" \
  SAVE_PREDICTIONS="$SAVE_PREDICTIONS" \
  bash "$ROOT_DIR/run_predict_eval_gpu.sh"
done

if [[ "$CLEANUP_FINAL_MODEL_AFTER_EVAL" == "1" ]]; then
  echo "=== Cleanup: removing final $FINAL_MODEL_LABEL after eval ==="
  rm -rf "$FINAL_MODEL_PATH" "$FINAL_RUN_DIR/checkpoints" "$FINAL_RUN_DIR/hf_checkpoints"
fi
if [[ "$CLEANUP_RL_ARTIFACTS_AFTER_EVAL" == "1" && "$ENABLE_RL" == "1" ]]; then
  echo "=== Cleanup: removing RL rollout and prepared subset cache after eval ==="
  rm -rf "$RL_OUTPUT_DIR/rl_rollouts.jsonl" "$OUTPUT_DIR/dataset_cache"
fi

echo "=== Workflow finished ==="
echo "Subset workspace: $OUTPUT_DIR"
if [[ "$ENABLE_RL" == "1" ]]; then
  echo "RL output dir: $RL_OUTPUT_DIR"
fi
