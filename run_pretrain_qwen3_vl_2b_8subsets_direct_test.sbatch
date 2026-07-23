#!/bin/bash
#SBATCH --account=p33222
#SBATCH --job-name=tsg-pre-q3vl2b-8sub
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=sxm
#SBATCH --mem=80G
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --chdir=/gpfs/projects/p33222/ybq9740/Thesis/Training-model-yilong
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

ROOT_DIR="${PROJECT_ROOT:-/gpfs/projects/p33222/ybq9740/Thesis/Training-model-yilong}"

if [[ -n "${SUBSETS_OVERRIDE:-}" ]]; then
  read -r -a SUBSETS <<< "$SUBSETS_OVERRIDE"
else
  SUBSETS=(
    Daphnet
    MSL
    NEK
    Power
    SED
    TAO
    TODS
    YAHOO
  )
fi

PYTHON_BIN="${PYTHON_BIN:-/gpfs/projects/p33222/ybq9740/envs/anomamind/bin/python}"
SOURCE_ROOT_BASE="${SOURCE_ROOT_BASE:-$ROOT_DIR/dataset/tsb_adu_subset_splits_raw_file_padded_256_128_7_1_2}"
MODEL_PATH="${MODEL_PATH:-/gpfs/projects/p33222/ybq9740/models/Qwen3-VL-2B-Instruct}"
CONFIG_PATH="${CONFIG_PATH:-$ROOT_DIR/configs/vlm_qwen3_vl_8b_anomaly_db_indexed_text_plain_image.yaml}"
OUTPUT_BASE="${OUTPUT_BASE:-$ROOT_DIR/outputs/pretrain_qwen3_vl_2b_8subsets_direct_test}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
SPLIT="${SPLIT:-test}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
RUN_SUMMARY="${RUN_SUMMARY:-1}"
DRY_RUN="${DRY_RUN:-0}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "PYTHON_BIN is not executable: $PYTHON_BIN" >&2
  exit 1
fi
if [[ ! -d "$SOURCE_ROOT_BASE" ]]; then
  echo "Dataset root not found: $SOURCE_ROOT_BASE" >&2
  exit 1
fi
if [[ ! -d "$MODEL_PATH" ]]; then
  echo "Model path not found: $MODEL_PATH" >&2
  exit 1
fi
if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Config file not found: $CONFIG_PATH" >&2
  exit 1
fi

batch_root="$OUTPUT_BASE/batch_$RUN_STAMP"
summary_root="$batch_root/_summary"
submission_manifest="$summary_root/submission_manifest.tsv"
mkdir -p "$batch_root" "$summary_root"
printf 'subset\ttrain_job_id\teval_job_id\trun_dir\n' > "$submission_manifest"

echo "Running pretrained Qwen3-VL-2B direct test on eight TSB subsets"
echo "  subsets: ${SUBSETS[*]}"
echo "  model: $MODEL_PATH"
echo "  config: $CONFIG_PATH"
echo "  batch_root: $batch_root"
echo "  max_new_tokens: $MAX_NEW_TOKENS"
echo "  slurm_job_id: ${SLURM_JOB_ID:-local}"

export PROJECT_ROOT="$ROOT_DIR"
export PYTHON_BIN
export MODEL_PATH
export BF16=1
export FP16=0
export SPLIT
export MAX_NEW_TOKENS
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TS_GROUNDER_PREDICT_TORCH_DTYPE="${TS_GROUNDER_PREDICT_TORCH_DTYPE:-bfloat16}"

for idx in "${!SUBSETS[@]}"; do
  subset="${SUBSETS[$idx]}"
  source_root="$SOURCE_ROOT_BASE/$subset"
  output_dir="$batch_root/$subset"

  if [[ ! -f "$source_root/test.json" ]]; then
    echo "Subset test split not found: $source_root/test.json" >&2
    exit 1
  fi

  mkdir -p "$output_dir"
  echo "==== [$((idx + 1))/${#SUBSETS[@]}] subset=$subset ===="
  echo "  source_root: $source_root"
  echo "  output_dir: $output_dir"

  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[DRY_RUN] would prepare test dataset and run direct pretrained eval"
  else
    PROJECT_ROOT="$ROOT_DIR" \
    CONFIG_PATH="$CONFIG_PATH" \
    OUTPUT_DIR="$output_dir" \
    MODE=prepare_only \
    PYTHON_BIN="$PYTHON_BIN" \
    MODEL_PATH="$MODEL_PATH" \
    SOURCE_ROOT="$source_root" \
    BF16="$BF16" \
    FP16="$FP16" \
    LAUNCHER=python \
    NUM_PROCS_PER_NODE=1 \
    bash "$ROOT_DIR/run_full_train_gpu.sh"

    PROJECT_ROOT="$ROOT_DIR" \
    PYTHON_BIN="$PYTHON_BIN" \
    RUN_DIR="$output_dir" \
    SPLIT="$SPLIT" \
    CONFIG_PATH="$output_dir/resolved_config.yaml" \
    MODEL_PATH="$MODEL_PATH" \
    DATASET_JSONL="$output_dir/dataset_cache/$SPLIT.jsonl" \
    EVAL_DIR="$output_dir/eval" \
    PREDICTIONS_PATH="$output_dir/eval/${SPLIT}_predictions.jsonl" \
    METRICS_PATH="$output_dir/eval/${SPLIT}_metrics.json" \
    MAX_NEW_TOKENS="$MAX_NEW_TOKENS" \
    bash "$ROOT_DIR/run_predict_eval_gpu.sh"
  fi

  printf '%s\t%s\t%s\t%s\n' \
    "$subset" "${SLURM_JOB_ID:-local}" "${SLURM_JOB_ID:-local}" "$output_dir" \
    >> "$submission_manifest"
done

if [[ "$RUN_SUMMARY" == "1" ]]; then
  echo "==== Summary ===="
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[DRY_RUN] would summarize: $submission_manifest -> $summary_root"
  else
    SUBMISSION_MANIFEST="$submission_manifest" \
    SUMMARY_OUTPUT_DIR="$summary_root" \
    bash "$ROOT_DIR/run_tsb_adu_subset_summary.sh"
  fi
fi

echo "Pretrained direct-test batch finished."
echo "Batch root: $batch_root"
echo "Submission manifest: $submission_manifest"
