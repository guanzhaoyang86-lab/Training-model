#!/bin/bash
#SBATCH --account=p33222
#SBATCH --job-name=tsg-q3vl-32b-8sub
#SBATCH --partition=gengpu
#SBATCH --constraint=quest12&sxm
#SBATCH --gres=gpu:2
#SBATCH --mem=400G
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=12
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
MODEL_PATH="${MODEL_PATH:-/gpfs/projects/p33222/ybq9740/models/Qwen3-VL-32B-Instruct}"
CONFIG_PATH="${CONFIG_PATH:-$ROOT_DIR/configs/vlm_qwen3_vl_32b_tsb_adu_indexed_text_plain_image_fsdp_2gpu.yaml}"
OUTPUT_BASE="${OUTPUT_BASE:-$ROOT_DIR/outputs/qwen3_vl_32b_tsb_adu_index_plain_image_single_job}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
SPLIT="${SPLIT:-test}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-29540}"
DRY_RUN="${DRY_RUN:-0}"
RUN_SUMMARY="${RUN_SUMMARY:-1}"

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

echo "Running Qwen3-VL-32B eight-subset single Slurm job"
echo "  subsets: ${SUBSETS[*]}"
echo "  model: $MODEL_PATH"
echo "  config: $CONFIG_PATH"
echo "  batch_root: $batch_root"
echo "  max_new_tokens: $MAX_NEW_TOKENS"
echo "  slurm_job_id: ${SLURM_JOB_ID:-local}"

export PROJECT_ROOT="$ROOT_DIR"
export PYTHON_BIN
export MODEL_PATH
export LAUNCHER=torchrun
export NUM_PROCS_PER_NODE=2
export BF16=1
export FP16=0
export SPLIT
export MAX_NEW_TOKENS
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TS_GROUNDER_PREDICT_DEVICE_MAP="${TS_GROUNDER_PREDICT_DEVICE_MAP:-auto}"
export TS_GROUNDER_PREDICT_TORCH_DTYPE="${TS_GROUNDER_PREDICT_TORCH_DTYPE:-bfloat16}"

for idx in "${!SUBSETS[@]}"; do
  subset="${SUBSETS[$idx]}"
  source_root="$SOURCE_ROOT_BASE/$subset"
  output_dir="$batch_root/$subset"

  if [[ ! -f "$source_root/train.json" ]]; then
    echo "Subset dataset not found: $source_root" >&2
    exit 1
  fi

  mkdir -p "$output_dir"
  export CONFIG_PATH
  export SOURCE_ROOT="$source_root"
  export OUTPUT_DIR="$output_dir"
  export MODE=train
  export MASTER_PORT="$((MASTER_PORT_BASE + idx))"

  echo "==== [$((idx + 1))/${#SUBSETS[@]}] subset=$subset ===="
  echo "  source_root: $SOURCE_ROOT"
  echo "  output_dir: $OUTPUT_DIR"
  echo "  master_port: $MASTER_PORT"

  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[DRY_RUN] would run: bash $ROOT_DIR/run_train_and_eval_gpu.sh"
  else
    bash "$ROOT_DIR/run_train_and_eval_gpu.sh"
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

echo "Single-job batch finished."
echo "Batch root: $batch_root"
echo "Submission manifest: $submission_manifest"
