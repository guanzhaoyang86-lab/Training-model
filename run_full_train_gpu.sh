#!/bin/bash
#SBATCH --account=p33222
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=ts_grounder_full
#SBATCH --output=/gpfs/projects/p33222/ybq9740/Thesis/ts_grounder/logs/%x-%j.out

set -euo pipefail

ROOT_DIR="/gpfs/projects/p33222/ybq9740/Thesis/ts_grounder"
PYTHON_BIN="${PYTHON_BIN:-/gpfs/projects/p33222/ybq9740/envs/anomllm/bin/python}"
DATASET_DIR="${DATASET_DIR:-$ROOT_DIR/dataset}"
DATASET_NAME="${DATASET_NAME:-}"
RAW_DATASET_ROOT="${RAW_DATASET_ROOT:-}"
RUN_TAG="${RUN_TAG:-full_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT_DIR/outputs/$RUN_TAG}"
CONFIG_PATH="${CONFIG_PATH:-$ROOT_DIR/configs/default.yaml}"
CONFIG_SNAPSHOT_PATH="${CONFIG_SNAPSHOT_PATH:-$OUTPUT_ROOT/train_full.yaml}"
MPLCONFIGDIR="${MPLCONFIGDIR:-$ROOT_DIR/.mplconfig}"

SEED="${SEED:-2026}"
EPOCHS="${EPOCHS:-200}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"
HIDDEN_DIM="${HIDDEN_DIM:-128}"
TEXT_LAYERS="${TEXT_LAYERS:-2}"
DROPOUT="${DROPOUT:-0.1}"
IMAGE_H="${IMAGE_H:-224}"
IMAGE_W="${IMAGE_W:-224}"
LOCAL_WINDOW="${LOCAL_WINDOW:-25}"
LR="${LR:-0.0003}"
MIN_LR="${MIN_LR:-0.00001}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0001}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
POINT_WEIGHT="${POINT_WEIGHT:-1.0}"
SEG_WEIGHT="${SEG_WEIGHT:-1.0}"
TYPE_WEIGHT="${TYPE_WEIGHT:-1.0}"
EVIDENCE_WEIGHT="${EVIDENCE_WEIGHT:-0.5}"
BC_WEIGHT="${BC_WEIGHT:-1.0}"
CONS_WEIGHT="${CONS_WEIGHT:-0.2}"
TV_WEIGHT="${TV_WEIGHT:-0.05}"

mkdir -p "$ROOT_DIR/logs" "$OUTPUT_ROOT" "$MPLCONFIGDIR"

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT_DIR/src"
export MPLCONFIGDIR

resolve_raw_dataset_root() {
  if [[ -n "$RAW_DATASET_ROOT" ]]; then
    echo "$RAW_DATASET_ROOT"
    return 0
  fi

  if [[ -n "$DATASET_NAME" ]]; then
    echo "$DATASET_DIR/$DATASET_NAME"
    return 0
  fi

  if [[ -f "$DATASET_DIR/train.json" ]]; then
    echo "$DATASET_DIR"
    return 0
  fi

  local candidate
  local dataset_roots=()
  while IFS= read -r candidate; do
    if [[ -f "$candidate/train.json" && -f "$candidate/val.json" ]]; then
      dataset_roots+=("$candidate")
    fi
  done < <(find "$DATASET_DIR" -mindepth 1 -maxdepth 1 -type d | sort)

  if [[ ${#dataset_roots[@]} -eq 1 ]]; then
    echo "${dataset_roots[0]}"
    return 0
  fi

  if [[ ${#dataset_roots[@]} -eq 0 ]]; then
    echo "[ERROR] No dataset with train.json/val.json found under $DATASET_DIR" >&2
  else
    echo "[ERROR] Multiple dataset roots found under $DATASET_DIR. Set DATASET_NAME or RAW_DATASET_ROOT explicitly." >&2
    printf ' - %s\n' "${dataset_roots[@]}" >&2
  fi
  return 1
}

has_npz_split() {
  local split_dir="$1"
  [[ -d "$split_dir" ]] && find "$split_dir" -maxdepth 1 -name '*.npz' -print -quit 2>/dev/null | grep -q .
}

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[ERROR] PYTHON_BIN not executable: $PYTHON_BIN" >&2
  exit 1
fi

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "[ERROR] Missing base config: $CONFIG_PATH" >&2
  exit 1
fi

if [[ ! -d "$DATASET_DIR" ]]; then
  echo "[ERROR] Missing dataset directory: $DATASET_DIR" >&2
  exit 1
fi

RAW_DATASET_ROOT="$(resolve_raw_dataset_root)"
CONVERTED_DATA_ROOT="${CONVERTED_DATA_ROOT:-$ROOT_DIR/data_full/$(basename "$RAW_DATASET_ROOT")}"

if [[ ! -f "$RAW_DATASET_ROOT/train.json" ]]; then
  echo "[ERROR] Missing raw PatternFinder split: $RAW_DATASET_ROOT/train.json" >&2
  exit 1
fi

echo "========== TS Grounder Full Training =========="
echo "Start time: $(date)"
echo "Host: $(hostname)"
echo "Python: $PYTHON_BIN"
echo "Dataset dir: $DATASET_DIR"
echo "Raw dataset root: $RAW_DATASET_ROOT"
echo "Converted data root: $CONVERTED_DATA_ROOT"
echo "Output root: $OUTPUT_ROOT"
echo "Base config: $CONFIG_PATH"
echo "Config snapshot: $CONFIG_SNAPSHOT_PATH"
echo "Batch size: $BATCH_SIZE"
echo "Epochs: $EPOCHS"
echo "Workers: $NUM_WORKERS"
echo "Hidden dim: $HIDDEN_DIM"
echo "Loss weights: point=$POINT_WEIGHT seg=$SEG_WEIGHT type=$TYPE_WEIGHT evidence=$EVIDENCE_WEIGHT bc=$BC_WEIGHT cons=$CONS_WEIGHT tv=$TV_WEIGHT"
echo "Dataset name: $(basename "$RAW_DATASET_ROOT")"
nvidia-smi || true

if ! has_npz_split "$CONVERTED_DATA_ROOT/train" || ! has_npz_split "$CONVERTED_DATA_ROOT/val"; then
  echo "[1/3] Converting PatternFinder JSON into ts_grounder .npz format..."
  "$PYTHON_BIN" "$ROOT_DIR/tools/convert_patternfinder_dataset.py" \
    --source-root "$RAW_DATASET_ROOT" \
    --output-root "$CONVERTED_DATA_ROOT"
else
  echo "[1/3] Reusing existing converted dataset at $CONVERTED_DATA_ROOT"
fi

echo "[2/3] Using base config with runtime overrides..."

echo "[3/3] Launching training..."
cd "$ROOT_DIR"
"$PYTHON_BIN" -u train.py \
  --config "$CONFIG_PATH" \
  --save-config "$CONFIG_SNAPSHOT_PATH" \
  --seed "$SEED" \
  --output-dir "$OUTPUT_ROOT" \
  --train-dir "$CONVERTED_DATA_ROOT/train" \
  --val-dir "$CONVERTED_DATA_ROOT/val" \
  --raw-dataset-root "$RAW_DATASET_ROOT" \
  --image-size "$IMAGE_H" "$IMAGE_W" \
  --local-window "$LOCAL_WINDOW" \
  --hidden-dim "$HIDDEN_DIM" \
  --text-layers "$TEXT_LAYERS" \
  --dropout "$DROPOUT" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --lr "$LR" \
  --min-lr "$MIN_LR" \
  --weight-decay "$WEIGHT_DECAY" \
  --grad-clip "$GRAD_CLIP" \
  --point-weight "$POINT_WEIGHT" \
  --seg-weight "$SEG_WEIGHT" \
  --type-weight "$TYPE_WEIGHT" \
  --evidence-weight "$EVIDENCE_WEIGHT" \
  --bc-weight "$BC_WEIGHT" \
  --cons-weight "$CONS_WEIGHT" \
  --tv-weight "$TV_WEIGHT"

echo "Training finished at $(date)"
echo "Base config: $CONFIG_PATH"
echo "Resolved config: $CONFIG_SNAPSHOT_PATH"
echo "Best checkpoint: $OUTPUT_ROOT/best.pt"
echo "Last checkpoint: $OUTPUT_ROOT/last.pt"
echo "History: $OUTPUT_ROOT/history.json"
echo "Summary: $OUTPUT_ROOT/summary.json"
