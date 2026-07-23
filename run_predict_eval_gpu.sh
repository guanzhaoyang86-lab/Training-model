#!/bin/bash
set -euo pipefail

ROOT_DIR="${PROJECT_ROOT:-}"
if [[ -z "$ROOT_DIR" ]]; then
  ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

RUN_DIR="${RUN_DIR:-$ROOT_DIR/outputs/qwen25vl_7b_single_a100_20260411_155921}"
SPLIT="${SPLIT:-test}"
PYTHON_BIN="${PYTHON_BIN:-}"
CONFIG_PATH="${CONFIG_PATH:-$RUN_DIR/resolved_config.yaml}"
MODEL_PATH="${MODEL_PATH:-$RUN_DIR/model}"
DATASET_JSONL="${DATASET_JSONL:-$RUN_DIR/dataset_cache/$SPLIT.jsonl}"
EVAL_DIR="${EVAL_DIR:-$RUN_DIR/eval}"
PREDICTIONS_PATH="${PREDICTIONS_PATH:-$EVAL_DIR/${SPLIT}_predictions.jsonl}"
METRICS_PATH="${METRICS_PATH:-$EVAL_DIR/${SPLIT}_metrics.json}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
SAVE_PREDICTIONS="${SAVE_PREDICTIONS:-1}"

pick_python() {
  local candidate
  local candidates=(
    "$ROOT_DIR/.venv/bin/python"
    "$ROOT_DIR/.venv_smoke/bin/python"
    "/gpfs/projects/p33222/ybq9740/envs/anomamind/bin/python"
    "/gpfs/projects/p33222/ybq9740/envs/anomllm/bin/python"
  )

  for candidate in "${candidates[@]}"; do
    if [[ -x "$candidate" ]]; then
      echo "$candidate"
      return 0
    fi
  done

  if command -v python3 >/dev/null 2>&1; then
    command -v python3
    return 0
  fi
  return 1
}

if [[ -z "$PYTHON_BIN" ]]; then
  if ! PYTHON_BIN="$(pick_python)"; then
    echo "未找到可用的 Python。请显式设置 PYTHON_BIN，或准备 Python 3.10+ 环境。" >&2
    exit 1
  fi
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "PYTHON_BIN 不可执行: $PYTHON_BIN" >&2
  exit 1
fi

if [[ ! -d "$RUN_DIR" ]]; then
  echo "RUN_DIR 不存在: $RUN_DIR" >&2
  exit 1
fi

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "配置文件不存在: $CONFIG_PATH" >&2
  exit 1
fi

if [[ ! -d "$MODEL_PATH" ]]; then
  echo "模型目录不存在: $MODEL_PATH" >&2
  exit 1
fi

if [[ ! -f "$DATASET_JSONL" ]]; then
  echo "数据集 JSONL 不存在: $DATASET_JSONL" >&2
  exit 1
fi

mkdir -p "$ROOT_DIR/outputs" "$ROOT_DIR/.cache" "$ROOT_DIR/.tmp" "$EVAL_DIR"
cd "$ROOT_DIR"

export HF_HOME="${HF_HOME:-$ROOT_DIR/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TMPDIR="${TMPDIR:-$ROOT_DIR/.tmp}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TS_GROUNDER_PREDICT_TORCH_DTYPE="${TS_GROUNDER_PREDICT_TORCH_DTYPE:-bfloat16}"

"$PYTHON_BIN" - <<'PY'
import importlib.util
import sys

required = ["yaml", "torch", "datasets", "transformers", "accelerate"]
missing = [name for name in required if importlib.util.find_spec(name) is None]

if sys.version_info < (3, 10):
    raise SystemExit(
        f"当前 Python 是 {sys.version.split()[0]}，至少需要 Python 3.10。"
        "可以把 PYTHON_BIN 设为 /gpfs/projects/p33222/ybq9740/envs/anomamind/bin/python。"
    )

if missing:
    joined = ", ".join(missing)
    raise SystemExit(
        f"当前环境缺少依赖: {joined}。"
        "可以先安装 requirements-lock.txt，或把 PYTHON_BIN 设为 /gpfs/projects/p33222/ybq9740/envs/anomamind/bin/python。"
    )
PY

echo "Using Python: $PYTHON_BIN"
echo "Run dir: $RUN_DIR"
echo "Config: $CONFIG_PATH"
echo "Model path: $MODEL_PATH"
echo "Dataset jsonl: $DATASET_JSONL"
echo "Split: $SPLIT"
echo "Predictions: $PREDICTIONS_PATH"
echo "Metrics: $METRICS_PATH"
echo "Max new tokens: $MAX_NEW_TOKENS"
echo "Predict torch dtype: $TS_GROUNDER_PREDICT_TORCH_DTYPE"
echo "Save predictions: $SAVE_PREDICTIONS"

"$PYTHON_BIN" "$ROOT_DIR/predict.py" \
  --config "$CONFIG_PATH" \
  --dataset-jsonl "$DATASET_JSONL" \
  --model-path "$MODEL_PATH" \
  --output "$PREDICTIONS_PATH" \
  --max-new-tokens "$MAX_NEW_TOKENS"

"$PYTHON_BIN" "$ROOT_DIR/scripts/eval_vlm_grounder.py" \
  --dataset-jsonl "$DATASET_JSONL" \
  --predictions "$PREDICTIONS_PATH" \
  --output "$METRICS_PATH"

if [[ "$SAVE_PREDICTIONS" == "0" ]]; then
  rm -f "$PREDICTIONS_PATH"
fi

echo "Prediction and evaluation finished."
if [[ "$SAVE_PREDICTIONS" == "0" ]]; then
  echo "Predictions removed after metrics were written."
else
  echo "Predictions written to: $PREDICTIONS_PATH"
fi
echo "Metrics written to: $METRICS_PATH"
