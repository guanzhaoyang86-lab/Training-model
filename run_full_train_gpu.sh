#!/bin/bash
set -euo pipefail

ROOT_DIR="${PROJECT_ROOT:-}"
if [[ -z "$ROOT_DIR" ]]; then
  ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

CONFIG_PATH="${CONFIG_PATH:-$ROOT_DIR/configs/vlm_7b_anomaly_db_indexed_text_plain_image.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/outputs/vlm_qwen25vl_7b_grounding_indexed_run}"
MODE="${MODE:-train}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
PYTHON_BIN="${PYTHON_BIN:-}"
LAUNCHER="${LAUNCHER:-python}"
NUM_PROCS_PER_NODE="${NUM_PROCS_PER_NODE:-1}"
MASTER_PORT="${MASTER_PORT:-29500}"

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

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "配置文件不存在: $CONFIG_PATH" >&2
  exit 1
fi

mkdir -p "$ROOT_DIR/outputs" "$ROOT_DIR/.cache" "$ROOT_DIR/.tmp"
cd "$ROOT_DIR"

export MODE
export HF_HOME="${HF_HOME:-$ROOT_DIR/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export TMPDIR="${TMPDIR:-$ROOT_DIR/.tmp}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ -n "${SOURCE_ROOT:-}" ]]; then
  export TS_GROUNDER_SOURCE_ROOT="$SOURCE_ROOT"
fi
if [[ -n "${MODEL_PATH:-}" ]]; then
  export TS_GROUNDER_MODEL_PATH="$MODEL_PATH"
fi
if [[ -n "${CHECKPOINT_OUTPUT_DIR:-}" ]]; then
  export TS_GROUNDER_CHECKPOINT_OUTPUT_DIR="$CHECKPOINT_OUTPUT_DIR"
fi
if [[ -n "${SFT_NUM_TRAIN_EPOCHS:-}" ]]; then
  export TS_GROUNDER_NUM_TRAIN_EPOCHS="$SFT_NUM_TRAIN_EPOCHS"
fi
if [[ -n "${BF16:-}" ]]; then
  export TS_GROUNDER_BF16="$BF16"
fi
if [[ -n "${FP16:-}" ]]; then
  export TS_GROUNDER_FP16="$FP16"
fi

"$PYTHON_BIN" - <<'PY'
import importlib.util
import os
import sys

required = ["yaml"]
if os.environ.get("MODE", "train") == "train":
    required.extend(["torch", "datasets", "transformers", "accelerate"])
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

cmd=()
if [[ "$LAUNCHER" == "torchrun" ]]; then
  cmd=(
    "$PYTHON_BIN"
    -m torch.distributed.run
    "--nproc_per_node=$NUM_PROCS_PER_NODE"
    "--master_port=$MASTER_PORT"
    "$ROOT_DIR/train.py"
    "--config" "$CONFIG_PATH"
    "--output-dir" "$OUTPUT_DIR"
    "--mode" "$MODE"
  )
else
  cmd=(
    "$PYTHON_BIN"
    "$ROOT_DIR/train.py"
    "--config" "$CONFIG_PATH"
    "--output-dir" "$OUTPUT_DIR"
    "--mode" "$MODE"
  )
fi

if [[ -n "$RESUME_FROM_CHECKPOINT" ]]; then
  cmd+=("--resume-from-checkpoint" "$RESUME_FROM_CHECKPOINT")
fi

echo "Using Python: $PYTHON_BIN"
echo "Config: $CONFIG_PATH"
echo "Output dir: $OUTPUT_DIR"
echo "Mode: $MODE"
echo "Launcher: $LAUNCHER"
echo "Num processes per node: $NUM_PROCS_PER_NODE"
echo "SFT num train epochs: ${SFT_NUM_TRAIN_EPOCHS:-config default}"

"${cmd[@]}"
