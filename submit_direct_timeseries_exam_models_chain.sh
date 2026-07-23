#!/bin/bash
set -euo pipefail

ROOT_DIR="${PROJECT_ROOT:-/gpfs/projects/p33222/ybq9740/Thesis/Training-model-yilong}"
MODELS_DIR="${MODELS_DIR:-/gpfs/projects/p33222/ybq9740/models}"
MODEL_JOB_SCRIPT="$ROOT_DIR/run_direct_timeseries_exam_model.sbatch"
SUMMARY_JOB_SCRIPT="$ROOT_DIR/run_direct_timeseries_exam_summary.sbatch"

ACCOUNT="${ACCOUNT:-p33222}"
PARTITION="${PARTITION:-gengpu}"
CONSTRAINT="${CONSTRAINT:-quest12&sxm}"
DEPENDENCY_TYPE="${DEPENDENCY_TYPE:-afterany}"
PYTHON_BIN="${PYTHON_BIN:-/gpfs/projects/p33222/ybq9740/envs/anomamind/bin/python}"
BATCH_ID="${BATCH_ID:-direct_timeseries_exam_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT_DIR/outputs/direct_timeseries_exam_models/$BATCH_ID}"
SUMMARY_OUTPUT_DIR="${SUMMARY_OUTPUT_DIR:-$OUTPUT_ROOT/_summary}"
MANIFEST_PATH="${MANIFEST_PATH:-$SUMMARY_OUTPUT_DIR/submission_manifest.tsv}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs}"
DRY_RUN="${DRY_RUN:-0}"
SUBMIT_SUMMARY="${SUBMIT_SUMMARY:-1}"

EXAM_DATA="${EXAM_DATA:-$ROOT_DIR/../TimeSeriesExam-main/output/round_3_folder/qa_dataset.json}"
EXAM_SEED="${EXAM_SEED:-2026}"
EXAM_MAX_NEW_TOKENS="${EXAM_MAX_NEW_TOKENS:-64}"
EXAM_TEMPERATURE="${EXAM_TEMPERATURE:-0.0}"
EXAM_DEVICE_MAP="${EXAM_DEVICE_MAP:-auto}"
EXAM_TORCH_DTYPE="${EXAM_TORCH_DTYPE:-bfloat16}"
EXAM_MAX_PIXELS="${EXAM_MAX_PIXELS:-131072}"
EXAM_LIMIT="${EXAM_LIMIT:-}"
EXAM_INDEXED_SERIES_TEXT="${EXAM_INDEXED_SERIES_TEXT:-1}"
EXAM_ADD_QUESTION_HINT="${EXAM_ADD_QUESTION_HINT:-0}"
EXAM_ADD_CONCEPTS="${EXAM_ADD_CONCEPTS:-0}"

GPUS_SMALL="${GPUS_SMALL:-1}"
MEM_SMALL="${MEM_SMALL:-120G}"
TIME_SMALL="${TIME_SMALL:-08:00:00}"
CPUS_SMALL="${CPUS_SMALL:-8}"

GPUS_32B="${GPUS_32B:-2}"
MEM_32B="${MEM_32B:-300G}"
TIME_32B="${TIME_32B:-16:00:00}"
CPUS_32B="${CPUS_32B:-12}"

GPUS_72B="${GPUS_72B:-2}"
MEM_72B="${MEM_72B:-400G}"
TIME_72B="${TIME_72B:-24:00:00}"
CPUS_72B="${CPUS_72B:-12}"

SUMMARY_GPUS="${SUMMARY_GPUS:-1}"
SUMMARY_MEM="${SUMMARY_MEM:-16G}"
SUMMARY_TIME="${SUMMARY_TIME:-00:30:00}"
SUMMARY_CPUS="${SUMMARY_CPUS:-2}"

if [[ ! -f "$MODEL_JOB_SCRIPT" ]]; then
  echo "Model job script not found: $MODEL_JOB_SCRIPT" >&2
  exit 1
fi
if [[ ! -f "$SUMMARY_JOB_SCRIPT" ]]; then
  echo "Summary job script not found: $SUMMARY_JOB_SCRIPT" >&2
  exit 1
fi
if [[ ! -d "$MODELS_DIR" ]]; then
  echo "MODELS_DIR not found: $MODELS_DIR" >&2
  exit 1
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "PYTHON_BIN is not executable: $PYTHON_BIN" >&2
  exit 1
fi
if [[ ! -f "$EXAM_DATA" ]]; then
  echo "EXAM_DATA not found: $EXAM_DATA" >&2
  exit 1
fi

contains_model() {
  local needle="$1"
  shift
  local item
  for item in "$@"; do
    if [[ "$item" == "$needle" ]]; then
      return 0
    fi
  done
  return 1
}

MODEL_NAMES=()
if [[ -n "${MODEL_NAMES_OVERRIDE:-}" ]]; then
  read -r -a MODEL_NAMES <<< "$MODEL_NAMES_OVERRIDE"
else
  preferred_order=(
    Qwen3-VL-2B-Instruct
    Qwen3-VL-4B-Instruct
    Qwen2.5-VL-7B-Instruct
    Qwen3-VL-8B-Instruct
    Qwen3-VL-32B-Instruct
    Qwen2.5-VL-72B-Instruct
  )
  for model_name in "${preferred_order[@]}"; do
    if [[ -d "$MODELS_DIR/$model_name" ]]; then
      MODEL_NAMES+=("$model_name")
    fi
  done
  while IFS= read -r model_name; do
    lower_name="${model_name,,}"
    if [[ "$lower_name" == "qwen3-8b" ]]; then
      continue
    fi
    if ! contains_model "$model_name" "${MODEL_NAMES[@]}"; then
      MODEL_NAMES+=("$model_name")
    fi
  done < <(find "$MODELS_DIR" -maxdepth 1 -mindepth 1 -type d -printf '%f\n' | sort)
fi

if [[ "${#MODEL_NAMES[@]}" -eq 0 ]]; then
  echo "No models selected under $MODELS_DIR" >&2
  exit 1
fi

mkdir -p "$OUTPUT_ROOT" "$SUMMARY_OUTPUT_DIR" "$LOG_DIR"
printf 'order\tmodel_name\tmodel_path\trun_dir\tjob_id\tdependency\tgres\tmem\ttime\tcpus\n' > "$MANIFEST_PATH"

echo "Submitting direct TimeSeriesExam QA dependency chain"
echo "  models_dir: $MODELS_DIR"
echo "  excluded: Qwen3-8B"
echo "  dependency_type: $DEPENDENCY_TYPE"
echo "  output_root: $OUTPUT_ROOT"
echo "  manifest: $MANIFEST_PATH"
echo "  models: ${MODEL_NAMES[*]}"

previous_job_id=""
for idx in "${!MODEL_NAMES[@]}"; do
  model_name="${MODEL_NAMES[$idx]}"
  model_path="$MODELS_DIR/$model_name"
  if [[ ! -d "$model_path" ]]; then
    echo "Selected model path does not exist: $model_path" >&2
    exit 1
  fi

  safe_model_name="${model_name//\//__}"
  safe_model_name="${safe_model_name// /_}"
  run_dir="$OUTPUT_ROOT/$safe_model_name"
  order="$((idx + 1))"

  gpus="$GPUS_SMALL"
  mem="$MEM_SMALL"
  time_limit="$TIME_SMALL"
  cpus="$CPUS_SMALL"
  if [[ "$model_name" == *"72B"* ]]; then
    gpus="$GPUS_72B"
    mem="$MEM_72B"
    time_limit="$TIME_72B"
    cpus="$CPUS_72B"
  elif [[ "$model_name" == *"32B"* ]]; then
    gpus="$GPUS_32B"
    mem="$MEM_32B"
    time_limit="$TIME_32B"
    cpus="$CPUS_32B"
  fi

  job_name="tsqa-${safe_model_name//_/-}"
  if [[ "${#job_name}" -gt 48 ]]; then
    job_name="${job_name:0:48}"
  fi

  dependency=""
  sbatch_cmd=(
    sbatch
    --parsable
    "--account=$ACCOUNT"
    "--partition=$PARTITION"
    "--job-name=$job_name"
    "--gres=gpu:$gpus"
    "--mem=$mem"
    "--time=$time_limit"
    "--cpus-per-task=$cpus"
    "--output=$LOG_DIR/%x-%j.out"
    "--error=$LOG_DIR/%x-%j.err"
  )
  if [[ -n "$CONSTRAINT" ]]; then
    sbatch_cmd+=("--constraint=$CONSTRAINT")
  fi
  if [[ -n "$previous_job_id" ]]; then
    dependency="$DEPENDENCY_TYPE:$previous_job_id"
    sbatch_cmd+=("--dependency=$dependency")
  fi
  sbatch_cmd+=("$MODEL_JOB_SCRIPT")

  env_kv=(
    "PROJECT_ROOT=$ROOT_DIR"
    "MODELS_DIR=$MODELS_DIR"
    "MODEL_NAME=$model_name"
    "MODEL_PATH=$model_path"
    "BATCH_ID=$BATCH_ID"
    "OUTPUT_ROOT=$OUTPUT_ROOT"
    "RUN_DIR=$run_dir"
    "PYTHON_BIN=$PYTHON_BIN"
    "EXAM_DATA=$EXAM_DATA"
    "EXAM_SEED=$EXAM_SEED"
    "EXAM_MAX_NEW_TOKENS=$EXAM_MAX_NEW_TOKENS"
    "EXAM_TEMPERATURE=$EXAM_TEMPERATURE"
    "EXAM_DEVICE_MAP=$EXAM_DEVICE_MAP"
    "EXAM_TORCH_DTYPE=$EXAM_TORCH_DTYPE"
    "EXAM_MAX_PIXELS=$EXAM_MAX_PIXELS"
    "EXAM_LIMIT=$EXAM_LIMIT"
    "EXAM_INDEXED_SERIES_TEXT=$EXAM_INDEXED_SERIES_TEXT"
    "EXAM_ADD_QUESTION_HINT=$EXAM_ADD_QUESTION_HINT"
    "EXAM_ADD_CONCEPTS=$EXAM_ADD_CONCEPTS"
  )

  echo "==== [$order/${#MODEL_NAMES[@]}] $model_name ===="
  echo "  model_path: $model_path"
  echo "  run_dir: $run_dir"
  echo "  gres: gpu:$gpus"
  echo "  mem: $mem"
  echo "  time: $time_limit"
  echo "  dependency: ${dependency:-none}"

  if [[ "$DRY_RUN" == "1" ]]; then
    job_id="DRYRUN-$order"
    printf '[DRY_RUN]'
    for kv in "${env_kv[@]}"; do
      printf ' %q' "$kv"
    done
    for arg in "${sbatch_cmd[@]}"; do
      printf ' %q' "$arg"
    done
    printf '\n'
  else
    job_id="$(env "${env_kv[@]}" "${sbatch_cmd[@]}")"
    job_id="${job_id%%;*}"
    echo "  submitted job_id: $job_id"
  fi

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$order" "$model_name" "$model_path" "$run_dir" "$job_id" "${dependency:-}" \
    "gpu:$gpus" "$mem" "$time_limit" "$cpus" >> "$MANIFEST_PATH"
  previous_job_id="$job_id"
done

if [[ "$SUBMIT_SUMMARY" == "1" ]]; then
  dependency=""
  summary_cmd=(
    sbatch
    --parsable
    "--account=$ACCOUNT"
    "--partition=$PARTITION"
    "--job-name=tsqa-summary"
    "--gres=gpu:$SUMMARY_GPUS"
    "--mem=$SUMMARY_MEM"
    "--time=$SUMMARY_TIME"
    "--cpus-per-task=$SUMMARY_CPUS"
    "--output=$LOG_DIR/%x-%j.out"
    "--error=$LOG_DIR/%x-%j.err"
  )
  if [[ -n "$CONSTRAINT" ]]; then
    summary_cmd+=("--constraint=$CONSTRAINT")
  fi
  if [[ -n "$previous_job_id" ]]; then
    dependency="$DEPENDENCY_TYPE:$previous_job_id"
    summary_cmd+=("--dependency=$dependency")
  fi
  summary_cmd+=("$SUMMARY_JOB_SCRIPT")

  summary_env=(
    "PROJECT_ROOT=$ROOT_DIR"
    "PYTHON_BIN=$PYTHON_BIN"
    "MANIFEST_PATH=$MANIFEST_PATH"
    "SUMMARY_OUTPUT_DIR=$SUMMARY_OUTPUT_DIR"
  )

  echo "==== Summary job ===="
  echo "  summary_dir: $SUMMARY_OUTPUT_DIR"
  echo "  dependency: ${dependency:-none}"
  if [[ "$DRY_RUN" == "1" ]]; then
    summary_job_id="DRYRUN-summary"
    printf '[DRY_RUN]'
    for kv in "${summary_env[@]}"; do
      printf ' %q' "$kv"
    done
    for arg in "${summary_cmd[@]}"; do
      printf ' %q' "$arg"
    done
    printf '\n'
  else
    summary_job_id="$(env "${summary_env[@]}" "${summary_cmd[@]}")"
    summary_job_id="${summary_job_id%%;*}"
    echo "  submitted summary_job_id: $summary_job_id"
  fi
  echo "Summary job id: $summary_job_id"
fi

echo "Submitted chain manifest: $MANIFEST_PATH"
echo "Output root: $OUTPUT_ROOT"
