#!/bin/bash
#SBATCH --account=p33222
#SBATCH --job-name=tsg-clean-metrics-only
#SBATCH --partition=short
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=1
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:?PROJECT_ROOT is required}"
CLEANUP_ROOT="${CLEANUP_ROOT:?CLEANUP_ROOT is required}"

project_root="$(readlink -f "$PROJECT_ROOT")"
cleanup_root="$(readlink -f "$CLEANUP_ROOT")"
allowed_root="$project_root/outputs"

case "$cleanup_root/" in
  "$allowed_root/"*) ;;
  *)
    echo "Refusing cleanup outside $allowed_root: $cleanup_root" >&2
    exit 1
    ;;
esac

if [[ ! -d "$cleanup_root" ]]; then
  echo "Cleanup root does not exist: $cleanup_root" >&2
  exit 1
fi

echo "Removing model weights, checkpoints, prepared caches, rollouts, and predictions."
echo "Cleanup root: $cleanup_root"

while IFS= read -r -d '' path; do
  echo "Removing directory: $path"
  rm -rf -- "$path"
done < <(
  find "$cleanup_root" -type d \
    \( -name model -o -name hf_checkpoints -o -name checkpoints -o -name dataset_cache \) \
    -print0
)

while IFS= read -r -d '' path; do
  echo "Removing file: $path"
  rm -f -- "$path"
done < <(
  find "$cleanup_root" -type f \
    \( -name '*.safetensors' \
       -o -name 'pytorch_model*.bin' \
       -o -name 'optimizer.pt' \
       -o -name 'scheduler.pt' \
       -o -name 'rng_state*.pth' \
       -o -name 'rl_rollouts.jsonl' \
       -o -name '*_predictions.jsonl' \) \
    -print0
)

remaining_weights="$(
  find "$cleanup_root" -type f \
    \( -name '*.safetensors' -o -name 'pytorch_model*.bin' -o -name 'optimizer.pt' \) \
    -print -quit
)"
if [[ -n "$remaining_weights" ]]; then
  echo "Weight cleanup verification failed; found: $remaining_weights" >&2
  exit 1
fi

echo "Cleanup finished. Metrics, summaries, resolved configs, manifests, and logs were retained."
