#!/bin/bash
#SBATCH --account=p33222
#SBATCH --job-name=tsqa-summary
#SBATCH --partition=gengpu
#SBATCH --constraint=quest12&sxm
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --chdir=/gpfs/projects/p33222/ybq9740/Thesis/Training-model-yilong
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

DEFAULT_ROOT_DIR="/gpfs/projects/p33222/ybq9740/Thesis/Training-model-yilong"
ROOT_DIR="${PROJECT_ROOT:-$DEFAULT_ROOT_DIR}"

if [[ ! -f "$ROOT_DIR/summarize_direct_timeseries_exam_models.sh" ]]; then
  echo "PROJECT_ROOT does not look like Training-model-yilong: $ROOT_DIR" >&2
  exit 1
fi

bash "$ROOT_DIR/summarize_direct_timeseries_exam_models.sh"
