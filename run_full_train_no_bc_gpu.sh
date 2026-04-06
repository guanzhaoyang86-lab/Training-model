#!/bin/bash
#SBATCH --account=p33222
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=ts_grounder_no_bc
#SBATCH --output=/gpfs/projects/p33222/ybq9740/Thesis/ts_grounder/logs/%x-%j.out

set -euo pipefail

ROOT_DIR="/gpfs/projects/p33222/ybq9740/Thesis/ts_grounder"
export BC_WEIGHT="${BC_WEIGHT:-0.0}"
export RUN_TAG="${RUN_TAG:-full_no_bc_$(date +%Y%m%d_%H%M%S)}"

bash "$ROOT_DIR/run_full_train_gpu.sh"
