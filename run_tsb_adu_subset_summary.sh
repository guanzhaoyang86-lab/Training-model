#!/bin/bash
set -euo pipefail

ROOT_DIR="${PROJECT_ROOT:-}"
if [[ -z "$ROOT_DIR" ]]; then
  ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

PYTHON_BIN="${PYTHON_BIN:-/gpfs/projects/p33222/ybq9740/envs/anomamind/bin/python}"
SUBMISSION_MANIFEST="${SUBMISSION_MANIFEST:-}"
SUMMARY_OUTPUT_DIR="${SUMMARY_OUTPUT_DIR:-}"

if [[ -z "$SUBMISSION_MANIFEST" ]]; then
  echo "SUBMISSION_MANIFEST is required." >&2
  exit 1
fi

if [[ -z "$SUMMARY_OUTPUT_DIR" ]]; then
  echo "SUMMARY_OUTPUT_DIR is required." >&2
  exit 1
fi

mkdir -p "$SUMMARY_OUTPUT_DIR"
cd "$ROOT_DIR"

echo "Using Python: $PYTHON_BIN"
echo "Submission manifest: $SUBMISSION_MANIFEST"
echo "Summary output dir: $SUMMARY_OUTPUT_DIR"

"$PYTHON_BIN" "$ROOT_DIR/tools/summarize_tsb_adu_subset_results.py" \
  --submission-manifest "$SUBMISSION_MANIFEST" \
  --output-dir "$SUMMARY_OUTPUT_DIR"
