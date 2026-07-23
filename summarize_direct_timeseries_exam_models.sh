#!/bin/bash
set -euo pipefail

DEFAULT_ROOT_DIR="/gpfs/projects/p33222/ybq9740/Thesis/Training-model-yilong"
ROOT_DIR="${PROJECT_ROOT:-$DEFAULT_ROOT_DIR}"
PYTHON_BIN="${PYTHON_BIN:-/gpfs/projects/p33222/ybq9740/envs/anomamind/bin/python}"
MANIFEST_PATH="${MANIFEST_PATH:?Set MANIFEST_PATH to the direct QA submission manifest.}"
SUMMARY_OUTPUT_DIR="${SUMMARY_OUTPUT_DIR:-$(dirname "$MANIFEST_PATH")}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "PYTHON_BIN is not executable: $PYTHON_BIN" >&2
  exit 1
fi
if [[ ! -f "$MANIFEST_PATH" ]]; then
  echo "Manifest not found: $MANIFEST_PATH" >&2
  exit 1
fi

mkdir -p "$SUMMARY_OUTPUT_DIR"

"$PYTHON_BIN" - "$MANIFEST_PATH" "$SUMMARY_OUTPUT_DIR" <<'PY'
import csv
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
summary_dir = Path(sys.argv[2])

with manifest_path.open("r", encoding="utf-8", newline="") as f:
    rows = list(csv.DictReader(f, delimiter="\t"))

summary_rows = []
for row in rows:
    run_dir = Path(row["run_dir"])
    metrics_path = run_dir / "timeseries_exam_round3_predictions.metrics.json"
    predictions_path = run_dir / "timeseries_exam_round3_predictions.json"
    official_path = run_dir / "timeseries_exam_round3_predictions.timeseriesexam.json"
    status_path = run_dir / "status.tsv"
    status = {}
    if status_path.exists():
        with status_path.open("r", encoding="utf-8") as f:
            for line in f:
                key, _, value = line.rstrip("\n").partition("\t")
                status[key] = value

    metrics = {}
    if metrics_path.exists():
        with metrics_path.open("r", encoding="utf-8") as f:
            metrics = json.load(f)

    if metrics and predictions_path.exists():
        with predictions_path.open("r", encoding="utf-8") as f:
            predictions_payload = json.load(f)
        data_file = Path(metrics["data_file"])
        with data_file.open("r", encoding="utf-8") as f:
            source_samples = json.load(f)

        official_results = []
        for result in predictions_payload.get("results", []):
            source_idx = int(result["index"])
            sample = dict(source_samples[source_idx])
            sample["response"] = result.get("response")
            sample["correct"] = bool(result.get("correct"))
            sample.pop("ts", None)
            sample.pop("ts1", None)
            sample.pop("ts2", None)
            sample.pop("examples", None)
            official_results.append(sample)
        official_results.append({"accuracy": metrics.get("accuracy")})
        with official_path.open("w", encoding="utf-8") as f:
            json.dump(official_results, f, ensure_ascii=False, indent=4)

    total = metrics.get("total")
    correct = metrics.get("correct")
    accuracy = metrics.get("accuracy")
    summary_rows.append(
        {
            "order": row.get("order", ""),
            "model_name": row.get("model_name", ""),
            "model_path": row.get("model_path", ""),
            "job_id": row.get("job_id", ""),
            "dependency": row.get("dependency", ""),
            "state": status.get("state", "missing_metrics" if not metrics else "completed"),
            "exit_code": status.get("exit_code", ""),
            "accuracy": "" if accuracy is None else f"{float(accuracy):.6f}",
            "correct": "" if correct is None else str(correct),
            "total": "" if total is None else str(total),
            "run_dir": str(run_dir),
            "metrics_path": str(metrics_path) if metrics_path.exists() else "",
            "predictions_path": str(predictions_path) if predictions_path.exists() else "",
            "timeseriesexam_format_path": str(official_path) if official_path.exists() else "",
        }
    )

summary_rows.sort(key=lambda item: int(item["order"]) if str(item["order"]).isdigit() else 10**9)

csv_path = summary_dir / "direct_timeseries_exam_summary.csv"
json_path = summary_dir / "direct_timeseries_exam_summary.json"
md_path = summary_dir / "direct_timeseries_exam_summary.md"

fieldnames = [
    "order",
    "model_name",
    "job_id",
    "dependency",
    "state",
    "exit_code",
    "accuracy",
    "correct",
    "total",
    "model_path",
    "run_dir",
    "metrics_path",
    "predictions_path",
    "timeseriesexam_format_path",
]
with csv_path.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(summary_rows)

with json_path.open("w", encoding="utf-8") as f:
    json.dump(summary_rows, f, ensure_ascii=False, indent=2)

with md_path.open("w", encoding="utf-8") as f:
    f.write("# Direct TimeSeriesExam QA Summary\n\n")
    f.write(f"Manifest: `{manifest_path}`\n\n")
    f.write("| # | model | state | accuracy | correct/total | job |\n")
    f.write("|---:|---|---|---:|---:|---:|\n")
    for item in summary_rows:
        correct_total = ""
        if item["correct"] and item["total"]:
            correct_total = f"{item['correct']}/{item['total']}"
        f.write(
            f"| {item['order']} | {item['model_name']} | {item['state']} | "
            f"{item['accuracy']} | {correct_total} | {item['job_id']} |\n"
        )

print(f"Summary CSV: {csv_path}")
print(f"Summary JSON: {json_path}")
print(f"Summary Markdown: {md_path}")
PY
