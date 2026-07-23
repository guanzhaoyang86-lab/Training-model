from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _load_manifest(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with open(path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            rows.append(dict(row))
    return rows


def _load_metrics(metrics_path: Path) -> dict:
    with open(metrics_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("| " + " | ".join(fieldnames) + " |\n")
        handle.write("|" + "|".join(["---"] * len(fieldnames)) + "|\n")
        for row in rows:
            handle.write("| " + " | ".join(str(row.get(key, "")) for key in fieldnames) + " |\n")


def _make_plots(rows: list[dict[str, object]], output_dir: Path) -> list[str]:
    import matplotlib.pyplot as plt

    subsets = [str(row["subset"]) for row in rows]
    x = list(range(len(subsets)))

    fig, axes = plt.subplots(3, 1, figsize=(max(12, len(subsets) * 0.9), 16), constrained_layout=True)

    width = 0.25
    axes[0].bar([i - width for i in x], [float(row["point_f1"]) for row in rows], width=width, label="point_f1")
    axes[0].bar(x, [float(row["mean_iou"]) for row in rows], width=width, label="mean_iou")
    axes[0].bar([i + width for i in x], [float(row["exact_match"]) for row in rows], width=width, label="exact_match")
    axes[0].set_title("TSB-AD-U Subset Localization Metrics")
    axes[0].set_ylabel("score")
    axes[0].set_ylim(0.0, 1.05)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(subsets, rotation=45, ha="right")
    axes[0].legend()

    axes[1].bar([i - width for i in x], [float(row["type_accuracy"]) for row in rows], width=width, label="type_accuracy")
    axes[1].bar(x, [float(row["strength_accuracy"]) for row in rows], width=width, label="strength_accuracy")
    axes[1].bar([i + width for i in x], [float(row["direction_accuracy"]) for row in rows], width=width, label="direction_accuracy")
    axes[1].set_title("TSB-AD-U Subset Classification Metrics")
    axes[1].set_ylabel("score")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(subsets, rotation=45, ha="right")
    axes[1].legend()

    axes[2].bar([i - width / 2 for i in x], [float(row["start_mae"]) for row in rows], width=width, label="start_mae")
    axes[2].bar([i + width / 2 for i in x], [float(row["end_mae"]) for row in rows], width=width, label="end_mae")
    axes[2].set_title("TSB-AD-U Subset Boundary Error")
    axes[2].set_ylabel("MAE")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(subsets, rotation=45, ha="right")
    axes[2].legend()

    plot_path = output_dir / "tsb_adu_subset_metrics_overview.png"
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)
    return [str(plot_path)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize per-subset TSB-AD-U training results.")
    parser.add_argument("--submission-manifest", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()

    manifest_path = Path(args.submission_manifest).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    submitted = _load_manifest(manifest_path)
    rows: list[dict[str, object]] = []

    for item in submitted:
        subset = item["subset"]
        run_dir = Path(item["run_dir"]).resolve()
        metrics_path = run_dir / "eval" / "test_metrics.json"
        status = "completed" if metrics_path.exists() else "missing_metrics"
        row: dict[str, object] = {
            "subset": subset,
            "run_dir": str(run_dir),
            "train_job_id": item.get("train_job_id", ""),
            "eval_job_id": item.get("eval_job_id", ""),
            "status": status,
        }
        if metrics_path.exists():
            payload = _load_metrics(metrics_path)
            metrics = payload.get("metrics", {})
            row["num_examples"] = payload.get("num_examples")
            for key in (
                "parse_success_rate",
                "exact_match",
                "summary_exact_match",
                "type_accuracy",
                "strength_accuracy",
                "direction_accuracy",
                "start_mae",
                "end_mae",
                "point_precision",
                "point_recall",
                "point_f1",
                "mean_iou",
            ):
                row[key] = metrics.get(key)
        rows.append(row)

    rows.sort(key=lambda item: str(item["subset"]).lower())

    json_path = output_dir / "tsb_adu_subset_metrics.json"
    csv_path = output_dir / "tsb_adu_subset_metrics.csv"
    md_path = output_dir / "tsb_adu_subset_metrics.md"
    summary_path = output_dir / "summary_manifest.json"

    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(rows, handle, ensure_ascii=False, indent=2)

    fieldnames = [
        "subset",
        "status",
        "num_examples",
        "parse_success_rate",
        "exact_match",
        "summary_exact_match",
        "type_accuracy",
        "strength_accuracy",
        "direction_accuracy",
        "start_mae",
        "end_mae",
        "point_precision",
        "point_recall",
        "point_f1",
        "mean_iou",
        "train_job_id",
        "eval_job_id",
        "run_dir",
    ]
    _write_csv(csv_path, rows, fieldnames)
    _write_markdown(md_path, rows, fieldnames)

    completed_rows = [row for row in rows if row["status"] == "completed"]
    plot_files: list[str] = []
    if completed_rows:
        plot_files = _make_plots(completed_rows, output_dir)

    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "submission_manifest": str(manifest_path),
                "num_subsets": len(rows),
                "num_completed": len(completed_rows),
                "artifacts": {
                    "json": str(json_path),
                    "csv": str(csv_path),
                    "markdown": str(md_path),
                    "plots": plot_files,
                },
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Wrote subset summary json to: {json_path}")
    print(f"Wrote subset summary csv to: {csv_path}")
    print(f"Wrote subset summary markdown to: {md_path}")
    if plot_files:
        print("Wrote plots:")
        for path in plot_files:
            print(f"  {path}")


if __name__ == "__main__":
    main()
