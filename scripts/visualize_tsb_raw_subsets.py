#!/usr/bin/env python3
"""Render one full-series overview image for each raw TSB-AD-U CSV file."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_SOURCE_ROOT = Path("/gpfs/projects/p33222/ybq9740/TSB-AD-U/TSB-AD-U")
DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parents[1]
    / "dataset"
    / "tsb_adu_raw_file_visualizations"
)
DEFAULT_SUBSETS = ("Daphnet", "MSL", "NEK", "Power", "SED", "TAO", "TODS", "YAHOO")


def _parse_subset_name(csv_path: Path) -> str:
    parts = csv_path.name.split("_")
    if len(parts) < 2:
        return "unknown"
    return parts[1]


def _load_series_and_labels(
    csv_path: Path,
    *,
    value_column: str,
    label_column: str,
) -> tuple[list[float], list[int]]:
    series: list[float] = []
    labels: list[int] = []
    with csv_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        if value_column not in fieldnames:
            raise ValueError(f"{csv_path} is missing value column {value_column!r}")
        for row in reader:
            series.append(float(row[value_column]))
            raw_label = row.get(label_column)
            labels.append(int(float(raw_label)) if raw_label not in (None, "") else 0)
    if not series:
        raise ValueError(f"{csv_path} has no sequence data.")
    if len(series) != len(labels):
        raise ValueError(f"{csv_path} has mismatched series and label lengths.")
    return series, labels


def _anomaly_spans(labels: list[int]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start: int | None = None
    for index, label in enumerate(labels + [0]):
        if label and start is None:
            start = index
        elif not label and start is not None:
            spans.append((start, index - 1))
            start = None
    return spans


def _draw_full_series(
    *,
    csv_path: Path,
    subset_name: str,
    series: list[float],
    labels: list[int],
    output_path: Path,
    dpi: int,
) -> dict[str, Any]:
    spans = _anomaly_spans(labels)
    x = list(range(len(series)))

    fig, ax = plt.subplots(1, 1, figsize=(14, 4))
    for start, end in spans:
        ax.axvspan(start - 0.5, end + 0.5, color="#ef4444", alpha=0.14, linewidth=0)

    ax.plot(x, series, color="#1f5fbf", linewidth=0.9)
    for start, end in spans:
        ax.plot(x[start : end + 1], series[start : end + 1], color="#dc2626", linewidth=1.25)

    ax.set_title(f"{subset_name}: {csv_path.name}", loc="left", fontsize=10, fontweight="bold")
    ax.text(
        0.99,
        0.95,
        (
            f"points={len(series)} | anomaly_points={sum(labels)} "
            f"| anomaly_segments={len(spans)}"
        ),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7,
        color="#334155",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.5},
    )
    ax.set_xlabel("raw file index")
    ax.set_ylabel("value")
    ax.set_xlim(0, max(len(series) - 1, 1))
    ax.grid(True, axis="x", color="#cbd5e1", linewidth=0.45, alpha=0.65)
    ax.grid(True, axis="y", color="#e2e8f0", linewidth=0.45, alpha=0.55)
    ax.tick_params(labelsize=7, length=2, colors="#475569")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#94a3b8")
    ax.spines["bottom"].set_color("#94a3b8")
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)

    return {
        "subset": subset_name,
        "source_file": csv_path.name,
        "source_path": str(csv_path),
        "image_path": str(output_path),
        "series_length": len(series),
        "anomaly_point_count": int(sum(labels)),
        "anomaly_segment_count": len(spans),
        "anomaly_spans": [{"start": start, "end": end} for start, end in spans],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Render complete raw TSB-AD-U files grouped by subset. "
            "This visualization does not split or window the input series."
        )
    )
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--subsets", nargs="*", default=list(DEFAULT_SUBSETS))
    parser.add_argument("--value-column", type=str, default="Data")
    parser.add_argument("--label-column", type=str, default="Label")
    parser.add_argument("--dpi", type=int, default=170)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    source_root = args.source_root.expanduser().resolve()
    output_dir = args.out_dir.expanduser().resolve()
    selected_subsets = set(args.subsets)

    if not source_root.exists():
        raise FileNotFoundError(f"Source directory does not exist: {source_root}")

    csv_files = [
        path
        for path in sorted(source_root.glob("*.csv"))
        if _parse_subset_name(path) in selected_subsets
    ]
    if not csv_files:
        raise ValueError(f"No matching CSV files found under {source_root}")

    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for index, csv_path in enumerate(csv_files, start=1):
        subset_name = _parse_subset_name(csv_path)
        subset_dir = output_dir / subset_name
        subset_dir.mkdir(parents=True, exist_ok=True)
        output_path = subset_dir / f"{csv_path.stem}.png"

        series, labels = _load_series_and_labels(
            csv_path,
            value_column=args.value_column,
            label_column=args.label_column,
        )
        if args.overwrite or not output_path.exists():
            record = _draw_full_series(
                csv_path=csv_path,
                subset_name=subset_name,
                series=series,
                labels=labels,
                output_path=output_path,
                dpi=int(args.dpi),
            )
        else:
            spans = _anomaly_spans(labels)
            record = {
                "subset": subset_name,
                "source_file": csv_path.name,
                "source_path": str(csv_path),
                "image_path": str(output_path),
                "series_length": len(series),
                "anomaly_point_count": int(sum(labels)),
                "anomaly_segment_count": len(spans),
                "anomaly_spans": [{"start": start, "end": end} for start, end in spans],
            }
        records.append(record)
        counts[subset_name] += 1
        if index % 50 == 0 or index == len(csv_files):
            print(f"processed {index}/{len(csv_files)}")

    manifest = {
        "source_root": str(source_root),
        "output_dir": str(output_dir),
        "visualization": "one complete raw file per image; no train/val/test split; no windowing",
        "num_files": len(records),
        "subsets": {subset: counts[subset] for subset in sorted(counts)},
        "files": records,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {len(records)} images and manifest: {manifest_path}")


if __name__ == "__main__":
    main()
