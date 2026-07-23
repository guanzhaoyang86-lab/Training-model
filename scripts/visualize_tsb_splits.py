#!/usr/bin/env python3
"""Create quick visual summaries for TSB-AD-U split JSON files."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_ROOT = (
    Path(__file__).resolve().parents[1]
    / "dataset"
    / "tsb_adu_subset_splits_raw_file_padded_256_128_7_1_2"
)
SPLITS = ("train", "val", "test")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def short_sample_id(sample_id: str, max_len: int = 44) -> str:
    if len(sample_id) <= max_len:
        return sample_id
    return sample_id[: max_len - 1] + "..."


def anomaly_spans(labels):
    spans = []  # type: List[Tuple[int, int]]
    start = None
    for idx, label in enumerate(labels + [0]):
        if label and start is None:
            start = idx
        elif not label and start is not None:
            spans.append((start, idx - 1))
            start = None
    return spans


def choose_representative(dataset_dir, split):
    data_path = dataset_dir / f"{split}.json"
    if not data_path.exists():
        raise FileNotFoundError(f"Missing split file: {data_path}")

    samples = load_json(data_path)
    if not samples:
        raise ValueError(f"No samples found in {data_path}")
    for sample in samples:
        if any(sample.get("point_labels", [])):
            return sample
    return samples[0]


def draw_window(ax, dataset_name, split, sample):
    series = sample.get("series", [])
    labels = sample.get("point_labels", [0] * len(series))
    x = list(range(len(series)))
    spans = anomaly_spans([int(v) for v in labels])

    for start, end in spans:
        ax.axvspan(start - 0.5, end + 0.5, color="#ef4444", alpha=0.12, linewidth=0)

    ax.plot(x, series, color="#1f5fbf", linewidth=1.15)
    for start, end in spans:
        span_x = x[start : end + 1]
        span_series = series[start : end + 1]
        if start == end:
            ax.scatter(span_x, span_series, color="#dc2626", s=18, zorder=3)
        else:
            ax.plot(span_x, span_series, color="#dc2626", linewidth=1.8, zorder=3)
    ax.axhline(0, color="#64748b", linewidth=0.5, alpha=0.35)
    ax.set_title(dataset_name, loc="left", fontsize=10, fontweight="bold", pad=4)
    ax.text(
        0.99,
        0.93,
        f"{split} | {'pos' if any(labels) else 'neg'}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7,
        color="#334155",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1.5},
    )
    ax.text(
        0.01,
        0.03,
        short_sample_id(str(sample.get("sample_id", ""))),
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.5,
        color="#475569",
    )
    ax.set_xlim(0, max(len(series) - 1, 1))
    ax.grid(True, axis="x", color="#cbd5e1", linewidth=0.45, alpha=0.65)
    ax.grid(True, axis="y", color="#e2e8f0", linewidth=0.45, alpha=0.55)
    ax.tick_params(labelsize=7, length=2, colors="#475569")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#94a3b8")
    ax.spines["bottom"].set_color("#94a3b8")


def ordered_datasets(root, selected):
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        manifest = load_json(manifest_path)
        names = list(manifest.get("datasets", {}).keys())
    else:
        names = sorted(p.name for p in root.iterdir() if p.is_dir())
    if selected:
        wanted = set(selected)
        names = [name for name in names if name in wanted]
        missing = sorted(wanted - set(names))
        if missing:
            raise ValueError(f"Dataset(s) not found: {', '.join(missing)}")
    return names


def collect_counts(root, names):
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        manifest = load_json(manifest_path)
        datasets = manifest.get("datasets", {})
        return {name: datasets[name] for name in names if name in datasets}

    counts = {}  # type: Dict[str, Dict[str, Dict[str, int]]]
    for name in names:
        counts[name] = {}
        for split in SPLITS:
            samples = load_json(root / name / f"{split}.json")
            positives = sum(1 for sample in samples if any(sample.get("point_labels", [])))
            counts[name][split] = {
                "num_windows": len(samples),
                "num_positive_windows": positives,
                "num_negative_windows": len(samples) - positives,
            }
    return counts


def plot_counts(root, names, out_dir):
    counts = collect_counts(root, names)
    y = list(range(len(names)))
    split_colors = {"train": "#2563eb", "val": "#f59e0b", "test": "#10b981"}

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(16, max(6, 0.43 * len(names) + 1.5)),
        gridspec_kw={"width_ratios": [2.2, 1]},
    )

    ax = axes[0]
    left = [0] * len(names)
    for split in SPLITS:
        values = [counts[name][split]["num_windows"] for name in names]
        ax.barh(y, values, left=left, color=split_colors[split], label=split, height=0.72)
        left = [a + b for a, b in zip(left, values)]
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("number of windows")
    ax.set_title("Split sizes", loc="left", fontweight="bold")
    ax.legend(loc="lower right", frameon=False, ncols=3, fontsize=8)
    ax.grid(True, axis="x", color="#e2e8f0", linewidth=0.6)
    ax.set_axisbelow(True)

    ax = axes[1]
    ratios = []
    labels = []
    for name in names:
        total = sum(counts[name][split]["num_windows"] for split in SPLITS)
        pos = sum(counts[name][split]["num_positive_windows"] for split in SPLITS)
        ratios.append(pos / total if total else 0.0)
        labels.append(f"{pos}/{total}")
    ax.barh(y, ratios, color="#ef4444", height=0.72)
    for yi, ratio, label in zip(y, ratios, labels):
        ax.text(min(ratio + 0.015, 0.98), yi, label, va="center", fontsize=7, color="#334155")
    ax.set_yticks(y)
    ax.set_yticklabels([])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("positive window ratio")
    ax.set_title("Anomaly coverage", loc="left", fontweight="bold")
    ax.grid(True, axis="x", color="#e2e8f0", linewidth=0.6)
    ax.set_axisbelow(True)

    fig.suptitle(root.name, fontsize=13, fontweight="bold")
    fig.tight_layout()
    out_path = out_dir / "split_counts_and_positive_ratio.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def plot_representatives(root, names, out_dir, prefix):
    output_paths = []  # type: List[Path]
    representative_dir = out_dir / prefix
    representative_dir.mkdir(parents=True, exist_ok=True)

    for name in names:
        dataset_dir = representative_dir / name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        for split in ("train", "test"):
            sample = choose_representative(root / name, split)
            fig, ax = plt.subplots(1, 1, figsize=(12, 4))
            draw_window(ax, name, split, sample)
            ax.set_xlabel("window index")
            ax.set_ylabel("value")
            fig.suptitle(
                f"Representative window: {name} ({split})",
                fontsize=13,
                fontweight="bold",
            )
            fig.tight_layout()
            out_path = dataset_dir / f"{split}.png"
            fig.savefig(out_path, dpi=170)
            plt.close(fig)
            output_paths.append(out_path)

    return output_paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--datasets", nargs="*", default=None)
    args = parser.parse_args()

    root = args.root.resolve()
    out_dir = (args.out_dir or root / "visualizations_overview").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    names = ordered_datasets(root, args.datasets)
    outputs = [plot_counts(root, names, out_dir)]
    outputs.extend(plot_representatives(root, names, out_dir, prefix="representative_windows"))

    print(f"datasets={len(names)}")
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
