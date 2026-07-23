from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

SPLITS = ("train", "val", "test")


def _read_file_list(file_list_csv: Path) -> list[str]:
    with open(file_list_csv, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [row["file_name"] for row in reader if row.get("file_name")]


def _normalize_dataset_filters(values: list[str] | None) -> set[str] | None:
    if not values:
        return None
    normalized: set[str] = set()
    for value in values:
        for part in str(value).split(","):
            item = part.strip()
            if item:
                normalized.add(item)
    return normalized or None


def _parse_dataset_name(file_name: str) -> str:
    parts = file_name.split("_")
    if len(parts) >= 2:
        return parts[1]
    return "unknown"


def _load_series_and_labels(
    csv_path: Path,
    *,
    value_column: str = "Data",
    label_column: str = "Label",
) -> tuple[list[float], list[int]]:
    series: list[float] = []
    labels: list[int] = []
    with open(csv_path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if value_column not in (reader.fieldnames or []):
            raise ValueError(f"{csv_path} is missing value column {value_column!r}")
        for row in reader:
            series.append(float(row[value_column]))
            if label_column in row and row[label_column] not in (None, ""):
                labels.append(int(float(row[label_column])))
            else:
                labels.append(0)
    if not series:
        raise ValueError(f"{csv_path} does not contain any usable series values.")
    if len(series) != len(labels):
        raise ValueError(f"{csv_path} has mismatched series and label lengths.")
    return series, labels


def _extract_segments(labels: list[int]) -> list[dict[str, int]]:
    segments: list[dict[str, int]] = []
    start: int | None = None
    for idx, value in enumerate(labels):
        if value == 1 and start is None:
            start = idx
        elif value == 0 and start is not None:
            segments.append({"start": start, "end": idx - 1})
            start = None
    if start is not None:
        segments.append({"start": start, "end": len(labels) - 1})
    return segments


def _draw_plain_plot(series: list[float], *, width: int, height: int) -> Image.Image:
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    margin_left = 22
    margin_right = 14
    margin_top = 18
    margin_bottom = 22

    plot_left = margin_left
    plot_top = margin_top
    plot_right = width - margin_right - 1
    plot_bottom = height - margin_bottom - 1
    plot_width = plot_right - plot_left
    plot_height = plot_bottom - plot_top

    values = [float(point) for point in series]
    value_min = min(values)
    value_max = max(values)
    if value_max == value_min:
        value_min -= 1.0
        value_max += 1.0
    value_pad = 0.06 * (value_max - value_min)
    value_min -= value_pad
    value_max += value_pad

    def x_to_px(index: int) -> float:
        if len(values) <= 1:
            return float(plot_left)
        return plot_left + (index / (len(values) - 1)) * plot_width

    def y_to_px(value: float) -> float:
        ratio = (value - value_min) / (value_max - value_min)
        return plot_bottom - ratio * plot_height

    draw.line([(plot_left, plot_top), (plot_left, plot_bottom)], fill=(192, 198, 210), width=1)
    draw.line([(plot_left, plot_bottom), (plot_right, plot_bottom)], fill=(192, 198, 210), width=1)
    points = [(x_to_px(idx), y_to_px(value)) for idx, value in enumerate(values)]
    draw.line(points, fill=(35, 96, 181), width=2)
    return image


def _compute_segment_boundaries(
    series_length: int,
    *,
    train_ratio: float,
    val_ratio: float,
) -> dict[str, tuple[int, int]]:
    train_end = int(series_length * train_ratio)
    val_end = train_end + int(series_length * val_ratio)
    train_end = max(0, min(train_end, series_length))
    val_end = max(train_end, min(val_end, series_length))
    return {
        "train": (0, train_end),
        "val": (train_end, val_end),
        "test": (val_end, series_length),
    }


def _iter_window_starts(segment_length: int, *, window_size: int, stride: int) -> list[int]:
    if segment_length <= 0:
        return []
    max_start = segment_length - window_size
    if max_start < 0:
        return [0]
    starts = list(range(0, max_start + 1, stride))
    if starts[-1] != max_start:
        starts.append(max_start)
    return starts


def _pad_window(
    series_slice: list[float],
    label_slice: list[int],
    *,
    window_size: int,
) -> tuple[list[float], list[int], int]:
    if len(series_slice) != len(label_slice):
        raise ValueError("series_slice and label_slice must have the same length.")
    if len(series_slice) > window_size:
        raise ValueError("series_slice is longer than the configured window size.")
    if len(series_slice) == window_size:
        return series_slice, label_slice, 0
    if not series_slice:
        raise ValueError("Cannot pad an empty segment into a window.")

    pad_count = window_size - len(series_slice)
    padded_series = list(series_slice) + [float(series_slice[-1])] * pad_count
    padded_labels = list(label_slice) + [0] * pad_count
    return padded_series, padded_labels, pad_count


def _intersect_segments(
    segments: list[dict[str, int]],
    window_start: int,
    window_end: int,
) -> list[dict[str, int]]:
    local_segments: list[dict[str, int]] = []
    for segment in segments:
        overlap_start = max(window_start, int(segment["start"]))
        overlap_end = min(window_end, int(segment["end"]))
        if overlap_start <= overlap_end:
            local_segments.append(
                {
                    "start": overlap_start - window_start,
                    "end": overlap_end - window_start,
                    "global_start": overlap_start,
                    "global_end": overlap_end,
                }
            )
    return local_segments


def _strength_from_segment(segment_length: int, window_size: int) -> str:
    ratio = segment_length / max(window_size, 1)
    if ratio >= 0.2:
        return "strong"
    if ratio >= 0.08:
        return "obvious"
    return "mild"


def _build_event(local_segment: dict[str, int], window_size: int) -> dict[str, Any]:
    local_start = int(local_segment["start"])
    local_end = int(local_segment["end"])
    length = local_end - local_start + 1
    return {
        "start": local_start,
        "end": local_end,
        "type": "range",
        "raw_params": {
            "global_start": int(local_segment["global_start"]),
            "global_end": int(local_segment["global_end"]),
            "window_size": int(window_size),
        },
        "verbal_tags": {
            "strength": _strength_from_segment(length, window_size),
            "direction": "becomes irregular",
        },
        "description": (
            f"An anomaly occurs from index {local_start} to {local_end}, "
            "and this segment becomes irregular."
        ),
    }


def _format_interval(start: int, end: int) -> str:
    if start == end:
        return str(start)
    return f"{start}-{end}"


def _build_sample_description(events: list[dict[str, Any]]) -> str:
    if not events:
        return "No anomaly is detected."
    if len(events) == 1:
        return str(events[0]["description"])

    preview = [_format_interval(int(event["start"]), int(event["end"])) for event in events[:4]]
    if len(preview) == 1:
        preview_text = preview[0]
    elif len(preview) == 2:
        preview_text = f"{preview[0]} and {preview[1]}"
    else:
        preview_text = ", ".join(preview[:-1]) + f", and {preview[-1]}"
    return (
        f"Multiple anomalies occur across {len(events)} intervals in this window, "
        f"including {preview_text}, and these segments become irregular."
    )


def _normal_keep(sample_key: str, split: str, window_start: int, keep_every: int) -> bool:
    if keep_every <= 1:
        return True
    token = f"{sample_key}:{split}:{window_start}"
    bucket = int(hashlib.sha1(token.encode("utf-8")).hexdigest()[:8], 16)
    return bucket % keep_every == 0


def _render_plain_image(
    *,
    series: list[float],
    plain_path: Path,
    width: int,
    height: int,
    overwrite: bool,
) -> None:
    plain_path.parent.mkdir(parents=True, exist_ok=True)

    if overwrite or not plain_path.exists():
        plain_image = _draw_plain_plot(series, width=width, height=height)
        plain_image.save(plain_path, format="PNG", optimize=True)

def _build_sample_payload(
    *,
    split: str,
    source_file: str,
    source_dataset: str,
    source_series_length: int,
    split_start: int,
    split_end: int,
    window_start_in_split: int,
    window_size: int,
    window_series: list[float],
    window_labels: list[int],
    window_padding_right: int,
    events: list[dict[str, Any]],
) -> dict[str, Any]:
    source_stem = Path(source_file).stem
    window_start_global = split_start + window_start_in_split
    actual_window_length = window_size - int(window_padding_right)
    window_end_global = window_start_global + actual_window_length - 1
    sample_id = f"{split}_{source_stem}_w{window_start_global:07d}"
    image_relpath = f"images_plain_768x384/{split}/{sample_id}.png"
    return {
        "sample_id": sample_id,
        "description": _build_sample_description(events),
        "series": [round(float(value), 6) for value in window_series],
        "point_labels": [int(label) for label in window_labels],
        "parameters": {
            "anomaly_type": "range" if events else "normal",
            "variant": "tsb_adu_segment_first_windowed",
        },
        "events": events,
        "context": {
            "series_length": len(window_series),
            "source_file": source_file,
            "source_dataset": source_dataset,
            "source_series_length": int(source_series_length),
            "window_start_global": int(window_start_global),
            "window_end_global": int(window_end_global),
            "window_start_in_split": int(window_start_in_split),
            "window_end_in_split": int(window_start_in_split + actual_window_length - 1),
            "window_padding_right": int(window_padding_right),
            "window_unpadded_length": int(actual_window_length),
            "split_segment_start_global": int(split_start),
            "split_segment_end_global": int(split_end - 1),
            "materialized_subset_split": split,
            "background_noise": "unknown",
            "source_model": "TSB-AD-U",
            "sampling_regime": "segment-first-irregular-length-windowed",
        },
        "image_path": image_relpath,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build per-subset TSB-AD-U roots by first splitting each source file into "
            "train/val/test segments, then windowing each split with the configured size and stride."
        )
    )
    parser.add_argument("--source-root", type=str, default="/gpfs/projects/p33222/ybq9740/TSB-AD-U")
    parser.add_argument("--csv-subdir", type=str, default="TSB-AD-U")
    parser.add_argument("--file-list-csv", type=str, default="File_List/TSB-AD-U-Eva.csv")
    parser.add_argument(
        "--output-root",
        type=str,
        default=(
            "/gpfs/projects/p33222/ybq9740/Thesis/Training-model-yilong/"
            "dataset/tsb_adu_subset_splits_segment_first_256_128_7_1_2"
        ),
    )
    parser.add_argument("--window-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--normal-keep-every", type=int, default=8)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--major-tick-step", type=int, default=8)
    parser.add_argument("--minor-tick-step", type=int, default=4)
    parser.add_argument("--value-column", type=str, default="Data")
    parser.add_argument("--label-column", type=str, default="Label")
    parser.add_argument("--source-dataset-filter", action="append", default=None)
    parser.add_argument("--max-source-files", type=int, default=None)
    parser.add_argument("--overwrite-existing-images", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    source_root = Path(args.source_root).expanduser().resolve()
    csv_root = (source_root / args.csv_subdir).resolve()
    file_list_csv = (source_root / args.file_list_csv).resolve()
    output_root = Path(args.output_root).expanduser().resolve()

    if not csv_root.exists():
        raise FileNotFoundError(f"CSV directory not found: {csv_root}")
    if not file_list_csv.exists():
        raise FileNotFoundError(f"File list not found: {file_list_csv}")
    if args.stride <= 0 or args.window_size <= 0:
        raise ValueError("window-size and stride must be positive integers.")

    dataset_filters = _normalize_dataset_filters(args.source_dataset_filter)
    file_names = _read_file_list(file_list_csv)
    if args.max_source_files is not None:
        file_names = file_names[: args.max_source_files]

    by_dataset: dict[str, list[str]] = defaultdict(list)
    for file_name in file_names:
        dataset_name = _parse_dataset_name(file_name)
        if dataset_filters and dataset_name not in dataset_filters:
            continue
        by_dataset[dataset_name].append(file_name)

    if not by_dataset:
        raise ValueError("No source files matched the requested dataset filters.")

    output_root.mkdir(parents=True, exist_ok=True)
    top_manifest: dict[str, Any] = {
        "source_root": str(source_root),
        "csv_root": str(csv_root),
        "file_list_csv": str(file_list_csv),
        "output_root": str(output_root),
        "split_strategy": "segment_each_source_file_then_window_each_split",
        "window_size": int(args.window_size),
        "stride": int(args.stride),
        "normal_keep_every": int(args.normal_keep_every),
        "train_ratio": float(args.train_ratio),
        "val_ratio": float(args.val_ratio),
        "test_ratio": float(1.0 - args.train_ratio - args.val_ratio),
        "datasets": {},
    }

    for dataset_name in sorted(by_dataset):
        subset_output_dir = output_root / dataset_name
        subset_output_dir.mkdir(parents=True, exist_ok=True)
        split_payloads: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
        per_file_manifest: dict[str, Any] = {}

        for file_index, file_name in enumerate(sorted(by_dataset[dataset_name]), start=1):
            csv_path = (csv_root / file_name).resolve()
            if not csv_path.exists():
                raise FileNotFoundError(f"Missing CSV file: {csv_path}")

            series, labels = _load_series_and_labels(
                csv_path,
                value_column=args.value_column,
                label_column=args.label_column,
            )
            segments = _extract_segments(labels)
            boundaries = _compute_segment_boundaries(
                len(series),
                train_ratio=float(args.train_ratio),
                val_ratio=float(args.val_ratio),
            )
            file_stats: dict[str, Any] = {
                "series_length": len(series),
                "anomaly_segment_count": len(segments),
                "splits": {},
            }

            for split in SPLITS:
                split_start, split_end = boundaries[split]
                segment_length = split_end - split_start
                local_starts = _iter_window_starts(
                    segment_length,
                    window_size=int(args.window_size),
                    stride=int(args.stride),
                )
                positive_candidates: list[dict[str, Any]] = []
                negative_candidates: list[dict[str, Any]] = []

                for window_start_in_split in local_starts:
                    window_start_global = split_start + window_start_in_split
                    window_end_global = window_start_global + int(args.window_size) - 1
                    local_segments = _intersect_segments(segments, window_start_global, window_end_global)

                    candidate = {
                        "window_start_in_split": int(window_start_in_split),
                        "window_start_global": int(window_start_global),
                        "events": [_build_event(segment, int(args.window_size)) for segment in local_segments],
                    }
                    if candidate["events"]:
                        positive_candidates.append(candidate)
                    else:
                        negative_candidates.append(candidate)

                kept_negative_candidates = [
                    candidate
                    for candidate in negative_candidates
                    if _normal_keep(file_name, split, int(candidate["window_start_global"]), int(args.normal_keep_every))
                ]
                if local_starts and not positive_candidates and not kept_negative_candidates and negative_candidates:
                    kept_negative_candidates = [negative_candidates[0]]

                kept_candidates = sorted(
                    positive_candidates + kept_negative_candidates,
                    key=lambda item: int(item["window_start_global"]),
                )

                for candidate in kept_candidates:
                    window_start_global = int(candidate["window_start_global"])
                    window_start_in_split = int(candidate["window_start_in_split"])
                    raw_window_end = min(window_start_global + int(args.window_size), split_end)
                    raw_window_series = series[window_start_global:raw_window_end]
                    raw_window_labels = labels[window_start_global:raw_window_end]
                    window_series, window_labels, window_padding_right = _pad_window(
                        raw_window_series,
                        raw_window_labels,
                        window_size=int(args.window_size),
                    )
                    payload = _build_sample_payload(
                        split=split,
                        source_file=file_name,
                        source_dataset=dataset_name,
                        source_series_length=len(series),
                        split_start=split_start,
                        split_end=split_end,
                        window_start_in_split=window_start_in_split,
                        window_size=int(args.window_size),
                        window_series=window_series,
                        window_labels=window_labels,
                        window_padding_right=window_padding_right,
                        events=list(candidate["events"]),
                    )
                    split_payloads[split].append(payload)

                    sample_id = payload["sample_id"]
                    plain_path = subset_output_dir / "images_plain_768x384" / split / f"{sample_id}.png"
                    _render_plain_image(
                        series=window_series,
                        plain_path=plain_path,
                        width=int(args.width),
                        height=int(args.height),
                        overwrite=bool(args.overwrite_existing_images),
                    )

                file_stats["splits"][split] = {
                    "segment_start_global": int(split_start),
                    "segment_end_global": int(split_end - 1) if split_end > split_start else None,
                    "segment_length": int(segment_length),
                    "num_candidate_windows": len(local_starts),
                    "num_kept_windows": len(kept_candidates),
                    "num_positive_windows": sum(1 for item in kept_candidates if item["events"]),
                    "num_negative_windows": sum(1 for item in kept_candidates if not item["events"]),
                }
            per_file_manifest[file_name] = file_stats

            if file_index % 25 == 0 or file_index == len(by_dataset[dataset_name]):
                print(f"{dataset_name}: processed source files {file_index}/{len(by_dataset[dataset_name])}")

        subset_manifest: dict[str, Any] = {
            "source_root": str(source_root),
            "csv_root": str(csv_root),
            "subset_dataset": dataset_name,
            "split_strategy": "segment_each_source_file_then_window_each_split",
            "window_size": int(args.window_size),
            "stride": int(args.stride),
            "normal_keep_every": int(args.normal_keep_every),
            "num_source_files": len(by_dataset[dataset_name]),
            "train_ratio": float(args.train_ratio),
            "val_ratio": float(args.val_ratio),
            "test_ratio": float(1.0 - args.train_ratio - args.val_ratio),
            "per_file_stats": per_file_manifest,
            "splits": {},
        }

        for split in SPLITS:
            records = sorted(
                split_payloads[split],
                key=lambda item: (
                    str(item.get("context", {}).get("source_file", "")),
                    int(item.get("context", {}).get("window_start_global", 0)),
                    str(item.get("sample_id", "")),
                ),
            )
            (subset_output_dir / f"{split}.json").write_text(
                json.dumps(records, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            subset_manifest["splits"][split] = {
                "num_windows": len(records),
                "num_positive_windows": sum(1 for item in records if item.get("events")),
                "num_negative_windows": sum(1 for item in records if not item.get("events")),
                "num_source_files": len({str(item["context"]["source_file"]) for item in records}),
                "num_source_files_with_no_windows": sum(
                    1 for stats in per_file_manifest.values() if stats["splits"][split]["num_kept_windows"] == 0
                ),
            }

        (subset_output_dir / "manifest.json").write_text(
            json.dumps(subset_manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        top_manifest["datasets"][dataset_name] = subset_manifest["splits"]

        print(
            f"{dataset_name}: windows train/val/test = "
            f"{subset_manifest['splits']['train']['num_windows']}/"
            f"{subset_manifest['splits']['val']['num_windows']}/"
            f"{subset_manifest['splits']['test']['num_windows']}"
        )
        print(
            f"{dataset_name}: source_files represented in train/val/test = "
            f"{subset_manifest['splits']['train']['num_source_files']}/"
            f"{subset_manifest['splits']['val']['num_source_files']}/"
            f"{subset_manifest['splits']['test']['num_source_files']}"
        )

    (output_root / "manifest.json").write_text(
        json.dumps(top_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote segment-first subset roots to: {output_root}")


if __name__ == "__main__":
    main()
