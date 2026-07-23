from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from build_tsb_adu_windowed_dataset import (
    _build_event,
    _build_sample_description,
    _draw_plain_plot,
)

SPLITS = ("train", "val", "test")


def _parse_dataset_name(file_name: str) -> str:
    parts = file_name.split("_")
    if len(parts) >= 2:
        return parts[1]
    return "unknown"


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
        raise ValueError(f"{csv_path} has no sequence data.")
    if len(series) != len(labels):
        raise ValueError(f"{csv_path} has mismatched series/labels lengths.")
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


def _normal_keep(sample_key: str, split: str, window_start_global: int, keep_every: int) -> bool:
    if keep_every <= 1:
        return True
    token = f"{split}:{sample_key}:{window_start_global}"
    bucket = int.from_bytes(token.encode("utf-8"), "little", signed=False)
    return bucket % keep_every == 0


def _compute_split_counts(n_items: int, train_ratio: float, val_ratio: float) -> tuple[int, int, int]:
    if n_items <= 0:
        return 0, 0, 0
    if n_items == 1:
        return 1, 0, 0
    if n_items == 2:
        return 1, 0, 1

    train_count = max(1, int(n_items * train_ratio))
    val_count = max(1, int(n_items * val_ratio))
    test_count = n_items - train_count - val_count

    if test_count < 1:
        deficit = 1 - test_count
        reducible_train = max(0, train_count - 1)
        take_train = min(deficit, reducible_train)
        train_count -= take_train
        deficit -= take_train

        reducible_val = max(0, val_count - 1)
        take_val = min(deficit, reducible_val)
        val_count -= take_val
        deficit -= take_val

        test_count = 1
        if deficit > 0:
            raise ValueError(f"Unable to allocate split counts for n_items={n_items}")

    assigned = train_count + val_count + test_count
    if assigned < n_items:
        test_count += n_items - assigned
    elif assigned > n_items:
        overflow = assigned - n_items
        reducible_test = max(0, test_count - 1)
        take_test = min(overflow, reducible_test)
        test_count -= take_test
        overflow -= take_test
        if overflow > 0:
            reducible_train = max(0, train_count - 1)
            take_train = min(overflow, reducible_train)
            train_count -= take_train
            overflow -= take_train
        if overflow > 0:
            raise ValueError(f"Split counts overflow for n_items={n_items}")

    return train_count, val_count, test_count


def _build_segments(
    series_length: int,
    *,
    train_ratio: float,
    val_ratio: float,
) -> dict[str, tuple[int, int]]:
    train_count, val_count, test_count = _compute_split_counts(
        series_length,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )
    train_start = 0
    val_start = train_start + train_count
    test_start = val_start + val_count
    if test_start + test_count != series_length:
        raise ValueError("Raw split segments do not cover the full sequence.")
    return {
        "train": (train_start, val_start),
        "val": (val_start, test_start),
        "test": (test_start, series_length),
    }


def _build_window_starts(segment_length: int, *, window_size: int, stride: int) -> list[int]:
    if segment_length <= 0:
        return []
    if segment_length <= window_size:
        return [0]

    max_start = segment_length - window_size
    starts = list(range(0, max_start + 1, stride))
    if starts[-1] != max_start:
        starts.append(max_start)
    return starts


def _pad_series(window_series: list[float], *, target_length: int) -> tuple[list[float], int]:
    if len(window_series) >= target_length:
        return [float(value) for value in window_series[:target_length]], 0
    if not window_series:
        return [0.0] * target_length, target_length
    pad_value = float(window_series[-1])
    pad_count = target_length - len(window_series)
    return [float(value) for value in window_series] + [pad_value] * pad_count, pad_count


def _pad_labels(window_labels: list[int], *, target_length: int) -> list[int]:
    if len(window_labels) >= target_length:
        return [int(value) for value in window_labels[:target_length]]
    return [int(value) for value in window_labels] + [0] * (target_length - len(window_labels))


def _render_plain_image(
    series: list[float],
    *,
    output_path: Path,
    width: int,
    height: int,
    overwrite: bool,
) -> None:
    if output_path.exists() and not overwrite:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = _draw_plain_plot(series, width=width, height=height)
    image.save(output_path, format="PNG", optimize=True)


def _build_sample_payload(
    *,
    split: str,
    source_path: Path,
    dataset_name: str,
    window_series: list[float],
    window_labels: list[int],
    window_start_global: int,
    window_end_global: int,
    segment_start_global: int,
    segment_end_global: int,
    original_window_length: int,
    padding_right: int,
    image_relpath: str,
) -> dict[str, Any]:
    local_segments = []
    for segment in _extract_segments(window_labels):
        local_segments.append(
            {
                "start": int(segment["start"]),
                "end": int(segment["end"]),
                "global_start": int(window_start_global + int(segment["start"])),
                "global_end": int(window_start_global + int(segment["end"])),
            }
        )
    event_payload = [_build_event(segment, len(window_series)) for segment in local_segments]
    sample_id = f"{split}_{source_path.stem}_w{window_start_global:07d}"
    return {
        "sample_id": sample_id,
        "description": _build_sample_description(event_payload),
        "series": [round(float(value), 6) for value in window_series],
        "point_labels": [int(label) for label in window_labels],
        "parameters": {
            "anomaly_type": "range" if event_payload else "normal",
            "variant": "tsb_adu_raw_split_padded_windowed",
        },
        "events": event_payload,
        "context": {
            "series_length": len(window_series),
            "source_file": source_path.name,
            "source_dataset": dataset_name,
            "window_start_global": int(window_start_global),
            "window_end_global": int(window_end_global),
            "segment_start_global": int(segment_start_global),
            "segment_end_global": int(segment_end_global),
            "segment_length_original": int(segment_end_global - segment_start_global + 1),
            "window_original_length": int(original_window_length),
            "padding_right": int(padding_right),
            "background_noise": "unknown",
            "source_model": "TSB-AD-U",
            "sampling_regime": "raw-file-split-windowed-right-padded",
        },
        "image_path": image_relpath,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build TSB-AD-U subset roots by first splitting each raw source file along the time axis "
            "into train/val/test segments, then windowing inside each split with right padding up to the window size."
        )
    )
    parser.add_argument(
        "--source-root",
        type=str,
        default="/gpfs/projects/p33222/ybq9740/TSB-AD-U/TSB-AD-U",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=(
            "/gpfs/projects/p33222/ybq9740/Thesis/Training-model-yilong/"
            "dataset/tsb_adu_subset_splits_raw_file_padded_256_128_7_1_2"
        ),
    )
    parser.add_argument("--source-dataset-filter", action="append", default=None)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--window-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--normal-keep-every", type=int, default=8)
    parser.add_argument("--value-column", type=str, default="Data")
    parser.add_argument("--label-column", type=str, default="Label")
    parser.add_argument("--plain-image-subdir", type=str, default="images_plain_768x384")
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--overwrite-images", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    source_root = Path(args.source_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    dataset_filters = _normalize_dataset_filters(args.source_dataset_filter)

    csv_paths = sorted(path for path in source_root.glob("*.csv") if path.is_file())
    if dataset_filters:
        csv_paths = [path for path in csv_paths if _parse_dataset_name(path.name) in dataset_filters]
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found under {source_root}")

    output_root.mkdir(parents=True, exist_ok=True)
    top_manifest: dict[str, Any] = {
        "source_root": str(source_root),
        "output_root": str(output_root),
        "split_strategy": "raw_file_time_axis_then_window_with_right_padding",
        "train_ratio": float(args.train_ratio),
        "val_ratio": float(args.val_ratio),
        "test_ratio": float(1.0 - args.train_ratio - args.val_ratio),
        "window_size": int(args.window_size),
        "stride": int(args.stride),
        "normal_keep_every": int(args.normal_keep_every),
        "padding_mode": "right_edge_repeat_to_window_size",
        "plain_image_subdir": args.plain_image_subdir,
        "datasets": {},
    }

    by_dataset: dict[str, list[Path]] = defaultdict(list)
    for csv_path in csv_paths:
        by_dataset[_parse_dataset_name(csv_path.name)].append(csv_path)

    for dataset_name in sorted(by_dataset):
        subset_output_dir = output_root / dataset_name
        subset_output_dir.mkdir(parents=True, exist_ok=True)

        split_payloads: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
        per_file_window_counts: dict[str, dict[str, int]] = {}
        per_file_padding_counts: dict[str, dict[str, int]] = {}

        for csv_path in sorted(by_dataset[dataset_name]):
            series, labels = _load_series_and_labels(
                csv_path,
                value_column=args.value_column,
                label_column=args.label_column,
            )
            raw_segments = _build_segments(
                len(series),
                train_ratio=float(args.train_ratio),
                val_ratio=float(args.val_ratio),
            )
            per_file_window_counts[csv_path.name] = {}
            per_file_padding_counts[csv_path.name] = {}

            for split in SPLITS:
                segment_start, segment_stop = raw_segments[split]
                segment_series = series[segment_start:segment_stop]
                segment_labels = labels[segment_start:segment_stop]
                starts = _build_window_starts(
                    len(segment_series),
                    window_size=int(args.window_size),
                    stride=int(args.stride),
                )
                candidate_payloads: list[dict[str, Any]] = []

                for local_start in starts:
                    local_stop = min(local_start + int(args.window_size), len(segment_series))
                    original_window_series = segment_series[local_start:local_stop]
                    original_window_labels = segment_labels[local_start:local_stop]
                    padded_series, padding_right = _pad_series(
                        original_window_series,
                        target_length=int(args.window_size),
                    )
                    padded_labels = _pad_labels(
                        original_window_labels,
                        target_length=int(args.window_size),
                    )
                    window_start_global = segment_start + local_start
                    window_end_global = segment_start + local_stop - 1
                    sample_id = f"{split}_{csv_path.stem}_w{window_start_global:07d}"
                    image_relpath = f"{args.plain_image_subdir}/{split}/{sample_id}.png"

                    payload = _build_sample_payload(
                        split=split,
                        source_path=csv_path,
                        dataset_name=dataset_name,
                        window_series=padded_series,
                        window_labels=padded_labels,
                        window_start_global=window_start_global,
                        window_end_global=window_end_global,
                        segment_start_global=segment_start,
                        segment_end_global=segment_stop - 1,
                        original_window_length=len(original_window_series),
                        padding_right=padding_right,
                        image_relpath=image_relpath,
                    )
                    payload["_padding_right"] = padding_right
                    payload["_keep_window"] = bool(payload["events"]) or _normal_keep(
                        csv_path.stem,
                        split,
                        window_start_global,
                        int(args.normal_keep_every),
                    )
                    candidate_payloads.append(payload)

                kept_payloads = [payload for payload in candidate_payloads if payload["_keep_window"]]
                if not kept_payloads and candidate_payloads:
                    kept_payloads = [candidate_payloads[0]]

                per_file_window_counts[csv_path.name][split] = len(kept_payloads)
                per_file_padding_counts[csv_path.name][split] = sum(
                    1 for payload in kept_payloads if int(payload["_padding_right"]) > 0
                )

                for payload in kept_payloads:
                    sample_id = str(payload["sample_id"])
                    padded_series = list(payload["series"])
                    plain_path = subset_output_dir / args.plain_image_subdir / split / f"{sample_id}.png"
                    _render_plain_image(
                        padded_series,
                        output_path=plain_path,
                        width=int(args.width),
                        height=int(args.height),
                        overwrite=bool(args.overwrite_images),
                    )
                    payload.pop("_padding_right", None)
                    payload.pop("_keep_window", None)
                    split_payloads[split].append(payload)

        subset_manifest: dict[str, Any] = {
            "source_root": str(source_root),
            "subset_dataset": dataset_name,
            "split_strategy": "raw_file_time_axis_then_window_with_right_padding",
            "num_source_files": len(by_dataset[dataset_name]),
            "train_ratio": float(args.train_ratio),
            "val_ratio": float(args.val_ratio),
            "test_ratio": float(1.0 - args.train_ratio - args.val_ratio),
            "window_size": int(args.window_size),
            "stride": int(args.stride),
            "padding_mode": "right_edge_repeat_to_window_size",
            "per_file_window_counts": per_file_window_counts,
            "per_file_padding_counts": per_file_padding_counts,
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
                "num_padded_windows": sum(
                    1 for item in records if int(item.get("context", {}).get("padding_right", 0)) > 0
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
            f"{dataset_name}: padded train/val/test = "
            f"{subset_manifest['splits']['train']['num_padded_windows']}/"
            f"{subset_manifest['splits']['val']['num_padded_windows']}/"
            f"{subset_manifest['splits']['test']['num_padded_windows']}"
        )

    (output_root / "manifest.json").write_text(
        json.dumps(top_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote raw-split padded subset roots to: {output_root}")


if __name__ == "__main__":
    main()
