from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


def _read_matches(matches_jsonl: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with open(matches_jsonl, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


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
            raise ValueError(f"{csv_path} 中缺少数值列 {value_column!r}")
        for row in reader:
            series.append(float(row[value_column]))
            if label_column in row and row[label_column] not in (None, ""):
                labels.append(int(float(row[label_column])))
            else:
                labels.append(0)
    if not series:
        raise ValueError(f"{csv_path} 没有可用的序列数据。")
    if len(series) != len(labels):
        raise ValueError(f"{csv_path} 的 series 与 labels 长度不一致。")
    return series, labels


def _assign_split(sample_key: str, train_ratio: float, val_ratio: float) -> str:
    bucket = int(hashlib.sha1(sample_key.encode("utf-8")).hexdigest()[:8], 16) % 10000
    threshold_train = int(train_ratio * 10000)
    threshold_val = int((train_ratio + val_ratio) * 10000)
    if bucket < threshold_train:
        return "train"
    if bucket < threshold_val:
        return "val"
    return "test"


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


def _normal_keep(sample_key: str, window_start: int, keep_every: int) -> bool:
    if keep_every <= 1:
        return True
    token = f"{sample_key}:{window_start}"
    bucket = int(hashlib.sha1(token.encode("utf-8")).hexdigest()[:8], 16)
    return bucket % keep_every == 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a windowed TSB-AD-U dataset compatible with the VLM pipeline.")
    parser.add_argument("--matches-jsonl", type=str, default="/gpfs/projects/p33222/ybq9740/TSB-AD-U/tsb_adu_eval_matches.jsonl")
    parser.add_argument("--output-root", type=str, default="/gpfs/projects/p33222/ybq9740/TSB-AD-U/windowed_vlm_dataset_256_128")
    parser.add_argument("--window-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--normal-keep-every", type=int, default=8)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--value-column", type=str, default="Data")
    parser.add_argument("--label-column", type=str, default="Label")
    parser.add_argument("--max-source-files", type=int, default=None)
    parser.add_argument("--overwrite-existing-images", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    matches_path = Path(args.matches_jsonl).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()

    records = _read_matches(matches_path)
    if args.max_source_files is not None:
        records = records[: args.max_source_files]

    output_root.mkdir(parents=True, exist_ok=True)
    image_root = output_root / "images_plain_768x384"
    for split in ("train", "val", "test"):
        (image_root / split).mkdir(parents=True, exist_ok=True)

    summary = {
        "source_matches": str(matches_path),
        "window_size": int(args.window_size),
        "stride": int(args.stride),
        "normal_keep_every": int(args.normal_keep_every),
        "num_source_files": len(records),
        "splits": {},
    }
    split_stats: dict[str, dict[str, int]] = {
        "train": {"num_windows": 0, "num_positive_windows": 0, "num_negative_windows": 0},
        "val": {"num_windows": 0, "num_positive_windows": 0, "num_negative_windows": 0},
        "test": {"num_windows": 0, "num_positive_windows": 0, "num_negative_windows": 0},
    }
    split_first_record = {"train": True, "val": True, "test": True}
    split_handles = {
        split: open(output_root / f"{split}.json", "w", encoding="utf-8")
        for split in ("train", "val", "test")
    }
    try:
        for handle in split_handles.values():
            handle.write("[\n")

        for source_index, record in enumerate(records, start=1):
            sample_key = str(record["sample_id"])
            split = _assign_split(sample_key, train_ratio=args.train_ratio, val_ratio=args.val_ratio)
            csv_path = Path(record["csv_path"]).resolve()
            anomaly_segments = list(record.get("anomaly_segments", []))
            series, labels = _load_series_and_labels(
                csv_path,
                value_column=args.value_column,
                label_column=args.label_column,
            )

            max_start = len(series) - args.window_size
            if max_start < 0:
                continue

            starts = list(range(0, max_start + 1, args.stride))
            if starts[-1] != max_start:
                starts.append(max_start)

            for window_start in starts:
                window_end = window_start + args.window_size - 1
                local_segments = _intersect_segments(anomaly_segments, window_start, window_end)
                if len(local_segments) == 0 and not _normal_keep(sample_key, window_start, args.normal_keep_every):
                    continue

                window_series = series[window_start : window_start + args.window_size]
                window_labels = labels[window_start : window_start + args.window_size]
                event_payload = [_build_event(segment, args.window_size) for segment in local_segments]

                sample_id = f"{split}_{sample_key}_w{window_start:07d}"
                image_relpath = f"images_plain_768x384/{split}/{sample_id}.png"
                image_path = output_root / image_relpath

                if args.overwrite_existing_images or not image_path.exists():
                    image = _draw_plain_plot(window_series, width=args.width, height=args.height)
                    image.save(image_path, format="PNG", optimize=True)

                payload = {
                    "sample_id": sample_id,
                    "description": _build_sample_description(event_payload),
                    "series": [round(float(value), 6) for value in window_series],
                    "point_labels": [int(label) for label in window_labels],
                    "parameters": {
                        "anomaly_type": "range" if event_payload else "normal",
                        "variant": "tsb_adu_windowed",
                    },
                    "events": event_payload,
                    "context": {
                        "series_length": len(window_series),
                        "source_file": record["file_name"],
                        "source_dataset": record["dataset_name"],
                        "window_start_global": int(window_start),
                        "window_end_global": int(window_end),
                        "background_noise": "unknown",
                        "source_model": "TSB-AD-U",
                        "sampling_regime": "irregular-length-windowed",
                    },
                    "image_path": image_relpath,
                }
                handle = split_handles[split]
                if not split_first_record[split]:
                    handle.write(",\n")
                handle.write(json.dumps(payload, ensure_ascii=False))
                split_first_record[split] = False
                split_stats[split]["num_windows"] += 1
                if event_payload:
                    split_stats[split]["num_positive_windows"] += 1
                else:
                    split_stats[split]["num_negative_windows"] += 1

            if source_index % 50 == 0 or source_index == len(records):
                print(f"processed source files {source_index}/{len(records)}")

        for split, handle in split_handles.items():
            handle.write("\n]\n")
            handle.close()
            summary["splits"][split] = split_stats[split]
    finally:
        for split, handle in split_handles.items():
            if not handle.closed:
                handle.close()

    with open(output_root / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(f"Wrote windowed dataset to: {output_root}")
    for split, info in summary["splits"].items():
        print(
            f"{split}: {info['num_windows']} windows "
            f"({info['num_positive_windows']} positive / {info['num_negative_windows']} negative)"
        )


if __name__ == "__main__":
    main()
