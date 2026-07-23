from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _read_file_list(file_list_csv: Path) -> list[str]:
    with open(file_list_csv, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [row["file_name"] for row in reader if row.get("file_name")]


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
    if len(labels) != len(series):
        raise ValueError(f"{csv_path} 的 labels 长度与 series 不一致。")
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


def _parse_dataset_name(file_name: str) -> str:
    parts = file_name.split("_")
    if len(parts) >= 2:
        return parts[1]
    return "unknown"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build csv-image alignment manifest for TSB-AD-U.")
    parser.add_argument("--source-root", type=str, default="/gpfs/projects/p33222/ybq9740/TSB-AD-U")
    parser.add_argument("--csv-subdir", type=str, default="TSB-AD-U")
    parser.add_argument("--image-subdir", type=str, default="images_plain_768x384")
    parser.add_argument("--file-list-csv", type=str, default="File_List/TSB-AD-U-Eva.csv")
    parser.add_argument("--output-jsonl", type=str, default="tsb_adu_eval_matches.jsonl")
    parser.add_argument("--output-summary", type=str, default="tsb_adu_eval_matches_summary.json")
    parser.add_argument("--value-column", type=str, default="Data")
    parser.add_argument("--label-column", type=str, default="Label")
    parser.add_argument("--limit", type=int, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    source_root = Path(args.source_root).expanduser().resolve()
    csv_dir = (source_root / args.csv_subdir).resolve()
    image_dir = (source_root / args.image_subdir).resolve()
    file_list_csv = (source_root / args.file_list_csv).resolve()
    output_jsonl = (source_root / args.output_jsonl).resolve()
    output_summary = (source_root / args.output_summary).resolve()

    if not csv_dir.exists():
        raise FileNotFoundError(f"CSV 目录不存在: {csv_dir}")
    if not image_dir.exists():
        raise FileNotFoundError(f"图片目录不存在: {image_dir}")
    if not file_list_csv.exists():
        raise FileNotFoundError(f"文件列表不存在: {file_list_csv}")

    file_names = _read_file_list(file_list_csv)
    if args.limit is not None:
        file_names = file_names[: args.limit]

    records: list[dict[str, Any]] = []
    dataset_counts: dict[str, int] = {}
    max_series_length = 0
    min_series_length: int | None = None

    for index, file_name in enumerate(file_names, start=1):
        csv_path = (csv_dir / file_name).resolve()
        if not csv_path.exists():
            raise FileNotFoundError(f"缺少 CSV: {csv_path}")
        image_path = (image_dir / f"{csv_path.stem}.png").resolve()
        if not image_path.exists():
            raise FileNotFoundError(f"缺少对应图片: {image_path}")

        series, labels = _load_series_and_labels(
            csv_path,
            value_column=args.value_column,
            label_column=args.label_column,
        )
        segments = _extract_segments(labels)
        series_length = len(series)
        dataset_name = _parse_dataset_name(file_name)
        dataset_counts[dataset_name] = dataset_counts.get(dataset_name, 0) + 1
        max_series_length = max(max_series_length, series_length)
        min_series_length = series_length if min_series_length is None else min(min_series_length, series_length)

        record = {
            "sample_id": csv_path.stem,
            "file_name": file_name,
            "dataset_name": dataset_name,
            "split": "eval",
            "csv_path": str(csv_path),
            "image_path": str(image_path),
            "series_length": series_length,
            "anomaly_segments": segments,
            "anomaly_point_count": int(sum(labels)),
            "anomaly_segment_count": len(segments),
            "value_column": args.value_column,
            "label_column": args.label_column,
        }
        records.append(record)

        if index % 100 == 0 or index == len(file_names):
            print(f"matched {index}/{len(file_names)}")

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(output_jsonl, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = {
        "num_records": len(records),
        "datasets": dataset_counts,
        "min_series_length": min_series_length,
        "max_series_length": max_series_length,
        "source_root": str(source_root),
        "csv_dir": str(csv_dir),
        "image_dir": str(image_dir),
        "file_list_csv": str(file_list_csv),
        "output_jsonl": str(output_jsonl),
    }
    with open(output_summary, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(f"Wrote matches: {output_jsonl}")
    print(f"Wrote summary: {output_summary}")


if __name__ == "__main__":
    main()
