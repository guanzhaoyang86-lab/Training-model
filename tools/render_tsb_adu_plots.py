from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from PIL import Image, ImageDraw


def _read_file_list(file_list_csv: Path) -> list[str]:
    with open(file_list_csv, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [row["file_name"] for row in reader if row.get("file_name")]


def _iter_csv_files(csv_dir: Path, file_list_csv: Path | None = None) -> list[Path]:
    if file_list_csv and file_list_csv.exists():
        ordered = []
        for file_name in _read_file_list(file_list_csv):
            csv_path = (csv_dir / file_name).resolve()
            if csv_path.exists():
                ordered.append(csv_path)
        return ordered
    return sorted(csv_dir.glob("*.csv"))


def _load_series(csv_path: Path, value_column: str = "Data", label_column: str = "Label") -> tuple[list[float], list[int]]:
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
        raise ValueError(f"{csv_path} 没有可绘制的数据点。")
    return series, labels


def _count_anomaly_segments(labels: list[int]) -> int:
    segments = 0
    previous = 0
    for value in labels:
        if value == 1 and previous == 0:
            segments += 1
        previous = value
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

    if plot_width <= 0 or plot_height <= 0:
        raise ValueError("Image canvas is too small for configured margins.")

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

    # Light baseline and border, matching the existing plain VLM plots.
    draw.line([(plot_left, plot_top), (plot_left, plot_bottom)], fill=(192, 198, 210), width=1)
    draw.line([(plot_left, plot_bottom), (plot_right, plot_bottom)], fill=(192, 198, 210), width=1)

    points = [(x_to_px(idx), y_to_px(value)) for idx, value in enumerate(values)]
    draw.line(points, fill=(35, 96, 181), width=2)
    return image


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render plain PNG plots for TSB-AD-U CSV files.")
    parser.add_argument("--source-root", type=str, default="/gpfs/projects/p33222/ybq9740/TSB-AD-U")
    parser.add_argument("--csv-subdir", type=str, default="TSB-AD-U")
    parser.add_argument("--file-list-csv", type=str, default="File_List/TSB-AD-U-Eva.csv")
    parser.add_argument("--output-image-dir", type=str, default="images_plain_768x384")
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--value-column", type=str, default="Data")
    parser.add_argument("--label-column", type=str, default="Label")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    source_root = Path(args.source_root).expanduser().resolve()
    csv_dir = (source_root / args.csv_subdir).resolve()
    output_dir = (source_root / args.output_image_dir).resolve()
    file_list_csv = (source_root / args.file_list_csv).resolve() if args.file_list_csv else None

    if not csv_dir.exists():
        raise FileNotFoundError(f"CSV 目录不存在: {csv_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_files = _iter_csv_files(csv_dir, file_list_csv=file_list_csv)
    if args.limit is not None:
        csv_files = csv_files[: args.limit]

    manifest_records = []
    for index, csv_path in enumerate(csv_files, start=1):
        output_path = output_dir / f"{csv_path.stem}.png"
        if output_path.exists() and not args.overwrite:
            continue

        series, labels = _load_series(
            csv_path,
            value_column=args.value_column,
            label_column=args.label_column,
        )
        image = _draw_plain_plot(series, width=args.width, height=args.height)
        image.save(output_path, format="PNG", optimize=True)

        manifest_records.append(
            {
                "file_name": csv_path.name,
                "csv_path": str(csv_path),
                "image_path": str(output_path),
                "series_length": len(series),
                "anomaly_point_count": int(sum(labels)),
                "anomaly_segment_count": _count_anomaly_segments(labels),
            }
        )

        if index % 100 == 0 or index == len(csv_files):
            print(f"rendered {index}/{len(csv_files)} -> {output_dir}")

    manifest_path = output_dir / "render_manifest.jsonl"
    with open(manifest_path, "w", encoding="utf-8") as handle:
        for record in manifest_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
