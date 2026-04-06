from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from ts_grounder.taxonomy import TYPE_TO_ID, canonicalize_type_name


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert PatternFinder JSON splits into ts_grounder .npz splits."
    )
    parser.add_argument(
        "--source-root",
        type=str,
        default="dataset/anomaly_db_v1",
        help="Directory containing PatternFinder train/val/test JSON files.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="data_full/anomaly_db_v1",
        help="Directory to write converted .npz splits.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=("train", "val", "test"),
        help="Splits to convert.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap per split for quick testing.",
    )
    return parser


def mask_to_segments(mask: np.ndarray) -> np.ndarray:
    segments: list[list[int]] = []
    start: int | None = None
    binary = mask > 0.5

    for idx, flag in enumerate(binary.tolist()):
        if flag and start is None:
            start = idx
        if start is not None and (not flag):
            segments.append([start, idx - 1])
            start = None

    if start is not None:
        segments.append([start, len(mask) - 1])

    if not segments:
        return np.zeros((0, 2), dtype=np.int64)
    return np.asarray(segments, dtype=np.int64)


def resolve_type_ids(sample: dict, num_segments: int) -> np.ndarray:
    if num_segments == 0:
        return np.zeros((0,), dtype=np.int64)

    default_name = canonicalize_type_name(sample["parameters"]["anomaly_type"])
    default_type_id = TYPE_TO_ID[default_name]
    events = sample.get("events", [])

    if len(events) == num_segments:
        type_ids = []
        for event in events:
            event_name = canonicalize_type_name(event.get("type", default_name))
            type_ids.append(TYPE_TO_ID.get(event_name, default_type_id))
        return np.asarray(type_ids, dtype=np.int64)

    return np.full((num_segments,), default_type_id, dtype=np.int64)


def convert_split(source_root: Path, output_root: Path, split_name: str, max_samples: int | None) -> int:
    split_json = source_root / f"{split_name}.json"
    if not split_json.exists():
        raise FileNotFoundError(f"Missing split JSON: {split_json}")

    samples = json.loads(split_json.read_text(encoding="utf-8"))
    if max_samples is not None:
        samples = samples[:max_samples]

    split_output_dir = output_root / split_name
    split_output_dir.mkdir(parents=True, exist_ok=True)

    for sample in samples:
        sample_id = sample["sample_id"]
        series = np.asarray(sample["series"], dtype=np.float32)
        mask = np.asarray(sample["point_labels"], dtype=np.float32)
        segments = mask_to_segments(mask)
        types = resolve_type_ids(sample, num_segments=segments.shape[0])

        np.savez_compressed(
            split_output_dir / f"{sample_id}.npz",
            series=series,
            mask=mask,
            segments=segments,
            types=types,
            image_path=sample.get("image_path", ""),
        )

    return len(samples)


def main() -> None:
    args = build_parser().parse_args()
    source_root = Path(args.source_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    summary = {}
    for split_name in args.splits:
        count = convert_split(
            source_root=source_root,
            output_root=output_root,
            split_name=split_name,
            max_samples=args.max_samples,
        )
        summary[split_name] = {"num_samples": count}

    (output_root / "conversion_summary.json").write_text(
        json.dumps(
            {
                "source_root": str(source_root),
                "output_root": str(output_root),
                "splits": summary,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    for split_name, split_summary in summary.items():
        print(f"{split_name}: converted {split_summary['num_samples']} samples")


if __name__ == "__main__":
    main()
