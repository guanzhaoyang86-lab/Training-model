from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from ts_grounder.vlm_data import build_vlm_sft_dataset
from ts_grounder.vlm_prompting import DEFAULT_SYSTEM_PROMPT, DEFAULT_USER_PROMPT


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build VLM SFT dataset from PatternFinder exports.")
    parser.add_argument("--source-root", type=str, default="dataset/anomaly_db_v1")
    parser.add_argument("--output-root", type=str, default="data_vlm/anomaly_db_v1")
    parser.add_argument("--image-subdir", type=str, default=None)
    parser.add_argument("--include-indexed-series-text", action="store_true")
    parser.add_argument("--indexed-series-precision", type=int, default=4)
    parser.add_argument("--indexed-series-compact", action="store_true")
    parser.add_argument("--series-normalization", type=str, default="none", choices=["none", "zscore", "robust_zscore"])
    parser.add_argument("--series-normalization-clip", type=float, default=8.0)
    parser.add_argument("--series-normalization-text-only", action="store_true")
    parser.add_argument("--series-normalization-overwrite-images", action="store_true")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--source-dataset-filter", action="append", default=None)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--no-balance", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    source_root = Path(args.source_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    manifest = build_vlm_sft_dataset(
        source_root=source_root,
        output_root=output_root,
        image_subdir=args.image_subdir,
        include_indexed_series_text=bool(args.include_indexed_series_text),
        indexed_series_precision=int(args.indexed_series_precision),
        indexed_series_compact=bool(args.indexed_series_compact),
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        max_test_samples=args.max_test_samples,
        balanced_by_type=not args.no_balance,
        source_dataset_filter=args.source_dataset_filter,
        series_normalization={
            "enabled": args.series_normalization != "none",
            "method": args.series_normalization,
            "clip": args.series_normalization_clip,
            "normalize_text": True,
            "normalize_image": not args.series_normalization_text_only,
            "overwrite_images": args.series_normalization_overwrite_images,
        },
        seed=args.seed,
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        user_prompt=DEFAULT_USER_PROMPT,
    )
    print(f"Wrote VLM dataset to: {output_root}")
    for split, info in manifest["splits"].items():
        print(f"{split}: {info['num_records']} records")


if __name__ == "__main__":
    main()
