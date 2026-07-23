from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

SPLITS = ("train", "val", "test")


def _load_split(source_root: Path, split: str) -> list[dict[str, Any]]:
    path = source_root / f"{split}.json"
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


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


def _iter_subset_dirs(source_root: Path, filters: set[str] | None) -> list[Path]:
    subset_dirs: list[Path] = []
    for child in sorted(source_root.iterdir()):
        if not child.is_dir():
            continue
        if filters and child.name not in filters:
            continue
        if any((child / f"{split}.json").exists() for split in SPLITS):
            subset_dirs.append(child)
    if not subset_dirs:
        raise FileNotFoundError(f"No subset directories with split json files were found under {source_root}")
    return subset_dirs


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
            raise ValueError(f"Unable to allocate train/val/test counts for n_items={n_items}")

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


def _assign_windows_within_file(
    samples: list[dict[str, Any]],
    *,
    train_ratio: float,
    val_ratio: float,
) -> dict[str, str]:
    ordered = sorted(
        samples,
        key=lambda item: (
            int(item.get("context", {}).get("window_start_global", 0)),
            int(item.get("context", {}).get("window_end_global", 0)),
            str(item.get("sample_id", "")),
        ),
    )
    train_count, val_count, test_count = _compute_split_counts(
        len(ordered),
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )
    split_map: dict[str, str] = {}

    cursor = 0
    for sample in ordered[cursor : cursor + train_count]:
        split_map[str(sample["sample_id"])] = "train"
    cursor += train_count
    for sample in ordered[cursor : cursor + val_count]:
        split_map[str(sample["sample_id"])] = "val"
    cursor += val_count
    for sample in ordered[cursor : cursor + test_count]:
        split_map[str(sample["sample_id"])] = "test"
    return split_map


def _resolve_existing_image(
    subset_root: Path,
    *,
    image_subdir: str,
    sample_id: str,
    stored_image_path: str | None = None,
) -> Path | None:
    candidates: list[Path] = []
    if stored_image_path:
        image_path = Path(stored_image_path)
        candidates.append(image_path if image_path.is_absolute() else subset_root / image_path)
    for split in SPLITS:
        candidates.append(subset_root / image_subdir / split / f"{sample_id}.png")

    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.expanduser()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists():
            return resolved
    return None


def _materialize_file(source_path: Path, target_path: Path, *, mode: str, overwrite: bool) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.exists():
        if not overwrite:
            return
        target_path.unlink()

    if mode == "copy":
        shutil.copy2(source_path, target_path)
        return
    if mode == "symlink":
        os.symlink(source_path, target_path)
        return
    if mode == "hardlink":
        os.link(source_path, target_path)
        return
    if mode != "auto":
        raise ValueError(f"Unsupported link mode: {mode}")

    try:
        os.link(source_path, target_path)
    except OSError:
        shutil.copy2(source_path, target_path)


def _build_target_sample_id(sample: dict[str, Any], *, target_split: str) -> str:
    source_file = str(sample.get("context", {}).get("source_file", "")).strip()
    if not source_file:
        raise KeyError(f"Sample {sample.get('sample_id')!r} is missing context.source_file")
    source_stem = Path(source_file).stem
    window_start = int(sample.get("context", {}).get("window_start_global", 0))
    return f"{target_split}_{source_stem}_w{window_start:07d}"


def _prepare_sample(
    sample: dict[str, Any],
    *,
    target_split: str,
    target_sample_id: str,
    plain_image_subdir: str,
) -> dict[str, Any]:
    payload = json.loads(json.dumps(sample, ensure_ascii=False))
    payload["sample_id"] = target_sample_id
    payload["image_path"] = f"{plain_image_subdir}/{target_split}/{target_sample_id}.png"
    payload.setdefault("context", {})
    payload["context"]["original_split"] = str(
        payload["context"].get("original_split") or str(sample.get("sample_id", "")).split("_", 1)[0]
    )
    payload["context"]["materialized_subset_split"] = target_split
    payload["context"]["original_sample_id"] = str(sample.get("sample_id", ""))
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild TSB-AD-U subset roots by splitting windows inside every source file, "
            "so each file contributes train/val/test samples."
        )
    )
    parser.add_argument(
        "--source-root",
        type=str,
        default="/gpfs/projects/p33222/ybq9740/thesis/TSB-AD-U-subset-splits",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=(
            "/gpfs/projects/p33222/ybq9740/Thesis/Training-model-yilong/"
            "dataset/tsb_adu_subset_splits_within_file_256_128_7_1_2"
        ),
    )
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--source-dataset-filter", action="append", default=None)
    parser.add_argument(
        "--link-mode",
        choices=("auto", "hardlink", "copy", "symlink"),
        default="auto",
        help="How to materialize PNGs into the rebuilt subset directory.",
    )
    parser.add_argument("--overwrite-images", action="store_true")
    parser.add_argument("--plain-image-subdir", type=str, default="images_plain_768x384")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    source_root = Path(args.source_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()

    output_root.mkdir(parents=True, exist_ok=True)
    dataset_filters = _normalize_dataset_filters(args.source_dataset_filter)
    subset_dirs = _iter_subset_dirs(source_root, dataset_filters)

    top_manifest: dict[str, Any] = {
        "source_root": str(source_root),
        "output_root": str(output_root),
        "split_strategy": "split_windows_within_each_source_file",
        "sample_id_strategy": "regenerate_with_target_split_prefix",
        "train_ratio": float(args.train_ratio),
        "val_ratio": float(args.val_ratio),
        "test_ratio": float(1.0 - args.train_ratio - args.val_ratio),
        "plain_image_subdir": args.plain_image_subdir,
        "link_mode": args.link_mode,
        "datasets": {},
    }

    for subset_source_dir in subset_dirs:
        dataset_name = subset_source_dir.name
        raw_samples: list[dict[str, Any]] = []
        for split in SPLITS:
            raw_samples.extend(_load_split(subset_source_dir, split))

        by_source_file: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for sample in raw_samples:
            source_file = str(sample.get("context", {}).get("source_file", "")).strip()
            if not source_file:
                raise KeyError(f"Sample {sample.get('sample_id')!r} is missing context.source_file")
            by_source_file[source_file].append(sample)

        subset_output_dir = output_root / dataset_name
        subset_output_dir.mkdir(parents=True, exist_ok=True)
        split_payloads: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
        per_file_window_counts: dict[str, dict[str, int]] = {}

        for source_file in sorted(by_source_file):
            source_samples = by_source_file[source_file]
            sample_split_map = _assign_windows_within_file(
                source_samples,
                train_ratio=float(args.train_ratio),
                val_ratio=float(args.val_ratio),
            )
            per_file_window_counts[source_file] = dict(Counter(sample_split_map.values()))

            for sample in source_samples:
                old_sample_id = str(sample["sample_id"])
                target_split = sample_split_map[old_sample_id]
                target_sample_id = _build_target_sample_id(sample, target_split=target_split)
                prepared_sample = _prepare_sample(
                    sample,
                    target_split=target_split,
                    target_sample_id=target_sample_id,
                    plain_image_subdir=args.plain_image_subdir,
                )
                split_payloads[target_split].append(prepared_sample)

                plain_source = _resolve_existing_image(
                    subset_source_dir,
                    image_subdir=args.plain_image_subdir,
                    sample_id=old_sample_id,
                    stored_image_path=str(sample.get("image_path", "")).strip() or None,
                )
                if plain_source is None:
                    raise FileNotFoundError(
                        f"Unable to locate plain image for sample {old_sample_id!r} under {subset_source_dir}"
                    )
                plain_target = subset_output_dir / args.plain_image_subdir / target_split / f"{target_sample_id}.png"
                _materialize_file(
                    plain_source,
                    plain_target,
                    mode=args.link_mode,
                    overwrite=bool(args.overwrite_images),
                )

        subset_manifest: dict[str, Any] = {
            "source_root": str(subset_source_dir),
            "subset_dataset": dataset_name,
            "split_strategy": "split_windows_within_each_source_file",
            "num_source_files": len(by_source_file),
            "train_ratio": float(args.train_ratio),
            "val_ratio": float(args.val_ratio),
            "test_ratio": float(1.0 - args.train_ratio - args.val_ratio),
            "per_file_window_counts": per_file_window_counts,
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
            f"{dataset_name}: source_files train/val/test coverage = "
            f"{subset_manifest['splits']['train']['num_source_files']}/"
            f"{subset_manifest['splits']['val']['num_source_files']}/"
            f"{subset_manifest['splits']['test']['num_source_files']}"
        )

    (output_root / "manifest.json").write_text(
        json.dumps(top_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote rebuilt subset roots to: {output_root}")


if __name__ == "__main__":
    main()
