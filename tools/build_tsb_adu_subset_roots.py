from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _load_split(source_root: Path, split: str) -> list[dict[str, Any]]:
    path = source_root / f"{split}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset split: {path}")
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


def _deterministic_order(items: list[str], *, salt: str) -> list[str]:
    return sorted(
        items,
        key=lambda item: (
            hashlib.sha1(f"{salt}:{item}".encode("utf-8")).hexdigest(),
            item,
        ),
    )


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
        train_count += n_items - assigned
    elif assigned > n_items:
        overflow = assigned - n_items
        reducible_train = max(0, train_count - 1)
        take_train = min(overflow, reducible_train)
        train_count -= take_train
        overflow -= take_train
        if overflow > 0:
            reducible_test = max(1, test_count - 1)
            take_test = min(overflow, reducible_test)
            test_count -= take_test
            overflow -= take_test
        if overflow > 0:
            raise ValueError(f"Split counts overflow for n_items={n_items}")

    return train_count, val_count, test_count


def _assign_source_file_splits(
    dataset_name: str,
    source_files: list[str],
    *,
    train_ratio: float,
    val_ratio: float,
) -> dict[str, str]:
    ordered = _deterministic_order(source_files, salt=dataset_name)
    train_count, val_count, test_count = _compute_split_counts(
        len(ordered),
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )
    split_map: dict[str, str] = {}

    cursor = 0
    for source_file in ordered[cursor : cursor + train_count]:
        split_map[source_file] = "train"
    cursor += train_count
    for source_file in ordered[cursor : cursor + val_count]:
        split_map[source_file] = "val"
    cursor += val_count
    for source_file in ordered[cursor : cursor + test_count]:
        split_map[source_file] = "test"
    return split_map


def _prepare_sample(sample: dict[str, Any], *, source_root: Path) -> dict[str, Any]:
    payload = json.loads(json.dumps(sample, ensure_ascii=False))
    stored_image_path = str(payload.get("image_path", "")).strip()
    if stored_image_path:
        image_path = Path(stored_image_path)
        if not image_path.is_absolute():
            payload["image_path"] = str((source_root / image_path).resolve())
    payload.setdefault("context", {})
    payload["context"]["original_split"] = sample.get("sample_id", "").split("_", 1)[0]
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build per-source-dataset TSB-AD-U subset roots with deterministic train/val/test splits."
    )
    parser.add_argument(
        "--source-root",
        type=str,
        default="/gpfs/projects/p33222/ybq9740/TSB-AD-U/windowed_vlm_dataset_256_128",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="/gpfs/projects/p33222/ybq9740/TSB-AD-U/TSB-AD-U-subset-splits",
    )
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--source-dataset-filter", action="append", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    source_root = Path(args.source_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    dataset_filters = _normalize_dataset_filters(args.source_dataset_filter)
    all_samples: list[dict[str, Any]] = []
    for split in ("train", "val", "test"):
        all_samples.extend(_load_split(source_root, split))

    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in all_samples:
        dataset_name = str(sample.get("context", {}).get("source_dataset", "")).strip()
        if not dataset_name:
            continue
        if dataset_filters and dataset_name not in dataset_filters:
            continue
        by_dataset[dataset_name].append(sample)

    top_manifest: dict[str, Any] = {
        "source_root": str(source_root),
        "output_root": str(output_root),
        "train_ratio": float(args.train_ratio),
        "val_ratio": float(args.val_ratio),
        "datasets": {},
    }

    for dataset_name in sorted(by_dataset):
        samples = by_dataset[dataset_name]
        source_files = sorted({str(sample["context"]["source_file"]) for sample in samples})
        split_map = _assign_source_file_splits(
            dataset_name,
            source_files,
            train_ratio=float(args.train_ratio),
            val_ratio=float(args.val_ratio),
        )
        subset_dir = output_root / dataset_name
        subset_dir.mkdir(parents=True, exist_ok=True)

        split_payloads: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
        for sample in samples:
            source_file = str(sample["context"]["source_file"])
            split = split_map[source_file]
            split_payloads[split].append(_prepare_sample(sample, source_root=source_root))

        for split in ("train", "val", "test"):
            records = sorted(
                split_payloads[split],
                key=lambda item: (
                    str(item.get("context", {}).get("source_file", "")),
                    int(item.get("context", {}).get("window_start_global", 0)),
                    str(item.get("sample_id", "")),
                ),
            )
            (subset_dir / f"{split}.json").write_text(
                json.dumps(records, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        subset_manifest = {
            "source_root": str(source_root),
            "subset_dataset": dataset_name,
            "num_source_files": len(source_files),
            "splits": {},
        }
        for split in ("train", "val", "test"):
            records = split_payloads[split]
            subset_manifest["splits"][split] = {
                "num_windows": len(records),
                "num_positive_windows": sum(1 for item in records if item.get("events")),
                "num_negative_windows": sum(1 for item in records if not item.get("events")),
                "num_source_files": len({str(item["context"]["source_file"]) for item in records}),
            }

        (subset_dir / "manifest.json").write_text(
            json.dumps(subset_manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        top_manifest["datasets"][dataset_name] = subset_manifest["splits"]

        source_file_counter = Counter(split_map.values())
        print(
            f"{dataset_name}: source_files train/val/test = "
            f"{source_file_counter.get('train', 0)}/"
            f"{source_file_counter.get('val', 0)}/"
            f"{source_file_counter.get('test', 0)}"
        )
        print(
            f"{dataset_name}: windows train/val/test = "
            f"{subset_manifest['splits']['train']['num_windows']}/"
            f"{subset_manifest['splits']['val']['num_windows']}/"
            f"{subset_manifest['splits']['test']['num_windows']}"
        )

    (output_root / "manifest.json").write_text(
        json.dumps(top_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote subset roots to: {output_root}")


if __name__ == "__main__":
    main()
