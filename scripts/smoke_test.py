from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from ts_grounder.vlm_eval import load_jsonl, parse_generated_result


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the prepared VLM smoke dataset.")
    parser.add_argument("--dataset-jsonl", type=str, default="_vlm_smoke_run/dataset_cache/val.jsonl")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    dataset_jsonl = Path(args.dataset_jsonl)
    if not dataset_jsonl.is_absolute():
        dataset_jsonl = repo_root / dataset_jsonl

    records = load_jsonl(dataset_jsonl)
    if not records:
        raise ValueError(f"空数据集：{dataset_jsonl}")
    for record in records[: min(16, len(records))]:
        if record.get("metadata", {}).get("task_type") == "qa":
            if record["assistant_text"] != record.get("target", {}).get("answer"):
                raise ValueError(f"QA target mismatch for {record['id']}")
        else:
            parsed = parse_generated_result(record["assistant_text"])
            if parsed.to_dict() != record["target"]:
                raise ValueError(f"schema round-trip failed for {record['id']}")
    print(f"Smoke test passed for {len(records)} records: {dataset_jsonl}")


if __name__ == "__main__":
    main()
