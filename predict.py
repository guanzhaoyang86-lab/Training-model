from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent / "src"))

from ts_grounder.utils import dump_json, load_yaml
from ts_grounder.vlm_training import generate_predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="Run VLM inference for TS Grounder")
    parser.add_argument("--config", type=str, default="configs/vlm_full.yaml")
    parser.add_argument("--dataset-jsonl", type=str, required=True)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--output", type=str, default="predictions.jsonl")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    _ = load_yaml(repo_root / args.config if not Path(args.config).is_absolute() else args.config)
    metrics = generate_predictions(
        model_path=args.model_path,
        dataset_jsonl=args.dataset_jsonl,
        output_path=args.output,
        max_new_tokens=args.max_new_tokens,
    )
    dump_json(metrics, Path(args.output).with_suffix(".metrics.json"))


if __name__ == "__main__":
    main()
