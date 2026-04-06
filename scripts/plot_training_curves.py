from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot training curves from a history.json file.")
    parser.add_argument("--history", type=str, required=True, help="Path to history.json")
    parser.add_argument("--output", type=str, default=None, help="Output PNG path")
    parser.add_argument("--title", type=str, default=None, help="Optional plot title")
    parser.add_argument("--include-bc", action="store_true", help="Force plotting train_bc when present.")
    parser.add_argument("--exclude-bc", action="store_true", help="Force hiding train_bc from the component panel.")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    root = Path(__file__).resolve().parents[1]
    sys.path.append(str(root / "src"))
    os.environ.setdefault("PYTHONNOUSERSITE", "1")
    os.environ.setdefault("MPLCONFIGDIR", str((root / ".mplconfig").resolve()))
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    from ts_grounder.plotting import save_training_curves_from_history_file

    history_path = Path(args.history)
    output_path = Path(args.output) if args.output else history_path.with_name("training_curves.png")
    include_bc = None
    if args.include_bc and args.exclude_bc:
        raise ValueError("Use only one of --include-bc or --exclude-bc.")
    if args.include_bc:
        include_bc = True
    if args.exclude_bc:
        include_bc = False
    save_training_curves_from_history_file(history_path, output_path, title=args.title, include_bc=include_bc)
    print(f"Saved training curves to: {output_path}")


if __name__ == "__main__":
    main()
