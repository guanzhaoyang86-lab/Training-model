from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare a local VLM smoke run with the current repository dataset.")
    parser.add_argument("--config", type=str, default="configs/vlm_smoke.yaml")
    parser.add_argument("--output-root", type=str, default="_vlm_smoke_run")
    parser.add_argument("--source-root", type=str, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = repo_root / output_root
    output_root.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        "train.py",
        "--config",
        args.config,
        "--output-dir",
        str(output_root),
        "--mode",
        "prepare_only",
    ]
    env = None
    if args.source_root:
        env = dict(**{"TS_GROUNDER_SOURCE_ROOT": args.source_root})
    subprocess.run(command, cwd=repo_root, check=True, env=env)
    print(f"Smoke preparation finished: {output_root}")


if __name__ == "__main__":
    main()
