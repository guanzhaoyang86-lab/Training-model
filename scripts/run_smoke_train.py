from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate tiny toy data and run a lightweight smoke training job."
    )
    parser.add_argument("--output-root", type=str, default="_smoke_run")
    parser.add_argument("--train-size", type=int, default=16)
    parser.add_argument("--val-size", type=int, default=4)
    parser.add_argument("--length", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--image-size", type=int, nargs=2, default=(224, 224))
    parser.add_argument("--local-window", type=int, default=25)
    parser.add_argument(
        "--skip-data",
        action="store_true",
        help="Reuse an existing smoke dataset under output-root/data.",
    )
    return parser


def prepare_env(root: Path) -> dict[str, str]:
    env = os.environ.copy()
    src_dir = str((root / "src").resolve())
    current_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = src_dir if not current_pythonpath else f"{src_dir}:{current_pythonpath}"
    env.setdefault("PYTHONNOUSERSITE", "1")
    env.setdefault("MPLCONFIGDIR", str((root / ".mplconfig").resolve()))
    Path(env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    return env


def format_yaml_number(value: int | float) -> str:
    if isinstance(value, int):
        return str(value)
    return format(value, ".10f").rstrip("0").rstrip(".")


def write_smoke_config(
    config_path: Path,
    *,
    seed: int,
    data_root: Path,
    output_dir: Path,
    image_size: tuple[int, int],
    local_window: int,
    hidden_dim: int,
    epochs: int,
    batch_size: int,
    lr: float,
    min_lr: float,
    weight_decay: float,
    grad_clip: float,
) -> None:
    config_path.write_text(
        f"""
seed: {seed}
output_dir: {output_dir.as_posix()}

data:
  train_dir: {data_root.joinpath("train").as_posix()}
  val_dir: {data_root.joinpath("val").as_posix()}
  raw_dataset_root: {data_root.as_posix()}
  image_size: [{image_size[0]}, {image_size[1]}]
  local_window: {local_window}

model:
  input_dim: 4
  hidden_dim: {hidden_dim}
  num_types: 4
  text_layers: 2
  dropout: 0.1

train:
  epochs: {epochs}
  batch_size: {batch_size}
  num_workers: 0
  lr: {format_yaml_number(lr)}
  min_lr: {format_yaml_number(min_lr)}
  weight_decay: {format_yaml_number(weight_decay)}
  grad_clip: {format_yaml_number(grad_clip)}

loss_weights:
  point: 1.0
  seg: 1.0
  type: 1.0
  evidence: 0.5
  bc: 1.0
  cons: 0.2
  tv: 0.05
""".strip()
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = build_parser().parse_args()

    root = Path(__file__).resolve().parents[1]
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = root / output_root

    data_root = output_root / "data"
    train_output_dir = output_root / "outputs"
    config_path = output_root / "smoke.yaml"
    output_root.mkdir(parents=True, exist_ok=True)

    env = prepare_env(root)

    if not args.skip_data:
        subprocess.run(
            [
                sys.executable,
                "tools/make_toy_dataset.py",
                "--output",
                str(data_root),
                "--train_size",
                str(args.train_size),
                "--val_size",
                str(args.val_size),
                "--length",
                str(args.length),
                "--seed",
                str(args.seed),
            ],
            cwd=root,
            env=env,
            check=True,
        )

    write_smoke_config(
        config_path,
        seed=args.seed,
        data_root=data_root,
        output_dir=train_output_dir,
        image_size=tuple(args.image_size),
        local_window=args.local_window,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        min_lr=args.min_lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
    )

    subprocess.run(
        [sys.executable, "train.py", "--config", str(config_path)],
        cwd=root,
        env=env,
        check=True,
    )

    print(f"Smoke training finished. Config: {config_path}")
    print(f"Outputs saved to: {train_output_dir}")


if __name__ == "__main__":
    main()
