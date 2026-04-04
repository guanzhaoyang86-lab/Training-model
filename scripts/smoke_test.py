from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


# 用 toy data 跑一个最小训练闭环，确保工程基本可运行。
def main() -> None:
    root = Path("_toy_run")
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        [
            sys.executable,
            "tools/make_toy_dataset.py",
            "--output",
            str(root / "data"),
            "--train_size",
            "16",
            "--val_size",
            "4",
            "--length",
            "256",
        ],
        check=True,
    )

    config_path = root / "smoke.yaml"
    config_path.write_text(
        """
seed: 2026
output_dir: _toy_run/outputs

data:
  train_dir: _toy_run/data/train
  val_dir: _toy_run/data/val
  image_size: [224, 224]
  local_window: 25

model:
  input_dim: 4
  hidden_dim: 64
  num_types: 4
  text_layers: 2
  dropout: 0.1

train:
  epochs: 1
  batch_size: 4
  num_workers: 0
  lr: 0.0005
  min_lr: 0.00001
  weight_decay: 0.0001
  grad_clip: 1.0

loss_weights:
  point: 1.0
  seg: 1.0
  type: 1.0
  bc: 1.0
  cons: 0.2
  tv: 0.05
""".strip(),
        encoding="utf-8",
    )

    subprocess.run([sys.executable, "train.py", "--config", str(config_path)], check=True)
    print("Smoke test finished.")


if __name__ == "__main__":
    main()
