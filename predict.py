from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parent / "src"))

from ts_grounder.data.dataset import GrounderDataset, grounder_collate
from ts_grounder.evidence_builder import build_evidence_package
from ts_grounder.models.grounder import Grounder
from ts_grounder.utils import load_yaml


# 用训练好的 grounder 对样本做推理，并输出结构化异常证据。
def main() -> None:
    parser = argparse.ArgumentParser(description="Predict with TS Grounder")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--input", type=str, required=True, help="单个 .npz 文件，或包含多个 .npz 的目录")
    parser.add_argument("--output", type=str, default="predictions.json")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    torch.backends.mkldnn.enabled = False
    torch.set_num_threads(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Grounder(
        input_dim=cfg["model"]["input_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
        num_types=cfg["model"]["num_types"],
        text_layers=cfg["model"]["text_layers"],
        dropout=cfg["model"]["dropout"],
    ).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    input_path = Path(args.input)
    temp_dir = None
    if input_path.is_file():
        temp_dir = input_path.parent
        dataset_root = temp_dir
        target_name = input_path.stem
    else:
        dataset_root = input_path
        target_name = None

    dataset = GrounderDataset(
        root=dataset_root,
        image_size=tuple(cfg["data"]["image_size"]),
        local_window=cfg["data"]["local_window"],
        enable_aug=False,
        seed=cfg["seed"],
    )

    items = []
    for i in range(len(dataset)):
        sample = dataset[i]
        if target_name is not None and sample["id"] != target_name:
            continue
        batch = grounder_collate([sample])
        for key, value in list(batch.items()):
            if torch.is_tensor(value):
                batch[key] = value.to(device)
        outputs = model(batch["text_features"], batch["images"], batch["text_valid"])
        decoded = model.decode(outputs, batch["deseasonalized"], batch["text_valid"])[0]
        items.append(
            build_evidence_package(
                sample_id=sample["id"],
                evidences=decoded,
                outputs=outputs,
                batch_index=0,
                num_types=cfg["model"]["num_types"],
            )
        )

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
