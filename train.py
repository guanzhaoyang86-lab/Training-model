from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parent / "src"))

from ts_grounder.data.dataset import GrounderDataset, grounder_collate
from ts_grounder.losses import LossWeights
from ts_grounder.models.grounder import Grounder
from ts_grounder.trainer import save_checkpoint, save_history, train_one_epoch, validate
from ts_grounder.utils import ensure_dir, load_yaml, set_seed


# 从配置文件启动训练。
def main() -> None:
    parser = argparse.ArgumentParser(description="Train TS Grounder")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    set_seed(cfg["seed"])

    output_dir = ensure_dir(cfg["output_dir"])
    torch.backends.mkldnn.enabled = False
    torch.set_num_threads(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = GrounderDataset(
        root=cfg["data"]["train_dir"],
        image_size=tuple(cfg["data"]["image_size"]),
        local_window=cfg["data"]["local_window"],
        enable_aug=True,
        seed=cfg["seed"],
    )
    val_dataset = GrounderDataset(
        root=cfg["data"]["val_dir"],
        image_size=tuple(cfg["data"]["image_size"]),
        local_window=cfg["data"]["local_window"],
        enable_aug=False,
        seed=cfg["seed"],
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["train"]["num_workers"],
        collate_fn=grounder_collate,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["train"]["num_workers"],
        collate_fn=grounder_collate,
    )

    model = Grounder(
        input_dim=cfg["model"]["input_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
        num_types=cfg["model"]["num_types"],
        text_layers=cfg["model"]["text_layers"],
        dropout=cfg["model"]["dropout"],
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg["train"]["epochs"],
        eta_min=cfg["train"]["min_lr"],
    )

    weights = LossWeights(**cfg["loss_weights"])
    best_score = float("-inf")
    history = []

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        train_stats = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            weights=weights,
            grad_clip=cfg["train"]["grad_clip"],
        )
        val_stats = validate(
            model=model,
            loader=val_loader,
            device=device,
            weights=weights,
        )
        scheduler.step()

        merged = {f"train_{k}": v for k, v in train_stats.items()}
        merged.update({f"val_{k}": v for k, v in val_stats.items()})
        merged["epoch"] = epoch
        merged["lr"] = optimizer.param_groups[0]["lr"]
        history.append(merged)

        print(
            f"[Epoch {epoch:03d}] "
            f"train_loss={train_stats['loss']:.4f} "
            f"val_loss={val_stats['loss']:.4f} "
            f"val_point_f1={val_stats['point_f1']:.4f} "
            f"val_event_f1={val_stats['event_f1']:.4f}"
        )

        current_score = val_stats["event_f1"]
        if current_score > best_score:
            best_score = current_score
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=val_stats,
                path=output_dir / "best.pt",
            )

        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            metrics=val_stats,
            path=output_dir / "last.pt",
        )
        save_history(history, output_dir / "history.json")


if __name__ == "__main__":
    os.environ.setdefault("PYTHONHASHSEED", "0")
    main()
