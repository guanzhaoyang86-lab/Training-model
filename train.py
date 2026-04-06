from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parent / "src"))

from ts_grounder.data.dataset import GrounderDataset, grounder_collate
from ts_grounder.losses import LossWeights
from ts_grounder.models.grounder import Grounder
from ts_grounder.plotting import save_training_curves
from ts_grounder.trainer import save_checkpoint, save_history, train_one_epoch, validate
from ts_grounder.utils import dump_json, dump_yaml, ensure_dir, load_yaml, set_seed


def _override(cfg: dict, section: str | None, key: str, value: object) -> None:
    if value is None:
        return
    if section is None:
        cfg[key] = value
        return
    cfg[section][key] = value


# 从配置文件启动训练。
def main() -> None:
    parser = argparse.ArgumentParser(description="Train TS Grounder")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--save-config", type=str, default=None)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-dir", type=str)
    parser.add_argument("--train-dir", type=str)
    parser.add_argument("--val-dir", type=str)
    parser.add_argument("--raw-dataset-root", type=str)
    parser.add_argument("--image-size", type=int, nargs=2, metavar=("HEIGHT", "WIDTH"))
    parser.add_argument("--local-window", type=int)
    parser.add_argument("--hidden-dim", type=int)
    parser.add_argument("--text-layers", type=int)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--min-lr", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--grad-clip", type=float)
    parser.add_argument("--point-weight", type=float)
    parser.add_argument("--seg-weight", type=float)
    parser.add_argument("--type-weight", type=float)
    parser.add_argument("--evidence-weight", type=float)
    parser.add_argument("--bc-weight", type=float)
    parser.add_argument("--cons-weight", type=float)
    parser.add_argument("--tv-weight", type=float)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    _override(cfg, None, "seed", args.seed)
    _override(cfg, None, "output_dir", args.output_dir)
    _override(cfg, "data", "train_dir", args.train_dir)
    _override(cfg, "data", "val_dir", args.val_dir)
    _override(cfg, "data", "raw_dataset_root", args.raw_dataset_root)
    _override(cfg, "data", "image_size", list(args.image_size) if args.image_size else None)
    _override(cfg, "data", "local_window", args.local_window)
    _override(cfg, "model", "hidden_dim", args.hidden_dim)
    _override(cfg, "model", "text_layers", args.text_layers)
    _override(cfg, "model", "dropout", args.dropout)
    _override(cfg, "train", "epochs", args.epochs)
    _override(cfg, "train", "batch_size", args.batch_size)
    _override(cfg, "train", "num_workers", args.num_workers)
    _override(cfg, "train", "lr", args.lr)
    _override(cfg, "train", "min_lr", args.min_lr)
    _override(cfg, "train", "weight_decay", args.weight_decay)
    _override(cfg, "train", "grad_clip", args.grad_clip)
    _override(cfg, "loss_weights", "point", args.point_weight)
    _override(cfg, "loss_weights", "seg", args.seg_weight)
    _override(cfg, "loss_weights", "type", args.type_weight)
    _override(cfg, "loss_weights", "evidence", args.evidence_weight)
    _override(cfg, "loss_weights", "bc", args.bc_weight)
    _override(cfg, "loss_weights", "cons", args.cons_weight)
    _override(cfg, "loss_weights", "tv", args.tv_weight)

    set_seed(cfg["seed"])

    output_dir = ensure_dir(cfg["output_dir"])
    resolved_config_path = Path(args.save_config) if args.save_config else output_dir / "resolved_config.yaml"
    ensure_dir(resolved_config_path.parent)
    dump_yaml(cfg, resolved_config_path)

    torch.backends.mkldnn.enabled = False
    torch.set_num_threads(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enable_aug = float(cfg["loss_weights"]["bc"]) > 0.0

    train_dataset = GrounderDataset(
        root=cfg["data"]["train_dir"],
        image_size=tuple(cfg["data"]["image_size"]),
        local_window=cfg["data"]["local_window"],
        enable_aug=enable_aug,
        seed=cfg["seed"],
        raw_dataset_root=cfg["data"].get("raw_dataset_root"),
    )
    val_dataset = GrounderDataset(
        root=cfg["data"]["val_dir"],
        image_size=tuple(cfg["data"]["image_size"]),
        local_window=cfg["data"]["local_window"],
        enable_aug=False,
        seed=cfg["seed"],
        raw_dataset_root=cfg["data"].get("raw_dataset_root"),
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
    best_epoch = 0
    best_record = {}
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
            f"val_event_f1={val_stats['event_f1']:.4f}",
            flush=True,
        )

        current_score = val_stats["event_f1"]
        if current_score > best_score:
            best_score = current_score
            best_epoch = epoch
            best_record = dict(merged)
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

    curves_path = output_dir / "training_curves.png"
    plot_error = None
    try:
        save_training_curves(history, curves_path, include_bc=enable_aug)
    except Exception as exc:
        plot_error = str(exc)
        print(f"[Warning] Failed to write training curves: {exc}", flush=True)

    summary = {
        "selection_metric": "val_event_f1",
        "best_epoch": int(best_epoch),
        "best_score": float(best_score),
        "best_checkpoint": str((output_dir / "best.pt").resolve()),
        "last_checkpoint": str((output_dir / "last.pt").resolve()),
        "history_path": str((output_dir / "history.json").resolve()),
        "training_curves_path": str(curves_path.resolve()),
        "config_path": str(resolved_config_path.resolve()),
        "config_source_path": str(Path(args.config).resolve()),
        "output_dir": str(output_dir.resolve()),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "data": dict(cfg["data"]),
        "model": dict(cfg["model"]),
        "train": dict(cfg["train"]),
        "loss_weights": dict(cfg["loss_weights"]),
        "augmentation_enabled": bool(enable_aug),
        "best_metrics": best_record,
    }
    if plot_error is not None:
        summary["training_curves_error"] = plot_error
    dump_json(summary, output_dir / "summary.json")


if __name__ == "__main__":
    os.environ.setdefault("PYTHONHASHSEED", "0")
    main()
