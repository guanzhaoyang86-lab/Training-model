from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _load_history(path: str | Path) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload["history"]


def save_training_curves_from_history_file(
    history_path: str | Path,
    output_path: str | Path,
    *,
    title: str | None = None,
    include_bc: bool | None = None,
) -> None:
    history = _load_history(history_path)
    save_training_curves(history, output_path, title=title, include_bc=include_bc)


def save_training_curves(
    history: list[dict[str, Any]],
    output_path: str | Path,
    *,
    title: str | None = None,
    include_bc: bool | None = None,
) -> None:
    if not history:
        raise ValueError("history is empty")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = [int(row["epoch"]) for row in history]
    best_epoch = max(history, key=lambda row: row.get("val_event_f1", float("-inf"))).get("epoch", epochs[-1])
    best_event = max(row.get("val_event_f1", float("-inf")) for row in history)

    fig, axes = plt.subplots(3, 1, figsize=(11, 12), sharex=True)
    fig.subplots_adjust(hspace=0.28)

    loss_ax = axes[0]
    loss_ax.plot(epochs, [row["train_loss"] for row in history], label="train_loss", linewidth=2.0)
    loss_ax.plot(epochs, [row["val_loss"] for row in history], label="val_loss", linewidth=2.0)
    loss_ax.axvline(best_epoch, color="tab:green", linestyle="--", linewidth=1.2, label=f"best epoch = {best_epoch}")
    loss_ax.set_ylabel("Total Loss")
    loss_ax.grid(True, alpha=0.25)
    loss_ax.legend(loc="best")

    component_ax = axes[1]
    component_keys = [
        ("train_bc", "train_bc"),
        ("train_seg", "train_seg"),
        ("train_point", "train_point"),
        ("train_evidence", "train_evidence"),
    ]
    for key, label in component_keys:
        if key == "train_bc" and include_bc is None:
            include_current = any(abs(float(row.get(key, 0.0))) > 1e-12 for row in history)
            if not include_current:
                continue
        elif key == "train_bc" and not include_bc:
            continue
        if key in history[0]:
            component_ax.plot(epochs, [row[key] for row in history], label=label, linewidth=1.8)
    component_ax.axvline(best_epoch, color="tab:green", linestyle="--", linewidth=1.2)
    component_ax.set_ylabel("Train Components")
    component_ax.grid(True, alpha=0.25)
    component_ax.legend(loc="best")

    metric_ax = axes[2]
    metric_keys = [
        ("val_event_f1", "val_event_f1"),
        ("val_point_f1", "val_point_f1"),
    ]
    for key, label in metric_keys:
        if key in history[0]:
            metric_ax.plot(epochs, [row[key] for row in history], label=label, linewidth=2.0)
    metric_ax.axvline(best_epoch, color="tab:green", linestyle="--", linewidth=1.2)
    metric_ax.set_xlabel("Epoch")
    metric_ax.set_ylabel("Validation Metrics")
    metric_ax.grid(True, alpha=0.25)
    metric_ax.legend(loc="best")

    fig.suptitle(
        title or f"Training Curves (best val_event_f1 = {best_event:.4f} @ epoch {best_epoch})",
        fontsize=14,
    )
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
