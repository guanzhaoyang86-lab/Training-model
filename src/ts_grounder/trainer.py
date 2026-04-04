from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader

from .losses import LossWeights, compute_total_loss
from .metrics import event_f1, point_f1_from_logits
from .utils import dump_json


# 将 batch 中的 tensor 挪到 device，其它字段保持原样。
def move_batch_to_device(batch: Dict, device: torch.device) -> Dict:
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


# 单个 epoch 训练。
def train_one_epoch(
    model,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    weights: LossWeights,
    grad_clip: float,
) -> Dict[str, float]:
    model.train()
    stats_sum = defaultdict(float)
    steps = 0

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)

        outputs = model(batch["text_features"], batch["images"], batch["text_valid"])
        aug_outputs = None
        if "aug_text_features" in batch:
            aug_outputs = model(batch["aug_text_features"], batch["aug_images"], batch["aug_text_valid"])

        loss, stats = compute_total_loss(outputs, batch, model, weights, aug_outputs=aug_outputs)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        for key, value in stats.items():
            stats_sum[key] += value
        steps += 1

    return {key: value / max(steps, 1) for key, value in stats_sum.items()}


# 验证：同时输出 point F1 与 event F1。
@torch.no_grad()
def validate(
    model,
    loader: DataLoader,
    device: torch.device,
    weights: LossWeights,
) -> Dict[str, float]:
    model.eval()
    loss_sum = defaultdict(float)
    point_metric_sum = defaultdict(float)
    event_metric_sum = defaultdict(float)
    steps = 0

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        outputs = model(batch["text_features"], batch["images"], batch["text_valid"])
        loss, stats = compute_total_loss(outputs, batch, model, weights, aug_outputs=None)
        decoded = model.decode(outputs, batch["deseasonalized"], batch["text_valid"])

        point_metrics = point_f1_from_logits(outputs["point_logits"], batch["point_mask"], batch["text_valid"])
        gt_segments = []
        for seg, valid in zip(batch["segments"], batch["segment_valid"]):
            count = int(valid.sum().item())
            gt_segments.append(seg[:count].cpu().numpy())
        event_metrics = event_f1(decoded, gt_segments)

        for key, value in stats.items():
            loss_sum[key] += value
        for key, value in point_metrics.items():
            point_metric_sum[f"point_{key}"] += value
        for key, value in event_metrics.items():
            event_metric_sum[f"event_{key}"] += value
        steps += 1

    merged = {key: value / max(steps, 1) for key, value in loss_sum.items()}
    merged.update({key: value / max(steps, 1) for key, value in point_metric_sum.items()})
    merged.update({key: value / max(steps, 1) for key, value in event_metric_sum.items()})
    return merged


# 保存 checkpoint 与训练日志。
def save_checkpoint(model, optimizer, epoch: int, metrics: Dict[str, float], path: str | Path) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "metrics": metrics,
        },
        path,
    )


# 把每个 epoch 的统计量写到 JSON，方便复查。
def save_history(history: list[Dict[str, float]], path: str | Path) -> None:
    dump_json({"history": history}, path)
