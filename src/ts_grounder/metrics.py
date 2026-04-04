from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import numpy as np
import torch

from .models.grounder import SegmentEvidence


# 逐点 F1，便于快速验证训练是否收敛。
def point_f1_from_logits(
    point_logits: torch.Tensor,
    point_mask: torch.Tensor,
    valid: torch.Tensor,
    threshold: float = 0.5,
) -> dict[str, float]:
    prob = torch.sigmoid(point_logits)
    pred = (prob >= threshold).float() * valid
    target = point_mask * valid

    tp = float((pred * target).sum().item())
    fp = float((pred * (1.0 - target)).sum().item())
    fn = float(((1.0 - pred) * target).sum().item())

    precision = tp / max(tp + fp, 1e-8)
    recall = tp / max(tp + fn, 1e-8)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-8)
    return {"precision": precision, "recall": recall, "f1": f1}


# 区间 IoU。
def segment_iou(seg_a: Tuple[int, int], seg_b: Tuple[int, int]) -> float:
    left = max(seg_a[0], seg_b[0])
    right = min(seg_a[1], seg_b[1])
    inter = max(0, right - left + 1)
    union = (seg_a[1] - seg_a[0] + 1) + (seg_b[1] - seg_b[0] + 1) - inter
    return inter / max(union, 1)


# 事件级 F1：只要预测区间与 GT 区间有足够重叠，就算命中一个事件。
def event_f1(
    predicted: Sequence[Sequence[SegmentEvidence]],
    ground_truth: Sequence[np.ndarray],
    iou_threshold: float = 0.1,
) -> dict[str, float]:
    tp = 0
    fp = 0
    fn = 0

    for pred_segments, gt_segments in zip(predicted, ground_truth):
        matched = set()
        for pred in pred_segments:
            current = (pred.start, pred.end)
            hit = False
            for idx, gt in enumerate(gt_segments.tolist()):
                gt_tuple = (int(gt[0]), int(gt[1]))
                if idx in matched:
                    continue
                if segment_iou(current, gt_tuple) >= iou_threshold:
                    matched.add(idx)
                    hit = True
                    tp += 1
                    break
            if not hit:
                fp += 1
        fn += len(gt_segments) - len(matched)

    precision = tp / max(tp + fp, 1e-8)
    recall = tp / max(tp + fn, 1e-8)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-8)
    return {"precision": precision, "recall": recall, "f1": f1}
