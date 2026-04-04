from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.nn.functional as F

from .data.augmentations import TransformMetadata


@dataclass
class LossWeights:
    point: float = 1.0
    seg: float = 1.0
    type: float = 1.0
    evidence: float = 0.5
    bc: float = 1.0
    cons: float = 0.2
    tv: float = 0.05


# 只在有效时间步上计算 BCE。
def masked_bce_with_logits(logits: torch.Tensor, targets: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    return (loss * valid).sum() / valid.sum().clamp_min(1.0)


# 将变换后长度的序列通过 alpha_tau^{-1} 对齐回原始长度。
def warp_to_source_length(sequence: torch.Tensor, meta: TransformMetadata) -> torch.Tensor:
    if meta.kind != "resample":
        return sequence
    warped = F.interpolate(
        sequence.unsqueeze(0).unsqueeze(0),
        size=meta.source_length,
        mode="linear",
        align_corners=True,
    )
    return warped.squeeze(0).squeeze(0)


# 计算概率分布的期望坐标，用于可微分的 boundary 对齐。
def expected_coordinate(prob: torch.Tensor) -> torch.Tensor:
    grid = torch.arange(prob.numel(), device=prob.device, dtype=prob.dtype)
    mass = prob.sum().clamp_min(1e-6)
    return (prob * grid).sum() / mass


# 在边界附近的局部窗口内计算期望坐标，避免多异常样本里不同事件互相干扰。
def localized_expected_coordinate(prob: torch.Tensor, left: int, right: int) -> torch.Tensor:
    offset = prob.new_tensor(float(left))
    return offset + expected_coordinate(prob[left : right + 1])


# 按异常段长度自适应设置边界窗口，让短异常和长异常都能稳定地做局部对齐。
def boundary_window_radius(start: int, end: int) -> int:
    segment_length = end - start + 1
    return max(2, int(round(segment_length * 0.25)))


def boundary_window(center: int, radius: int, length: int) -> tuple[int, int]:
    return max(0, center - radius), min(length - 1, center + radius)


# 用 GT 区间作为监督锚点，在边界附近做可微分的坐标回归。
def segment_interval_loss(
    start_logits: torch.Tensor,
    end_logits: torch.Tensor,
    segments: torch.Tensor,
    segment_valid: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    start_prob = torch.sigmoid(start_logits)
    end_prob = torch.sigmoid(end_logits)

    interval_terms = []
    batch_size, max_segments, _ = segments.shape
    for b in range(batch_size):
        source_len = int(valid[b].sum().item())
        for k in range(max_segments):
            if segment_valid[b, k] <= 0:
                continue

            start = int(segments[b, k, 0].item())
            end = int(segments[b, k, 1].item())
            radius = boundary_window_radius(start, end)

            start_left, start_right = boundary_window(start, radius, source_len)
            end_left, end_right = boundary_window(end, radius, source_len)

            pred_start = localized_expected_coordinate(start_prob[b, :source_len], start_left, start_right)
            pred_end = localized_expected_coordinate(end_prob[b, :source_len], end_left, end_right)

            target_start = start_prob.new_tensor(float(start))
            target_end = end_prob.new_tensor(float(end))
            interval_terms.append(torch.abs(pred_start - target_start) + torch.abs(pred_end - target_end))

    if not interval_terms:
        return start_logits.new_tensor(0.0)
    return torch.stack(interval_terms).mean()


@dataclass
class PredictedInterval:
    start: torch.Tensor
    end: torch.Tensor


def map_coordinate_between_lengths(coord: torch.Tensor, from_length: int, to_length: int) -> torch.Tensor:
    if from_length <= 1 or to_length <= 1:
        return coord.new_tensor(0.0)
    if from_length == to_length:
        return coord
    scale = coord.new_tensor(float(to_length - 1) / float(from_length - 1))
    upper = coord.new_tensor(float(max(to_length - 1, 0)))
    return torch.clamp(coord * scale, min=0.0, max=upper)


def decode_binary_segments(
    point_prob: torch.Tensor,
    threshold: float = 0.5,
    min_segment_length: int = 1,
) -> List[tuple[int, int]]:
    binary = point_prob.detach() >= threshold
    segments = []
    start = None
    for idx, flag in enumerate(binary.tolist()):
        if flag and start is None:
            start = idx
        if (not flag) and start is not None:
            if idx - start >= min_segment_length:
                segments.append((start, idx - 1))
            start = None
    if start is not None and point_prob.numel() - start >= min_segment_length:
        segments.append((start, point_prob.numel() - 1))
    return segments


def extract_predicted_intervals(
    point_prob: torch.Tensor,
    start_prob: torch.Tensor,
    end_prob: torch.Tensor,
    threshold: float = 0.5,
    min_segment_length: int = 1,
) -> List[PredictedInterval]:
    intervals = []
    for start, end in decode_binary_segments(point_prob, threshold=threshold, min_segment_length=min_segment_length):
        radius = boundary_window_radius(start, end)
        start_left, start_right = boundary_window(start, radius, point_prob.numel())
        end_left, end_right = boundary_window(end, radius, point_prob.numel())

        start_coord = localized_expected_coordinate(start_prob, start_left, start_right)
        end_coord = localized_expected_coordinate(end_prob, end_left, end_right)
        if float(end_coord.detach().item()) < float(start_coord.detach().item()):
            start_coord, end_coord = end_coord, start_coord
        intervals.append(PredictedInterval(start=start_coord, end=end_coord))
    return intervals


def map_intervals_to_source(intervals: List[PredictedInterval], meta: TransformMetadata) -> List[PredictedInterval]:
    mapped = []
    for interval in intervals:
        mapped.append(
            PredictedInterval(
                start=map_coordinate_between_lengths(interval.start, meta.target_length, meta.source_length),
                end=map_coordinate_between_lengths(interval.end, meta.target_length, meta.source_length),
            )
        )
    return mapped


def interval_iou(a: PredictedInterval, b: PredictedInterval) -> float:
    a_start = float(a.start.detach().item())
    a_end = float(a.end.detach().item())
    b_start = float(b.start.detach().item())
    b_end = float(b.end.detach().item())
    intersection = max(0.0, min(a_end, b_end) - max(a_start, b_start))
    union = max(a_end, b_end) - min(a_start, b_start)
    return 0.0 if union <= 0.0 else intersection / union


def interval_center_distance(a: PredictedInterval, b: PredictedInterval) -> float:
    a_center = 0.5 * float((a.start + a.end).detach().item())
    b_center = 0.5 * float((b.start + b.end).detach().item())
    return abs(a_center - b_center)


def match_interval_pairs(
    origin_intervals: List[PredictedInterval],
    warped_intervals: List[PredictedInterval],
) -> List[tuple[PredictedInterval, PredictedInterval]]:
    matches = []
    next_search_start = 0
    for origin in origin_intervals:
        best_index = None
        best_iou = -1.0
        best_center_gap = float("inf")
        for idx in range(next_search_start, len(warped_intervals)):
            candidate = warped_intervals[idx]
            candidate_iou = interval_iou(origin, candidate)
            candidate_gap = interval_center_distance(origin, candidate)
            if candidate_iou > best_iou or (
                abs(candidate_iou - best_iou) <= 1e-6 and candidate_gap < best_center_gap
            ):
                best_index = idx
                best_iou = candidate_iou
                best_center_gap = candidate_gap
        if best_index is None:
            continue
        matches.append((origin, warped_intervals[best_index]))
        next_search_start = best_index + 1
        if next_search_start >= len(warped_intervals):
            break
    return matches


def gt_anchored_interval_consistency(
    origin_start: torch.Tensor,
    origin_end: torch.Tensor,
    aug_start: torch.Tensor,
    aug_end: torch.Tensor,
    meta: TransformMetadata,
    source_len: int,
    segments: torch.Tensor,
    segment_valid: torch.Tensor,
) -> List[torch.Tensor]:
    warped_start = warp_to_source_length(aug_start, meta)
    warped_end = warp_to_source_length(aug_end, meta)
    terms = []

    max_segments = segments.size(0)
    for k in range(max_segments):
        if segment_valid[k] <= 0:
            continue

        start = int(segments[k, 0].item())
        end = int(segments[k, 1].item())
        radius = boundary_window_radius(start, end)

        start_left, start_right = boundary_window(start, radius, source_len)
        end_left, end_right = boundary_window(end, radius, source_len)

        origin_start_coord = localized_expected_coordinate(origin_start, start_left, start_right)
        warped_start_coord = localized_expected_coordinate(warped_start, start_left, start_right)
        origin_end_coord = localized_expected_coordinate(origin_end, end_left, end_right)
        warped_end_coord = localized_expected_coordinate(warped_end, end_left, end_right)

        terms.append(torch.abs(origin_start_coord - warped_start_coord) + torch.abs(origin_end_coord - warped_end_coord))
    return terms


# Boundary-consistent loss：点级比较 warped mask，区间级优先比较映射回原轴后的预测起止点。
def boundary_consistency_loss(
    outputs: Dict[str, torch.Tensor],
    aug_outputs: Dict[str, torch.Tensor],
    valid: torch.Tensor,
    aug_valid: torch.Tensor,
    transforms: List[TransformMetadata],
    segments: torch.Tensor,
    segment_valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    point_prob = torch.sigmoid(outputs["point_logits"])
    start_prob = torch.sigmoid(outputs["start_logits"])
    end_prob = torch.sigmoid(outputs["end_logits"])

    aug_point_prob = torch.sigmoid(aug_outputs["point_logits"])
    aug_start_prob = torch.sigmoid(aug_outputs["start_logits"])
    aug_end_prob = torch.sigmoid(aug_outputs["end_logits"])

    bc_mask_terms = []
    bc_int_terms = []

    batch_size = point_prob.size(0)
    for b in range(batch_size):
        source_len = int(valid[b].sum().item())
        target_len = int(aug_valid[b].sum().item())
        meta = transforms[b]

        origin_mask = point_prob[b, :source_len]
        origin_start = start_prob[b, :source_len]
        origin_end = end_prob[b, :source_len]

        aug_mask = aug_point_prob[b, :target_len]
        aug_start = aug_start_prob[b, :target_len]
        aug_end = aug_end_prob[b, :target_len]

        warped_mask = warp_to_source_length(aug_mask, meta)
        bc_mask_terms.append(torch.abs(origin_mask - warped_mask).mean())

        origin_intervals = extract_predicted_intervals(origin_mask, origin_start, origin_end)
        warped_aug_intervals = map_intervals_to_source(
            extract_predicted_intervals(aug_mask, aug_start, aug_end),
            meta,
        )
        matched_pairs = match_interval_pairs(origin_intervals, warped_aug_intervals)
        if matched_pairs:
            sample_interval_loss = origin_mask.new_tensor(0.0)
            for origin_interval, warped_interval in matched_pairs:
                sample_interval_loss = sample_interval_loss + torch.abs(origin_interval.start - warped_interval.start) + torch.abs(origin_interval.end - warped_interval.end)
            bc_int_terms.append(sample_interval_loss)
            continue

        fallback_terms = gt_anchored_interval_consistency(
            origin_start=origin_start,
            origin_end=origin_end,
            aug_start=aug_start,
            aug_end=aug_end,
            meta=meta,
            source_len=source_len,
            segments=segments[b],
            segment_valid=segment_valid[b],
        )
        if fallback_terms:
            bc_int_terms.append(torch.stack(fallback_terms).sum())

    bc_mask = point_prob.new_tensor(0.0) if not bc_mask_terms else torch.stack(bc_mask_terms).mean()
    bc_int = point_prob.new_tensor(0.0) if not bc_int_terms else torch.stack(bc_int_terms).mean()
    return bc_mask + bc_int, bc_mask, bc_int


# 类型分类损失：在 GT 区间上池化，再做交叉熵。
def type_classification_loss(
    type_logits: torch.Tensor,
    types: torch.Tensor,
    segment_valid: torch.Tensor,
) -> torch.Tensor:
    batch_size, max_segments, num_types = type_logits.shape
    flat_logits = type_logits.reshape(batch_size * max_segments, num_types)
    flat_targets = types.reshape(batch_size * max_segments)
    flat_valid = segment_valid.reshape(batch_size * max_segments)
    ce = F.cross_entropy(flat_logits, flat_targets, reduction="none")
    return (ce * flat_valid).sum() / flat_valid.sum().clamp_min(1.0)


# 显式监督 evidence head，让模型直接学习 confidence / attributes。
def evidence_regression_loss(
    confidence_logits: torch.Tensor,
    attribute_values: torch.Tensor,
    confidence_targets: torch.Tensor,
    attribute_targets: torch.Tensor,
    segment_valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    confidence = masked_bce_with_logits(confidence_logits, confidence_targets, segment_valid)

    attr_loss = F.smooth_l1_loss(attribute_values, attribute_targets, reduction="none").mean(dim=-1)
    attribute = (attr_loss * segment_valid).sum() / segment_valid.sum().clamp_min(1.0)
    return confidence + attribute, confidence, attribute


# 文本模态与图像模态对同一异常证据不要差太远。
def cross_modal_consistency_loss(text_evidence: torch.Tensor, image_evidence: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(text_evidence, image_evidence)


# point-wise mask 的时间总变差正则。
def temporal_smoothness_loss(point_logits: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    point_prob = torch.sigmoid(point_logits)
    diff = torch.abs(point_prob[:, 1:] - point_prob[:, :-1])
    valid_diff = valid[:, 1:] * valid[:, :-1]
    return (diff * valid_diff).sum() / valid_diff.sum().clamp_min(1.0)


# 汇总总损失。
def compute_total_loss(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    model,
    weights: LossWeights,
    aug_outputs: Dict[str, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, Dict[str, float]]:
    point = masked_bce_with_logits(outputs["point_logits"], batch["point_mask"], batch["text_valid"])

    seg_start = masked_bce_with_logits(outputs["start_logits"], batch["start_target"], batch["text_valid"])
    seg_end = masked_bce_with_logits(outputs["end_logits"], batch["end_target"], batch["text_valid"])
    seg_interval = segment_interval_loss(
        outputs["start_logits"],
        outputs["end_logits"],
        batch["segments"],
        batch["segment_valid"],
        batch["text_valid"],
    )
    seg = seg_start + seg_end + seg_interval

    pooled_segments = model.pool_segments(outputs["fused_h"], batch["segments"], batch["segment_valid"])
    type_logits = model.classify_segments(pooled_segments)
    type_loss = type_classification_loss(type_logits, batch["types"], batch["segment_valid"])
    evidence_outputs = model.predict_segment_evidence(pooled_segments)
    evidence_loss, evidence_conf, evidence_attr = evidence_regression_loss(
        evidence_outputs["confidence_logits"],
        evidence_outputs["attribute_values"],
        batch["evidence_confidence"],
        batch["evidence_attributes"],
        batch["segment_valid"],
    )

    cons = cross_modal_consistency_loss(outputs["text_evidence"], outputs["image_evidence"])
    tv = temporal_smoothness_loss(outputs["point_logits"], batch["text_valid"])

    if aug_outputs is not None:
        bc, bc_mask, bc_int = boundary_consistency_loss(
            outputs,
            aug_outputs,
            batch["text_valid"],
            batch["aug_text_valid"],
            batch["transforms"],
            batch["segments"],
            batch["segment_valid"],
        )
    else:
        bc = point.new_tensor(0.0)
        bc_mask = point.new_tensor(0.0)
        bc_int = point.new_tensor(0.0)

    total = (
        weights.point * point
        + weights.seg * seg
        + weights.type * type_loss
        + weights.evidence * evidence_loss
        + weights.bc * bc
        + weights.cons * cons
        + weights.tv * tv
    )

    stats = {
        "loss": float(total.item()),
        "point": float(point.item()),
        "seg": float(seg.item()),
        "seg_boundary": float((seg_start + seg_end).item()),
        "seg_interval": float(seg_interval.item()),
        "type": float(type_loss.item()),
        "evidence": float(evidence_loss.item()),
        "evidence_conf": float(evidence_conf.item()),
        "evidence_attr": float(evidence_attr.item()),
        "bc": float(bc.item()),
        "bc_mask": float(bc_mask.item()),
        "bc_int": float(bc_int.item()),
        "cons": float(cons.item()),
        "tv": float(tv.item()),
    }
    return total, stats
