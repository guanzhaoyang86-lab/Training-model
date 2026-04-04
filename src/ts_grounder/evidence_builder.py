from __future__ import annotations

from typing import Sequence

import torch

from .models.grounder import SegmentEvidence
from .taxonomy import type_name_from_id


# 先把段级 evidence 统一整理成可序列化的结构，作为最终证据集合 E。
def build_evidence_list(evidences: Sequence[SegmentEvidence]) -> list[dict]:
    return [
        {
            "start": ev.start,
            "end": ev.end,
            "type": ev.anomaly_type,
            "type_name": type_name_from_id(ev.anomaly_type),
            "confidence": ev.confidence,
            "attributes": {
                "length": ev.length,
                "peak_amplitude": ev.peak_amplitude,
                "mean_deviation": ev.mean_deviation,
            },
        }
        for ev in evidences
    ]


# 连续度看的是异常覆盖长度占整体异常跨度的比例；越接近 1 表示越连续，越接近 0 表示越碎片化。
def compute_continuity_features(evidences: Sequence[SegmentEvidence]) -> tuple[float, float]:
    if not evidences:
        return 0.0, 0.0

    ordered = sorted(evidences, key=lambda item: (item.start, item.end))
    total_length = float(sum(ev.length for ev in ordered))
    span = float(ordered[-1].end - ordered[0].start + 1)
    continuity_degree = total_length / max(span, 1.0)
    fragmentation_degree = 1.0 - continuity_degree
    return continuity_degree, fragmentation_degree


def build_type_distribution(evidences: Sequence[SegmentEvidence], num_types: int) -> dict[str, dict[str, float]]:
    counts = [0 for _ in range(num_types)]
    for ev in evidences:
        if 0 <= ev.anomaly_type < num_types:
            counts[ev.anomaly_type] += 1

    total = max(len(evidences), 1)
    distribution = {}
    for type_idx in range(num_types):
        type_name = type_name_from_id(type_idx)
        distribution[type_name] = {
            "count": int(counts[type_idx]),
            "ratio": float(counts[type_idx] / total),
        }
    return distribution


# 这里直接复用训练时的跨模态一致性形式，用 MSE 度量 text / image 证据嵌入分歧。
def compute_text_image_divergence(outputs: dict[str, torch.Tensor], batch_index: int) -> float:
    text_evidence = outputs["text_evidence"][batch_index]
    image_evidence = outputs["image_evidence"][batch_index]
    return float(torch.mean((text_evidence - image_evidence) ** 2).item())


# 从段级证据 E 提取固定顺序的摘要向量 q = phi(E)，便于后续 agent 或规则层直接消费。
def build_summary(
    evidences: Sequence[SegmentEvidence],
    outputs: dict[str, torch.Tensor],
    batch_index: int,
    num_types: int,
) -> tuple[dict, list[float]]:
    anomaly_count = len(evidences)
    average_segment_length = float(sum(ev.length for ev in evidences) / anomaly_count) if evidences else 0.0
    continuity_degree, fragmentation_degree = compute_continuity_features(evidences)
    max_confidence = float(max((ev.confidence for ev in evidences), default=0.0))
    text_image_divergence = compute_text_image_divergence(outputs, batch_index=batch_index)
    type_distribution = build_type_distribution(evidences, num_types=num_types)

    type_ratio_vector = [
        float(type_distribution[type_name_from_id(type_idx)]["ratio"])
        for type_idx in range(num_types)
    ]
    summary_vector = [
        float(anomaly_count),
        average_segment_length,
        *type_ratio_vector,
        continuity_degree,
        fragmentation_degree,
        max_confidence,
        text_image_divergence,
    ]

    summary = {
        "anomaly_count": int(anomaly_count),
        "average_segment_length": average_segment_length,
        "type_distribution": type_distribution,
        "continuity_degree": continuity_degree,
        "fragmentation_degree": fragmentation_degree,
        "max_confidence": max_confidence,
        "text_image_divergence": text_image_divergence,
    }
    return summary, summary_vector


def build_evidence_package(
    sample_id: str,
    evidences: Sequence[SegmentEvidence],
    outputs: dict[str, torch.Tensor],
    batch_index: int,
    num_types: int,
) -> dict:
    evidence_list = build_evidence_list(evidences)
    summary, summary_vector = build_summary(
        evidences,
        outputs=outputs,
        batch_index=batch_index,
        num_types=num_types,
    )
    return {
        "id": sample_id,
        "evidence": evidence_list,
        "summary": summary,
        "q": summary_vector,
    }
