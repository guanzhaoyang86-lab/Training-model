from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
import torch.nn.functional as F
from torch import nn

from .encoders import ImageEncoder, TextEncoder


@dataclass
class SegmentEvidence:
    start: int
    end: int
    anomaly_type: int
    confidence: float
    length: int
    peak_amplitude: float
    mean_deviation: float


# 轻量门控融合：text 主导定位，image 提供全局形态补充。
class GatedFusion(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )

    def forward(self, text_h: torch.Tensor, image_h: torch.Tensor) -> torch.Tensor:
        image_expanded = image_h.unsqueeze(1).expand(-1, text_h.size(1), -1)
        gate = self.gate(torch.cat([text_h, image_expanded], dim=-1))
        return gate * text_h + (1.0 - gate) * image_expanded


class Grounder(nn.Module):
    def __init__(
        self,
        input_dim: int = 4,
        hidden_dim: int = 128,
        num_types: int = 4,
        text_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.text_encoder = TextEncoder(
            in_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=text_layers,
            dropout=dropout,
        )
        self.image_encoder = ImageEncoder(in_channels=1, hidden_dim=hidden_dim)
        self.fusion = GatedFusion(hidden_dim=hidden_dim)

        self.point_head = nn.Linear(hidden_dim, 1)
        self.start_head = nn.Linear(hidden_dim, 1)
        self.end_head = nn.Linear(hidden_dim, 1)
        self.type_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_types),
        )
        self.confidence_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.attribute_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
        )

        self.text_evidence_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.image_evidence_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

    def forward(
        self,
        text_features: torch.Tensor,
        images: torch.Tensor,
        text_valid: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        text_h = self.text_encoder(text_features)
        image_h = self.image_encoder(images)
        fused_h = self.fusion(text_h, image_h)

        point_logits = self.point_head(fused_h).squeeze(-1)
        start_logits = self.start_head(fused_h).squeeze(-1)
        end_logits = self.end_head(fused_h).squeeze(-1)

        mask = text_valid > 0
        point_logits = point_logits.masked_fill(~mask, -1e4)
        start_logits = start_logits.masked_fill(~mask, -1e4)
        end_logits = end_logits.masked_fill(~mask, -1e4)

        pooled_text = self.masked_mean(text_h, text_valid)
        pooled_fused = self.masked_mean(fused_h, text_valid)

        return {
            "text_h": text_h,
            "image_h": image_h,
            "fused_h": fused_h,
            "point_logits": point_logits,
            "start_logits": start_logits,
            "end_logits": end_logits,
            "text_evidence": self.text_evidence_proj(pooled_text),
            "image_evidence": self.image_evidence_proj(image_h),
            "fused_evidence": pooled_fused,
        }

    @staticmethod
    def masked_mean(x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        denom = valid.sum(dim=1, keepdim=True).clamp_min(1.0)
        return (x * valid.unsqueeze(-1)).sum(dim=1) / denom

    # 基于给定区间池化特征，供 type loss 和 evidence head 使用。
    def pool_segments(self, fused_h: torch.Tensor, segments: torch.Tensor, segment_valid: torch.Tensor) -> torch.Tensor:
        batch_size, max_segments, _ = segments.shape
        hidden_dim = fused_h.size(-1)
        pooled = fused_h.new_zeros(batch_size, max_segments, hidden_dim)
        for b in range(batch_size):
            for k in range(max_segments):
                if segment_valid[b, k] <= 0:
                    continue
                start = int(segments[b, k, 0].item())
                end = int(segments[b, k, 1].item())
                pooled[b, k] = fused_h[b, start : end + 1].mean(dim=0)
        return pooled

    # 对区间特征做异常类型分类。
    def classify_segments(self, pooled_segments: torch.Tensor) -> torch.Tensor:
        return self.type_head(pooled_segments)

    # 显式的 evidence head：学习预测段级置信度和结构化属性，而不是只靠 decode 时手工统计。
    def predict_segment_evidence(self, pooled_segments: torch.Tensor) -> dict[str, torch.Tensor]:
        confidence_logits = self.confidence_head(pooled_segments).squeeze(-1)
        attribute_values = F.softplus(self.attribute_head(pooled_segments))
        return {
            "confidence_logits": confidence_logits,
            "attribute_values": attribute_values,
        }

    # 从 point/start/end 输出中解码区间，并生成结构化证据。
    @torch.no_grad()
    def decode(
        self,
        outputs: dict[str, torch.Tensor],
        deseasonalized: torch.Tensor,
        valid_mask: torch.Tensor,
        point_threshold: float = 0.5,
        min_segment_length: int = 1,
    ) -> List[List[SegmentEvidence]]:
        point_prob = torch.sigmoid(outputs["point_logits"])
        start_prob = torch.sigmoid(outputs["start_logits"])
        end_prob = torch.sigmoid(outputs["end_logits"])
        fused_h = outputs["fused_h"]

        results: List[List[SegmentEvidence]] = []
        batch_size = point_prob.size(0)
        for b in range(batch_size):
            current_valid = int(valid_mask[b].sum().item())
            binary = point_prob[b, :current_valid] >= point_threshold
            segments: List[tuple[int, int]] = []
            start = None
            for i, flag in enumerate(binary.tolist()):
                if flag and start is None:
                    start = i
                if (not flag) and start is not None:
                    if i - start >= min_segment_length:
                        segments.append((start, i - 1))
                    start = None
            if start is not None and current_valid - start >= min_segment_length:
                segments.append((start, current_valid - 1))

            evidences: List[SegmentEvidence] = []
            for s, e in segments:
                refined_s = int(torch.argmax(start_prob[b, s : e + 1]).item()) + s
                refined_e = int(torch.argmax(end_prob[b, s : e + 1]).item()) + s
                if refined_e < refined_s:
                    refined_s, refined_e = min(refined_s, refined_e), max(refined_s, refined_e)
                seg_feat = fused_h[b, refined_s : refined_e + 1].mean(dim=0, keepdim=True)
                type_logits = self.type_head(seg_feat)
                evidence_pred = self.predict_segment_evidence(seg_feat)
                anomaly_type = int(torch.argmax(type_logits, dim=-1).item())
                confidence = float(torch.sigmoid(evidence_pred["confidence_logits"]).item())
                peak_amplitude = float(evidence_pred["attribute_values"][0, 0].item())
                mean_deviation = float(evidence_pred["attribute_values"][0, 1].item())
                evidences.append(
                    SegmentEvidence(
                        start=refined_s,
                        end=refined_e,
                        anomaly_type=anomaly_type,
                        confidence=confidence,
                        length=refined_e - refined_s + 1,
                        peak_amplitude=peak_amplitude,
                        mean_deviation=mean_deviation,
                    )
                )
            results.append(evidences)
        return results
