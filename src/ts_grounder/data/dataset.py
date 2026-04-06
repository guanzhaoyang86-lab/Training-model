from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .augmentations import TransformMetadata, apply_style_perturbation, sample_semantics_preserving_transform
from .preprocessing import preprocess_series, project_period_to_length


@dataclass
class SampleRecord:
    path: Path


# 训练数据里没有人工标注的 evidence score，这里用 GT 区间上的异常强度构造弱监督目标。
def build_segment_evidence_targets(
    segments: np.ndarray,
    deseasonalized: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    confidence = np.zeros((segments.shape[0],), dtype=np.float32)
    attributes = np.zeros((segments.shape[0], 2), dtype=np.float32)

    for idx, (start, end) in enumerate(segments):
        seg_series = deseasonalized[start : end + 1]
        peak_amplitude = float(np.max(np.abs(seg_series)))
        mean_deviation = float(np.mean(np.abs(seg_series)))

        # 用异常幅值和局部偏差构造 0~1 的软置信度标签，避免把所有 GT 段都压成常数 1。
        severity = 0.5 * peak_amplitude + 0.5 * mean_deviation
        confidence[idx] = float(1.0 - np.exp(-severity))
        attributes[idx, 0] = peak_amplitude
        attributes[idx, 1] = mean_deviation

    return confidence, attributes


# Grounder 的数据集格式约定：每个样本一个 .npz 文件。
class GrounderDataset(Dataset[Dict[str, Any]]):
    def __init__(
        self,
        root: str | Path,
        image_size: tuple[int, int] = (224, 224),
        local_window: int = 25,
        enable_aug: bool = True,
        seed: int = 42,
        raw_dataset_root: str | Path | None = None,
    ) -> None:
        self.root = Path(root)
        self.records = [SampleRecord(path=p) for p in sorted(self.root.glob("*.npz"))]
        self.image_size = image_size
        self.local_window = local_window
        self.enable_aug = enable_aug
        self.rng = np.random.default_rng(seed)
        self.raw_dataset_root = None if raw_dataset_root in (None, "") else Path(raw_dataset_root)
        if not self.records:
            raise ValueError(f"未在 {self.root} 下找到任何 .npz 样本。")

    def __len__(self) -> int:
        return len(self.records)

    @staticmethod
    def _read_optional_string(raw: np.lib.npyio.NpzFile, key: str) -> str | None:
        if key not in raw.files:
            return None
        value = raw[key]
        if np.asarray(value).size == 0:
            return None
        text = value.item()
        if isinstance(text, bytes):
            text = text.decode("utf-8")
        return str(text)

    def _resolve_external_image_path(self, record: SampleRecord, raw: np.lib.npyio.NpzFile) -> Path | None:
        stored_path = self._read_optional_string(raw, "image_path")
        if stored_path:
            candidate = Path(stored_path)
            if candidate.is_absolute():
                return candidate
            if self.raw_dataset_root is not None:
                return self.raw_dataset_root / candidate
            return None

        if self.raw_dataset_root is None:
            return None

        return self.raw_dataset_root / "images_plain_768x384" / record.path.parent.name / f"{record.path.stem}.png"

    def _load_image_view(self, record: SampleRecord, raw: np.lib.npyio.NpzFile) -> np.ndarray:
        external_path = self._resolve_external_image_path(record, raw)
        if external_path is None:
            raise FileNotFoundError(
                f"External image path is unresolved for sample {record.path.stem}. "
                "Set data.raw_dataset_root or store an absolute image_path in the .npz."
            )
        if not external_path.exists():
            raise FileNotFoundError(f"Missing external image for sample {record.path.stem}: {external_path}")
        image = Image.open(external_path).convert("L").resize(self.image_size)
        image_np = np.asarray(image, dtype=np.float32) / 255.0
        return image_np[None, ...]

    def __getitem__(self, index: int) -> Dict[str, Any]:
        record = self.records[index]
        raw = np.load(record.path, allow_pickle=False)
        series = raw["series"].astype(np.float32)
        mask = raw["mask"].astype(np.float32)
        segments = raw["segments"].astype(np.int64)
        types = raw["types"].astype(np.int64)

        processed = preprocess_series(
            series,
            local_window=self.local_window,
        )
        image = self._load_image_view(record, raw)

        start_target = np.zeros_like(mask, dtype=np.float32)
        end_target = np.zeros_like(mask, dtype=np.float32)
        for s, e in segments:
            start_target[s] = 1.0
            end_target[e] = 1.0
        evidence_confidence, evidence_attributes = build_segment_evidence_targets(
            segments,
            processed.deseasonalized_series,
        )

        item: Dict[str, Any] = {
            "id": record.path.stem,
            "series": series,
            "mask": mask,
            "segments": segments,
            "types": types,
            "text_features": processed.text_features,
            "image": image,
            "start_target": start_target,
            "end_target": end_target,
            "evidence_confidence": evidence_confidence,
            "evidence_attributes": evidence_attributes,
            "period": processed.period,
            "deseasonalized": processed.deseasonalized_series,
        }

        if self.enable_aug:
            aug_series, meta = sample_semantics_preserving_transform(series, self.rng)
            # 增强分支复用原样本的缩放锚点，避免 scale/shift 在重新标准化后被直接抵消。
            # 对 resample 额外投影周期长度，避免一致性损失学到的是预处理带来的周期漂移。
            aug_period = processed.period
            if meta.kind == "resample":
                aug_period = project_period_to_length(
                    period=processed.period,
                    source_length=meta.source_length,
                    target_length=meta.target_length,
                )
            aug_processed = preprocess_series(
                aug_series,
                period=aug_period,
                local_window=self.local_window,
                scale_mean=processed.scale_mean,
                scale_std=processed.scale_std,
            )
            # 运行时只使用外部 PNG；增强分支默认复用绑定图像，只在 style 分支上做轻量扰动。
            aug_image = image.copy()
            if meta.kind == "style":
                aug_image = apply_style_perturbation(aug_image, self.rng)
            item["aug_text_features"] = aug_processed.text_features
            item["aug_image"] = aug_image
            item["transform"] = meta
            item["aug_deseasonalized"] = aug_processed.deseasonalized_series

        return item


# 方便在 batch 维度上对变长序列做 padding。
def _pad_sequence(arrays: List[np.ndarray], pad_value: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(arr.shape[0] for arr in arrays)
    padded = []
    valid = []
    for arr in arrays:
        current_len = arr.shape[0]
        pad_width = [(0, max_len - current_len)] + [(0, 0)] * (arr.ndim - 1)
        padded_arr = np.pad(arr, pad_width, mode="constant", constant_values=pad_value)
        padded.append(padded_arr)
        valid_mask = np.zeros(max_len, dtype=np.float32)
        valid_mask[:current_len] = 1.0
        valid.append(valid_mask)
    return torch.tensor(np.stack(padded)), torch.tensor(np.stack(valid))


# 将不同样本的 segment 列表也 pad 到统一长度。
def _pad_segments(segments_list: List[np.ndarray], types_list: List[np.ndarray]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    max_segments = max(1, max(seg.shape[0] for seg in segments_list))
    padded_segments = []
    padded_types = []
    valid = []
    for segments, types in zip(segments_list, types_list):
        current = segments.shape[0]
        seg_source = segments if current > 0 else np.zeros((0, 2), dtype=np.int64)
        type_source = types if current > 0 else np.zeros((0,), dtype=np.int64)
        seg_pad = np.pad(seg_source, ((0, max_segments - current), (0, 0)), mode="constant")
        type_pad = np.pad(type_source, (0, max_segments - current), mode="constant")
        valid_pad = np.zeros(max_segments, dtype=np.float32)
        valid_pad[:current] = 1.0
        padded_segments.append(seg_pad)
        padded_types.append(type_pad)
        valid.append(valid_pad)
    return (
        torch.tensor(np.stack(padded_segments), dtype=torch.long),
        torch.tensor(np.stack(padded_types), dtype=torch.long),
        torch.tensor(np.stack(valid), dtype=torch.float32),
    )


# 将按 segment 组织的目标张量 pad 到统一长度，便于和 segment_valid 一起做监督。
def _pad_segment_targets(values_list: List[np.ndarray], max_segments: int) -> torch.Tensor:
    padded = []
    for values in values_list:
        pad_width = [(0, max_segments - values.shape[0])] + [(0, 0)] * (values.ndim - 1)
        padded_values = np.pad(values, pad_width, mode="constant", constant_values=0.0)
        padded.append(padded_values)
    return torch.tensor(np.stack(padded), dtype=torch.float32)


# DataLoader 的 collate_fn。
def grounder_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    text_features, valid_mask = _pad_sequence([sample["text_features"] for sample in batch])
    point_mask, _ = _pad_sequence([sample["mask"][:, None] for sample in batch])
    start_target, _ = _pad_sequence([sample["start_target"][:, None] for sample in batch])
    end_target, _ = _pad_sequence([sample["end_target"][:, None] for sample in batch])
    deseasonalized, _ = _pad_sequence([sample["deseasonalized"][:, None] for sample in batch])
    images = torch.tensor(np.stack([sample["image"] for sample in batch]), dtype=torch.float32)

    segments, types, segment_valid = _pad_segments(
        [sample["segments"] for sample in batch],
        [sample["types"] for sample in batch],
    )
    max_segments = int(segments.shape[1])
    evidence_confidence = _pad_segment_targets(
        [sample["evidence_confidence"] for sample in batch],
        max_segments=max_segments,
    )
    evidence_attributes = _pad_segment_targets(
        [sample["evidence_attributes"] for sample in batch],
        max_segments=max_segments,
    )

    collated: Dict[str, Any] = {
        "ids": [sample["id"] for sample in batch],
        "text_features": text_features.float(),
        "text_valid": valid_mask.float(),
        "point_mask": point_mask.squeeze(-1).float(),
        "start_target": start_target.squeeze(-1).float(),
        "end_target": end_target.squeeze(-1).float(),
        "images": images,
        "segments": segments,
        "types": types,
        "segment_valid": segment_valid,
        "evidence_confidence": evidence_confidence,
        "evidence_attributes": evidence_attributes,
        "deseasonalized": deseasonalized.squeeze(-1).float(),
    }

    if "aug_text_features" in batch[0]:
        aug_text, aug_valid = _pad_sequence([sample["aug_text_features"] for sample in batch])
        aug_deseasoned, _ = _pad_sequence([sample["aug_deseasonalized"][:, None] for sample in batch])
        aug_images = torch.tensor(np.stack([sample["aug_image"] for sample in batch]), dtype=torch.float32)
        transforms: List[TransformMetadata] = [sample["transform"] for sample in batch]
        collated.update(
            {
                "aug_text_features": aug_text.float(),
                "aug_text_valid": aug_valid.float(),
                "aug_images": aug_images,
                "aug_deseasonalized": aug_deseasoned.squeeze(-1).float(),
                "transforms": transforms,
            }
        )

    return collated
