from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


TransformKind = Literal["scale", "shift", "resample", "style", "smooth"]


@dataclass
class TransformMetadata:
    """记录变换类型与时间坐标映射信息。"""

    kind: TransformKind
    source_length: int
    target_length: int

    @property
    def changes_time_axis(self) -> bool:
        return self.source_length != self.target_length

    def source_to_target_coordinate(self, index: float) -> float:
        return linear_time_coordinate_mapping(index, self.source_length, self.target_length)

    def target_to_source_coordinate(self, index: float) -> float:
        return linear_time_coordinate_mapping(index, self.target_length, self.source_length)


# 0-based 线性时间坐标映射；与文档中的 1-based alpha_tau / alpha_tau^{-1} 等价。
def linear_time_coordinate_mapping(index: float, source_length: int, target_length: int) -> float:
    if source_length <= 1 or target_length <= 1:
        return 0.0
    if source_length == target_length:
        return float(index)
    scale = float(target_length - 1) / float(source_length - 1)
    mapped = float(index) * scale
    return float(np.clip(mapped, 0.0, max(float(target_length - 1), 0.0)))


# 线性插值重采样，并返回新序列长度。
def _resample_linear(series: np.ndarray, new_length: int) -> np.ndarray:
    old_index = np.linspace(0.0, 1.0, num=len(series), dtype=np.float32)
    new_index = np.linspace(0.0, 1.0, num=new_length, dtype=np.float32)
    return np.interp(new_index, old_index, series).astype(np.float32)


# 温和平滑，对异常语义影响较小，但能测试边界稳定性。
def _mild_smooth(series: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    assert kernel_size % 2 == 1, "平滑核大小必须是奇数。"
    pad = kernel_size // 2
    padded = np.pad(series, (pad, pad), mode="edge")
    kernel = np.ones(kernel_size, dtype=np.float32) / kernel_size
    return np.convolve(padded, kernel, mode="valid").astype(np.float32)


# 图像风格扰动：这里只做轻量亮度和少量噪声扰动。
def apply_style_perturbation(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    brightness = rng.uniform(0.9, 1.1)
    noise = rng.normal(0.0, 0.02, size=image.shape).astype(np.float32)
    perturbed = np.clip(image * brightness + noise, 0.0, 1.0)
    return perturbed.astype(np.float32)


# 从变换族中采样一个变换，供 boundary-consistent training 使用。
def sample_semantics_preserving_transform(
    series: np.ndarray,
    rng: np.random.Generator,
    resample_ratio_range: tuple[float, float] = (0.9, 1.1),
) -> tuple[np.ndarray, TransformMetadata]:
    kind: TransformKind = rng.choice(["scale", "shift", "resample", "style", "smooth"])
    length = len(series)

    if kind == "scale":
        factor = rng.uniform(0.85, 1.15)
        transformed = (series * factor).astype(np.float32)
        return transformed, TransformMetadata(kind=kind, source_length=length, target_length=length)

    if kind == "shift":
        bias = rng.uniform(-0.25, 0.25)
        transformed = (series + bias).astype(np.float32)
        return transformed, TransformMetadata(kind=kind, source_length=length, target_length=length)

    if kind == "resample":
        ratio = rng.uniform(*resample_ratio_range)
        new_length = max(8, int(round(length * ratio)))
        transformed = _resample_linear(series, new_length)
        return transformed, TransformMetadata(kind=kind, source_length=length, target_length=new_length)

    if kind == "style":
        transformed = series.astype(np.float32)
        return transformed, TransformMetadata(kind=kind, source_length=length, target_length=length)

    transformed = _mild_smooth(series.astype(np.float32), kernel_size=5)
    return transformed, TransformMetadata(kind=kind, source_length=length, target_length=length)
