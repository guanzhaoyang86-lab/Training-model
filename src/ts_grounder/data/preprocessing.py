from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf


@dataclass
class PreprocessOutput:
    """单条时间序列经过预处理后的结果。"""

    original_series: np.ndarray
    scaled_series: np.ndarray
    deseasonalized_series: np.ndarray
    seasonal_component: np.ndarray
    text_features: np.ndarray
    period: int
    scale_mean: float
    scale_std: float


# 标准化缩放，落实方法里的 scaling。
def zscore_scale(
    series: np.ndarray,
    eps: float = 1e-6,
    mean: float | None = None,
    std: float | None = None,
) -> tuple[np.ndarray, float, float]:
    resolved_mean = float(series.mean()) if mean is None else float(mean)
    resolved_std = float(series.std()) if std is None else float(std)
    scaled = (series - resolved_mean) / (resolved_std + eps)
    return scaled.astype(np.float32), resolved_mean, resolved_std


# 轻微重采样后，把原时间轴上的周期投影到新长度，避免增强分支重新估周期引入额外噪声。
def project_period_to_length(period: int, source_length: int, target_length: int) -> int:
    projected = int(round(period * target_length / source_length))
    upper_bound = max(target_length // 2, 2)
    return int(np.clip(projected, 2, upper_bound))


# 利用 ACF 自动发现主周期，用于去季节化。
def infer_period_via_acf(series: np.ndarray, min_period: int = 2, max_period: int | None = None) -> int:
    length = len(series)
    if max_period is None:
        max_period = max(min(length // 2, 256), min_period + 1)
    if length < 2 * min_period:
        raise ValueError("序列长度过短，无法进行基于 ACF 的周期估计。")
    acf_values = acf(series, nlags=max_period, fft=True)
    candidate = acf_values[min_period:]
    return int(np.argmax(candidate) + min_period)


# 经典加性分解，只去掉 seasonality，保留 trend 与 residual。
def deseasonalize(series: np.ndarray, period: int) -> tuple[np.ndarray, np.ndarray]:
    if len(series) < 2 * period:
        raise ValueError("序列长度必须至少为 2 * period，才能执行 seasonal_decompose。")
    result = seasonal_decompose(series, period=period, model="additive", extrapolate_trend="freq")
    seasonal = np.asarray(result.seasonal, dtype=np.float32)
    deseasonalized = np.asarray(series - seasonal, dtype=np.float32)
    return deseasonalized, seasonal


# 构造文本视图：索引、去季节值、一阶差分、局部 z-score。
def build_text_features(series: np.ndarray, local_window: int = 25) -> np.ndarray:
    t = np.arange(len(series), dtype=np.float32)
    t = t / max(len(series) - 1, 1)

    delta = np.zeros_like(series, dtype=np.float32)
    delta[1:] = series[1:] - series[:-1]

    local_z = np.zeros_like(series, dtype=np.float32)
    half = max(local_window // 2, 1)
    for i in range(len(series)):
        left = max(0, i - half)
        right = min(len(series), i + half + 1)
        window = series[left:right]
        local_z[i] = (series[i] - window.mean()) / (window.std() + 1e-6)

    features = np.stack([t, series.astype(np.float32), delta, local_z], axis=-1)
    return features.astype(np.float32)


# 对单条序列执行完整预处理。
def preprocess_series(
    series: np.ndarray,
    period: int | None = None,
    local_window: int = 25,
    scale_mean: float | None = None,
    scale_std: float | None = None,
) -> PreprocessOutput:
    series = np.asarray(series, dtype=np.float32)
    scaled, resolved_mean, resolved_std = zscore_scale(series, mean=scale_mean, std=scale_std)
    resolved_period = infer_period_via_acf(scaled) if period is None else period
    deseasoned, seasonal = deseasonalize(scaled, resolved_period)
    text_features = build_text_features(deseasoned, local_window=local_window)
    return PreprocessOutput(
        original_series=series,
        scaled_series=scaled.astype(np.float32),
        deseasonalized_series=deseasoned.astype(np.float32),
        seasonal_component=seasonal.astype(np.float32),
        text_features=text_features,
        period=resolved_period,
        scale_mean=resolved_mean,
        scale_std=resolved_std,
    )
