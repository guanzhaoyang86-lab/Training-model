from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from ts_grounder.taxonomy import TYPE_NAMES, TYPE_TO_ID


def render_series_image(
    series: np.ndarray,
    size: tuple[int, int] = (768, 384),
    line_width: float = 1.4,
    add_grid: bool = True,
) -> np.ndarray:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8, 4), dpi=96)
    ax = fig.add_subplot(111)
    ax.plot(series, linewidth=line_width)
    if add_grid:
        ax.grid(True, alpha=0.25)
    ax.set_xlim(0, len(series) - 1)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout(pad=0.2)

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    buffer.seek(0)
    image = Image.open(buffer).convert("L").resize(size)
    image_np = np.asarray(image, dtype=np.float32) / 255.0
    return image_np[None, ...]


# 生成一个基础周期序列，后面往里面注入不同类型的异常。
def make_base_series(length: int, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(length, dtype=np.float32)
    season = np.sin(2.0 * np.pi * t / 48.0)
    trend = 0.002 * t
    noise = rng.normal(0.0, 0.08, size=length).astype(np.float32)
    return (season + trend + noise).astype(np.float32)


# 按 PatternFinder 的顺序注入 point / freq / trend / range 四类异常。
def inject_anomaly(series: np.ndarray, anomaly_type: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    length = len(series)
    start = int(rng.integers(40, length - 80))
    duration = int(rng.integers(8, 32))
    end = min(length - 1, start + duration - 1)
    out = series.copy()

    if anomaly_type == TYPE_TO_ID["point"]:
        pos = int(rng.integers(start, end + 1))
        out[pos] += float(rng.uniform(2.5, 3.5))
        return out, np.array([[pos, pos]], dtype=np.int64)

    if anomaly_type == TYPE_TO_ID["freq"]:
        # freq anomaly：只在局部区域提高频率。
        local_t = np.arange(end - start + 1, dtype=np.float32)
        out[start : end + 1] += 0.8 * np.sin(2.0 * np.pi * local_t / 8.0)
        return out, np.array([[start, end]], dtype=np.int64)

    if anomaly_type == TYPE_TO_ID["trend"]:
        ramp = np.linspace(0.0, float(rng.uniform(1.5, 3.0)), end - start + 1, dtype=np.float32)
        out[start : end + 1] += ramp
        return out, np.array([[start, end]], dtype=np.int64)

    if anomaly_type == TYPE_TO_ID["range"]:
        out[start : end + 1] += float(rng.uniform(1.5, 2.2))
        return out, np.array([[start, end]], dtype=np.int64)

    raise ValueError(f"Unsupported anomaly type id: {anomaly_type}")


# 构造单个样本文件。
def make_sample(length: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    base = make_base_series(length, rng)
    anomaly_type = int(rng.integers(0, len(TYPE_NAMES)))
    series, segments = inject_anomaly(base, anomaly_type, rng)
    mask = np.zeros(length, dtype=np.float32)
    for s, e in segments:
        mask[s : e + 1] = 1.0
    types = np.array([anomaly_type], dtype=np.int64)
    return series.astype(np.float32), mask, segments, types


# 生成 train / val 两个目录。
def main() -> None:
    parser = argparse.ArgumentParser(description="Create toy dataset for TS Grounder")
    parser.add_argument("--output", type=str, default="data")
    parser.add_argument("--train_size", type=int, default=128)
    parser.add_argument("--val_size", type=int, default=32)
    parser.add_argument("--length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    output_root = Path(args.output)
    train_dir = output_root / "train"
    val_dir = output_root / "val"
    image_root = output_root / "images_plain_768x384"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    (image_root / "train").mkdir(parents=True, exist_ok=True)
    (image_root / "val").mkdir(parents=True, exist_ok=True)

    for split_name, split_dir, count in [("train", train_dir, args.train_size), ("val", val_dir, args.val_size)]:
        for idx in range(count):
            series, mask, segments, types = make_sample(args.length, rng)
            sample_id = f"sample_{idx:04d}"
            image_rel_path = Path("images_plain_768x384") / split_name / f"{sample_id}.png"
            image_abs_path = output_root / image_rel_path
            image_np = render_series_image(series, size=(768, 384))
            image_uint8 = np.clip(image_np[0] * 255.0, 0.0, 255.0).astype(np.uint8)
            Image.fromarray(image_uint8, mode="L").save(image_abs_path)
            np.savez_compressed(
                split_dir / f"{sample_id}.npz",
                series=series,
                mask=mask,
                segments=segments,
                types=types,
                image_path=image_rel_path.as_posix(),
            )


if __name__ == "__main__":
    main()
