# TS Grounder

A lightweight training pipeline for multimodal time-series anomaly grounding.

This repository implements the **grounder only**. It does not include the agent layer, rule orchestration, or downstream decision modules.

## Features

- Preprocessing with scaling, indexing, de-seasonalization, and multi-view construction
- Lightweight multimodal grounder with text encoder, image encoder, and gated fusion
- Multi-head prediction for point-wise anomaly scores, boundaries, and anomaly types
- Boundary-consistent training under semantic-preserving perturbations
- Offline training and offline inference pipeline

## Project Structure

```text
ts_grounder/
├── configs/
│   └── default.yaml
├── dataset/
│   └── anomaly_db_v1/
├── scripts/
│   └── smoke_test.py
├── src/
│   └── ts_grounder/
│       ├── data/
│       │   ├── augmentations.py
│       │   ├── dataset.py
│       │   └── preprocessing.py
│       ├── models/
│       │   ├── encoders.py
│       │   └── grounder.py
│       ├── losses.py
│       ├── metrics.py
│       ├── trainer.py
│       └── utils.py
├── tools/
│   └── make_toy_dataset.py
├── predict.py
├── train.py
└── requirements.txt
```

## Method Overview

The current implementation follows this pipeline:

1. **Preprocessing**

   * z-score scaling
   * normalized positional indexing
   * de-seasonalization with ACF-based period estimation
   * two input views: time-series text features and rendered image features

2. **Grounder**

   * text encoder for fine-grained temporal modeling
   * image encoder for global pattern understanding
   * gated fusion between text and image features
   * prediction heads for anomaly mask, boundary maps, and type classification

3. **Boundary-Consistent Training**

   * perturbations: scale, shift, resample, style, smooth
   * transformed predictions are aligned back to the original time axis with `alpha_tau^{-1}`
   * consistency enforced at both point-mask and interval-boundary levels

4. **Outputs**

   * point-wise anomaly scores
   * start and end boundary maps
   * anomaly type predictions
   * structured anomaly evidence

## Repository Data Layout

- `dataset/` stores raw PatternFinder exports moved from `../PatternFinder-main/datasets/`.
- `dataset/anomaly_db_v1/` keeps the original JSON splits, PNG renders, manifest, and checksums.
- `data/` is reserved for converted `.npz` files that `train.py` and `predict.py` consume directly.
- Anomaly types follow the PatternFinder-aligned order: `point`, `freq`, `trend`, `range`.

## Data Format

Each sample should be stored as an `.npz` file with the following fields:

```python
series:   float32 [T]
mask:     float32 [T]
segments: int64   [K, 2]
types:    int64   [K]
```

Where:

* `series` is the raw time series
* `mask` is the point-wise anomaly label
* `segments` contains anomaly intervals as `[start, end]`
* `types` uses the PatternFinder-aligned convention:

  * `0`: point
  * `1`: freq
  * `2`: trend
  * `3`: range

Recommended layout:

```text
data/
├── train/
│   ├── sample_0000.npz
│   ├── sample_0001.npz
│   └── ...
└── val/
    ├── sample_0000.npz
    ├── sample_0001.npz
    └── ...
```

The raw PatternFinder dataset is not loaded directly by the current `GrounderDataset`.
It should stay under `dataset/`, and be converted into the `.npz` layout above before training.

For normal samples without anomalies:

* `segments.shape == (0, 2)`
* `types.shape == (0,)`
* `mask` is all zeros

## Installation

```bash
cd ts_grounder
pip install -r requirements.txt
export PYTHONPATH=./src
```

For Windows PowerShell:

```powershell
$env:PYTHONPATH = "./src"
```

## Quick Start

### 1. Generate toy data

```bash
python tools/make_toy_dataset.py --output data --train_size 128 --val_size 32 --length 256
```

### 2. Train

```bash
python train.py --config configs/default.yaml
```

Outputs are saved to:

```text
outputs/default_run/
├── best.pt
├── last.pt
└── history.json
```

### 3. Run a smoke test

```bash
python scripts/smoke_test.py
```

This script generates a small toy dataset, runs one training epoch, and checks that the full pipeline works end to end.

## Inference

Run inference on a directory:

```bash
python predict.py \
  --config configs/default.yaml \
  --checkpoint outputs/default_run/best.pt \
  --input data/val \
  --output predictions.json
```

Run inference on a single sample:

```bash
python predict.py \
  --config configs/default.yaml \
  --checkpoint outputs/default_run/best.pt \
  --input data/val/sample_0000.npz \
  --output one_sample.json
```

Example output:

```json
[
  {
    "id": "sample_0000",
    "evidence": [
      {
        "start": 81,
        "end": 95,
        "type": 2,
        "confidence": 0.86,
        "attributes": {
          "length": 15,
          "peak_amplitude": 2.14,
          "mean_deviation": 1.22
        }
      }
    ]
  }
]
```

## Configuration

Main options in `configs/default.yaml`:

```yaml
data:
  train_dir: data/train
  val_dir: data/val

model:
  hidden_dim: 128
  text_layers: 2
  num_types: 4

train:
  epochs: 30
  batch_size: 8
  lr: 0.0003

loss_weights:
  point: 1.0
  seg: 1.0
  type: 1.0
  bc: 1.0
  cons: 0.2
  tv: 0.05
```

## Training Objective

The current implementation includes:

* `L_point`: BCE loss for point-wise anomaly prediction
* `L_seg`: BCE loss on start/end boundary maps
* `L_type`: cross-entropy loss for anomaly type prediction
* `L_bc`: boundary consistency loss
* `L_cons`: consistency loss between text and image evidence
* `L_tv`: temporal total variation regularization

`L_seg` is implemented as boundary-map supervision in the current codebase. This matches the start/end heads and keeps optimization stable. Replacing it with interval L1 or IoU only requires modifying `losses.py`.

`L_bc` is decomposed as:

* `L_bc-mask`: `L1` distance between the original point mask and the transformed mask warped back by `alpha_tau^{-1}`
* `L_bc-int`: `L1` distance between predicted start/end coordinates after mapping transformed intervals back to the original time axis

For `scale`, `shift`, `style`, and `smooth`, `alpha_tau` is identity. For `resample`, the implementation uses the linear endpoint-preserving mapping described in the thesis notes.

## Notes

* This repository intentionally does **not** include fallback logic.
* Invalid inputs are expected to raise errors directly.
* The current codebase is designed for **grounder training only**.

## Extension Points

Likely places to modify:

* `src/ts_grounder/models/encoders.py`

  * replace BiGRU with a Transformer-based text encoder
* `ImageEncoder`

  * swap in a stronger visual backbone
* `src/ts_grounder/losses.py`

  * replace boundary-map supervision with interval regression or IoU
* `Grounder.decode(...)`

  * add richer anomaly attributes

## Scope

Included in this repository:

* preprocessing
* multimodal grounder
* training losses
* offline training
* offline inference

Not included:

* agent modules
* reviewer / planner / validator
* context-aware filtering
* action decision layer
* online serving system

## Summary

TS Grounder is a compact baseline for the **perception layer** of the proposed framework. It takes raw time series as input, builds text and image views, predicts anomaly evidence, and improves boundary stability through consistency training.
