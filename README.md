# TS Grounder

A lightweight training pipeline for the multimodal time-series anomaly grounding module.

This repository implements the **grounder / perception module only**. It does not include the agent layer, rule orchestration, retrieval, or downstream decision modules in the larger system.

In the current codebase, some internal names still use `text_features` or `text encoder`. In the system-level terminology, these refer to the **temporal branch** derived from the time series itself, not to natural-language text.

## System Role

Within the broader system, TS Grounder is responsible for:

- consuming aligned time-series samples plus their bound PNG views
- predicting anomaly existence, boundaries, and anomaly type
- emitting structured anomaly evidence for downstream modules
- serving as the perception-stage baseline for `with BC` and `no-BC` training ablations

## Features

- Temporal preprocessing with scaling, indexing, de-seasonalization, and multi-view construction
- Lightweight multimodal grounder with temporal encoder, image encoder, and gated fusion
- Multi-head prediction for point-wise anomaly scores, boundaries, anomaly types, and evidence
- Boundary-consistent training under semantic-preserving perturbations
- Strict `with BC` vs `no-BC` experiment support for ablation studies
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
   * two input views: time-series temporal features and bound external PNG image features

2. **Grounder**

   * temporal encoder for fine-grained sequence modeling
   * image encoder for global pattern understanding
   * gated fusion between temporal and image features
   * prediction heads for anomaly mask, boundary maps, type classification, and evidence regression

3. **Training Regimes**

   * default baseline: boundary-consistent training with perturbations `scale`, `shift`, `resample`, `style`, `smooth`
   * strict `no-BC` ablation: same data, optimizer, and epochs, but the augmentation / boundary-consistency branch is disabled
   * transformed predictions are aligned back to the original time axis with `alpha_tau^{-1}`
   * consistency is enforced at both point-mask and interval-boundary levels when BC is enabled

4. **Outputs**

   * point-wise anomaly scores
   * start and end boundary maps
   * anomaly type predictions
   * structured anomaly evidence

## Repository Data Layout

- `dataset/` stores raw PatternFinder exports moved from `../PatternFinder-main/datasets/`.
- `dataset/anomaly_db_v1/` keeps the original JSON splits, PNG renders, manifest, and checksums.
- `data/` is the generic converted `.npz` layout used by local smoke / toy runs.
- `data_full/` is the converted `.npz` cache used by the full Slurm training scripts.
- `outputs/` stores run artifacts such as checkpoints, resolved configs, summaries, and training curves.
- Runtime training and inference read converted `.npz` samples plus bound PNG views from disk; if an external image is missing, the loader fails fast.
- Anomaly types follow the PatternFinder-aligned order: `point`, `freq`, `trend`, `range`.

## Data Format

Each sample should be stored as an `.npz` file with the following fields:

```python
series:   float32 [T]
mask:     float32 [T]
segments: int64   [K, 2]
types:    int64   [K]
```

Optional field:

```python
image_path: str scalar
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
* `image_path` optionally points to the bound PNG under `images_plain_768x384/`; when present, the loader resolves this path against `raw_dataset_root`

In practice, the runtime expects each sample to be paired with a PNG image either by:

* storing `image_path` in the `.npz`, or
* placing the image under `raw_dataset_root/images_plain_768x384/{split}/{stem}.png`

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

The raw dataset is not loaded directly as JSON by the current `GrounderDataset`.
Training expects converted `.npz` samples plus bound PNG files on disk.
For PatternFinder data, keep the raw export under `dataset/`, convert the JSON splits into `.npz`, and treat `dataset/ -> data_full/ -> outputs/` as the standard full-training data flow.

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

### 1. Run a lightweight smoke train

```bash
python scripts/run_smoke_train.py --output-root _smoke_run
```

This command generates a tiny toy dataset, writes a temporary smoke config, and runs a 1-epoch training job.
The smoke dataset now follows the same image pipeline as full training: it writes `.npz` files plus bound PNG views under `images_plain_768x384/`, then trains by reading those PNGs directly.
By default it also sets safer local runtime defaults through the script for `PYTHONPATH`, `PYTHONNOUSERSITE`, and `MPLCONFIGDIR`.

### 2. Generate toy data

```bash
python tools/make_toy_dataset.py --output data --train_size 128 --val_size 32 --length 256
```

This toy generator now writes both:

* `data/train/*.npz`, `data/val/*.npz`
* `data/images_plain_768x384/train/*.png`, `data/images_plain_768x384/val/*.png`

### 3. Train

```bash
python train.py --config configs/default.yaml
```

Outputs are saved to:

```text
outputs/default_run/
├── best.pt
├── last.pt
├── history.json
├── resolved_config.yaml
├── training_curves.png
└── summary.json
```

`summary.json` records the best epoch selected by `val_event_f1`, not by `val_loss`.

For cluster training, `run_full_train_gpu.sh` uses `configs/default.yaml` as the base config, applies dataset/output/runtime overrides from the Slurm environment, and saves the resolved run config as `train_full.yaml` inside that run's output directory.

### 3a. Full-system training modes

Baseline `with BC`:

```bash
sbatch run_full_train_gpu.sh
```

Strict `no-BC` ablation:

```bash
sbatch run_full_train_no_bc_gpu.sh
```

The two runs are aligned for comparison:

* same dataset
* same model size
* same optimizer and scheduler
* same number of epochs
* only BC-related behavior changes

For the strict `no-BC` run, `loss_weights.bc = 0.0` and the augmentation / boundary-consistency branch is disabled entirely.

You can also regenerate the plot for any existing run:

```bash
python scripts/plot_training_curves.py \
  --history outputs/default_run/history.json \
  --output outputs/default_run/training_curves.png
```

When BC is disabled, the default plot omits `train_bc` automatically. You can still force the behavior with `--include-bc` or `--exclude-bc`.

You can also override the BC weight directly from the main script:

```bash
sbatch --export=ALL,BC_WEIGHT=0.0,RUN_TAG=full_no_bc_custom run_full_train_gpu.sh
```

### 4. Run a smoke test

```bash
python scripts/smoke_test.py
```

This script generates a small toy dataset, runs one training epoch, and checks that the full pipeline works end to end.
If you only want a quicker training-only sanity check, use `scripts/run_smoke_train.py` instead.

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
  raw_dataset_root: dataset/anomaly_db_v1

model:
  hidden_dim: 128
  text_layers: 2
  num_types: 4

train:
  epochs: 200
  batch_size: 8
  lr: 0.0003

loss_weights:
  point: 1.0
  seg: 1.0
  type: 1.0
  evidence: 0.5
  bc: 1.0
  cons: 0.2
  tv: 0.05
```

Notes:

* `configs/default.yaml` is the base local config.
* Full Slurm training overrides dataset paths, output paths, worker count, and optional ablation weights at launch time.
* The strict `no-BC` script preserves all other settings and only removes the BC branch.

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

The image branch always uses the bound PNG on disk for the original sample.
In the augmentation branch, `scale`, `shift`, `resample`, and `smooth` reuse the same bound PNG, while `style` applies a lightweight brightness-and-noise perturbation to that PNG.

Important interpretation note for curves:

* when BC is enabled, `train_loss` includes the BC term but `val_loss` does not, because validation does not run the augmentation branch
* as a result, `train_loss` and `val_loss` are not directly comparable in magnitude
* the default model-selection metric is `val_event_f1`, which is the main comparison target for experiments

## Notes

* This repository intentionally does **not** include fallback image rendering logic in the training or inference runtime.
* Invalid inputs are expected to raise errors directly.
* The current codebase is designed for **grounder training only**.

## Extension Points

Likely places to modify:

* `src/ts_grounder/models/encoders.py`

  * replace the current 1D Conv temporal encoder with a Transformer-based temporal encoder
* `ImageEncoder`

  * swap in a stronger visual backbone
* `src/ts_grounder/losses.py`

  * replace boundary-map supervision with interval regression or IoU
* `train.py` / `run_full_train_gpu.sh`

  * change experiment protocol, loss weighting, or model-selection metric
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

TS Grounder is a compact baseline for the **perception / grounding layer** of the proposed framework. It takes raw time series plus aligned PNG views as input, builds temporal and image representations, predicts anomaly evidence, and supports both `with BC` and strict `no-BC` experiments for system-level comparison.
