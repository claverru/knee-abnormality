# 2026-10-12 middle-slice preparation and loader speed

Six changes to image preparation that leave every prepared image bit-identical. **Training steps go
from 3.05 to 4.27 per second on the measured host; on Kaggle at ×4 the model loop is 4% shorter and
peak GPU memory about 23% lower.** They edit hashed preprocessing sources, so checkpoints trained
before them refuse the new code ([what a candidate costs](../../backlog/untried-ideas.md#inference-loading)).

## The changes

All apply to `model.slice_channels: repeated_middle`, where the model reads only the middle slice of
each three-slice input. The neighbouring slices still set the per-series intensity bounds, so they
are still read and resampled where the bounds are taken. With `adjacent` slices every path prepares
all three channels as before.

| Change | File | Acts on |
|---|---|---|
| Loader normalises and stores only the middle slices, written straight into the final shared array | `data/preprocessing/device.py`, `data/dataset.py` | training, validation |
| Axis resampling filter built once per (source, target) size and sliced per crop | `data/preprocessing/spatial.py` | every path |
| Neighbour slices resampled only in the central window the intensity bounds are taken from | `data/preprocessing/device.py`, `spatial.py` | loader preparation |
| GPU preparation normalises and stores only the middle slices | `data/preprocessing/device.py`, `inference/engine.py` | inference |
| The four intensity ranks from two `topk` selections instead of a full sort | `data/preprocessing/device.py` | inference |
| The ×4 final prediction of a training asks the loader for middle slices | `scripts/pipeline/train.py` | end of each fold |

## Parity

Old code at `0bd598f` against the changes, main ResNet-50 recipe, fold 1 of `all_5fold_scanner_v2`:
every tensor and metadata field of every batch hashed and compared.

| Pass | Batches | Studies | Differing fields |
|---|---:|---:|---:|
| Training loader, epochs 0 and 5 | 442 each | 3,533 | 0 |
| Training loader with `adjacent` slices | 442 | 3,533 | 0 |
| Validation loader | 110 | 874 | 0 |
| ×4 prediction loader, three channels | 219 | 874 | 0 |
| ×4 prediction loader, middle only against the old middle channel | 219 | 874 | 0 |
| GPU preparation at ×4, three channels | 219 | 874 | 0 |
| GPU preparation at ×4, middle-slice images (fold 3 validation) | 221 | 882 | 0 |

- Predictions of the fold 3 checkpoint, old against new: mean absolute difference 6e-5, the same as
  two runs of the old code (5e-5; largest single difference between two old runs 0.0175).
- Forty training steps in each slice mode run on both versions with first losses equal to the third
  decimal.
- No series with a constant intensity range occurs in the data, so that branch was not exercised.

## Training speed

One rented host: RTX 5090, i7-14700K (8 fast and 12 small cores), 125 GB. Main ResNet-50 recipe,
batch 8, 110 steps under the real trainer with `scripts/tools/bench_training_steps.py`.

| | Old | New |
|---|---:|---:|
| Loader alone, 8 workers, batches/s | 3.4 | 5.75 |
| Loader alone, 16 workers, batches/s | 2.9 | 5.3 |
| Training steps/s, 16 workers, pinned memory | 3.05 | 4.27 |
| Validation wait for data per step, ms | 208 | 95 |

- **The loader was the limit, and memory bandwidth was the loader's limit.** The model alone runs
  4.97 steps/s on a repeated batch. The old loader delivered about 3 batches/s at 4, 8, 16 or 24
  workers, cold or cached, while one process alone needs 1 core-second per batch and sixteen
  together 5.6. On that host pure computation scales 15x at 16 processes but memory copying stays at
  about 9 GB/s in total, and preparing a study allocates about 100 MB.
- With the new code 8 and 16 workers are within 3% and pinned memory is slightly faster than
  unpinned (4.12 against 3.94 steps/s with the first four changes).
- With `adjacent` slices training speed is unchanged (3.04 against 3.13 steps/s).

## Kaggle

`profile_kaggle_inference`, condition `fp16-b4-w8`, 1,300 studies, two T4s, random-weight timing
bundles of the main ResNet-50 recipe. One pair of runs per row; hosts differ by about ±15 s on
identical code ([how to measure](loading-measurement-rules.md#separate-host-variance-from-the-change)).

| Multiplier | Version | Wall, s | Forward per GPU, s | Loader wait per GPU, s | Peak GiB | iowait |
|---|---|---:|---:|---:|---:|---:|
| ×4 | old | 312.6 | 254.2, 249.9 | 18.6, 18.1 | 3.93, 3.40 | 11.4% |
| ×4 | all six changes | 304.3 | 240.7, 241.8 | 17.6, 17.8 | 2.73, 2.89 | 11.7% |
| ×2 | old | 254.2 | 179.8, 179.7 | 38.9, 36.3 | 2.70, 2.62 | 9.8% |
| ×2 | first, second, fourth and sixth | 282.2 | 173.1, 179.0 | 71.2, 63.8 | 2.30, 2.27 | 18.4% |

- **×4:** the forward loop is 66 ms per batch shorter, 4.3%, with equal storage wait on both hosts.
  The 8 s on the wall is inside host variance.
- **×2:** the new arm landed on a host with twice the storage wait, 45 batches waiting over a second
  against 19; the median batch wait is equal (102 and 105 ms). The pair says nothing about the
  total.
- Predictions of the two bundles agree to 0.002.
- Outputs: `outputs/profiling/kaggle-loader-opt6-x4-{base,new}-20261012` and
  `outputs/profiling/kaggle-loader-opt-{base,new}-20261012`.
