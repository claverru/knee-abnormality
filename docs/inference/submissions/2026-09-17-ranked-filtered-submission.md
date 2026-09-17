# Ranked-filtered submission

The first submission of the series filtering and deterministic source ranking code
(preprocessing identity `ranked_filtered_physical_gaussian_overlap_spatial_tokens_reflection_v10`):
run `outputs/training/ranked-filtered-seed42`, epoch 10, **0.889573 fold-0 macro AUC**
([result](../../validation/results/2026-09-17-ranked-filtered-result.md)). It was exported on
2026-09-17 to `outputs/submission/exports/knee-ranked-filtered-seed42-20260917` (checkpoint
SHA-256 `194a19322e639219257f58111a3be9dbb39207bd894dac3049084250a3bb3d11`) with the default fast
runtime: 2× images, four studies per GPU, 16 total loader workers and automatic FP16 backbone
chunks on T4.

Competition submission **56303522 completed with public AUC 0.933**. The dataset is
`claverru/knee-ranked-filtered-seed42-20260917` and the notebook
`claverru/knee-ranked-filtered-seed42-20260917-inference`; the
[workflow record](../../../outputs/submission/records/knee-ranked-filtered-seed42-20260917.json)
holds the transitions. The dataset upload took 7 s, the visible notebook ran for 3m 32s, and
scoring took 9m 04s after submission (13m 16s from launch to score).

| Model | Fold-0 macro AUC | Public AUC |
|---|---:|---:|
| Ranked-filtered, seed 42 | 0.889573 | 0.933 |
| Seed 42 control (old code) | 0.892995 (0.892697 re-scored) | 0.928 |
| Seed 43 control (old code) | 0.889837 (0.889796 re-scored) | 0.931 |
| Seed 44 control (old code) | 0.891627 (0.891729 re-scored) | 0.931 |
| `00-baseline` (2026-09-15) | 0.892763 | 0.934 |

- **The new indexing, drops, guard and ranking ran on the full hidden test without failure.**
- The public score lies inside the 0.928–0.934 range of the same recipe on the old code, above
  all three seed controls; with the lowest fold-0 score of the four, fold-0 order again does not
  predict public order. One run cannot attribute the difference to the change.
